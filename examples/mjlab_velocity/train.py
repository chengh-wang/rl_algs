"""Three-way cold-start native velocity benchmark entry point."""
import time
PROCESS_START=time.perf_counter()
import argparse
import dataclasses
from enum import Enum
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from mbdac.runners import ModelBasedRunner
from .environment import NativeVelocityEnv,VelocityAdapter
from .model_task import NativeVelocityTask
from .evaluation import TrainingBudgetReached,TrainingConverged


def stable_value(value):
    if dataclasses.is_dataclass(value):return {f.name:stable_value(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,dict):return {str(k):stable_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [stable_value(v) for v in value]
    if isinstance(value,Enum):return value.value
    if isinstance(value,slice):return {'slice':[value.start,value.stop,value.step]}
    if callable(value):return f'{value.__module__}:{value.__qualname__}'
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,Path):return str(value)
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    raise TypeError(f'Cannot fingerprint {type(value)}')


class BenchmarkRunner(ModelBasedRunner):
    observer=None
    def get_inference_policy(self,device=None):
        policy=super().get_inference_policy(device)
        policy.actor.precision='fp32'
        return policy
    def _log(self,logs):
        if self.alg.replay.size>=self.alg.cfg.wm_window:
            from mbdac.utils.diagnostics import world_model_diagnostics
            rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            try:logs.update(world_model_diagnostics(self.alg.world_model,self.alg.task,self.alg.replay,window=8,batch=128))
            finally:torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
        super()._log(logs)
        if self.observer:
            before=time.perf_counter()
            try:self.observer.observe(self.current_learning_iteration,self.real_transitions)
            finally:
                if self._active_start is not None:self._active_start+=time.perf_counter()-before
            if self.observer.converged:raise TrainingConverged()
            if self.observer.active_elapsed()>=self.cfg.get('max_train_seconds',7200):raise TrainingBudgetReached()


def training_config(task,args):
    algorithm=dict(class_name='MBDAC' if args.method=='mbdac' else 'MBDACFastSAC',
        history=1,horizon=args.horizon,compile=args.compile,compile_task=False,
        wm_batch=128,pi_batch=256,wm_only_iters=500,buffer_steps=256,
        action_clip=max(20.,float(task.action_bounds.max())))
    if args.method=='wm_fastsac':
        algorithm['fast_sac']=dict(action_scale=task.action_bounds.cpu().tolist(),
            critic_batch=8192,actor_lr=5e-5,critic_warmup_updates=800,
            action_std_min=.05,action_std_max=.8,precision='bf16',fused_optimizers=True,
            num_atoms=101,tau=.125,critic_updates=8,policy_frequency=4,target_entropy_ratio=0.,
            pretrain_critic_during_wm_warmup=True)
        algorithm['gamma']=.97
    if args.smoke:
        algorithm.update(wm_only_iters=0,warmup_steps=8,wm_batches=1,wm_window=4,
            buffer_steps=16,wm_hidden=(128,128),wm_batch=16,pi_batch=16,
            actor_hidden=(128,64),critic_hidden=(64,64),critic_epochs=1,critic_minibatches=1,precision='fp32')
        if args.method=='wm_fastsac':
            algorithm['fast_sac'].update(actor_hidden_dim=64,critic_hidden_dim=64,num_atoms=51,critic_batch=32,
                critic_updates=2,critic_warmup_updates=0,precision='fp32',fused_optimizers=False)
    return dict(algorithm=algorithm,logger=args.logger,wandb_project='mbdac-native-velocity',batch_scalar_logging=True,
        run_name=args.log_dir.name,save_interval=args.save_interval,log_interval=1 if args.smoke else 10,
        task=f'native_mjlab_{args.robot}_flat_velocity',robot=args.robot,seed=args.seed,num_envs=args.num_envs,
        initialized_from='scratch',physics_dt=task.env.physics_dt,control_dt=task.dt,
        evaluation_protocol='doc/native_velocity_benchmark.md',actor_history_frames=1,max_train_seconds=args.max_train_seconds)


def main():
    process_start=PROCESS_START
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=('g1','go1'),default='go1')
    p.add_argument('--method',choices=('mbdac','fastsac','wm_fastsac'),default='wm_fastsac')
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--iterations',type=int,default=5000)
    p.add_argument('--horizon',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-interval',type=int,default=250);p.add_argument('--compile',action='store_true')
    p.add_argument('--max-train-seconds',type=float,default=7200.,help='Active process budget, excluding evaluation')
    p.add_argument('--logger',choices=('wandb','tensorboard'),default='wandb')
    p.add_argument('--log-dir',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if (args.log_dir/'config.json').exists():p.error('Cold starts require a fresh log directory')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    env=NativeVelocityEnv(args.robot,args.num_envs,seed=args.seed);task=NativeVelocityTask(env);env.model_task=task
    args.log_dir.mkdir(parents=True,exist_ok=True)
    # Freeze the actually launched code, including uncommitted edits.
    import tarfile
    repo=Path(__file__).resolve().parents[2]
    with tarfile.open(args.log_dir/'source_snapshot.tar.gz','w:gz') as archive:
        for folder in ('src/mbdac','examples/mjlab_velocity'):
            for path in sorted((repo/folder).rglob('*.py')):archive.add(path,arcname=str(path.relative_to(repo)))
        for name in ('pyproject.toml','AGENTS.md','LICENSE','README.md','doc/native_velocity_benchmark.md',
                     'src/mbdac/vendor/holosoma/source.json','scripts/setup_native_velocity_baseline.sh'):
            archive.add(repo/name,arcname=name)
        for path in sorted((repo/'licenses').rglob('*')):
            if path.is_file():archive.add(path,arcname=str(path.relative_to(repo)))
    native_task=stable_value(env.cfg)
    native_task['rewards'].pop('_model_capture',None)
    task_json=json.dumps(native_task,sort_keys=True,default=str)
    (args.log_dir/'native_task.json').write_text(task_json+'\n')
    manifest=dict(method=args.method,robot=args.robot,initialization='scratch',seed=args.seed,
        task_sha256=hashlib.sha256(task_json.encode()).hexdigest(),process_start_unix=time.time()-(time.perf_counter()-process_start),
        setup_s=time.perf_counter()-process_start,
        source_snapshot_sha256=hashlib.sha256((args.log_dir/'source_snapshot.tar.gz').read_bytes()).hexdigest())
    (args.log_dir/'timing.json').write_text(json.dumps(manifest,indent=2)+'\n')
    try:
        if args.method=='fastsac':
            from .upstream import train_upstream
            train_upstream(env,task,args,manifest,process_start)
        else:
            from .evaluation import BenchmarkObserver
            runner=BenchmarkRunner(VelocityAdapter(env,task),training_config(task,args),str(args.log_dir),'cuda:0')
            observer=BenchmarkObserver(args,manifest,process_start,runner.get_inference_policy,runner.save,runner.wandb)
            runner.observer=observer
            try:
                observer.observe(0,0,force=True)
                try:runner.learn(args.iterations);manifest['status']='completed'
                except TrainingBudgetReached:manifest['status']='budget_reached'
                except TrainingConverged:manifest['status']='converged'
                observer.observe(runner.current_learning_iteration,runner.real_transitions,force=True)
            finally:
                manifest.setdefault('status','interrupted_or_error')
                manifest.update(process_elapsed_s=time.perf_counter()-process_start,training_s=runner._elapsed(),
                    evaluation_s=observer.evaluation_s,active_process_s=observer.active_elapsed(),
                    iteration=runner.current_learning_iteration,real_transitions=runner.real_transitions)
                (args.log_dir/'timing.json').write_text(json.dumps(manifest,indent=2)+'\n');runner.close()
    finally:env.close()


if __name__=='__main__':main()
