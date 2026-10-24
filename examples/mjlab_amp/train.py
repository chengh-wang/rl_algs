"""Run MBDAC/MBDACAMP against the retained AMP baseline's native velocity task."""
import argparse
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from mbdac.runners import ModelBasedRunner
from .environment import BaselineVelocityEnv,VelocityModelAdapter
from .model_task import BaselineVelocityTask


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--num-envs',type=int,default=1024); p.add_argument('--iterations',type=int,default=5000)
    p.add_argument('--algorithm',choices=['MBDAC','MBDACAMP'],default='MBDACAMP')
    p.add_argument('--log-dir',type=Path,default=Path('logs/velocity_mbdacamp_1024'))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--logger',choices=['wandb','tensorboard'],default='wandb')
    p.add_argument('--smoke',action='store_true'); p.add_argument('--resume'); p.add_argument('--compile',action='store_true')
    p.add_argument('--compile-task',action='store_true',help='Also compile analytic task heads (may hit Triton backward limitations)')
    p.add_argument('--prior',choices=['upstream','independent'],help='New runs reuse the upstream AMP modules; legacy resumes retain their prior')
    p.add_argument('--reset-prior',action='store_true',help='Resume actor/WM/critic while initializing a new discriminator')
    args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    native=BaselineVelocityEnv(args.num_envs,args.device,args.seed)
    prior=args.prior
    if prior is None:
        prior=torch.load(args.resume,map_location='cpu',weights_only=False)['cfg']['algorithm']['amp'].get('backend','independent') if args.resume else 'upstream'
    if prior=='upstream':
        from .upstream_prior import UpstreamVelocityTask
        task=UpstreamVelocityTask(native)
    else: task=BaselineVelocityTask(native)
    native.model_task=task; env=VelocityModelAdapter(native,task)
    config=dict(algorithm=dict(class_name=args.algorithm,history=4,actor_hidden=(512,256,128),critic_hidden=(512,256,128),
                              wm_batch=128,pi_batch=256,compile=args.compile,compile_task=args.compile_task,
                              amp=dict(weight=.25,reward_scale=.1,backend=prior,max_grad_norm=None if prior=='upstream' else 1.)),
                logger=args.logger,wandb_project='mbdac-velocity',run_name=args.log_dir.name,
                task='Unitree-G1-AMP-Flat',num_envs=args.num_envs,seed=args.seed,save_interval=500,log_interval=10,
                physical_heads=[asdict(spec) for spec in task.primitive_specs],
                task_source_commit='6c7a2947fccc973e4af8e6d90e550400f1b6fcfc')
    if prior=='upstream':
        from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
        config['algorithm']['amp']['lr']=g1_amp_ppo_runner_cfg().algorithm.learning_rate
    if args.smoke:
        config['algorithm'].update(wm_only_iters=0,warmup_steps=8,wm_batches=1,horizon=4,wm_window=4,history=3,
                                  buffer_steps=16,wm_batch=16,pi_batch=16,actor_hidden=(128,64),wm_hidden=(128,128),
                                  critic_hidden=(64,32),critic_epochs=1,critic_minibatches=1,precision='fp32')
        config['log_interval']=1
    runner=ModelBasedRunner(env,config,str(args.log_dir),args.device)
    if args.resume:
        load_cfg=None
        if args.reset_prior:
            load_cfg=dict(actor=True,critic=True,world_model=True,optimizer=True,replay=True,iteration=True,rng=True,prior=False)
        runner.load(args.resume,load_cfg=load_cfg)
    try: runner.learn(args.iterations,init_at_random_ep_len=True)
    finally: runner.close(); env.close()


if __name__=='__main__': main()
