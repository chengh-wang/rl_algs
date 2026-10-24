"""Shared deterministic physical evaluation; preserves the learner's RNG state."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from .environment import NativeVelocityEnv

COMMANDS=((0.,0.,0.),(.8,0.,0.),(-.5,0.,0.),(0.,.5,0.),(0.,0.,.5))

class TrainingBudgetReached(Exception):pass
class TrainingConverged(Exception):pass


class EvaluationEnv(NativeVelocityEnv):
    def __init__(self,*args,**kwargs):
        self.allow_reset=True
        super().__init__(*args,evaluation=True,**kwargs)
        self.allow_reset=False
    def _reset_idx(self,ids):
        if self.allow_reset:super()._reset_idx(ids)


@torch.no_grad()
def evaluate(policy,robot,replicas=8,seconds=20.,seed=1701,trajectory=None):
    rng=(torch.get_rng_state(),torch.cuda.get_rng_state_all(),np.random.get_state(),random.getstate())
    start=time.perf_counter();e=None
    try:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        e=EvaluationEnv(robot,replicas*len(COMMANDS),seed=seed)
        command=torch.tensor(COMMANDS,device=e.device).repeat_interleave(replicas,dim=0)
        term=e.command_manager.get_term('twist');term.vel_command_b.copy_(command)
        e.obs_buf=e.observation_manager.compute(update_history=True)
        alive=torch.ones(e.num_envs,device=e.device,dtype=torch.bool)
        errors=[];yaw=[];returns=torch.zeros(e.num_envs,device=e.device);states=[]
        first_failure=torch.full((e.num_envs,),-1,device=e.device,dtype=torch.long)
        for step in range(round(seconds/e.step_dt)):
            obs=TensorDict({'actor':e.obs_buf['actor'],'goal':command},batch_size=[e.num_envs])
            action=policy(obs)
            _,reward,failed,_,_=e.step(action)
            returns+=reward*alive
            newly=alive&failed;first_failure[newly]=step
            alive&=~failed
            r=e.scene['robot'].data
            errors.append((r.root_link_lin_vel_b[:,:2]-command[:,:2]).square().sum(-1))
            yaw.append((r.root_link_ang_vel_b[:,2]-command[:,2]).square())
            if trajectory:states.append(e.sim.data.qpos.clone())
        start_index=round(2/e.step_dt)
        xy=torch.stack(errors)[start_index:].mean(0).sqrt();wr=torch.stack(yaw)[start_index:].mean(0).sqrt()
        per=[]
        for i,c in enumerate(COMMANDS):
            idx=slice(i*replicas,(i+1)*replicas)
            per.append(dict(command=c,xy_rmse_m_s=float(xy[idx].square().mean().sqrt()),
                yaw_rmse_rad_s=float(wr[idx].square().mean().sqrt()),survival=float(alive[idx].float().mean()),
                mean_return=float(returns[idx].mean())))
        result=dict(xy_rmse_m_s=float(xy.square().mean().sqrt()),yaw_rmse_rad_s=float(wr.square().mean().sqrt()),
            survival=float(alive.float().mean()),mean_return=float(returns.mean()),per_command=per,
            first_failure_steps=first_failure.cpu().tolist(),replicas=replicas,seconds=seconds,
            evaluation_s=time.perf_counter()-start)
        result['pass']=result['survival']==1. and result['xy_rmse_m_s']<=.20 and result['yaw_rmse_rad_s']<=.30
        result['all_command_pass']=all(p['survival']==1. and p['xy_rmse_m_s']<=.20 and p['yaw_rmse_rad_s']<=.30 for p in per)
        if trajectory:
            target=Path(trajectory);target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,qpos=torch.stack(states).cpu().numpy(),command=command.cpu().numpy(),
                                env_origins=e.scene.env_origins.cpu().numpy(),dt=e.step_dt)
        return result
    finally:
        if e:e.close()
        torch.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);np.random.set_state(rng[2]);random.setstate(rng[3])


class BenchmarkObserver:
    """Evaluate every 300 active seconds, plus initialization and the final policy."""
    def __init__(self,args,manifest,process_start,policy_factory,save,wandb_run=None):
        self.args=args;self.manifest=manifest;self.process_start=process_start
        self.policy_factory=policy_factory;self.save=save;self.wandb_run=wandb_run
        self.evaluation_s=0.;self.last_active=-float('inf');self.records=[]
        self.converged=False

    def active_elapsed(self):return time.perf_counter()-self.process_start-self.evaluation_s

    def observe(self,iteration,transitions,force=False):
        if force and self.records and self.records[-1]['iteration']==iteration:return
        if not force and self.active_elapsed()-self.last_active<300.:return
        pre=time.perf_counter();active=self.active_elapsed()
        result=evaluate(self.policy_factory(),self.args.robot,seconds=2.2 if self.args.smoke else 20.,replicas=1 if self.args.smoke else 8)
        result.update(iteration=iteration,real_transitions=transitions,active_process_s=active,
            process_elapsed_s=time.perf_counter()-self.process_start,method=self.args.method,robot=self.args.robot,
            task_sha256=self.manifest['task_sha256'])
        self.evaluation_s+=time.perf_counter()-pre;self.last_active=active;self.records.append(result)
        if len(self.records)>=2:
            recent=self.records[-2:]
            self.converged=all(r['all_command_pass'] for r in recent) and abs(recent[1]['mean_return']-recent[0]['mean_return'])/max(1.,abs(recent[0]['mean_return']))<.05
        path=self.args.log_dir/'evaluations.jsonl'
        with path.open('a') as f:f.write(json.dumps(result)+'\n')
        print('BENCHMARK_EVALUATION '+json.dumps(result),flush=True)
        self.save(self.args.log_dir/f'eval_model_{iteration}.pt')
        if self.wandb_run:
            metrics={f'evaluation/{k}':result[k] for k in ('xy_rmse_m_s','yaw_rmse_rad_s','survival','mean_return','active_process_s','real_transitions','pass')}
            # Same training iteration keeps the source learner's W&B step axis valid.
            self.wandb_run.log(metrics,step=iteration)


def upstream_policy(agent):
    actor=copy.deepcopy(agent.actor).eval();norm=copy.deepcopy(agent.obs_normalizer).eval()
    @torch.no_grad()
    def policy(obs):return actor(norm(obs['actor'],update=False))[0]
    return policy
