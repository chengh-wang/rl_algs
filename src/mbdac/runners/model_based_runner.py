"""RSL-RL-style public runner; real collection and imagined batches have independent sizes."""
import copy
from dataclasses import asdict
import csv
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from mbdac.algorithms import MBDAC, MBDACAMP


class InferencePolicy(nn.Module):
    def __init__(self, actor):
        super().__init__(); self.actor=actor; self.history=None; self.fresh=None

    def reset(self, dones=None):
        if dones is None: self.history=None; self.fresh=None
        else: self.fresh=dones.bool().clone()

    def forward(self, obs):
        o=obs['actor']
        if o.ndim==3: return self.actor(o,obs['goal'])
        if self.history is None or len(self.history)!=len(o): self.history=o[:,None].repeat(1,self.actor.history,1)
        else:
            self.history=torch.cat((self.history[:,1:],o[:,None]),1)
            if self.fresh is not None:
                self.history=torch.where(self.fresh[:,None,None],o[:,None].expand_as(self.history),self.history)
        self.fresh=None
        return self.actor(self.history,obs['goal'])


class ModelBasedRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        self.env=env; self.cfg=copy.deepcopy(train_cfg); self.device=device
        name=self.cfg.get('algorithm',{}).get('class_name','MBDAC').split(':')[-1]
        classes={'MBDAC':MBDAC,'MBDACAMP':MBDACAMP}
        self.alg=classes[name].construct_algorithm(env.get_observations(),env,self.cfg,device)
        self.cfg['algorithm']={'class_name':name,**asdict(self.alg.cfg)}
        self.current_learning_iteration=0; self.real_transitions=0; self._step_acc=0.; self.elapsed_s=0.
        self._active_start=None
        self.log_dir=Path(log_dir) if log_dir else None
        self.writer=self.wandb=None; self.log_file=None
        if self.log_dir:
            self.log_dir.mkdir(parents=True,exist_ok=True)
            (self.log_dir/'config.json').write_text(json.dumps(self.cfg,indent=2,default=str)+'\n')
            from mbdac.utils.provenance import runtime_provenance
            (self.log_dir/'runtime.json').write_text(json.dumps(runtime_provenance(),indent=2)+'\n')
            from torch.utils.tensorboard import SummaryWriter
            self.writer=SummaryWriter(str(self.log_dir))
            if self.cfg.get('logger')=='wandb':
                import wandb
                self.wandb=wandb.init(project=self.cfg.get('wandb_project','mbdac'),name=self.cfg.get('run_name'),dir=str(self.log_dir),config=self.cfg)
                (self.log_dir/'wandb.json').write_text(json.dumps({'id':self.wandb.id,'url':self.wandb.url})+'\n')

    def _sync(self):
        if str(self.device).startswith('cuda'): torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def _collect(self, n):
        logs={}; count={}
        for _ in range(n):
            obs=self.env.get_observations()
            action=self.alg.act(obs)
            obs,reward,dones,extras=self.env.step(action)
            self.alg.process_env_step(obs,reward,dones,extras)
            self.real_transitions+=self.env.num_envs
            logs['sim_reward']=logs.get('sim_reward',0.)+float(reward.mean())/n
            for k,v in extras.get('log',{}).items():
                value=float(v.float().mean()) if isinstance(v,torch.Tensor) else float(v)
                logs[k]=logs.get(k,0.)+value; count[k]=count.get(k,0)+1
        for k,nc in count.items(): logs[k]/=nc
        return logs

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            randomize=getattr(self.env,'randomize_episode_lengths',None)
            if randomize is None:
                raise ValueError('This task owns episode phases; its adapter must explicitly support random episode lengths')
            randomize()
        self.alg.train_mode(); self._active_start=time.perf_counter(); c=self.alg.cfg
        aggregated=[]
        try:
            needed=min(c.buffer_steps,max(c.warmup_steps,c.wm_window))
            while self.alg.replay.size<needed: self._collect(1)
            final=self.current_learning_iteration+num_learning_iterations
            for it in range(self.current_learning_iteration,final):
                self._step_acc+=1/c.utd; n=int(self._step_acc); self._step_acc-=n
                self._sync(); t=time.perf_counter(); logs=self._collect(n)
                self._sync(); logs['time/collect_s']=time.perf_counter()-t; t=time.perf_counter()
                logs.update(self.alg.update(it)); self._sync(); logs['time/update_s']=time.perf_counter()-t
                self.current_learning_iteration=it+1
                logs.update(iteration=it+1,real_transitions=self.real_transitions,wall_s=self._elapsed(),learning_rate=self.alg.learning_rate)
                if str(self.device).startswith('cuda'): logs['gpu/peak_allocated_bytes']=torch.cuda.max_memory_allocated(self.device)
                aggregated.append(logs)
                if (it+1)%self.cfg.get('log_interval',10)==0 or it+1==final:
                    mean={k:sum(row[k] for row in aggregated if k in row)/sum(k in row for row in aggregated) for k in set().union(*(r.keys() for r in aggregated))}
                    for k in ['iteration','real_transitions','wall_s']: mean[k]=logs[k]
                    self._log(mean); aggregated=[]
                if self.log_dir and (it+1)%self.cfg.get('save_interval',500)==0: self.save(str(self.log_dir/f'model_{it+1}.pt'))
            if self.log_dir: self.save(str(self.log_dir/f'model_{self.current_learning_iteration}.pt'))
        except BaseException:
            if self.log_dir: self.save(str(self.log_dir/'interrupted.pt'))
            raise
        finally:
            self.elapsed_s=self._elapsed(); self._active_start=None
            if self.writer: self.writer.flush()

    def _elapsed(self):
        return self.elapsed_s+(time.perf_counter()-self._active_start if self._active_start is not None else 0.)

    def _log(self, logs):
        print(json.dumps(logs,sort_keys=True),flush=True)
        if self.writer:
            for k,v in logs.items(): self.writer.add_scalar(k,v,self.current_learning_iteration)
        if self.wandb: self.wandb.log(logs,step=self.current_learning_iteration)
        if self.log_dir:
            with (self.log_dir/'metrics.jsonl').open('a') as out: out.write(json.dumps(logs)+'\n')

    def save(self, path, infos=None, include_replay=False):
        state=self.alg.save(include_replay)
        state.update(schema_version=1,iter=self.current_learning_iteration,real_transitions=self.real_transitions,elapsed_s=self._elapsed(),
                     step_acc=self._step_acc,cfg=self.cfg,infos=infos,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        if torch.cuda.is_available(): state['cuda_rng']=torch.cuda.get_rng_state_all()
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(state,str(path)+'.tmp'); os.replace(str(path)+'.tmp',path)
        if self.wandb: self.wandb.save(str(path.resolve()),base_path=str(self.log_dir.resolve()),policy='now')

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        state=torch.load(path,map_location=map_location or self.device,weights_only=False)
        if state.get('schema_version')!=1: raise ValueError('Unsupported checkpoint schema')
        load_iteration=self.alg.load(state,load_cfg,strict)
        if load_iteration:
            self.current_learning_iteration=state['iter']; self.real_transitions=state['real_transitions']
            self.elapsed_s=state['elapsed_s']; self._step_acc=state['step_acc']
        if load_cfg is None or load_cfg.get('rng',load_iteration):
            torch.set_rng_state(state['torch_rng'].cpu()); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
            if 'cuda_rng' in state and torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng'][:torch.cuda.device_count()]])
        return state.get('infos')

    def get_inference_policy(self, device=None):
        # Export/inference never changes the live training module's mode or device.
        return InferencePolicy(copy.deepcopy(self.alg.actor).to(device or self.device).eval())

    def export_jit(self,path):
        """Export a stateless CPU policy: (observation_history, goal) -> action.

        The caller maintains history and applies the environment's PD action scale.
        No simulator, task, critic or world model is needed at inference time.
        """
        actor=copy.deepcopy(self.alg.actor).cpu().eval(); actor.precision='fp32'
        c=self.alg.cfg; task=self.alg.task
        inputs=(torch.zeros(1,c.history,task.obs_dim),torch.zeros(1,task.goal_dim))
        with torch.no_grad(): exported=torch.jit.trace(actor,inputs)
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); exported.save(str(path))
        return exported

    def close(self):
        if self.writer: self.writer.close()
        if self.wandb: self.wandb.finish()
