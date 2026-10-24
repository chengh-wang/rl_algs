"""MJLab protocol and logging bridge; Holosoma's learner is inherited unchanged.

Pinned upstream d18d6cc. No physics SDK from Holosoma is constructed. Its original
replay, normalization, optimizer, C51, temperature, policy and update loops run.
"""
import copy
import json
import time
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
from holosoma.config_types.algo import FastSACConfig


class HolosomaBridge:
    def __init__(self,env,task):
        self.native=env;self.task=task;self.device=env.device;self.num_envs=env.num_envs
        names=task.action_joint_names; ids=env.scene['robot'].indexing.joint_ids[task.action_joint_ids].cpu().numpy()
        limits=env.sim.mj_model.jnt_range[ids]
        self.robot_config=NS(actions_dim=task.action_dim,dof_names=names,
            dof_pos_lower_limit_list=limits[:,0].tolist(),dof_pos_upper_limit_list=limits[:,1].tolist(),
            init_state=NS(default_joint_angles={n:float(v) for n,v in zip(names,task.default_q[task.action_joint_ids])}),
            control=NS(action_scale=task.pd_scale))
        self.observation_manager=NS(get_obs_dims=lambda:{'actor_obs':task.obs_dim,'critic_obs':task.critic_dim},
            cfg=NS(groups={k:NS(history_length=1,terms={k:None}) for k in ('actor_obs','critic_obs')}))
        self.episode_keys=tuple(env.extras.get('log',{}))

    @staticmethod
    def observations(obs):return {'actor_obs':obs['actor'],'critic_obs':obs['critic']}

    def reset_all(self):return self.observations(self.native.reset()[0])

    def step(self,action_dict):
        obs,r,term,trunc,extras=self.native.step(action_dict['actions'])
        tr=extras['model_transition']
        final_critic=tr['next_critic_features']
        logs={k:v for k,v in extras.get('log',{}).items() if k.startswith(('Episode_','Metrics/twist'))}
        info=dict(time_outs=trunc&~term,final_observations={'actor_obs':tr['terminal_actor_observation'],'critic_obs':final_critic},
            episode={k:logs.get(k,torch.empty(0,device=self.device)) for k in self.episode_keys},
            episode_all=logs,to_log=self.native.command_manager.get_term('twist').metrics)
        return self.observations(obs),r.clone(),(term|trunc).long(),info

    def get_checkpoint_state(self):return {}
    def load_checkpoint_state(self,state):
        if state:raise ValueError('Physical episodes reset on reload')


class BenchmarkFastSACAgent(FastSACAgent):
    """Only export/metadata/timing differ; no training method is overridden."""
    def save(self,path):
        super().save(str(path))
        elapsed=time.perf_counter()-self.process_start
        data=dict(self.manifest,process_elapsed_s=elapsed,global_step=self.global_step,
                  real_transitions=(self.global_step+1)*self.env.num_envs)
        Path(path).with_suffix('.timing.json').write_text(json.dumps(data,indent=2)+'\n')

    def export(self,onnx_file_path):
        # MJLab deployment export is separate; no training env reset for ONNX.
        return None


def train_upstream(env,task,args,manifest,process_start):
    # Upstream locomotion FastSAC defaults, symmetry disabled for all methods.
    cfg=FastSACConfig(num_learning_iterations=args.iterations,compile=args.compile,
        save_interval=args.save_interval,logging_interval=1 if args.smoke else 10,use_symmetry=False)
    if args.smoke:cfg=replace(cfg,learning_starts=1,num_updates=2,batch_size=32,buffer_size=16,
        actor_hidden_dim=64,critic_hidden_dim=64,num_atoms=51,amp=False,policy_frequency=2)
    record={**manifest,'algorithm':asdict(cfg),'source':'amazon-far/holosoma@d18d6cc50f872c15e904a22ceac22313cec955c8'}
    from mbdac.utils.provenance import runtime_provenance
    (args.log_dir/'runtime.json').write_text(json.dumps(runtime_provenance(),indent=2)+'\n')
    (args.log_dir/'config.json').write_text(json.dumps(record,indent=2)+'\n')
    run=None
    if args.logger=='wandb':
        import wandb
        run=wandb.init(project='mbdac-native-velocity',name=args.log_dir.name,dir=str(args.log_dir),config=record)
        (args.log_dir/'wandb.json').write_text(json.dumps(dict(id=run.id,url=run.url))+'\n')
    agent=BenchmarkFastSACAgent(HolosomaBridge(env,task),cfg,env.device,str(args.log_dir))
    agent.process_start=process_start;agent.manifest=manifest
    agent.attach_checkpoint_metadata(NS(to_serializable_dict=lambda:record))
    try:
        from .evaluation import BenchmarkObserver,upstream_policy,TrainingBudgetReached,TrainingConverged
        agent.setup()
        observer=BenchmarkObserver(args,manifest,process_start,lambda:upstream_policy(agent),agent.save,run)
        original_log=agent.logging_helper.post_epoch_logging
        def log_and_evaluate(*a,**kw):
            original_log(*a,**kw)
            observer.observe(agent.global_step,(agent.global_step+1)*env.num_envs)
            if observer.converged:raise TrainingConverged()
            if observer.active_elapsed()>=args.max_train_seconds:
                raise TrainingBudgetReached()
        agent.logging_helper.post_epoch_logging=log_and_evaluate
        observer.observe(0,0,force=True)
        try:agent.learn();manifest['status']='completed'
        except TrainingBudgetReached:
            manifest['status']='budget_reached';agent.save(args.log_dir/'budget_checkpoint.pt')
        except TrainingConverged:
            manifest['status']='converged';agent.save(args.log_dir/'converged_checkpoint.pt')
        observer.observe(agent.global_step,(agent.global_step+1)*env.num_envs,force=True)
    finally:
        manifest.setdefault('status','interrupted_or_error')
        manifest.update(process_elapsed_s=time.perf_counter()-process_start,global_step=agent.global_step,
                        real_transitions=(agent.global_step+1)*env.num_envs)
        (args.log_dir/'timing.json').write_text(json.dumps(manifest,indent=2)+'\n')
        agent.writer.close()
        if run:run.finish()
