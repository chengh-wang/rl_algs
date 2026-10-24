"""Physical contract checks against both unchanged native velocity tasks."""
import argparse
import json
from pathlib import Path
import torch
from .environment import NativeVelocityEnv,VelocityAdapter
from .model_task import NativeVelocityTask


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=('g1','go1'),default='go1')
    p.add_argument('--num-envs',type=int,default=16);p.add_argument('--steps',type=int,default=32)
    p.add_argument('--out',type=Path,default=Path('logs/native_velocity_contract.json'))
    args=p.parse_args();torch.set_num_threads(4)
    e=NativeVelocityEnv(args.robot,args.num_envs);task=NativeVelocityTask(e);e.model_task=task;e.audit=True
    adapter=VelocityAdapter(e,task);result=dict(robot=args.robot,num_envs=e.num_envs,physics_dt=e.physics_dt,control_dt=e.step_dt,
        obs_dim=task.obs_dim,critic_dim=task.critic_dim,state_dim=task.codec.state_dim,action_dim=task.action_dim,
        primitive_specs=[vars(s) for s in task.primitive_specs],formula_error={},end_state_reward_error={},
        observation_error=0.,fk_position_error_m=0.,fk_velocity_error_m_s=0.,boundary_count=0)
    try:
        for i in range(args.steps):
            action=torch.randn(e.num_envs,task.action_dim,device=e.device)*.2
            if i==args.steps-1:
                e.episode_length_buf[:e.num_envs//2]=e.max_episode_length-1
                e.sim.data.qpos[e.num_envs//2:,3:7]=torch.tensor([0.,1.,0.,0.],device=e.device)
                e.sim.forward()
            obs,r,d,extras=adapter.step(action);tr=extras['model_transition']
            assert torch.equal(action,tr['a'])
            labels={k.split('/',1)[1]:v for k,v in tr.items() if k.startswith('primitive/')}
            _,exact=task.reward(tr['s1'],tr['condition'],tr['a'],tr['a_prev'],labels,tr['s'],soft=False,overrides=e._captured['audit'],terms=True)
            _,end=task.reward(tr['s1'],tr['condition'],tr['a'],tr['a_prev'],labels,tr['s'],soft=False,terms=True)
            native={name:e._captured['native_terms'][:,j] for j,name in enumerate(task.reward_cfg)}
            for name in exact:
                result['formula_error'][name]=max(result['formula_error'].get(name,0.),float((exact[name]-native[name]).abs().max()))
                result['end_state_reward_error'][name]=max(result['end_state_reward_error'].get(name,0.),float((end[name]-native[name]).abs().max()))
            # Actor observations are refreshed after physics; compare without noise.
            state=e.physical_state();prediction=task.observation(state,e.action_manager.action,e.physical_context(),e.condition())
            e.observation_manager.cfg['actor'].enable_corruption=False
            for cfg in e.observation_manager._group_obs_term_cfgs['actor']: cfg.noise=None
            native_o=e.observation_manager.compute_group('actor',update_history=False)
            result['observation_error']=max(result['observation_error'],float((prediction-native_o).abs().max()))
            sites=task.kin.sites(task.geometry(state));data=e.scene['robot'].data
            result['fk_position_error_m']=max(result['fk_position_error_m'],float((sites[0]-(data.site_pos_w-e.scene.env_origins[:,None])).abs().max()))
            result['fk_velocity_error_m_s']=max(result['fk_velocity_error_m_s'],float((sites[2]-data.site_lin_vel_w).abs().max()))
            result['boundary_count']+=int(tr['model_boundary'].sum())
            if i==args.steps-1:
                result['timeouts']=int(tr['truncated'].sum());result['terminations']=int(tr['terminated'].sum())
                result['pre_reset_gap']=float((tr['s1'][d.bool()]-state[d.bool()]).abs().max()) if d.any() else 0.
        # Exact native functions, supplied the native lagged data, must agree.
        result['formula_pass']=max(result['formula_error'].values())<1e-5
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));assert result['formula_pass']
        assert result['observation_error']<1e-4 and result['fk_position_error_m']<1e-4 and result['fk_velocity_error_m_s']<1e-3
    finally: e.close()


if __name__=='__main__':main()
