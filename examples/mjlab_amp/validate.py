"""Separate exact reuse of native reward formulas from one-substep timing error."""
import argparse
import json
from pathlib import Path
import torch
from .environment import BaselineVelocityEnv,VelocityModelAdapter
from .model_task import BaselineVelocityTask


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('logs/velocity_validation.json'))
    args=p.parse_args()
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg()
    for group in cfg.observations.values(): group.enable_corruption=False
    native=BaselineVelocityEnv(32,cfg=cfg); task=BaselineVelocityTask(native)
    native.model_task=task; env=VelocityModelAdapter(native,task); native.audit_snapshots=True
    try:
        with torch.no_grad():
            def analytic_observation_error():
                actual=native.obs_buf['actor'].reshape(32,-1,task.obs_dim)[:,-1]
                predicted=task.observation(native.physical_state(),native.action_manager.action,
                                           native.physical_context(),native.condition())
                bounds={'gyro':(0,3),'gravity':(3,6),'command':(6,9),'joint_position':(9,38),
                        'joint_velocity':(38,67),'previous_action':(67,96)}
                return {name:float((actual[:,a:b]-predicted[:,a:b]).abs().max()) for name,(a,b) in bounds.items()}
            analytic_observation_errors=analytic_observation_error()
            native_actor=native.obs_buf['actor'].reshape(32,-1,task.obs_dim)[:,-1]
            observation_error=float((native_actor-env.get_observations()['actor']).abs().max())
            assert observation_error==0.
            action=torch.randn(32,29,device=native.device)*.1
            _,r,_,extras=env.step(action); r=r.clone()
            after_step=analytic_observation_error()
            analytic_observation_errors={name:max(value,after_step[name]) for name,value in analytic_observation_errors.items()}
            assert max(analytic_observation_errors.values())<3e-5,analytic_observation_errors
            tr=extras['model_transition']; c=tr['condition_reward']
            labels={k.split('/',1)[1]:v for k,v in tr.items() if k.startswith('primitive/')}
            reward,terms=task.reward(tr['s1'],c,tr['a'],tr['a_prev'],labels,tr['s'],soft=False,overrides=native.audit_snapshot,terms=True)
            original={name:torch.tensor([dict(native.reward_manager.get_active_iterable_terms(i))[name][0] for i in range(32)],device=native.device)*native.step_dt for name in terms}
            errors={name:float((v-original[name]).abs().max()) for name,v in terms.items()}
            assert max(errors.values())<2e-6,errors
            assert (reward-r).abs().max()<3e-6
            _,current=task.reward(tr['s1'],c,tr['a'],tr['a_prev'],labels,tr['s'],soft=False,terms=True)
            timing={name:float((current[name]-terms[name]).abs().max()) for name in terms}
            # Compare analytic body FK with fresh post-step native kinematics.
            state=native.physical_state(); p0,R0,v0,w0,q0,qd0=task.codec.unpack(state)
            bp,bR,bv,bw=task.kin.forward(p0,R0,v0,w0,q0,qd0,all_bodies=True)
            body=native.scene['robot'].data
            fk_errors=dict(position_m=float((bp+native.scene.env_origins[:,None]-body.body_link_pos_w[:,task.moving_ids]).abs().max()),
                           velocity_m_s=float((bv-body.body_link_lin_vel_w[:,task.moving_ids]).abs().max()))
            assert max(fk_errors.values())<3e-5,fk_errors
            amp_error=float((task.amp_features(state)-native.obs_buf['amp']).abs().max())
            assert amp_error<3e-5,amp_error
        # The reused math must also carry state/action gradients.
        s1=tr['s1'].detach().requires_grad_(); a=tr['a'].detach().requires_grad_()
        reward=task.reward(s1,c,a,tr['a_prev'],labels,tr['s'])
        gs,ga=torch.autograd.grad(reward.sum(),(s1,a))
        assert torch.isfinite(gs).all() and torch.isfinite(ga).all()
        assert gs.abs().sum()>0 and ga.abs().sum()>0
        with torch.no_grad():
            native.termination_manager._delay_env_mask.zero_()
            robot=native.scene['robot']; root=robot.data.default_root_state.clone()
            root[:,:3]+=native.scene.env_origins; root[:,2]=.4; root[:,7:]=0
            robot.write_root_state_to_sim(root); native.sim.forward()
            _,_,done,extra=env.step(torch.zeros(32,29,device=native.device))
            terminal=extra['model_transition']
            assert terminal['terminated'].all() and done.all()
            pre_reset_gap=float((terminal['s1'][:,2]-native.physical_state()[:,2]).abs().min())
            assert pre_reset_gap>.1,pre_reset_gap
        result=dict(native_task='Unitree-G1-AMP-Flat',num_envs=32,reward_formula_max_abs=errors,
                    native_observation_max_abs=observation_error,
                    analytic_observation_term_max_abs=analytic_observation_errors,
                    amp_feature_max_abs=amp_error,forced_terminal_capture=True,pre_reset_min_height_gap_m=pre_reset_gap,
                    soft_imagined_reward_max_abs=float((reward.detach()-r).abs().max()),
                    hard_reward_total_max_abs=float((sum(terms.values())-r).abs().max()),
                    post_step_fk_max_abs=fk_errors,one_substep_timing_term_max_abs=timing,
                    finite_state_and_action_gradients=True,head_units={s.name:s.units for s in task.primitive_specs})
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally: env.close()


if __name__=='__main__': main()
