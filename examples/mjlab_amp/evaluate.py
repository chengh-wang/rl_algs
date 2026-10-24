"""Matched real-physics evaluation of MBDAC(AMP) and the retained AMP+PPO actor."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from mjlab.envs.mdp import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mbdac.modules.networks import Actor
from mbdac.runners.model_based_runner import InferencePolicy
from mbdac.utils.provenance import runtime_provenance
from .environment import BaselineVelocityEnv


def evaluation_config(seconds):
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
    cfg=g1_amp_flat_env_cfg(play=True)
    failures={k:v for k,v in cfg.terminations.items() if not v.time_out}
    cfg.episode_length_s=seconds+1
    cfg.events={'reset_to_default':EventTermCfg(func=reset_scene_to_default,mode='reset')}
    cfg.curriculum={}; cfg.terminations={}
    for group in cfg.observations.values(): group.enable_corruption=False
    cmd=cfg.commands['twist']; cmd.heading_command=False; cmd.ranges.heading=None
    cmd.rel_heading_envs=0.; cmd.rel_standing_envs=0.; cmd.init_velocity_prob=0.
    cmd.resampling_time_range=(1e8,1e8)
    cmd.ranges.lin_vel_x=(0.,0.); cmd.ranges.lin_vel_y=(0.,0.); cmd.ranges.ang_vel_z=(0.,0.)
    cmd.debug_vis=False
    cfg.viewer.height=480; cfg.viewer.width=640; cfg.viewer.max_extra_envs=0
    from mjlab.viewer.viewer_config import ViewerConfig
    cfg.viewer.origin_type=ViewerConfig.OriginType.ASSET_BODY
    cfg.viewer.entity_name='robot'; cfg.viewer.body_name='torso_link'
    cfg.viewer.distance=3.; cfg.viewer.elevation=-18; cfg.viewer.azimuth=140
    return cfg,failures


def load_policy(checkpoint):
    if 'schema_version' in checkpoint:
        cfg=checkpoint['cfg']['algorithm']
        actor=Actor(96,3,29,cfg['history'],cfg['actor_hidden']).eval()
        actor.load_state_dict(checkpoint['actor'],strict=True)
        policy=InferencePolicy(actor)
        def action(obs):
            frame=obs['actor'].reshape(5,4,96)[:,-1].cpu()
            return policy(TensorDict(dict(actor=frame,goal=frame[:,6:9]),batch_size=[5]))
        return action,cfg['action_clip'],checkpoint['cfg']['algorithm']['class_name']
    # Original fork's modules and saved observation statistics; no learner needed.
    from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
    from rsl_rl.modules import ActorCritic,EmpiricalNormalization
    from rsl_rl.runners.amp_on_policy_runner import _migrate_train_cfg
    cfg=asdict(g1_amp_ppo_runner_cfg()); _migrate_train_cfg(cfg)
    options=cfg['policy']; options.pop('class_name')
    state=checkpoint['model_state_dict']
    actor=ActorCritic(state['actor.0.weight'].shape[1],state['critic.0.weight'].shape[1],29,**options).eval()
    actor.load_state_dict(state,strict=True)
    normalizer=EmpiricalNormalization([state['actor.0.weight'].shape[1]],until=1e8).eval()
    normalizer.load_state_dict(checkpoint['obs_norm_state_dict'],strict=True)
    return lambda obs:actor.act_inference(normalizer(obs['actor'].cpu())),cfg.get('clip_actions'),'AMP+PPO'


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path)
    p.add_argument('--seconds',type=float,default=20.); p.add_argument('--seed',type=int,default=1042)
    p.add_argument('--output',type=Path,default=Path('logs/velocity_evaluation'))
    p.add_argument('--model-diagnostics',action='store_true')
    p.add_argument('--video',action='store_true'); args=p.parse_args()
    torch.set_num_threads(4); args.output.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    policy,clip,algorithm=load_policy(checkpoint)
    cfg,failures=evaluation_config(args.seconds)
    env=BaselineVelocityEnv(5,seed=args.seed,cfg=cfg,render_mode='rgb_array' if args.video else None)
    model=task=replay=None; diagnostics={}
    if args.model_diagnostics and 'world_model' in checkpoint:
        from mbdac.modules.world_model import WorldModel
        from mbdac.storage.replay import ReplayBuffer
        from .model_task import BaselineVelocityTask
        task=BaselineVelocityTask(env); env.model_task=task
        options=checkpoint['cfg']['algorithm']
        model=WorldModel(task.codec,29,33,options['wm_hidden'],options['precision'],task.primitive_specs).to(env.device)
        model.load_state_dict(checkpoint['world_model'],strict=True); model.eval()
        replay=ReplayBuffer(256,5,env.device)
    commands=np.array([[.8,0,0],[-.4,0,0],[0,.4,0],[0,0,.6],[0,0,0]],dtype=np.float32)
    labels=['forward','backward','lateral_left','turn_left','stand']
    env.reset(seed=args.seed); command=env.command_manager.get_term('twist')
    command.vel_command_b[:]=torch.as_tensor(commands,device=env.device); command.time_left.fill_(1e8)
    for _ in range(cfg.observations['actor'].history_length):
        obs=env.observation_manager.compute(update_history=True)
    env.obs_buf=obs; robot=env.scene['robot']
    steps=round(args.seconds/env.step_dt); first_failure=np.full(5,steps+1,dtype=int); causes=[None]*5
    traces={key:[] for key in ['velocity','position','joint_pos','joint_vel','action','contact','foot_velocity','foot_position','qpos','qvel','actual_command']}
    feet=[robot.site_names.index(name) for name in ('left_foot','right_foot')]
    writer=None
    if args.video:
        import imageio.v2 as imageio
        writer=imageio.get_writer(args.output/'forward.mp4',fps=25,codec='libx264',quality=7)
    try:
        for step in range(1,steps+1):
            actual_command=command.command.cpu().numpy().copy()
            np.testing.assert_allclose(actual_command,commands,atol=1e-6,rtol=0)
            np.testing.assert_allclose(obs['actor'].reshape(5,4,96)[:,-1,6:9].cpu().numpy(),commands,atol=1e-6,rtol=0)
            action=policy(obs).to(env.device)
            if clip is not None: action=action.clamp(-clip,clip)
            action[torch.as_tensor(first_failure<step,device=env.device)]=0
            obs,_,terminated,truncated,extra=env.step(action)
            if replay is not None and replay.size<256:
                tr=dict(extra['model_transition'])
                tr['eval_active']=torch.as_tensor(first_failure>=step,device=env.device)
                replay.add(tr)
            assert not (terminated|truncated).any(),'Evaluation must not auto-reset'
            bad=~(torch.isfinite(env.sim.data.qpos).all(-1)&torch.isfinite(env.sim.data.qvel).all(-1))
            for name,term in failures.items():
                mask=term.func(env,**term.params); bad|=mask
                for index in np.flatnonzero(mask.cpu().numpy()&(first_failure>step)): causes[index]=name
            for index in np.flatnonzero(bad.cpu().numpy()&(first_failure>step)):
                first_failure[index]=step; causes[index]=causes[index] or 'non_finite_physics'
            values=dict(velocity=torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),-1),
                position=robot.data.root_link_pos_w,joint_pos=robot.data.joint_pos,joint_vel=robot.data.joint_vel,
                action=action,contact=(env.scene['feet_ground_contact'].data.found>0).reshape(5,-1),
                foot_velocity=robot.data.site_lin_vel_w[:,feet,:2],foot_position=robot.data.site_pos_w[:,feet]-env.scene.env_origins[:,None],
                qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,actual_command=command.command)
            for key,value in values.items(): traces[key].append(value.cpu().numpy().copy())
            if writer and step%2==0 and step<=first_failure[0]+round(1/env.step_dt):
                from PIL import Image,ImageDraw
                frame=Image.fromarray(env.render()); draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,62),fill=(20,20,20))
                draw.text((10,6),f'{algorithm} | checkpoint {checkpoint["iter"]} | native MJLab | t={step*env.step_dt:.2f}s',fill='white')
                draw.text((10,24),'command: vx=0.80 m/s, vy=0, yaw=0',fill='white')
                velocity=values['velocity'][0].cpu().tolist()
                draw.text((10,42),f'actual: vx={velocity[0]:.2f}, vy={velocity[1]:.2f}, yaw={velocity[2]:.2f}',fill='white')
                writer.append_data(np.asarray(frame))
            if (first_failure<=step).all(): break
        if model is not None:
            from mbdac.utils.diagnostics import world_model_diagnostics
            diagnostics=world_model_diagnostics(model,task,replay)
    finally:
        if writer: writer.close()
        env.close()
    arrays={k:np.stack(v) for k,v in traces.items()}
    np.savez_compressed(args.output/'traces.npz',commands=commands,first_failure_step=first_failure,**arrays)
    cases=[]; times=(np.arange(len(arrays['velocity']))+1)*env.step_dt
    for index,name in enumerate(labels):
        steady=(np.arange(1,len(times)+1)<first_failure[index])&(times>2)
        row=dict(name=name,command=commands[index].tolist(),survived_horizon=bool(first_failure[index]>steps),
                 first_failure_s=float(first_failure[index]*env.step_dt) if first_failure[index]<=steps else None,
                 failure_cause=causes[index],steady_seconds=float(steady.sum()*env.step_dt))
        if steady.any():
            velocity=arrays['velocity'][steady,index]; error=velocity-commands[index]
            contact=arrays['contact'][steady,index]; foot_v=arrays['foot_velocity'][steady,index]
            row.update(mean_velocity=velocity.mean(0).tolist(),rmse_per_axis=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                horizontal_velocity_rmse=float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
                foot_slip_rms_m_s=float(np.sqrt((np.sum(foot_v**2,axis=-1)*contact).sum()/contact.sum())) if contact.any() else None,
                both_feet_contact_fraction=float(contact.all(1).mean()),contact_fraction_per_foot=contact.mean(0).tolist(),
                foot_height_range_m=np.ptp(arrays['foot_position'][steady,index,:,2],axis=0).tolist(),
                liftoffs_per_foot=(contact[:-1]&~contact[1:]).sum(0).tolist(),
                touchdowns_per_foot=(~contact[:-1]&contact[1:]).sum(0).tolist())
            if steady.sum()>1: row['joint_acceleration_rms_rad_s2']=float(np.sqrt(np.mean((np.diff(arrays['joint_vel'][steady,index],axis=0)/env.step_dt)**2)))
        cases.append(row)
    result=dict(algorithm=algorithm,checkpoint=str(args.checkpoint.resolve()),iteration=checkpoint['iter'],
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        training_elapsed_s=checkpoint.get('elapsed_s'),real_transitions=checkpoint.get('real_transitions'),
        task='Unitree-G1-AMP-Flat',seed=args.seed,requested_horizon_s=args.seconds,physics_dt=.005,control_dt=env.step_dt,
        initialization='standing',native_physics=True,auto_reset=False,domain_randomization=False,
        observation_noise=False,steady_start_s=2.,cases=cases,world_model=diagnostics,runtime=runtime_provenance())
    (args.output/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='runtime'},indent=2))


if __name__=='__main__': main()
