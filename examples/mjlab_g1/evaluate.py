"""Evaluate an actor in native physics, with no model rollouts or automatic reset."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from mbdac.runners import ModelBasedRunner
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.storage.replay import ReplayBuffer
from mbdac.utils.diagnostics import world_model_diagnostics
from .train import build_task
from .environment import G1TrackingEnv
from .robot import TRACKED_BODIES


class Video:
    def __init__(self,model,path,fps=25):
        import mujoco
        import imageio.v2 as imageio
        self.mj=mujoco; self.data=mujoco.MjData(model); self.model=model
        self.renderer=mujoco.Renderer(model,height=480,width=640)
        self.writer=imageio.get_writer(path,fps=fps,codec='libx264',quality=7)
        self.camera=mujoco.MjvCamera(); self.camera.distance=3.5; self.camera.azimuth=135; self.camera.elevation=-15
        self.option=mujoco.MjvOption(); self.option.geomgroup[:]=1

    def append(self,qpos_actual,qpos_reference,policy_failed=False):
        from PIL import Image,ImageDraw
        frames=[]; self.camera.lookat[:]=qpos_reference[:3]; self.camera.lookat[2]=.9
        for q in [qpos_reference,qpos_actual]:
            self.data.qpos[:]=q; self.mj.mj_forward(self.model,self.data)
            self.renderer.update_scene(self.data,camera=self.camera,scene_option=self.option)
            frames.append(self.renderer.render().copy())
        frame=Image.fromarray(np.concatenate(frames,axis=1)); draw=ImageDraw.Draw(frame)
        for x,label in [(12,'REFERENCE: kinematic playback'),(652,'POLICY: native MuJoCo physics')]:
            draw.rectangle((x-4,8,x+315,34),fill=(0,0,0)); draw.text((x,14),label,fill=(255,255,255))
        if policy_failed:
            draw.rectangle((648,38,1270,64),fill=(140,0,0))
            draw.text((652,44),'FAILURE DETECTED - physical rollout continues',fill=(255,255,255))
        self.writer.append_data(np.asarray(frame))

    def close(self): self.writer.close(); self.renderer.close()


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path)
    p.add_argument('--seconds',type=float,default=20.); p.add_argument('--device',default='cuda:0')
    p.add_argument('--motion-dir',type=Path,default=Path('data/lafan1_g1'))
    p.add_argument('--output',type=Path,default=Path('logs/evaluation'))
    p.add_argument('--video-clip',default='walk1_subject1'); p.add_argument('--video-seconds',type=float,default=12.)
    p.add_argument('--continue-after-failure',action='store_true',help='Keep physical simulation/video running, while metrics stop at the first failure')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    task=build_task(args.device,args.motion_dir); motion=task.motion; n=len(motion.names)
    if ck['cfg']['algorithm']['class_name'].split(':')[-1]=='MBDACFastSAC':
        from .fast_sac import G1FastSACTask
        task=G1FastSACTask(task.kin,task.motion,task.default_q,finite_reference=ck['cfg'].get('finite_reference',False))
    native=G1TrackingEnv(task,n,args.device,eval_mode=True); env=MjlabModelAdapter(native,task)
    native._fixed_frames=motion.clip_start.clone(); native.reset()
    runner=ModelBasedRunner(env,ck['cfg'],device=args.device)
    # Evaluation loads actor only: it does not restore training RNG/optimizers/replay.
    runner.alg.actor.load_state_dict(ck['actor']); policy=runner.get_inference_policy(args.device)
    runner.alg.world_model.load_state_dict(ck['world_model'])
    diagnostic_replay=ReplayBuffer(256,n,args.device)
    horizon=torch.minimum(motion.clip_len-1,torch.full_like(motion.clip_len,int(args.seconds/.02)))
    active=torch.ones(n,dtype=torch.bool,device=args.device); failed=torch.zeros_like(active)
    survived=torch.zeros(n,device=args.device); total_reward=torch.zeros_like(survived)
    sum_joint_error=torch.zeros_like(survived); sum_body_error=torch.zeros_like(survived)
    video_id=motion.names.index(args.video_clip) if args.video_clip in motion.names else next(i for i,k in enumerate(motion.names) if k.startswith('walk'))
    video=Video(native.sim.mj_model,str(args.output/'reference_left_policy_right.mp4'))
    actual_buffer=[]; reference_buffer=[]; failed_buffer=[]
    try:
        for step in range(int(horizon.max())):
            action=policy(env.get_observations()).clamp(-runner.alg.cfg.action_clip,runner.alg.cfg.action_clip)
            _,r,_,extras=env.step(action)
            transition=dict(extras['model_transition'])
            transition.update(terminated=native.last_failure.clone(),eval_active=active.clone())
            if active.any(): diagnostic_replay.add(transition)
            s=extras['model_transition']['s1']; f=native.f
            total_reward+=r*active; survived+=active.float()
            q=task.codec.unpack(s)[4]; sum_joint_error+=(q-motion.q[f]).square().mean(-1)*active
            bp=task.kin.forward(*task.codec.unpack(s))[0]
            sum_body_error+=(bp-motion.body_p[f]).square().sum(-1).mean(-1)*active
            failed|=native.last_failure&active
            if step%2==0 and step*.02<args.video_seconds:
                actual=native.sim.data.qpos[video_id].cpu().numpy().copy()
                ref=motion.ref_state(f[video_id:video_id+1]); refq=task.codec.pack(ref)[0]
                reference=torch.cat((refq[:7],refq[13:42])).cpu().numpy()
                actual_buffer.append(actual); reference_buffer.append(reference)
                failed_buffer.append(bool(failed[video_id]))
            active&=~native.last_failure & (step+1<horizon)
            if (step+1)%1000==0:
                print(json.dumps(dict(evaluation_step=step+1,maximum_steps=int(horizon.max()),
                                      active_clips=int(active.sum()),failed_clips=int(failed.sum()))),flush=True)
            if not active.any() and not args.continue_after_failure: break
        # Render after stepping, avoiding interference between renderer and GPU physics.
        for actual,reference,failure in zip(actual_buffer,reference_buffer,failed_buffer): video.append(actual,reference,failure)
        rows=[]
        for i,name in enumerate(motion.names):
            rows.append(dict(clip=name,evaluated_horizon_s=float(horizon[i]*.02),survival_s=float(survived[i]*.02),
                             failed=bool(failed[i]),normalized_return=float(total_reward[i]/(.1*horizon[i])),
                             joint_rmse_rad=float((sum_joint_error[i]/survived[i].clamp_min(1)).sqrt()),
                             body_position_rmse_m=float((sum_body_error[i]/survived[i].clamp_min(1)).sqrt()),
                             full_clip=bool(horizon[i]==motion.clip_len[i]-1)))
        diagnostics=world_model_diagnostics(runner.alg.world_model,task,diagnostic_replay)
        result=dict(checkpoint=str(args.checkpoint.resolve()),iteration=ck['iter'],training_elapsed_s=ck['elapsed_s'],
                    algorithm=ck['cfg']['algorithm']['class_name'],evaluation_device=args.device,evaluation_seed=0,
                    fast_sac_state_version=ck.get('fast_sac_state_version'),continued_after_failure=args.continue_after_failure,
                    real_transitions=ck['real_transitions'],native_physics=True,auto_reset=False,domain_randomization=False,
                    mean_normalized_return=float(np.mean([r['normalized_return'] for r in rows])),
                    survived_fraction=float(np.mean([not r['failed'] for r in rows])),
                    video_clip=motion.names[video_id],video_layout='reference left, policy in real physics right',world_model=diagnostics,clips=rows)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='clips'},indent=2))
    finally: video.close(); runner.close(); env.close()


if __name__=='__main__': main()
