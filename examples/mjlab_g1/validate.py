"""Numerical checks against MuJoCo, including native pre-reset transitions."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from .train import build_task
from .environment import G1TrackingEnv
from .robot import TRACKED_BODIES,G1_JOINT_NAMES,action_scale,motor_of,_kp,_kd
from mbdac.utils.rotations import mat_to_quat,so3_exp


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0')
    p.add_argument('--output',type=Path,default=Path('logs/validation.json')); args=p.parse_args()
    torch.manual_seed(31); task=build_task(args.device,Path('data/lafan1_g1'))
    env=G1TrackingEnv(task,16,args.device,eval_mode=True)
    try:
        env.reset(); n=env.num_envs; codec=task.codec; dev=args.device
        f=task.motion.clip_start[:n]; S=list(task.motion.ref_state(f)); S[0]=S[0].clone(); S[0][:,2]+=1.5
        S[1]=so3_exp(torch.randn(n,3,device=dev)*.3)@S[1]
        S[2]=torch.randn(n,3,device=dev); S[3]=torch.randn(n,3,device=dev)
        S[4]=torch.maximum(torch.minimum(S[4],task.kin.q_hi),task.kin.q_lo)
        S[5]=torch.randn(n,29,device=dev)
        robot=env.scene['robot']; robot.write_root_state_to_sim(torch.cat((S[0],mat_to_quat(S[1]),S[2],S[3]),-1))
        robot.write_joint_state_to_sim(S[4],S[5]); env.scene.write_data_to_sim(); env.sim.forward()
        actual=env.physical_state(); root_error=(actual-codec.pack(S)).abs().max().item()
        assert root_error<2e-5,root_error
        fk=task.kin.forward(*codec.unpack(actual)); m=env.sim.mj_model; d=mujoco.MjData(m)
        ids=[m.body('robot/'+name).id for name in TRACKED_BODIES]
        cpu=[[],[],[],[]]
        qpos=env.sim.data.qpos.cpu().numpy(); qvel=env.sim.data.qvel.cpu().numpy()
        for i in range(n):
            d.qpos[:]=qpos[i]; d.qvel[:]=qvel[i]; mujoco.mj_forward(m,d)
            velocity=[]
            for b in ids:
                # XBODY is the link origin; BODY reports the inertial/COM origin.
                out=np.zeros(6); mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_XBODY,b,out,0); velocity.append(out)
            velocity=np.array(velocity)
            for dst,value in zip(cpu,[d.xpos[ids],d.xmat[ids].reshape(-1,3,3),velocity[:,3:],velocity[:,:3]]): dst.append(value.copy())
        errors={name:float(np.abs(a.cpu().numpy()-np.array(b)).max()) for name,a,b in zip(['body_position','body_rotation','body_velocity','body_angular_velocity'],fk,cpu)}
        assert max(errors.values())<3e-5,errors
        action_scale_error=max(abs(float(env.alpha[i])-action_scale(name)) for i,name in enumerate(G1_JOINT_NAMES))
        assert action_scale_error<1e-6
        gain_error=0.
        for i,name in enumerate(G1_JOINT_NAMES):
            aid=m.actuator('robot/'+name).id
            arm=motor_of(name)[2]
            gain_error=max(gain_error,abs(m.actuator_gainprm[aid,0]-_kp(arm)),abs(m.actuator_biasprm[aid,2]+_kd(arm)))
        assert gain_error<1e-4,gain_error
        # Use train mode termination to force a fall and confirm s1 precedes reset.
        env.eval_mode=False; env._fixed_frames=f; env.reset()
        env.sim.data.qpos[0,2]+=2.; before=env.physical_state().clone()
        action=torch.randn(n,29,device=dev)*.02
        _,r,done,trunc,extra=env.step(action); tr=extra['model_transition']
        expected=task.reward(tr['s1'],tr['condition']+1,tr['a'],tr['a_prev'])
        reward_error=(expected-r).abs().max().item(); assert reward_error<1e-7
        assert done[0] and not trunc[0]
        reset_gap=float((tr['s1'][0]-env.physical_state()[0]).abs().max()); assert reset_gap>1.
        assert torch.equal(tr['s'],before)
        applied=task.default_q+tr['context'][:,1:30]+env.alpha*action
        live=~(done|trunc)
        assert live.any() and torch.allclose(env.action_manager.action[live],applied[live])
        result=dict(native_manager_based_env=True,num_envs=n,root_state_roundtrip_max_abs=root_error,
                    fk_max_abs=errors,action_scale_max_abs=action_scale_error,pd_gain_max_abs=gain_error,
                    reward_max_abs=reward_error,pre_reset_gap=reset_gap,pre_reset_capture_passed=True)
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally: env.close()


if __name__=='__main__': main()
