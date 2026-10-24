"""Evaluate one FastSAC motion clip from distinct initial phases in native physics."""
import argparse
import json
from pathlib import Path
import torch
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.runners import ModelBasedRunner
from .train import build_task
from .fast_sac import G1FastSACTask
from .environment import G1TrackingEnv


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--motion-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-seconds',type=float,nargs='+',default=[0,1,2,3,4,5])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['cfg']['algorithm']['class_name'].split(':')[-1]!='MBDACFastSAC':
        parser.error('This evaluator expects MBDACFastSAC')
    original=build_task(args.device,args.motion_dir)
    if len(original.motion.names)!=1:parser.error('Provide a directory with exactly one reference clip')
    task=G1FastSACTask(original.kin,original.motion,original.default_q,
                     finite_reference=saved['cfg'].get('finite_reference',False))
    frames=torch.tensor([round(t/.02) for t in args.start_seconds],device=args.device)
    if (frames<0).any() or (frames>=task.motion.clip_end[0]).any():
        parser.error('Each start must precede the reference endpoint')
    n=len(frames)
    native=G1TrackingEnv(task,n,args.device,eval_mode=True,seed=args.seed)
    env=MjlabModelAdapter(native,task);native._fixed_frames=frames;native.reset()
    runner=ModelBasedRunner(env,saved['cfg'],device=args.device)
    runner.alg.actor.load_state_dict(saved['actor']);policy=runner.get_inference_policy(args.device)
    horizon=task.motion.clip_end[0]-frames
    active=torch.ones(n,dtype=torch.bool,device=args.device);failed=torch.zeros_like(active)
    count=torch.zeros(n,device=args.device);returns=torch.zeros_like(count)
    qerr=torch.zeros_like(count);berr=torch.zeros_like(count)
    try:
        for step in range(int(horizon.max())):
            action=policy(env.get_observations()).clamp(-runner.alg.cfg.action_clip,runner.alg.cfg.action_clip)
            _,reward,_,extras=env.step(action);state=extras['model_transition']['s1'];f=native.f
            returns+=reward*active;count+=active
            q=task.codec.unpack(state)[4]
            qerr+=(q-task.motion.q[f]).square().mean(-1)*active
            bp=task.kin.forward(*task.codec.unpack(state))[0]
            berr+=(bp-task.motion.body_p[f]).square().sum(-1).mean(-1)*active
            failed|=native.last_failure&active
            active&=~native.last_failure&(step+1<horizon)
            if not active.any():break
        rows=[dict(start_frame=int(frames[i]),horizon_s=float(horizon[i]*.02),
            survival_s=float(count[i]*.02),failed=bool(failed[i]),
            normalized_return=float(returns[i]/(.1*horizon[i])),
            joint_rmse_rad=float((qerr[i]/count[i]).sqrt()),
            body_position_rmse_m=float((berr[i]/count[i]).sqrt())) for i in range(n)]
        result=dict(checkpoint=str(args.checkpoint.resolve()),native_manager_based_env=True,
            device=args.device,num_parallel_worlds=n,auto_reset=False,domain_randomization=False,
            seed=args.seed,observation_dim=task.obs_dim,cases=rows)
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:runner.close();env.close()


if __name__=='__main__':main()
