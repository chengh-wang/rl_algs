"""Train Holosoma FastSAC + differentiable WM on one native G1 motion clip."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.runners import ModelBasedRunner
from .fast_sac import G1FastSACTask,training_config
from .kinematics import G1Kinematics
from .motion import MotionLib
from .robot import default_joint_pos_list
from .environment import G1TrackingEnv


def prepare_motion(source, output, seconds, start_seconds=0.):
    output.mkdir(parents=True,exist_ok=True)
    target=output/source.name
    if start_seconds<0: raise ValueError('Motion start must be nonnegative')
    lines=source.read_bytes().splitlines(keepends=True)
    start=int(start_seconds*30)
    selected=lines[start:start+int(seconds*30)+1] if seconds>0 else lines[start:]
    if len(selected)<3: raise ValueError('Motion subset is too short')
    selected_bytes=b''.join(selected)
    if target.exists() and target.read_bytes()!=selected_bytes:
        raise ValueError('Existing motion subset differs; use a fresh output directory')
    if not target.exists():
        # Subset frames only: do not smooth, speed up, mirror or retarget again.
        target.write_bytes(selected_bytes)
        (output/'source.json').write_text(json.dumps(dict(source=str(source.resolve()),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),start_frame=start,
            frames=len(selected),source_fps=30,subset_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,default=Path('data/lafan1_g1/walk1_subject1.csv'))
    p.add_argument('--clip-seconds',type=float,default=10.)
    p.add_argument('--start-seconds',type=float,default=0.)
    p.add_argument('--log-dir',type=Path,default=Path('logs/g1_wm_fastsac'))
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--horizon',type=int,default=3);p.add_argument('--device',default='cuda:0')
    p.add_argument('--logger',choices=['wandb','tensorboard'],default='wandb')
    p.add_argument('--seed',type=int,default=42)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--resume',type=Path)
    group.add_argument('--wm-checkpoint',type=Path,help='Warm-start only a compatible G1 physical WM; actor/Q/temperature start fresh')
    p.add_argument('--critic-batch',type=int)
    p.add_argument('--wm-only-iters',type=int)
    p.add_argument('--compile',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.log_dir.exists() and (args.log_dir/'config.json').exists() and not args.resume:
        p.error('Use a new output directory; existing experiments are preserved')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    motion_file=prepare_motion(args.motion,args.log_dir/'motion',args.clip_seconds,args.start_seconds)
    kin=G1Kinematics(args.device);motion=MotionLib([str(motion_file)],kin,args.device)
    task=G1FastSACTask(kin,motion,torch.tensor(default_joint_pos_list(),device=args.device))
    cfg=training_config(task,args.horizon,args.compile,args.logger,args.log_dir.name)
    cfg.update(seed=args.seed,num_envs=args.num_envs,motion_dir=str(motion_file.parent.resolve()))
    if args.critic_batch is not None: cfg['algorithm']['fast_sac']['critic_batch']=args.critic_batch
    if args.wm_only_iters is not None: cfg['algorithm']['wm_only_iters']=args.wm_only_iters
    cfg['initialized_from']='scratch'
    if args.wm_checkpoint:
        cfg['initialized_from']=dict(world_model=str(args.wm_checkpoint.resolve()),
            sha256=hashlib.sha256(args.wm_checkpoint.read_bytes()).hexdigest(),actor='scratch',critic='scratch')
    if args.smoke:
        cfg['algorithm'].update(wm_only_iters=0,warmup_steps=8,wm_batches=1,wm_window=4,
            buffer_steps=16,wm_hidden=(128,128),wm_batch=32,pi_batch=32,precision='fp32')
        cfg['algorithm']['fast_sac'].update(actor_hidden_dim=64,critic_hidden_dim=64,num_atoms=51,critic_batch=32,critic_updates=2)
        cfg['log_interval']=1
    native=G1TrackingEnv(task,args.num_envs,args.device,seed=args.seed)
    env=MjlabModelAdapter(native,task)
    runner=ModelBasedRunner(env,cfg,str(args.log_dir),args.device)
    try:
        if args.resume:
            saved=torch.load(args.resume,map_location='cpu',weights_only=False)
            previous=Path(saved['cfg']['motion_dir'])/motion_file.name
            if not previous.is_file() or previous.read_bytes()!=motion_file.read_bytes():
                raise ValueError('Full resume requires the identical reference subset; use a WM-only warm start for another clip')
            for key,value in saved['cfg']['algorithm'].items():
                if key=='compile': continue
                if runner.cfg['algorithm'].get(key)!=value:
                    raise ValueError(f'Full resume changes algorithm option {key}; pass the original settings')
            runner.load(args.resume)
        if args.wm_checkpoint:
            saved=torch.load(args.wm_checkpoint,map_location='cpu',weights_only=False)
            if saved['cfg'].get('task') not in ('g1_lafan1_motion_tracking','g1_lafan1_single_clip_tracking'):
                raise ValueError('WM warm start requires the same native G1 physical task')
            runner.load(args.wm_checkpoint,load_cfg={'world_model':True})
        runner.learn(args.iterations)
    finally: runner.close();env.close()


if __name__=='__main__':
    main()
