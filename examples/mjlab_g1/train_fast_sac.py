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


def prepare_motion(source, output, seconds):
    output.mkdir(parents=True,exist_ok=True)
    target=output/source.name
    if not target.exists():
        # Subset frames only: do not smooth, speed up, mirror or retarget again.
        lines=source.read_bytes().splitlines(keepends=True)
        selected=lines[:min(len(lines),int(seconds*30)+1)] if seconds>0 else lines
        if len(selected)<3: raise ValueError('Motion subset is too short')
        target.write_bytes(b''.join(selected))
        (output/'source.json').write_text(json.dumps(dict(source=str(source.resolve()),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),start_frame=0,
            frames=len(selected),source_fps=30,subset_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
    return target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,default=Path('data/lafan1_g1/walk1_subject1.csv'))
    p.add_argument('--clip-seconds',type=float,default=10.)
    p.add_argument('--log-dir',type=Path,default=Path('logs/g1_wm_fastsac'))
    p.add_argument('--num-envs',type=int,default=1024);p.add_argument('--iterations',type=int,default=3000)
    p.add_argument('--horizon',type=int,default=3);p.add_argument('--device',default='cuda:0')
    p.add_argument('--logger',choices=['wandb','tensorboard'],default='wandb')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--resume',type=Path)
    p.add_argument('--compile',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.log_dir.exists() and (args.log_dir/'config.json').exists() and not args.resume:
        p.error('Use a new output directory; existing experiments are preserved')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    motion_file=prepare_motion(args.motion,args.log_dir/'motion',args.clip_seconds)
    kin=G1Kinematics(args.device);motion=MotionLib([str(motion_file)],kin,args.device)
    task=G1FastSACTask(kin,motion,torch.tensor(default_joint_pos_list(),device=args.device))
    cfg=training_config(task,args.horizon,args.compile,args.logger,args.log_dir.name)
    cfg.update(seed=args.seed,num_envs=args.num_envs,motion_dir=str(motion_file.parent.resolve()))
    if args.smoke:
        cfg['algorithm'].update(wm_only_iters=0,warmup_steps=8,wm_batches=1,wm_window=4,
            buffer_steps=16,wm_hidden=(128,128),wm_batch=32,pi_batch=32,precision='fp32')
        cfg['algorithm']['fast_sac'].update(actor_hidden_dim=64,critic_hidden_dim=64,num_atoms=51,critic_batch=32,critic_updates=2)
        cfg['log_interval']=1
    native=G1TrackingEnv(task,args.num_envs,args.device,seed=args.seed)
    env=MjlabModelAdapter(native,task)
    runner=ModelBasedRunner(env,cfg,str(args.log_dir),args.device)
    if args.resume: runner.load(args.resume)
    try: runner.learn(args.iterations)
    finally: runner.close();env.close()


if __name__=='__main__':
    main()
