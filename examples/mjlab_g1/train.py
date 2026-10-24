"""Run from the repository root: python -m examples.mjlab_g1.train."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.runners import ModelBasedRunner
from .kinematics import G1Kinematics
from .motion import MotionLib
from .robot import default_joint_pos_list
from .model_task import G1ModelTask
from .environment import G1TrackingEnv


def build_task(device,motion_dir):
    kin=G1Kinematics(device)
    motion=MotionLib.from_dir(str(motion_dir),kin,device)
    return G1ModelTask(kin,motion,torch.tensor(default_joint_pos_list(),device=device))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--num-envs',type=int,default=1024); p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--motion-dir',type=Path,default=Path('data/lafan1_g1')); p.add_argument('--log-dir',type=Path,default=Path('logs/g1_mbdac'))
    p.add_argument('--device',default='cuda:0'); p.add_argument('--logger',choices=['wandb','tensorboard'],default='wandb')
    p.add_argument('--algorithm',choices=['MBDAC','MBDACAMP'],default='MBDAC'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--wm-batch',type=int,default=128); p.add_argument('--pi-batch',type=int,default=256)
    p.add_argument('--wm-only-iters',type=int,default=500); p.add_argument('--wm-batches',type=int,default=10)
    p.add_argument('--log-interval',type=int,default=10)
    p.add_argument('--resume'); p.add_argument('--smoke',action='store_true'); p.add_argument('--compile',action='store_true')
    args=p.parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    task=build_task(args.device,args.motion_dir)
    env=MjlabModelAdapter(G1TrackingEnv(task,args.num_envs,args.device,seed=args.seed),task)
    config={'algorithm':{'class_name':args.algorithm,'wm_batch':args.wm_batch,'pi_batch':args.pi_batch,'compile':args.compile,
                         'wm_only_iters':args.wm_only_iters,'wm_batches':args.wm_batches},
            'logger':args.logger,'wandb_project':'mbdac-g1','run_name':args.log_dir.name,'save_interval':500,'log_interval':args.log_interval,
            'task':'g1_lafan1_motion_tracking','seed':args.seed,'num_envs':args.num_envs,'source':'SuperMimic-vibed a1508d4; native MJLab1.2 adapter'}
    if args.smoke:
        config['algorithm'].update(wm_only_iters=0,warmup_steps=8,wm_batches=1,horizon=4,wm_window=4,history=3,buffer_steps=16,
                                   actor_hidden=(128,64),wm_hidden=(128,128),critic_hidden=(64,32),critic_epochs=1,critic_minibatches=1,precision='fp32')
        config['log_interval']=1
    args.log_dir.mkdir(parents=True,exist_ok=True)
    runner=ModelBasedRunner(env,config,str(args.log_dir),args.device)
    if args.resume: runner.load(args.resume)
    try: runner.learn(args.iterations)
    finally: runner.close(); env.close()


if __name__=='__main__': main()
