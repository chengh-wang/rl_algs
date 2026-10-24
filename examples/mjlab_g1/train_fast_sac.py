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


def state_digest(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


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
    p.add_argument('--real-actor-batch',type=int,help='Real-state SAC batch independent of the memory-limited WM actor batch')
    p.add_argument('--critic-warmup-updates',type=int,help='Fit this many real Q updates before enabling the actor')
    p.add_argument('--actor-lr',type=float)
    p.add_argument('--policy-frequency',type=int,help='Actor update interval measured in Q updates')
    p.add_argument('--wm-only-iters',type=int)
    p.add_argument('--model-actor-weight',type=float)
    p.add_argument('--log-std-min',type=float)
    p.add_argument('--log-std-max',type=float)
    p.add_argument('--action-std-min',type=float,help='Exploration std in raw WM/replay control units')
    p.add_argument('--action-std-max',type=float,help='Exploration std upper bound in raw WM/replay control units')
    p.add_argument('--compile',action='store_true');p.add_argument('--compile-task',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if (args.action_std_min is not None or args.action_std_max is not None) and (args.log_std_min is not None or args.log_std_max is not None):
        p.error('Choose raw-action std bounds or pre-tanh log-std bounds, not both')
    if args.log_dir.exists() and (args.log_dir/'config.json').exists() and not args.resume:
        p.error('Use a new output directory; existing experiments are preserved')
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    motion_file=prepare_motion(args.motion,args.log_dir/'motion',args.clip_seconds,args.start_seconds)
    kin=G1Kinematics(args.device);motion=MotionLib([str(motion_file)],kin,args.device)
    task=G1FastSACTask(kin,motion,torch.tensor(default_joint_pos_list(),device=args.device))
    cfg=training_config(task,args.horizon,args.compile,args.logger,args.log_dir.name)
    cfg.update(seed=args.seed,num_envs=args.num_envs,motion_dir=str(motion_file.parent.resolve()))
    cfg['finite_reference']=True
    if args.critic_batch is not None: cfg['algorithm']['fast_sac']['critic_batch']=args.critic_batch
    if args.real_actor_batch is not None: cfg['algorithm']['fast_sac']['real_actor_batch']=args.real_actor_batch
    if args.critic_warmup_updates is not None: cfg['algorithm']['fast_sac']['critic_warmup_updates']=args.critic_warmup_updates
    if args.actor_lr is not None: cfg['algorithm']['fast_sac']['actor_lr']=args.actor_lr
    if args.policy_frequency is not None: cfg['algorithm']['fast_sac']['policy_frequency']=args.policy_frequency
    if args.compile_task:cfg['algorithm']['compile_task']=True
    if args.wm_only_iters is not None: cfg['algorithm']['wm_only_iters']=args.wm_only_iters
    for name in ('model_actor_weight','log_std_min','log_std_max','action_std_min','action_std_max'):
        value=getattr(args,name)
        if value is not None: cfg['algorithm']['fast_sac'][name]=value
    cfg['initialized_from']='scratch'
    if args.resume:
        cfg['initialized_from']=dict(full_checkpoint=str(args.resume.resolve()),
            sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest())
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
            if saved['cfg'].get('finite_reference',False)!=cfg['finite_reference']:
                raise ValueError('Reference endpoint semantics changed; use a new run with a WM-only warm start')
            previous=Path(saved['cfg']['motion_dir'])/motion_file.name
            if not previous.is_file() or previous.read_bytes()!=motion_file.read_bytes():
                raise ValueError('Full resume requires the identical reference subset; use a WM-only warm start for another clip')
            for key,value in saved['cfg']['algorithm'].items():
                if key=='compile': continue
                if key=='fast_sac':
                    from mbdac.algorithms.mbdac_fast_sac import FAST_SAC_DEFAULTS
                    value={**FAST_SAC_DEFAULTS,**value}
                if runner.cfg['algorithm'].get(key)!=value:
                    raise ValueError(f'Full resume changes algorithm option {key}; pass the original settings')
            runner.load(args.resume)
        if args.wm_checkpoint:
            saved=torch.load(args.wm_checkpoint,map_location='cpu',weights_only=False)
            if saved['cfg'].get('task') not in ('g1_lafan1_motion_tracking','g1_lafan1_single_clip_tracking'):
                raise ValueError('WM warm start requires the same native G1 physical task')
            actor_before=state_digest(runner.alg.actor);critic_before=state_digest(runner.alg.critic)
            runner.load(args.wm_checkpoint,load_cfg={'world_model':True})
            wm=runner.alg.world_model.state_dict()
            audit=dict(world_model_equal=all(torch.equal(wm[k].cpu(),v.cpu()) for k,v in saved['world_model'].items()),
                actor_unchanged=actor_before==state_digest(runner.alg.actor),
                critic_unchanged=critic_before==state_digest(runner.alg.critic),
                fresh_wm_optimizer=not runner.alg.opt_wm.state,initial_iteration=runner.current_learning_iteration,
                source_iteration=saved['iter'],source_training_s=saved['elapsed_s'])
            assert audit['world_model_equal'] and audit['actor_unchanged'] and audit['critic_unchanged']
            (args.log_dir/'initialization_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        runner.learn(args.iterations)
    finally: runner.close();env.close()


if __name__=='__main__':
    main()
