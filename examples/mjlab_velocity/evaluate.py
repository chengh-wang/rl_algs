"""Load only an actor, evaluate in native GPU physics and optionally render it."""
import argparse
import json
from pathlib import Path
import torch
from mbdac.modules.networks import Actor as MBDACActor
from mbdac.modules.holosoma import HolosomaActor
from mbdac.algorithms.mbdac_fast_sac import FAST_SAC_DEFAULTS
from mbdac.runners.model_based_runner import InferencePolicy
from mbdac.vendor.holosoma.fast_sac import Actor as SourceActor
from mbdac.vendor.holosoma.normalization import EmpiricalNormalization
from .evaluation import evaluate


def load_policy(path,device='cuda:0'):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if 'actor_state_dict' in state:
        cfg=state['args'];weights=state['actor_state_dict']
        obs_dim=state['obs_normalizer_state']['_mean'].shape[-1]
        act_dim=len(weights['action_scale'])
        actor=SourceActor({'actor_obs':dict(start=0,end=obs_dim,size=obs_dim)},['actor_obs'],act_dim,1,
            cfg['actor_hidden_dim'],cfg['log_std_max'],cfg['log_std_min'],device='cpu',
            action_scale=weights['action_scale'],use_tanh=cfg['use_tanh'],use_layer_norm=cfg['use_layer_norm'])
        actor.load_state_dict(weights);actor=actor.to(device).eval()
        norm=EmpiricalNormalization((obs_dim,),device);norm.load_state_dict(state['obs_normalizer_state']);norm.eval()
        @torch.no_grad()
        def policy(obs):return actor(norm(obs['actor'],update=False))[0]
        method='fastsac';iteration=state['global_step']
    else:
        cfg=state['cfg']['algorithm'];weights=state['actor']
        if cfg['class_name']=='MBDACFastSAC':
            options={**FAST_SAC_DEFAULTS,**cfg['fast_sac']}
            actor=HolosomaActor(weights['obs_norm._mean'].shape[-1],weights['goal_norm._mean'].shape[-1],
                len(weights['net.action_scale']),cfg['history'],options)
            method='wm_fastsac'
        elif cfg['class_name']=='MBDAC':
            last=max(int(k.split('.')[1]) for k in weights if k.startswith('decoder.') and k.endswith('.weight'))
            actor=MBDACActor(len(weights['obs_norm.mean']),len(weights['goal_norm.mean']),
                weights[f'decoder.{last}.weight'].shape[0],cfg['history'],cfg['actor_hidden'],precision='fp32')
            method='mbdac'
        else:raise ValueError('Unsupported benchmark algorithm')
        actor.load_state_dict(weights);actor.precision='fp32'
        policy=InferencePolicy(actor.to(device).eval());iteration=state['iter']
    return policy,dict(method=method,iteration=iteration)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path)
    p.add_argument('--robot',choices=('g1','go1'),default='go1');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trajectory',action='store_true');args=p.parse_args();torch.set_num_threads(4)
    args.out.mkdir(parents=True,exist_ok=True)
    policy,metadata=load_policy(args.checkpoint)
    metrics=evaluate(policy,args.robot,trajectory=args.out/'trajectory.npz' if args.trajectory else None)
    metrics.update(metadata,checkpoint=str(args.checkpoint.resolve()))
    (args.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
