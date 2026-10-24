"""Counterfactual native-physics audit of learned Q versus model reward gradients.

No policy update, teacher action, or simulator gradient is used. Paired worlds
start from identical references, perturb only the first action, then use the
same deterministic actor. A short diagnostic is not a benchmark.
"""
import argparse
import json
from pathlib import Path
import torch
from mbdac.adapters.mjlab import MjlabModelAdapter
from mbdac.runners import ModelBasedRunner
from mbdac.algorithms.mbdac import frozen
from .train import build_task
from .fast_sac import G1FastSACTask
from .environment import G1TrackingEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path)
    p.add_argument('--motion-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=32)
    p.add_argument('--steps',type=int,default=32)
    p.add_argument('--epsilon',type=float,default=.05)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(17);dev='cuda:0'
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    t=build_task(dev,args.motion_dir);task=G1FastSACTask(t.kin,t.motion,t.default_q)
    n=args.frames;groups=7
    native=G1TrackingEnv(task,n*groups,dev,eval_mode=True,seed=17)
    env=MjlabModelAdapter(native,task);runner=ModelBasedRunner(env,ck['cfg'],device=dev)
    alg=runner.alg
    for name in ['actor','critic','world_model']:getattr(alg,name).load_state_dict(ck[name])
    alg.world_model.precision='fp32'
    alg.eval_mode()
    frames=torch.linspace(0,int(task.motion.clip_end[0])-max(32,args.steps)-1,n,device=dev).long()
    native._fixed_frames=frames.repeat(groups);native.reset()
    obs=env.get_observations();history=obs['actor'][:,None].expand(-1,alg.cfg.history,-1).clone()
    with torch.no_grad():base=alg.actor(history[:n],obs['goal'][:n]);state=native.physical_state()[:n].clone();ctx=native.context[:n].clone()
    previous=torch.zeros_like(base);directions={};norms={};first=base.detach().requires_grad_(True)
    with frozen(alg.actor,alg.critic,alg.world_model):
        features=alg._q_features(state,frames,ctx,previous,history[:n])
        q=alg.critic.value(features,first)
        grad=torch.autograd.grad(q.sum(),first)[0]
        directions['Q']=grad.detach();norms['Q']=float(grad.norm(dim=-1).mean())
        for horizon in [3,16]:
            s=state;cond=frames;hist=history[:n];prev=previous;alive=torch.ones(n,device=dev);objective=torch.zeros(n,device=dev)
            for j in range(horizon):
                a=first if j==0 else alg.actor(hist,task.goal(cond))
                s1=alg.world_model(s,a,task.normalize_context(ctx));c1=task.advance(cond,state=s1,context=ctx)
                r,d=alg._reward_and_termination(s1,c1,a,prev,s,ctx)
                objective=objective+alg.cfg.gamma**j*alive*r;alive=alive*(1-d)
                o=task.observation(s1,a,ctx,c1);hist=torch.cat((hist[:,1:],o[:,None]),1)
                s,cond,prev=s1,c1,a
            grad=torch.autograd.grad(objective.sum(),first)[0]
            directions['model'+str(horizon)]=grad.detach();norms['model'+str(horizon)]=float(grad.norm(dim=-1).mean())
    actions=[base]
    for d in directions.values():
        delta=args.epsilon*d/d.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        actions.extend((base+delta,base-delta))
    returns=torch.zeros(groups*n,device=dev);alive=torch.ones(groups*n,dtype=torch.bool,device=dev)
    first_action=torch.cat(actions);survival=torch.zeros_like(returns)
    try:
        with torch.no_grad():
            for step in range(args.steps):
                action=first_action if step==0 else alg.actor(history,env.get_observations()['goal'])
                nxt,r,_,extra=env.step(action)
                returns+=alg.cfg.gamma**step*alive*r;survival+=alive
                alive&=~native.last_failure
                history=torch.cat((history[:,1:],nxt['actor'][:,None]),1)
        rows={};r=returns.reshape(groups,n);survival=survival.reshape(groups,n)*.02
        for i,name in enumerate(directions):
            diff=r[1+2*i]-r[2+2*i]
            rows[name]=dict(plus_minus_return_mean=float(diff.mean()),fraction_positive=float((diff>0).float().mean()),
                plus_return=float(r[1+2*i].mean()),minus_return=float(r[2+2*i].mean()),
                plus_survival_s=float(survival[1+2*i].mean()),minus_survival_s=float(survival[2+2*i].mean()))
        cosine={}
        for name in ['model3','model16']:
            cosine['Q_vs_'+name]=float(torch.nn.functional.cosine_similarity(directions['Q'],directions[name],dim=-1).mean())
        result=dict(checkpoint=str(args.checkpoint),iteration=ck['iter'],native_physics=True,auto_reset=False,
            phase_frames=frames.cpu().tolist(),paired_frames=n,groups=groups,rollout_steps=args.steps,
            epsilon_raw_action_rms=args.epsilon,baseline_return=float(r[0].mean()),gradient_l2_norm_mean=norms,
            gradient_cosine=cosine,directional_physical_results=rows,
            caveats='Local finite perturbations at reference initial states; deterministic continuation; no training and no simulator differentiation.')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    finally:runner.close();env.close()


if __name__=='__main__':main()
