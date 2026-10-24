"""Optional AMP prior. Experts and real policy transitions remain separate from imagination."""
import torch
from torch import nn
from .mbdac import MBDAC
from mbdac.modules.networks import mlp, RunningNorm


class AmpPrior(nn.Module):
    def __init__(self, feature_dim, hidden=(1024,512,256)):
        super().__init__()
        self.norm=RunningNorm(feature_dim)
        self.net=mlp(2*feature_dim,hidden,1)

    def score(self, phi0, phi1):
        return self.net(torch.cat((self.norm(phi0),self.norm(phi1)),-1)).squeeze(-1)

    def reward(self, phi0, phi1):
        return (1-.25*(self.score(phi0,phi1)-1).square()).clamp_min(0)


class MBDACAMP(MBDAC):
    def __init__(self, task, num_envs, cfg=None, device='cpu'):
        super().__init__(task,num_envs,cfg,device)
        self.prior=AmpPrior(task.amp_feature_dim,self.cfg.amp.get('hidden',(1024,512,256))).to(device)
        self.opt_prior=torch.optim.Adam(self.prior.parameters(),lr=self.cfg.lr)

    def _reward(self, state, condition, action, previous, state0, context):
        task_reward=super()._reward(state,condition,action,previous,state0,context)
        p0,p1=self.task.amp_features(state0),self.task.amp_features(state)
        weight=self.cfg.amp.get('weight',.25); scale=self.cfg.amp.get('reward_scale',.1)
        return (1-weight)*task_reward+weight*scale*self.prior.reward(p0,p1)

    def update_prior(self):
        count=self.cfg.amp.get('batch_size',self.cfg.pi_batch)
        real,_=self.replay.sample_starts(count,1)
        with torch.no_grad():
            p0,p1=self.task.amp_features(real['s']),self.task.amp_features(real['s1'])
            e0,e1=self.task.sample_expert_features(count)
            self.prior.norm.update(torch.cat((p0,p1,e0,e1)))
        expert=torch.cat((self.prior.norm(e0),self.prior.norm(e1)),-1).detach().requires_grad_(True)
        policy=torch.cat((self.prior.norm(p0),self.prior.norm(p1)),-1).detach()
        dp=self.prior.net(policy).squeeze(-1); de=self.prior.net(expert).squeeze(-1)
        grad=torch.autograd.grad(de.sum(),expert,create_graph=True)[0]
        penalty=self.cfg.amp.get('gradient_penalty',10.)*grad.square().sum(-1).mean()
        loss=.5*((dp+1).square().mean()+(de-1).square().mean())+penalty
        self.opt_prior.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(self.prior.parameters(),1.,error_if_nonfinite=True); self.opt_prior.step()
        return {'amp_loss':float(loss.detach()),'amp_gradient_penalty':float(penalty.detach()),'amp_policy_score':float(dp.detach().mean()),'amp_expert_score':float(de.detach().mean())}

    def update(self, iteration=0):
        logs=self.update_prior(); logs.update(super().update(iteration)); return logs

    def save(self, include_replay=False):
        state=super().save(include_replay); state.update(prior=self.prior.state_dict(),opt_prior=self.opt_prior.state_dict()); return state

    def load(self, state, load_cfg=None, strict=True):
        load_iteration=super().load(state,load_cfg,strict)
        if load_cfg is None or load_cfg.get('prior'):
            self.prior.load_state_dict(state['prior'],strict=strict)
            if load_cfg is None or load_cfg.get('optimizer'): self.opt_prior.load_state_dict(state['opt_prior'])
        return load_iteration
