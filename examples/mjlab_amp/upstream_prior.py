"""Reuse the pinned AMP fork's modules; keep its state outside the learning core."""
import numpy as np
import torch
from torch import nn
from .model_task import BaselineVelocityTask


class UpstreamNormalizer(nn.Module):
    def __init__(self,dim):
        super().__init__()
        from rsl_rl.utils.utils import Normalizer
        self.original=Normalizer(dim)

    def forward(self,x): return self.original.normalize_torch(x,x.device)

    @torch.no_grad()
    def update(self,x): self.original.update(x.detach().cpu().numpy())

    def get_extra_state(self):
        n=self.original
        return dict(mean=n.mean.tolist(),var=n.var.tolist(),count=float(n.count))

    def set_extra_state(self,state):
        n=self.original
        n.mean=np.asarray(state['mean'],dtype=np.float64)
        n.var=np.asarray(state['var'],dtype=np.float64); n.count=state['count']


class UpstreamAmpPrior(nn.Module):
    normalize_before_update=False

    def __init__(self,feature_dim,hidden=(1024,512,256),device='cpu'):
        super().__init__()
        from rsl_rl.modules.discriminator import Discriminator
        self.feature_dim=feature_dim; self.norm=UpstreamNormalizer(feature_dim)
        self.discriminator=Discriminator(2*feature_dim,.1,list(hidden),device)

    @property
    def net(self): return self.discriminator

    def score(self,p0,p1):
        return self.net(torch.cat((self.norm(p0),self.norm(p1)),-1)).squeeze(-1)

    def reward(self,p0,p1):
        # The upstream prediction helper uses no_grad for PPO collection.
        # Preserve its forward formula while allowing model-based input gradients.
        return (1-.25*(self.score(p0,p1)-1).square()).clamp_min(0)

    def optimizer_parameters(self):
        return [dict(params=self.discriminator.trunk.parameters(),weight_decay=1e-3),
                dict(params=self.discriminator.amp_linear.parameters(),weight_decay=.1)]

    def gradient_penalty(self,e0,e1,strength):
        return self.discriminator.compute_grad_pen(e0,e1,lambda_=strength)

    @torch.no_grad()
    def after_update(self,policy,expert):
        # Match the fork's update ordering and data exactly: it updates statistics
        # from the normalized current-state tensors after its optimizer step.
        self.norm.update(policy[:,:self.feature_dim])
        self.norm.update(expert[:,:self.feature_dim])


class UpstreamVelocityTask(BaselineVelocityTask):
    def make_amp_prior(self,device,options):
        from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_ppo_runner_cfg
        cfg=g1_amp_ppo_runner_cfg()
        return UpstreamAmpPrior(self.amp_feature_dim,options.get('hidden',cfg.amp_discr_hidden_dims),device)
