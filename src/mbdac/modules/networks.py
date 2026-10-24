"""Networks adapted from SuperMimic-vibed (MIT, ETH Zurich/Chong Zhang)."""
import torch
from torch import nn


def mlp(inp, hidden, out, activation=nn.ELU, layer_norm=False):
    layers=[]
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        inp=width
    return nn.Sequential(*layers, nn.Linear(inp, out))


class RunningNorm(nn.Module):
    """Statistics change only through explicit update calls, never forward().

    Actor/world-model/primitive statistics use real data. The critic follows
    the reference method and normalizes its detached imagined features.
    """
    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x):
        x=x.reshape(-1, x.shape[-1]).float()
        if not len(x) or self.count >= self.max_count:
            return
        if not torch.isfinite(x).all():
            raise FloatingPointError('Non-finite normalization input')
        n=len(x); total=self.count+n; delta=x.mean(0)-self.mean
        self.var.copy_((self.var*self.count+x.var(0,unbiased=False)*n+delta.square()*self.count*n/total)/total)
        self.mean.add_(delta*n/total); self.count.copy_(total)

    def std(self):
        return torch.sqrt(self.var+self.eps**2)

    def forward(self, x):
        return (x-self.mean)/self.std()


def run_network(net, x, precision='fp32'):
    if precision=='bf16' and x.is_cuda:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return net(x).float()
    return net(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, history=10, hidden=(2048,1024,512,256), latent_dim=32, precision='fp32'):
        super().__init__()
        self.history, self.precision=history,precision
        self.encoder=mlp(goal_dim,hidden,latent_dim)
        self.decoder=mlp(latent_dim+obs_dim*history,hidden,action_dim)
        self.obs_norm=RunningNorm(obs_dim); self.goal_norm=RunningNorm(goal_dim)
        with torch.no_grad():
            self.decoder[-1].weight.mul_(.01); self.decoder[-1].bias.zero_()

    def forward(self, obs, goal):
        z=torch.tanh(run_network(self.encoder,self.goal_norm(goal),self.precision))
        return run_network(self.decoder,torch.cat((z,self.obs_norm(obs).flatten(-2)),-1),self.precision)


class Critic(nn.Module):
    def __init__(self, dim, hidden=(256,128), precision='fp32'):
        super().__init__()
        self.norm=RunningNorm(dim); self.net=mlp(dim,hidden,1); self.precision=precision

    def forward(self, x):
        return run_network(self.net,self.norm(x),self.precision).squeeze(-1)
