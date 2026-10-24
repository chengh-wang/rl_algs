"""History/goal and differentiable-normalization adapters for Holosoma math.

The upstream actor, distributional Q and categorical projection remain unchanged.
No simulator, environment registry or Holosoma runtime is imported.
"""
import torch
from torch import nn
from mbdac.vendor.holosoma.fast_sac import Actor, Critic
from mbdac.vendor.holosoma.normalization import EmpiricalNormalization


class DifferentiableNormalization(EmpiricalNormalization):
    """Preserve source statistics/epsilon, allow input gradients with fixed stats."""
    def __init__(self, dim, device='cpu'):
        super().__init__((dim,), device)

    @torch.no_grad()
    def update(self, x):
        super().update(x.reshape(-1, x.shape[-1]).detach())

    def forward(self, x, center=True, update=False):
        if update:
            raise ValueError('Update normalization explicitly from real data only')
        return (x-self._mean)/(self._std+self.eps) if center else x/(self._std+self.eps)


class HolosomaActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, history, options):
        super().__init__()
        self.history=history; self.precision='fp32'
        self.obs_norm=DifferentiableNormalization(obs_dim)
        self.goal_norm=DifferentiableNormalization(goal_dim)
        dim=obs_dim*history+goal_dim
        self.net=Actor({'policy':dict(start=0,end=dim,size=dim)},['policy'],action_dim,1,
                       options['actor_hidden_dim'],options['log_std_max'],options['log_std_min'],
                       device='cpu',action_scale=torch.as_tensor(options['action_scale'],dtype=torch.float32).expand(action_dim).clone())
        if options.get('action_std_min') is not None:
            # tanh is 1-Lipschitz: scaled pre-tanh std upper-bounds action std.
            # Convert control-space bounds per joint; leave mean/PD range intact.
            for suffix in ['min','max']:
                value=torch.as_tensor(options['action_std_'+suffix],dtype=torch.float32).expand(action_dim)
                log_bound=torch.log(value/self.net.action_scale)
                delattr(self.net,'log_std_'+suffix)
                self.net.register_buffer('log_std_'+suffix,log_bound)

    def inputs(self, history, goal):
        # Flatten explicitly: source normalizer stats have a leading batch axis.
        normalized=self.obs_norm(history.reshape(-1,history.shape[-1])).reshape_as(history)
        return torch.cat((normalized.flatten(-2),self.goal_norm(goal)),-1)

    def forward(self, history, goal):
        return self.net(self.inputs(history,goal))[0]

    def sample(self, history, goal):
        return self.net.get_actions_and_log_probs(self.inputs(history,goal))


class HolosomaCritic(nn.Module):
    def __init__(self, dim, action_dim, options):
        super().__init__()
        self.norm=DifferentiableNormalization(dim)
        self.net=Critic({'critic':dict(start=0,end=dim,size=dim)},['critic'],action_dim,
                        options['num_atoms'],options['v_min'],options['v_max'],
                        options['critic_hidden_dim'],device='cpu',num_q_networks=options['num_q_networks'])

    def forward(self, features, actions):
        return self.net(self.norm(features),actions)

    def value(self, features, actions):
        probabilities=torch.softmax(self(features,actions),dim=-1)
        return self.net.get_value(probabilities).mean(dim=0)

    def projection(self, features, actions, rewards, bootstrap, discount):
        discount=torch.as_tensor(discount,device=rewards.device,dtype=rewards.dtype).expand_as(rewards)
        return self.net.projection(self.norm(features),actions,rewards,bootstrap,discount)
