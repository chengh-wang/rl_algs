"""Holosoma FastSAC with a short differentiable world-model actor objective.

Real-only C51 fitting anchors Q. WM is fit only to real physical transitions.
The actor follows an entropy-regularized model rollout bootstrapped by mean Q.
H=0 recovers Holosoma's actor objective; H>0 is an explicit algorithm extension.
"""
import copy
import math
from dataclasses import dataclass, field
import torch
from torch import nn
from .mbdac import MBDAC, MBDACConfig, frozen
from mbdac.modules.holosoma import HolosomaActor, HolosomaCritic


FAST_SAC_DEFAULTS=dict(actor_hidden_dim=512,critic_hidden_dim=768,num_atoms=501,
    v_min=-20.,v_max=20.,num_q_networks=2,log_std_min=-5.,log_std_max=0.,
    actor_lr=3e-4,critic_lr=3e-4,alpha_lr=3e-4,alpha_init=.001,
    target_entropy_ratio=.5,tau=.05,critic_updates=4,policy_frequency=2,
    critic_batch=256,actor_grad_clip=1.,critic_grad_clip=0.,action_scale=1.,
    weight_decay=.001,alpha_weight_decay=.01,model_actor_weight=1.,
    action_std_min=None,action_std_max=None)


@dataclass
class MBDACFastSACConfig(MBDACConfig):
    horizon: int = 3
    fast_sac: dict = field(default_factory=dict)


class MBDACFastSAC(MBDAC):
    def __init__(self, task, num_envs, cfg=None, device='cpu'):
        cfg=copy.deepcopy(cfg or MBDACFastSACConfig())
        unknown=set(cfg.fast_sac)-set(FAST_SAC_DEFAULTS)
        if unknown: raise ValueError(f'Unknown FastSAC options: {sorted(unknown)}')
        cfg.fast_sac={**FAST_SAC_DEFAULTS,**cfg.fast_sac}
        f=cfg.fast_sac
        bounds=torch.as_tensor(f['action_scale'],dtype=torch.float32)
        if (not torch.isfinite(bounds).all() or (bounds<=0).any() or
                bounds.numel() not in (1,task.action_dim) or bounds.max()>cfg.action_clip):
            raise ValueError('FastSAC action bounds must be finite, positive and within action_clip')
        if (cfg.horizon<0 or cfg.history<1 or f['critic_updates']<1 or f['policy_frequency']<1 or
                f['num_atoms']<2 or f['v_min']>=f['v_max'] or not 0<f['tau']<=1 or f['alpha_init']<=0 or
                f['num_q_networks']<1 or f['critic_batch']<1 or f['actor_hidden_dim']<4 or f['critic_hidden_dim']<4 or
                f['log_std_min']>f['log_std_max'] or min(f['actor_lr'],f['critic_lr'],f['alpha_lr'])<=0):
            raise ValueError('Invalid FastSAC horizon, update schedule or value support')
        if not 0<=f['model_actor_weight']<=1:
            raise ValueError('model_actor_weight must be in [0,1]')
        if (f['action_std_min'] is None)!=(f['action_std_max'] is None):
            raise ValueError('Specify both raw-action standard deviation bounds')
        if f['action_std_min'] is not None:
            lo,hi=(torch.as_tensor(f[name],dtype=torch.float32) for name in ['action_std_min','action_std_max'])
            if (lo.numel() not in (1,task.action_dim) or hi.numel() not in (1,task.action_dim) or
                    not torch.isfinite(lo).all() or not torch.isfinite(hi).all() or (lo<=0).any() or (hi<lo).any()):
                raise ValueError('Invalid raw-action standard deviation bounds')
        super().__init__(task,num_envs,cfg,device)
        self.opt_actor=torch.optim.AdamW(self.actor.parameters(),lr=f['actor_lr'],betas=(.9,.95),weight_decay=f['weight_decay'])
        self.opt_critic=torch.optim.AdamW(self.critic.parameters(),lr=f['critic_lr'],betas=(.9,.95),weight_decay=f['weight_decay'])
        self.target_critic=copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.log_alpha=nn.Parameter(torch.tensor([math.log(f['alpha_init'])],device=self.device))
        self.opt_alpha=torch.optim.AdamW([self.log_alpha],lr=f['alpha_lr'],betas=(.9,.95),weight_decay=f['alpha_weight_decay'])
        self.target_entropy=-task.action_dim*f['target_entropy_ratio']
        self.critic_updates=0; self.actor_updates=0

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        options=dict(cfg['algorithm']); options.pop('class_name',None)
        return cls(env.model_task,env.num_envs,MBDACFastSACConfig(**options),device)

    def _make_actor(self):
        t=self.task
        return HolosomaActor(t.obs_dim,t.goal_dim,t.action_dim,self.cfg.history,self.cfg.fast_sac)

    def _make_critic(self):
        return HolosomaCritic(self.task.critic_dim,self.task.action_dim,self.cfg.fast_sac)

    @torch.no_grad()
    def act(self, obs):
        o,g=obs['actor'],obs['goal']
        self.actor.obs_norm.update(o); self.actor.goal_norm.update(g)
        if self.history is None:
            self.history=o[:,None].expand(-1,self.cfg.history,-1).clone()
        else:
            h=torch.cat((self.history[:,1:],o[:,None]),1)
            self.history=torch.where(self.fresh[:,None,None],o[:,None].expand_as(h),h)
        # No additive exploration noise or post-sampling clipping: preserve density.
        return self.actor.sample(self.history,g)[0]

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        tr={**extras['model_transition'],'reward':rewards.detach().clone()}
        super().process_env_step(obs,rewards,dones,{**extras,'model_transition':tr})
        c1=self._next_condition(tr)
        current=self._critic_features(tr['s'],tr['condition'],tr['context'])
        future=self._critic_features(tr['s1'],c1,tr['context'])
        self.critic.norm.update(torch.cat((current,future)))
        self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())

    def _next_condition(self, transition):
        # Exogenous command changes belong to the adapter, not the dynamics WM.
        # Deterministic motion clocks can use the analytic advance fallback.
        if 'condition1' in transition: return transition['condition1']
        return self.task.advance(transition['condition'],state=transition['s1'],context=transition['context'])

    def _history(self, history):
        return self.task.add_noise(self.task.observation(history['s'],history['a_prev'],history['context'],
            history['condition'],primitives=self._primitives(history['s'],history['context'])))

    @torch.no_grad()
    def _real_batch(self, batch):
        tr,h=self.replay.sample_starts(batch,self.cfg.history)
        hist=self._history(h)
        cond1=self._next_condition(tr)
        obs1=self.task.add_noise(self.task.observation(tr['s1'],tr['a'],tr['context'],cond1,
                                  primitives=self._primitives(tr['s1'],tr['context'])))
        return tr,hist,torch.cat((hist[:,1:],obs1[:,None]),1),cond1

    def update_real_critic(self):
        f=self.cfg.fast_sac
        tr,hist,next_hist,cond1=self._real_batch(f['critic_batch'])
        with torch.no_grad():
            features=self._critic_features(tr['s'],tr['condition'],tr['context'])
            next_features=self._critic_features(tr['s1'],cond1,tr['context'])
            action1,logp1=self.actor.sample(next_hist,self.task.goal(cond1))
            # True terminations stop bootstrap; time limits keep the physical s1.
            bootstrap=(~tr['terminated'].bool()).float()
            targets=self.target_critic.projection(next_features,action1,
                tr['reward']-self.cfg.gamma*bootstrap*self.log_alpha.exp()*logp1,
                bootstrap,self.cfg.gamma)
        logits=self.critic(features,tr['a'])
        # Match Holosoma: mean over batch, then sum over Q ensemble.
        loss=-(targets*torch.log_softmax(logits,dim=-1)).sum(-1).mean(dim=1).sum()
        self.opt_critic.zero_grad(set_to_none=True); loss.backward()
        norm=nn.utils.clip_grad_norm_(self.critic.parameters(),f['critic_grad_clip'] or float('inf'),error_if_nonfinite=True)
        self.opt_critic.step()
        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss=(-self.log_alpha.exp()*(logp1.detach()+self.target_entropy)).mean()
        alpha_loss.backward()
        if not torch.isfinite(self.log_alpha.grad).all(): raise FloatingPointError('Non-finite temperature gradient')
        self.opt_alpha.step()
        with torch.no_grad():
            for source,target in zip(self.critic.parameters(),self.target_critic.parameters()):
                target.lerp_(source,f['tau'])
            self.target_critic.norm.load_state_dict(self.critic.norm.state_dict())
        self.critic_updates+=1
        return dict(critic_loss=float(loss.detach()),critic_grad_norm=float(norm),
                    alpha_loss=float(alpha_loss.detach()),alpha=float(self.log_alpha.exp().detach()),
                    real_reward=float(tr['reward'].mean()),target_edge_mass=float((targets[...,[0,-1]].sum(-1)).mean()))

    def actor_objective(self, start, history):
        """Return differentiable mean finite-horizon soft return; starts are fixed."""
        s=start['s']; cond=start['condition']; ctx=start['context']; previous=start['a_prev']
        hist=history; alive=torch.ones(len(s),device=self.device); objective=torch.zeros_like(alive)
        entropies=[]; rewards=[]
        for t in range(self.cfg.horizon):
            action,logp=self.actor.sample(hist,self.task.goal(cond))
            s1=self._wm_step(s,action,self.task.normalize_context(ctx))
            cond1=self.task.advance(cond,state=s1,context=ctx)
            r,d=self._reward_and_termination(s1,cond1,action,previous,s,ctx)
            objective=objective+self.cfg.gamma**t*alive*(r-self.log_alpha.exp().detach()*logp)
            alive=alive*(1-d)
            obs=self.task.add_noise(self.task.observation(s1,action,ctx,cond1,primitives=self._primitives(s1,ctx)))
            hist=torch.cat((hist[:,1:],obs[:,None]),1)
            s,cond,previous=s1,cond1,action
            entropies.append(-logp); rewards.append(r)
        terminal_action,terminal_logp=self.actor.sample(hist,self.task.goal(cond))
        value=self.critic.value(self._critic_features(s,cond,ctx),terminal_action)
        objective=objective+self.cfg.gamma**self.cfg.horizon*alive*(value-self.log_alpha.exp().detach()*terminal_logp)
        entropies.append(-terminal_logp)
        return objective.mean(),dict(imagined_reward=float(torch.stack(rewards).detach().mean()) if rewards else 0.,
            imagined_survival=float(alive.detach().mean()),entropy=float(torch.stack(entropies).detach().mean()))

    def update_actor(self):
        start,h=self._sample_starts()
        with torch.no_grad(): history=self._history(h)
        self.opt_wm.zero_grad(set_to_none=True); self.opt_critic.zero_grad(set_to_none=True)
        self.opt_alpha.zero_grad(set_to_none=True)
        with frozen(self.world_model,self.critic):
            weight=self.cfg.fast_sac['model_actor_weight']
            logs={}
            if weight>0 or self.cfg.horizon==0:
                objective,logs=self.actor_objective(start,history)
                logs['model_actor_loss']=float(-objective.detach())
            if weight<1 and self.cfg.horizon>0:
                action,logp=self.actor.sample(history,self.task.goal(start['condition']))
                value=self.critic.value(self._critic_features(start['s'],start['condition'],start['context']),action)
                real_objective=(value-self.log_alpha.exp().detach()*logp).mean()
                objective=weight*objective+(1-weight)*real_objective if weight else real_objective
                logs.update(real_actor_loss=float(-real_objective.detach()),real_actor_entropy=float(-logp.detach().mean()))
            loss=-objective
            self.opt_actor.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(self.actor.parameters(),self.cfg.fast_sac['actor_grad_clip'] or float('inf'),error_if_nonfinite=True)
            self.opt_actor.step()
        with torch.no_grad():
            _,_,log_std=self.actor.net(self.actor.inputs(history,self.task.goal(start['condition'])))
            std_bound=self.actor.net.action_scale*log_std.exp()
            logs.update(action_std_bound_mean=float(std_bound.mean()),action_std_bound_max=float(std_bound.max()))
        self.actor_updates+=1
        return {**logs,'actor_loss':float(loss.detach()),'actor_grad_norm':float(norm),'model_actor_weight':weight}

    def update_actor_critic(self):
        """Common learner entry point, using real Q fits and model policy steps."""
        c=self.cfg;rows=[]
        for _ in range(c.fast_sac['critic_updates']):
            row=self.update_real_critic()
            if self.critic_updates%c.fast_sac['policy_frequency']==0: row.update(self.update_actor())
            rows.append(row)
        return {key:sum(r[key] for r in rows if key in r)/sum(key in r for r in rows)
                for key in set().union(*(r.keys() for r in rows))}

    def update(self, iteration=0):
        c=self.cfg; factor=.03+.97*.5*(1+math.cos(math.pi*min(iteration/c.lr_decay_iters,1)))
        self.learning_rate=c.lr*factor
        for group in self.opt_wm.param_groups: group['lr']=self.learning_rate
        logs={}
        for _ in range(c.wm_batches):
            for k,v in self.update_world_model().items(): logs[k]=logs.get(k,0.)+v/c.wm_batches
        if iteration>=c.wm_only_iters:
            logs.update(self.update_actor_critic())
        return {**logs,'optimizer/critic_updates':self.critic_updates,'optimizer/actor_updates':self.actor_updates,
                'actor_learning_rate':self.opt_actor.param_groups[0]['lr']}

    def save(self, include_replay=False):
        state=super().save(include_replay)
        state.update(target_critic=self.target_critic.state_dict(),log_alpha=self.log_alpha.detach().clone(),
                     opt_alpha=self.opt_alpha.state_dict(),critic_updates=self.critic_updates,actor_updates=self.actor_updates)
        return state

    def load(self, state, load_cfg=None, strict=True):
        load_iteration=super().load(state,load_cfg,strict)
        if load_cfg is None or load_cfg.get('critic'):
            self.target_critic.load_state_dict(state['target_critic'],strict=strict)
        if load_cfg is None or load_cfg.get('temperature'):
            with torch.no_grad(): self.log_alpha.copy_(state['log_alpha'])
            if load_cfg is None or load_cfg.get('optimizer'): self.opt_alpha.load_state_dict(state['opt_alpha'])
        if load_iteration:
            self.critic_updates=state['critic_updates']; self.actor_updates=state['actor_updates']
        self.learning_rate=self.opt_wm.param_groups[0]['lr']
        return load_iteration
