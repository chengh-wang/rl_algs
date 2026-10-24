"""Model-based differentiable actor-critic, following SuperMimic equations 1-3.

The simulator is used only for data collection. No simulator SDK is imported here.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch
from torch import nn
from mbdac.modules.networks import Actor, Critic
from mbdac.modules.world_model import WorldModel
from mbdac.storage.replay import ReplayBuffer


@dataclass
class MBDACConfig:
    history: int = 10
    horizon: int = 32
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 128
    pi_batch: int = 256
    buffer_steps: int = 256
    wm_only_iters: int = 500
    warmup_steps: int = 64
    utd: float = .4
    lr: float = 2e-4
    lr_decay_iters: int = 250000
    gamma: float = .99
    lam: float = .95
    action_noise: float = .1
    action_clip: float = 20.
    critic_epochs: int = 5
    critic_minibatches: int = 4
    precision: str = 'bf16'
    actor_hidden: tuple = (2048,1024,512,256)
    wm_hidden: tuple = (2048,2048,2048)
    critic_hidden: tuple = (256,128)
    compile: bool = False
    amp: dict = field(default_factory=dict)


@contextmanager
def frozen(*modules):
    saved=[[(p,p.requires_grad) for p in m.parameters()] for m in modules]
    for rows in saved:
        for p,_ in rows: p.requires_grad_(False)
    try:
        yield
    finally:
        for rows in saved:
            for p,flag in rows: p.requires_grad_(flag)


def lambda_returns(rewards, termination_probability, values, gamma, lam):
    target=values[-1]; result=[]
    for t in range(len(rewards)-1,-1,-1):
        target=rewards[t]+gamma*(1-termination_probability[t])*((1-lam)*values[t+1]+lam*target)
        result.append(target)
    return torch.stack(result[::-1])


class MBDAC:
    def __init__(self, task, num_envs, cfg=None, device='cpu'):
        self.task=task; self.cfg=cfg or MBDACConfig(); c=self.cfg
        if c.buffer_steps<c.wm_window or c.utd<=0: raise ValueError('Invalid replay window or update/data ratio')
        self.device=torch.device(device); self.num_envs=num_envs
        duplicates=set(getattr(task,'analytic_primitives',())) & {p.name for p in getattr(task,'primitive_specs',())}
        if duplicates:
            raise ValueError(f'Primitives already available from state/kinematics: {sorted(duplicates)}')
        self.world_model=WorldModel(task.codec,task.action_dim,task.context_dim,c.wm_hidden,c.precision,getattr(task,'primitive_specs',())).to(device)
        self.actor=Actor(task.obs_dim,task.goal_dim,task.action_dim,c.history,c.actor_hidden,precision=c.precision).to(device)
        self.critic=Critic(task.critic_dim,c.critic_hidden,c.precision).to(device)
        self.opt_wm=torch.optim.Adam(self.world_model.parameters(),lr=c.lr)
        self.opt_actor=torch.optim.Adam(self.actor.parameters(),lr=c.lr,betas=(.9,.95))
        self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=c.lr)
        self.replay=ReplayBuffer(c.buffer_steps,num_envs,device)
        self.history=None; self.fresh=torch.ones(num_envs,dtype=torch.bool,device=device)
        self.learning_rate=c.lr
        self._wm_step=torch.compile(self.world_model.forward) if c.compile else self.world_model.forward
        self._reward_term=torch.compile(self._reward_and_termination) if c.compile else self._reward_and_termination
        self._critic_input=torch.compile(self._critic_features) if c.compile else self._critic_features

    @classmethod
    def construct_algorithm(cls, obs, env, cfg, device):
        options=dict(cfg['algorithm']); options.pop('class_name',None)
        return cls(env.model_task,env.num_envs,MBDACConfig(**options),device)

    @torch.no_grad()
    def act(self, obs):
        o,g=obs['actor'],obs['goal']
        self.actor.obs_norm.update(o); self.actor.goal_norm.update(g)
        if self.history is None: self.history=o[:,None].expand(-1,self.cfg.history,-1).clone()
        else:
            history=torch.cat((self.history[:,1:],o[:,None]),1)
            self.history=torch.where(self.fresh[:,None,None],o[:,None].expand_as(history),history)
        a=self.actor(self.history,g)+self.cfg.action_noise*torch.randn(len(o),self.task.action_dim,device=self.device)
        return a.clamp(-self.cfg.action_clip,self.cfg.action_clip)

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        if 'model_transition' not in extras: raise KeyError('Adapter must provide pre-reset model_transition')
        tr=extras['model_transition']; self.replay.add(tr)
        ctx=self.task.normalize_context(tr['context'])
        self.world_model.input_norm.update(self.world_model.inputs(tr['s'],tr['a'],ctx))
        self.world_model.output_norm.update(self.task.codec.targets(tr['s'],tr['s1']))
        labels={k.split('/',1)[1]:v for k,v in tr.items() if k.startswith('primitive/')}
        self.world_model.primitive_heads.update_normalization(labels,self.world_model.primitive_inputs(tr['s1'],ctx))
        self.fresh=dones.bool().clone()

    def _wm_rollout(self, w):
        state=w['s'][:,0]; preds=[]
        for j in range(self.cfg.wm_window):
            state=self._wm_step(state,w['a'][:,j],self.task.normalize_context(w['context'][:,j]))
            preds.append(state)
        return torch.stack(preds,1)

    def update_world_model(self):
        c=self.cfg; w=self.replay.sample_windows(c.wm_batch,c.wm_window)
        predicted=self._wm_rollout(w)
        scale=torch.maximum(self.world_model.output_norm.std(),self.task.codec.error_floor)
        error=(self.task.codec.error(predicted,w['s1'])/scale).square().mean(-1)
        loss=(error*w['mask']).sum()/c.wm_batch
        # Primitive labels describe s1 before reset. Fit physical measurements on
        # real states, so a head cannot compensate for errors in predicted dynamics.
        features=self.world_model.primitive_inputs(w['s1'],self.task.normalize_context(w['context']))
        labels={k.split('/',1)[1]:v for k,v in w.items() if k.startswith('primitive/')}
        loss=loss+self.world_model.primitive_heads.loss(features,labels,w['mask'])
        if not torch.isfinite(loss): raise FloatingPointError('World-model loss is non-finite')
        self.opt_wm.zero_grad(set_to_none=True); loss.backward()
        norm=nn.utils.clip_grad_norm_(self.world_model.parameters(),1.,error_if_nonfinite=True)
        self.opt_wm.step()
        mask=w['mask'][:,-1]
        return {'wm_loss':float(loss.detach()/c.wm_window),'wm_error_last':float((error[:,-1].detach()*mask).sum()/mask.sum().clamp_min(1)), 'wm_grad_norm':float(norm)}

    def _sample_starts(self):
        c=self.cfg; starts=[]; histories=[]; found=0
        for _ in range(100):
            s,h=self.replay.sample_starts(2*c.pi_batch,c.history)
            valid=self.task.valid_start(s['condition'],c.horizon)
            idx=valid.nonzero().flatten()[:c.pi_batch-found]
            starts.append({k:v[idx] for k,v in s.items()}); histories.append({k:v[idx] for k,v in h.items()}); found+=len(idx)
            if found==c.pi_batch:
                cat=lambda rows:{k:torch.cat([r[k] for r in rows]) for k in rows[0]}
                return cat(starts),cat(histories)
        raise RuntimeError('No sufficient valid imagination starts; check goal/clip boundaries')

    def _primitives(self, state, context):
        return self.world_model.primitives(state,self.task.normalize_context(context))

    def _reward(self, state, condition, action, previous, state0, context):
        return self.task.reward(state,condition,action,previous,primitives=self._primitives(state,context))

    def _critic_features(self,state,condition,context):
        return self.task.critic_features(state,condition,context,primitives=self._primitives(state,context))

    def _reward_and_termination(self,state,condition,action,previous,state0,context):
        return (self._reward(state,condition,action,previous,state0,context),
                self.task.termination_probability(state,condition,primitives=self._primitives(state,context)))

    def update_actor_critic(self):
        c=self.cfg; start,h=self._sample_starts(); s=start['s']; cond=start['condition']; ctx=start['context']; a_prev=start['a_prev']
        with torch.no_grad():
            hist=self.task.observation(h['s'],h['a_prev'],h['context'],h['condition'],primitives=self._primitives(h['s'],h['context']))
        hist=self.task.add_noise(hist)
        states=[s]; conditions=[cond]; actions=[]; previous=[]
        self.opt_wm.zero_grad(set_to_none=True); self.opt_critic.zero_grad(set_to_none=True)
        frozen_modules=[self.world_model,self.critic]
        if hasattr(self,'prior'):
            self.opt_prior.zero_grad(set_to_none=True); frozen_modules.append(self.prior)
        with frozen(*frozen_modules):
            for _ in range(c.horizon):
                action=self.actor(hist,self.task.goal(cond)).clamp(-c.action_clip,c.action_clip)
                s1=self._wm_step(s,action,self.task.normalize_context(ctx)); cond1=self.task.advance(cond)
                prim=self._primitives(s1,ctx)
                obs=self.task.add_noise(self.task.observation(s1,action,ctx,cond1,primitives=prim))
                hist=torch.cat((hist[:,1:],obs[:,None]),1)
                states.append(s1); conditions.append(cond1); actions.append(action); previous.append(a_prev)
                s,cond,a_prev=s1,cond1,action
            # Evaluate analytic heads over the whole horizon in one batched call.
            flat=lambda rows:torch.stack(rows).flatten(0,1)
            all_ctx=ctx[None].expand(c.horizon,*ctx.shape).flatten(0,1)
            r,d=self._reward_term(flat(states[1:]),flat(conditions[1:]),flat(actions),flat(previous),flat(states[:-1]),all_ctx)
            r,d=r.reshape(c.horizon,-1),d.reshape(c.horizon,-1)
            alive=torch.cumprod(1-d,0); ct=torch.cat((torch.ones_like(alive[:1]),alive[:-1]),0)
            discount=c.gamma**torch.arange(c.horizon,device=self.device)[:,None]
            terminal=self.critic(self._critic_input(s,cond,ctx))
            objective=(discount*ct*r).sum(0)+c.gamma**c.horizon*alive[-1]*terminal
            loss=-objective.mean()
            self.opt_actor.zero_grad(set_to_none=True); loss.backward()
            norm=nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
            nn.utils.clip_grad_value_(self.actor.parameters(),.1); self.opt_actor.step()
        with torch.no_grad():
            all_ctx=ctx[None].expand(c.horizon+1,*ctx.shape).flatten(0,1)
            x=self._critic_input(flat([st.detach() for st in states]),flat(conditions),all_ctx).reshape(c.horizon+1,c.pi_batch,-1)
            self.critic.norm.update(x); values=self.critic(x)
            targets=lambda_returns(r.detach(),d.detach(),values,c.gamma,c.lam)
        xf,yf=x[:-1].flatten(0,1),targets.flatten(); n=len(yf); vloss=0.
        for _ in range(c.critic_epochs):
            for idx in torch.randperm(n,device=self.device).chunk(c.critic_minibatches):
                l=(self.critic(xf[idx])-yf[idx]).square().mean()
                self.opt_critic.zero_grad(set_to_none=True); l.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),1.,error_if_nonfinite=True); self.opt_critic.step(); vloss+=float(l.detach())
        return {'actor_loss':float(loss.detach()),'critic_loss':vloss/(c.critic_epochs*c.critic_minibatches),
                'actor_grad_norm':float(norm),'imagined_reward':float((ct*r).detach().mean()),'imagined_survival':float(alive[-1].detach().mean())}

    def update(self, iteration=0):
        import math
        c=self.cfg; factor=.03+.97*.5*(1+math.cos(math.pi*min(iteration/c.lr_decay_iters,1)))
        self.learning_rate=c.lr*factor
        for opt in [self.opt_wm,self.opt_actor,self.opt_critic]:
            for group in opt.param_groups: group['lr']=self.learning_rate
        logs={}
        for _ in range(c.wm_batches):
            for k,v in self.update_world_model().items(): logs[k]=logs.get(k,0)+v/c.wm_batches
        if iteration>=c.wm_only_iters: logs.update(self.update_actor_critic())
        return logs

    def train_mode(self):
        for m in [self.actor,self.critic,self.world_model]: m.train()

    def eval_mode(self):
        for m in [self.actor,self.critic,self.world_model]: m.eval()

    def get_policy(self): return self.actor

    def save(self, include_replay=False):
        result={'actor':self.actor.state_dict(),'critic':self.critic.state_dict(),'world_model':self.world_model.state_dict(),
                'opt_actor':self.opt_actor.state_dict(),'opt_critic':self.opt_critic.state_dict(),'opt_wm':self.opt_wm.state_dict()}
        if include_replay: result['replay']=self.replay.state_dict()
        return result

    def load(self, state, load_cfg=None, strict=True):
        options=load_cfg if load_cfg is not None else dict(actor=True,critic=True,world_model=True,optimizer=True,replay=True,iteration=True)
        for name in ['actor','critic','world_model']:
            if options.get(name): getattr(self,name).load_state_dict(state[name],strict=strict)
        if options.get('optimizer'):
            for name in ['opt_actor','opt_critic','opt_wm']: getattr(self,name).load_state_dict(state[name])
            self.learning_rate=self.opt_actor.param_groups[0]['lr']
        self.replay=ReplayBuffer(self.cfg.buffer_steps,self.num_envs,self.device)
        if options.get('replay') and 'replay' in state: self.replay.load_state_dict(state['replay'])
        self.history=None; self.fresh.fill_(True)
        return options.get('iteration',False)
