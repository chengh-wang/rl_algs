"""Bounded potential shaping with explicit episode-boundary contracts.

This preserves the original task objective under the stated assumptions.
Ideal policy invariance requires the same discount, correct terminal handling,
consistent endpoint potentials, a temporally fixed positive reward divisor,
and no downstream nonlinear reward changes. A divisor that changes between
transitions generally prevents the potential terms from telescoping; the
public FlashSAC integration therefore requires fixed reward normalization for
this provider. The divisor must be fixed for the full shaped-return objective,
not merely shared across rows of one minibatch.
Function approximation, entropy, and distributional target clipping remain
properties of the learner; this module supplies no performance guarantee.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class PotentialShapingConfig:
    potential_id: str
    version: str
    gamma: float = .99
    beta: float = 1.
    max_abs_potential: float = 1.

    def __post_init__(self):
        for name in ('potential_id','version'):
            value=getattr(self,name)
            if not isinstance(value,str) or not value.strip():
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('gamma','beta','max_abs_potential'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.gamma <= 1 or self.beta < 0 or self.max_abs_potential <= 0:
            raise ValueError('Require gamma in [0,1], beta >= 0, max_abs_potential > 0')


def _vector(value,name,reference=None,mask=False):
    if not isinstance(value,torch.Tensor) or value.ndim!=1 or not value.numel():
        raise ValueError(f'{name} must be a nonempty tensor [N]')
    if mask:
        if value.dtype!=torch.bool:
            raise ValueError(f'{name} must have boolean dtype')
    elif not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite floating point')
    if reference is not None:
        if value.shape!=reference.shape or value.device!=reference.device:
            raise ValueError(f'{name} must match reward shape and device')
        if not mask and value.dtype!=reference.dtype:
            raise ValueError(f'{name} must match reward dtype')


def compose_potential_reward(task_reward,current,next_potential,terminated,config):
    """Differentiable r + beta*(gamma*(1-terminal)*Phi_next - Phi).

    Timeouts retain their real pre-reset endpoint potential. A history with no
    valid potential must supply Phi=0; do not mask the entire difference.
    Bounds are validated, never silently clipped. gamma=1 requires finite
    episodes with zero terminal potential for the telescoping guarantee.
    """
    _vector(task_reward,'task_reward')
    _vector(current,'current',task_reward)
    _vector(next_potential,'next',task_reward)
    _vector(terminated,'terminated',task_reward,mask=True)
    if (current.abs()>config.max_abs_potential).any() or (next_potential.abs()>config.max_abs_potential).any():
        raise ValueError('Potential exceeds declared max_abs_potential')
    endpoint=torch.where(terminated,torch.zeros_like(next_potential),next_potential)
    shaped=task_reward+config.beta*(config.gamma*endpoint-current)
    if not torch.isfinite(shaped).all():
        raise ValueError('Potential composition produced nonfinite reward')
    return shaped


class PotentialRewardProvider:
    """Sequential provider for public runners; original task reward stays visible.

    The caller explicitly declares pre-reset endpoint provenance. This module
    cannot inspect a simulator to verify it. Version identifies the actual
    potential/noise/normalizer definition. Changes require a new provider/run.
    Calls must include even invalid-history and terminal transitions; otherwise
    exact cached boundary continuity cannot be maintained.
    """
    def __init__(self,config: PotentialShapingConfig,*,next_potential_is_pre_reset: bool):
        if not isinstance(config,PotentialShapingConfig):
            raise ValueError('Expected PotentialShapingConfig')
        if next_potential_is_pre_reset is not True:
            raise ValueError('Explicit pre-reset next-potential declaration is required')
        self.config=config
        self.calls=0
        self._next=None
        self._done=None
        self.last_task_reward=None

    @property
    def contract(self):
        c=self.config
        return dict(kind='potential_based_shaping',version=1,potential_id=c.potential_id,
            potential_version=c.version,algorithm_discount=c.gamma,beta=c.beta,
            max_abs_potential=c.max_abs_potential,next_potential_semantics='pre_reset',
            normalization_base='task_reward',next_potential_is_pre_reset=True,reward_units='same_as_task_reward',
            continuity='exact_cached_boundary',terminal_next_potential=0.)

    def __call__(self,reward,extras):
        if not isinstance(extras,dict):
            raise ValueError('extras must be a mutable dict')
        if 'terminated' not in extras or 'truncated' not in extras:
            raise ValueError('Both explicit terminated and truncated boolean masks are required')
        _vector(reward,'task_reward')
        for key in ('terminated','truncated'):
            _vector(extras[key],key,reward,mask=True)
        payload=extras.get('reward_potential')
        if not isinstance(payload,Mapping) or payload.get('version')!=self.config.version:
            raise ValueError('Missing or mismatched reward_potential version')
        current=payload.get('current');nxt=payload.get('next')
        # Validate before reading cache or publishing any persistent state.
        shaped=compose_potential_reward(reward,current,nxt,extras['terminated'],self.config)
        if self._next is not None:
            if current.shape!=self._next.shape or current.device!=self._next.device or current.dtype!=self._next.dtype:
                raise ValueError('Potential batch changed; explicitly reset provider before a fresh rollout')
            active=~self._done
            if not torch.equal(current[active].detach(),self._next[active]):
                raise ValueError('Potential continuity mismatch: current must equal cached previous next on continuing rows')
        self._next=nxt.detach().clone()
        self._done=(extras['terminated']|extras['truncated']).detach().clone()
        self.last_task_reward=reward.detach().clone()
        extras['task_reward']=reward.detach().clone()
        self.calls+=1
        return shaped

    def reset(self):
        """Start fresh simulator histories; retain persistent accounting."""
        self._next=None
        self._done=None
        self.last_task_reward=None

    def state_dict(self):
        return dict(version=1,contract=self.contract,calls=self.calls,
                    next_potential=None if self._next is None else self._next.clone(),
                    done=None if self._done is None else self._done.clone(),
                    last_task_reward=None if self.last_task_reward is None else self.last_task_reward.clone())

    def load_state_dict(self,state):
        if not isinstance(state,Mapping) or state.get('version')!=1 or state.get('contract')!=self.contract:
            raise ValueError('Potential provider checkpoint contract mismatch')
        calls=state.get('calls')
        if isinstance(calls,bool) or not isinstance(calls,int) or calls<0:
            raise ValueError('Invalid provider call counter')
        nxt,done,last=(state.get(k) for k in ('next_potential','done','last_task_reward'))
        if (nxt is None)!=(done is None) or (nxt is None)!=(last is None):
            raise ValueError('Incomplete potential continuity checkpoint')
        if nxt is not None:
            _vector(nxt,'next_potential');_vector(done,'done',nxt,mask=True);_vector(last,'last_task_reward',nxt)
            if (nxt.abs()>self.config.max_abs_potential).any():
                raise ValueError('Checkpoint potential exceeds configured bound')
        self.calls=calls
        self._next=None if nxt is None else nxt.detach().clone()
        self._done=None if done is None else done.detach().clone()
        self.last_task_reward=None if last is None else last.detach().clone()
