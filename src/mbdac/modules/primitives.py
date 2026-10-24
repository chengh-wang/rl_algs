"""Predict physical primitives, never goal-conditioned or weighted reward totals."""
from dataclasses import dataclass
import torch
from torch import nn
from .networks import mlp, RunningNorm


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    dim: int
    units: str
    frame: str
    binary: bool = False
    transition: bool = False

    def __post_init__(self):
        if not self.name or self.dim < 1 or not self.units or not self.frame:
            raise ValueError('A primitive needs a name, positive dimension, units and coordinate frame')


class PhysicalSignals:
    """Declare physical dependencies once; prefer analytic providers over heads.

    A measurement reader takes the native environment and returns a batched
    tensor. An analytic provider takes (state, context, **transition). Reward
    weights, commands and tolerances are deliberately absent from this registry.
    """
    def __init__(self): self._signals={}

    def register(self,spec,*,measure=None,analytic=None):
        if spec.name in self._signals: raise ValueError(f'Duplicate physical signal: {spec.name}')
        if measure is None and analytic is None:
            raise ValueError(f'{spec.name}: provide an analytic function or a real measurement reader')
        self._signals[spec.name]=(spec,measure,analytic)
        return self

    def _required(self,names):
        result=[]
        for name in dict.fromkeys(names):
            if name not in self._signals:
                raise KeyError(f'Unknown physical dependency: {name}; register its meaning and provider')
            result.append(self._signals[name])
        return result

    def learned_specs(self,required):
        return tuple(spec for spec,_,analytic in self._required(required) if analytic is None)

    def analytic_names(self,required):
        return tuple(spec.name for spec,_,analytic in self._required(required) if analytic is not None)

    def capture(self,env,required):
        labels={}
        for spec,reader,analytic in self._required(required):
            if analytic is not None: continue
            value=reader(env)
            if not isinstance(value,torch.Tensor) or value.ndim<1:
                raise TypeError(f'{spec.name}: measurement must be a batched Tensor')
            if hasattr(env,'num_envs') and value.shape[0]!=env.num_envs:
                raise ValueError(f'{spec.name}: measurement batch does not match the native environment')
            value=value.reshape(value.shape[0],-1)
            if value.shape[-1]!=spec.dim:
                raise ValueError(f'{spec.name}: expected {spec.dim} physical values, got {value.shape[-1]}')
            labels['primitive/'+spec.name]=value.detach().float().clone()
        return labels

    def resolve(self,state,context,predictions,required,**transition):
        values={}
        for spec,_,analytic in self._required(required):
            if analytic is not None: value=analytic(state,context,**transition)
            elif spec.name in predictions: value=predictions[spec.name]
            else: raise KeyError(f'Missing physical prediction: {spec.name}')
            if not isinstance(value,torch.Tensor) or value.ndim<1:
                raise TypeError(f'{spec.name}: physical provider must return a batched Tensor')
            if value.shape[:-1]!=state.shape[:-1] or value.shape[-1]!=spec.dim:
                raise ValueError(f'{spec.name}: expected last dimension {spec.dim}, got {value.shape[-1]}')
            values[spec.name]=value
        return values


class PrimitiveHeads(nn.Module):
    def __init__(self, input_dim, specs=(), hidden=(128,128)):
        super().__init__()
        self.specs=tuple(specs)
        if len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError('Duplicate physical primitive names')
        self.input_norm=RunningNorm(input_dim)
        self.nets=nn.ModuleDict({s.name:mlp(input_dim,hidden,s.dim) for s in specs})
        self.norms=nn.ModuleDict({s.name:RunningNorm(s.dim) for s in specs if not s.binary})

    def forward(self, features):
        if not self.specs: return {}
        features=self.input_norm(features)
        result={}
        for s in self.specs:
            y=self.nets[s.name](features)
            result[s.name]=torch.sigmoid(y) if s.binary else y*self.norms[s.name].std()+self.norms[s.name].mean
        return result

    @torch.no_grad()
    def update_normalization(self, labels, features=None):
        if features is not None and self.specs:
            self.input_norm.update(features)
        for s in self.specs:
            if not s.binary and s.name in labels:
                self.norms[s.name].update(labels[s.name])

    def loss(self, features, labels, mask):
        if not self.specs: return features.new_zeros(())
        features=self.input_norm(features)
        result=features.new_zeros(())
        for s in self.specs:
            if s.name not in labels:
                raise KeyError(f'Missing physical primitive labels: {s.name}')
            y=self.nets[s.name](features); target=labels[s.name]
            if s.binary:
                error=torch.nn.functional.binary_cross_entropy_with_logits(y,target,reduction='none')
            else:
                error=(y-self.norms[s.name](target)).square()
            result=result+(error.mean(-1)*mask).sum()/mask.shape[0]
        return result
