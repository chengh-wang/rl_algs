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

    def __post_init__(self):
        if not self.name or self.dim < 1 or not self.units or not self.frame:
            raise ValueError('A primitive needs a name, positive dimension, units and coordinate frame')


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
