"""Explicit continuous-time velocity residual scoring, independent of tasks."""
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
from torch import nn
from .normalization import FlowErrorNormalizer
from .prior import FlowMotionPrior, linear_path


@dataclass(frozen=True)
class FlowScore:
    reward: torch.Tensor
    raw_error: torch.Tensor
    normalized_error: torch.Tensor
    per_time_error: torch.Tensor  # [K,B], before normalization


class FlowMotionScorer(nn.Module):
    """Count-mean normalized velocity residual; normalizer updates are opt-in."""
    def __init__(self, prior, times, ws=4.0, normalize=True,
                 min_value=1e-4, max_count=None):
        super().__init__()
        self.times = tuple(times)
        if not self.times or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0 < t < 1 for t in self.times):
            raise ValueError('times must be nonempty finite continuous times in (0,1)')
        if isinstance(ws,bool) or not isinstance(ws,(int,float)) or not math.isfinite(ws) or ws < 0:
            raise ValueError('ws must be finite and nonnegative')
        if not isinstance(prior, FlowMotionPrior) or not isinstance(normalize, bool):
            raise ValueError('Expected FlowMotionPrior and boolean normalize')
        self.prior, self.ws, self.normalize = prior, float(ws), normalize
        self.normalizer = FlowErrorNormalizer(len(self.times), min_value, max_count).to(prior.q_low.device)

    def get_extra_state(self):
        return dict(version=1,times=self.times,ws=self.ws,normalize=self.normalize,
                    min_value=self.normalizer.min_value,max_count=self.normalizer.max_count)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Scorer configuration differs from checkpoint')

    def normalizer_state_dict(self):
        return {'configuration':self.get_extra_state(), 'normalizer':deepcopy(self.normalizer.state_dict())}

    def load_normalizer_state_dict(self, state):
        if not isinstance(state,dict) or set(state) != {'configuration','normalizer'}:
            raise ValueError('Invalid scorer normalizer state')
        self.set_extra_state(state['configuration'])
        self.normalizer.load_state_dict(state['normalizer'], strict=True)

    @torch.no_grad()
    def score(self, features, *, noise=None, generator=None, update_normalizer=False):
        if not isinstance(update_normalizer,bool):
            raise ValueError('update_normalizer must be boolean')
        p = self.prior
        if not isinstance(features,torch.Tensor) or not features.is_floating_point() or features.ndim != 3 or not len(features) or features.shape[1:] != (p.window_size,p.feature_dim) or not torch.isfinite(features).all():
            raise ValueError('Expected finite floating nonempty features [B,W,F]')
        features = features.to(p.q_low.device,torch.float32)
        if not torch.isfinite(features).all():
            raise ValueError('features overflow float32')
        if noise is not None:
            if generator is not None or not isinstance(noise,torch.Tensor) or not noise.is_floating_point() or noise.shape != (len(self.times),*features.shape) or not torch.isfinite(noise).all():
                raise ValueError('noise must be finite floating [K,B,W,F], without generator')
            noise = noise.to(features.device,torch.float32)
            if not torch.isfinite(noise).all():
                raise ValueError('noise overflows float32')
        if generator is not None and (not isinstance(generator,torch.Generator) or torch.device(generator.device) != features.device):
            raise ValueError('generator must match prior device')
        x = 2*(features-p.q_low)/(p.q_high-p.q_low+1e-8)-1
        if not torch.isfinite(x).all():
            raise ValueError('Feature normalization overflow')
        errors, normalized = [], []
        for i,t in enumerate(self.times):
            eps = noise[i] if noise is not None else torch.randn(x.shape,device=x.device,generator=generator)
            tau = torch.full((len(x),),t,device=x.device)
            point,target = linear_path(x,eps,tau)
            error = (p.model(point,tau)-target).square().mean((-1,-2))
            if not torch.isfinite(error).all():
                raise ValueError('Nonfinite flow prediction error')
            if update_normalizer:
                self.normalizer.update(i,error)
            errors.append(error)
            normalized.append(self.normalizer.normalize(i,error))
        per_time = torch.stack(errors)
        raw, norm = per_time.mean(0),torch.stack(normalized).mean(0)
        return FlowScore(torch.exp(-self.ws*(norm if self.normalize else raw)),raw,norm,per_time)

    forward = score
