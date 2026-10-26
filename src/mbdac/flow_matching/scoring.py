"""Frozen motion prior and explicitly configured continuous-time residual scorer."""
from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
import math
import torch
from torch import nn
from mbdac.smp.prior import DiffNormalizer
from .prior import FrozenFlowPrior, linear_path


def legacy_score_times(labels=(8, 15, 22)):
    """Map historical cosine-50 labels to continuous linear-flow times."""
    labels = tuple(labels)
    if not labels or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < 50 for t in labels):
        raise ValueError('labels must be nonempty integers in [0,49]')
    def alpha(t):
        return math.cos((t + .008) / 1.008 * math.pi / 2) ** 2
    betas = torch.tensor([min(1-alpha((i+1)/50)/alpha(i/50), .999) for i in range(50)], dtype=torch.float32)
    a = torch.cumprod(1-betas, 0)
    times = a.sqrt() / (a.sqrt() + (1-a).sqrt())
    return tuple(times[t].item() for t in labels)


class FlowMotionPrior(nn.Module):
    """Immutable model/quantiles, independent of scorer and scheduler state."""
    def __init__(self, model, q_low, q_high, cfg=None, feature_schema=None):
        super().__init__()
        self.feature_dim, self.window_size = model.feature_dim, model.window_size
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (self.feature_dim, self.window_size)):
            raise ValueError('Prior dimensions must be positive integers')
        low, high = [torch.as_tensor(q, dtype=torch.float32).detach().clone() for q in (q_low,q_high)]
        if low.shape != (self.feature_dim,) or high.shape != low.shape or not all(torch.isfinite(q).all() for q in (low,high,high-low)) or not (high > low).all():
            raise ValueError('Quantiles must have finite positive feature spans')
        if feature_schema is not None and not isinstance(feature_schema, Mapping):
            raise ValueError('feature_schema must be a mapping or None')
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite model parameters')
        self.model = model.float().eval().requires_grad_(False)
        self.register_buffer('q_low',low)
        self.register_buffer('q_high',high)
        self.checkpoint_cfg = deepcopy(dict(cfg or {}))
        self.feature_schema = deepcopy(dict(feature_schema)) if feature_schema is not None else None
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @classmethod
    def load_checkpoint(cls, path, device='cpu', expected_feature_dim=None,
                        expected_window_size=None, expected_feature_schema=None):
        old = FrozenFlowPrior.load_checkpoint(path, device, expected_feature_dim, expected_window_size)
        if expected_feature_schema is not None and old.feature_schema != expected_feature_schema:
            raise ValueError('Checkpoint feature_schema differs from expected feature_schema')
        return cls(old.model, old.q_low, old.q_high, old.checkpoint_cfg, old.feature_schema).to(device)


@dataclass(frozen=True)
class FlowScore:
    reward: torch.Tensor
    raw_error: torch.Tensor
    normalized_error: torch.Tensor
    per_time_error: torch.Tensor  # [K,B], before normalization


class FlowMotionScorer(nn.Module):
    """Count-mean normalized velocity residual; normalizer updates are opt-in."""
    def __init__(self, prior, times, ws=4.0, normalize=True,
                 min_value=1e-4, max_count=50_000_000):
        super().__init__()
        self.times = tuple(times)
        if not self.times or any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or not 0 < t < 1 for t in self.times):
            raise ValueError('times must be nonempty finite continuous times in (0,1)')
        if not math.isfinite(ws) or ws < 0:
            raise ValueError('ws must be finite and nonnegative')
        self.prior, self.ws, self.normalize = prior, float(ws), bool(normalize)
        self.normalizer = DiffNormalizer(len(self.times), min_value, max_count).to(prior.q_low.device)

    def get_extra_state(self):
        return dict(version=1,times=self.times,ws=self.ws,normalize=self.normalize,
                    min_value=self.normalizer.min_value,max_count=self.normalizer.max_count)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Scorer configuration differs from checkpoint')

    def normalizer_state_dict(self):
        return {'configuration':self.get_extra_state(), 'normalizer':deepcopy(self.normalizer.state_dict())}

    def load_normalizer_state_dict(self, state):
        self.set_extra_state(state['configuration'])
        self.normalizer.load_state_dict(state['normalizer'], strict=True)

    @torch.no_grad()
    def score(self, features, *, noise=None, generator=None, update_normalizer=False):
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
        if generator is not None and torch.device(generator.device) != features.device:
            raise ValueError('generator must match prior device')
        x = 2*(features-p.q_low)/(p.q_high-p.q_low+1e-8)-1
        errors, normalized = [], []
        for i,t in enumerate(self.times):
            eps = noise[i] if noise is not None else torch.randn(x.shape,device=x.device,generator=generator)
            tau = torch.full((len(x),),t,device=x.device)
            point,target = linear_path(x,eps,tau)
            error = (p.model(point,tau*50)-target).square().mean((-1,-2))
            if update_normalizer:
                self.normalizer.update(i,error)
            errors.append(error)
            normalized.append(self.normalizer.normalize(i,error))
        per_time = torch.stack(errors)
        raw, norm = per_time.mean(0),torch.stack(normalized).mean(0)
        return FlowScore(torch.exp(-self.ws*(norm if self.normalize else raw)),raw,norm,per_time)

    forward = score
