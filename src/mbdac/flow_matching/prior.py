"""Legacy v1 experiment bridge; scheduler labels and 50-slot normalizer are compatibility APIs."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import torch
from torch import nn
from mbdac.smp.prior import DiffNormalizer, SMPScore, _positive_int
from mbdac.smp.vendor.model import DiffusionDenoiser
from mbdac.smp.vendor.scheduler import DDPMScheduler

PATH_CONTRACT = {
    "prior_type": "linear_flow", "version": 1,
    "path": "x_tau=(1-tau)*epsilon+tau*x", "target": "x-epsilon",
    "tau_distribution": "Uniform[0,1)", "conditioning": "continuous_tau_times_50",
    "loss": "velocity_mse", "use_ema": False,
    "score_mapping": "cosine_ddpm_50_label:tau=sqrt(alpha_bar)/(sqrt(alpha_bar)+sqrt(1-alpha_bar))",
    "score_input": "tau*x+(1-tau)*epsilon", "score_target": "x-epsilon",
    "score_error": "mean_W_F_squared_velocity_residual",
}

def linear_path(x, noise, tau):
    """Return affine path and exact conditional velocity; tau is [B]."""
    if x.ndim != 3 or noise.shape != x.shape or tau.shape != (len(x),):
        raise ValueError("Expected x/noise [B,W,F] and tau [B]")
    if not torch.isfinite(tau).all() or ((tau < 0) | (tau > 1)).any():
        raise ValueError("tau must be finite in [0,1]")
    t = tau[:, None, None]
    return (1-t)*noise+t*x, x-noise

@dataclass(frozen=True)
class FlowScoreLabelDomain:
    """Legacy bridge validation domain only; no diffusion or sampling API."""
    num_timesteps: int = 50

class FrozenFlowPrior(nn.Module):
    def __init__(self, model, q_low, q_high, cfg=None, feature_schema=None):
        super().__init__()
        self.checkpoint_cfg = dict(cfg or {})
        self.feature_schema = dict(feature_schema) if feature_schema is not None else None
        self.feature_dim, self.window_size = model.feature_dim, model.window_size
        low, high = [torch.as_tensor(q, dtype=torch.float32).detach().clone() for q in (q_low, q_high)]
        if low.shape != (self.feature_dim,) or high.shape != low.shape:
            raise ValueError("Quantile dimensions differ from feature_dim")
        if not all(torch.isfinite(q).all() for q in (low, high, high-low)) or not (high > low).all():
            raise ValueError("Quantiles must have finite positive spans")
        self.model = model.float().eval().requires_grad_(False)
        self.register_buffer("q_low", low)
        self.register_buffer("q_high", high)
        # SMPRewardEnv validates labels through this legacy duck-typed field.
        # GSI must retain its separate original DDPM prior.
        self.scheduler = FlowScoreLabelDomain()
        scheduler = DDPMScheduler(self.scheduler.num_timesteps)
        a, sigma = scheduler.sqrt_alphas_cumprod, scheduler.sqrt_one_minus_alphas_cumprod
        self.register_buffer("label_tau", a/(a+sigma))
        self.normalizer = DiffNormalizer(50).to(low.device)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @classmethod
    def load_checkpoint(cls, path: str | Path, device="cpu", expected_feature_dim=None, expected_window_size=None):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, Mapping) or ckpt.get("prior_type") != "linear_flow":
            raise ValueError("Expected an explicit linear_flow checkpoint")
        if ckpt.get("checkpoint_version", 1) != 1:
            raise ValueError("Unsupported flow checkpoint version")
        if ckpt.get("path_contract") != PATH_CONTRACT:
            raise ValueError("Unsupported linear-flow path/score contract")
        cfg = ckpt.get("cfg")
        if not isinstance(cfg, Mapping) or cfg.get("prior_type") != "linear_flow" or cfg.get("use_ema") is not False:
            raise ValueError("Invalid linear-flow config or EMA selection")
        f, w = _positive_int(cfg, "feature_dim"), _positive_int(cfg, "window_size")
        for actual, expected in ((f, expected_feature_dim), (w, expected_window_size)):
            if expected is not None and actual != expected:
                raise ValueError("Checkpoint dimensions differ from expected dimensions")
        d, h, n = [_positive_int(cfg, k) for k in ("d_model", "nhead", "num_layers")]
        dropout = float(cfg.get("dropout", 0.0))
        if d % 2 or d % h or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("Invalid architecture")
        model = DiffusionDenoiser(f,w,d,h,n,dropout)
        model.load_state_dict(ckpt["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters")
        schema = ckpt.get("feature_schema")
        if schema is not None and not isinstance(schema, Mapping):
            raise ValueError("feature_schema must be mapping or None")
        return cls(model, ckpt["q_low"], ckpt["q_high"], cfg, schema).to(device)

    @torch.no_grad()
    def score(self, features: torch.Tensor, fixed_timesteps: Sequence[int]=(8,15,22), ws=4.0,
              normalize=True, update_normalizer=True, *, generator=None, noise=None):
        steps = tuple(fixed_timesteps)
        if not steps or any(isinstance(t,bool) or not isinstance(t,int) or not 0 <= t < 50 for t in steps):
            raise ValueError("fixed_timesteps must contain integer cosine-DDPM labels in [0,49]")
        if not math.isfinite(ws) or ws < 0:
            raise ValueError("ws must be finite and nonnegative")
        if features.ndim != 3 or len(features)==0 or features.shape[1:] != (self.window_size,self.feature_dim):
            raise ValueError("Expected nonempty features [B,W,F]")
        features = features.to(self.q_low.device, torch.float32)
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        if noise is not None:
            if generator is not None or noise.shape != (len(steps),*features.shape):
                raise ValueError("noise must be [K,B,W,F] and cannot accompany generator")
            noise = noise.to(features.device, torch.float32)
            if not torch.isfinite(noise).all():
                raise ValueError("Noise must be finite")
        x = 2*(features-self.q_low)/(self.q_high-self.q_low+1e-8)-1
        raw, normalized = torch.zeros(len(x),device=x.device), torch.zeros(len(x),device=x.device)
        for i,label in enumerate(steps):
            eps = noise[i] if noise is not None else torch.randn(x.shape, device=x.device, generator=generator)
            tau = self.label_tau[label].expand(len(x))
            point,target = linear_path(x,eps,tau)
            error = (self.model(point,tau*50)-target).square().mean((-1,-2))
            raw += error
            if normalize and update_normalizer:
                self.normalizer.update(label,error)
            normalized += self.normalizer.normalize(label,error)
        raw, normalized = raw/len(steps), normalized/len(steps)
        return SMPScore(torch.exp(-ws*(normalized if normalize else raw)),raw,normalized)
