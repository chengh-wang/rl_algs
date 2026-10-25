"""Frozen, environment-independent SMP diffusion reward evaluation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import torch
from torch import nn

from .vendor.model import DiffusionDenoiser
from .vendor.scheduler import DDPMScheduler


@dataclass(frozen=True)
class SMPScore:
    reward: torch.Tensor
    raw_error: torch.Tensor
    normalized_error: torch.Tensor


class DiffNormalizer(nn.Module):
    """Upstream count-weighted per-timestep mean, persisted as module buffers."""

    def __init__(self, num_timesteps: int, min_value: float = 1e-4,
                 max_count: int = 50_000_000) -> None:
        super().__init__()
        if num_timesteps <= 0 or not math.isfinite(min_value) or min_value <= 0 or max_count < 0:
            raise ValueError("Invalid normalizer settings")
        self.min_value = min_value
        self.max_count = max_count
        self.register_buffer("mean", torch.ones(num_timesteps, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(num_timesteps, dtype=torch.long))

    @torch.no_grad()
    def update(self, t: int, values: torch.Tensor) -> None:
        # Preserve upstream's strictly-greater-than cap (including the final batch).
        if self.count[t] > self.max_count:
            return
        n = values.numel()
        old_count = self.count[t].item()
        new_count = old_count + n
        batch_mean = values.mean()
        if old_count == 0:
            self.mean[t] = batch_mean
        else:
            self.mean[t] = (old_count / new_count) * self.mean[t] + (n / new_count) * batch_mean
        self.count[t] = new_count

    def normalize(self, t: int, values: torch.Tensor) -> torch.Tensor:
        return values / self.mean[t].clamp(min=self.min_value)

    def update_and_normalize(self, t: int, values: torch.Tensor) -> torch.Tensor:
        self.update(t, values)
        return self.normalize(t, values)


def _positive_int(cfg: Mapping, key: str, default: int | None = None) -> int:
    value = cfg.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


class FrozenSMPPrior(nn.Module):
    """SMP denoiser with immutable weights and explicit normalizer updates.

    Features use upstream order and quantile scaling without clipping. Noise may
    be supplied as [K,B,W,F] for reproducible comparisons with upstream.
    """

    def __init__(self, model: DiffusionDenoiser, scheduler: DDPMScheduler,
                 q_low: torch.Tensor, q_high: torch.Tensor,
                 cfg: Mapping | None = None, feature_schema: Mapping | None = None) -> None:
        super().__init__()
        self.checkpoint_cfg = dict(cfg or {})
        self.feature_schema = dict(feature_schema) if feature_schema is not None else None
        self.feature_dim = model.feature_dim
        self.window_size = model.window_size
        low = torch.as_tensor(q_low, dtype=torch.float32).detach().clone()
        high = torch.as_tensor(q_high, dtype=torch.float32).detach().clone()
        if low.shape != (self.feature_dim,) or high.shape != low.shape:
            raise ValueError(f"Quantiles must have shape ({self.feature_dim},)")
        span = high - low
        if (not torch.isfinite(low).all() or not torch.isfinite(high).all()
                or not torch.isfinite(span).all() or not (span > 0).all()):
            raise ValueError("Quantiles must be finite with strictly positive spans")
        self.model = model.float().eval().requires_grad_(False)
        self.scheduler = scheduler.float()
        self.register_buffer("q_low", low)
        self.register_buffer("q_high", high)
        self.normalizer = DiffNormalizer(scheduler.num_timesteps).to(low.device)
        self.eval()

    def train(self, mode: bool = True) -> FrozenSMPPrior:
        super().train(False)
        return self

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu",
                        expected_feature_dim: int | None = None,
                        expected_window_size: int | None = None) -> FrozenSMPPrior:
        # Scoped, narrow compatibility allowlist for upstream NumPy quantiles and
        # pathlib metadata; never retry with arbitrary pickle execution enabled.
        safe = [np.core.multiarray._reconstruct, np.ndarray, np.dtype,
                type(np.dtype(np.float32)), Path, PosixPath, WindowsPath]
        with torch.serialization.safe_globals(safe):
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, Mapping) or not isinstance(ckpt.get("cfg"), Mapping):
            raise ValueError("Checkpoint must contain a cfg mapping")
        cfg = ckpt["cfg"]
        feature_dim = _positive_int(cfg, "feature_dim")
        window_size = _positive_int(cfg, "window_size")
        for key, actual, expected in (("feature_dim", feature_dim, expected_feature_dim),
                                      ("window_size", window_size, expected_window_size)):
            if expected is not None and actual != expected:
                raise ValueError(f"Checkpoint {key}={actual}, expected {expected}")
        d_model = _positive_int(cfg, "d_model", 256)
        nhead = _positive_int(cfg, "nhead", 8)
        layers = _positive_int(cfg, "num_layers", 2)
        timesteps = _positive_int(cfg, "num_timesteps", 50)
        dropout = float(cfg.get("dropout", 0.0))
        if not math.isfinite(dropout) or not 0 <= dropout <= 1 or d_model % 2 or d_model % nhead:
            raise ValueError("Invalid denoiser architecture or dropout")
        model = DiffusionDenoiser(feature_dim, window_size, d_model, nhead, layers, dropout)
        state = ckpt.get("model_ema")
        if state is None or (isinstance(state, Mapping) and not state):
            state = ckpt.get("model")
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint must contain model or model_ema state")
        model.load_state_dict(state, strict=True)
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Checkpoint model contains nonfinite weights")
        schema = ckpt.get("feature_schema")
        if schema is not None and not isinstance(schema, Mapping):
            raise ValueError("feature_schema must be a mapping or None")
        return cls(model, DDPMScheduler(timesteps), ckpt["q_low"], ckpt["q_high"], cfg, schema).to(device)

    @torch.no_grad()
    def score(self, features: torch.Tensor, fixed_timesteps: Sequence[int] = (8, 15, 22),
              ws: float = 4.0, normalize: bool = True, update_normalizer: bool = True,
              *, generator: torch.Generator | None = None,
              noise: torch.Tensor | None = None) -> SMPScore:
        steps = tuple(fixed_timesteps)
        if not steps or any(isinstance(t, bool) or not isinstance(t, int) or
                            not 0 <= t < self.scheduler.num_timesteps for t in steps):
            raise ValueError("fixed_timesteps must be nonempty integer timesteps within the schedule")
        if not math.isfinite(ws) or ws < 0:
            raise ValueError("ws must be finite and nonnegative")
        if features.ndim != 3 or features.shape[0] == 0 or features.shape[1:] != (self.window_size, self.feature_dim):
            raise ValueError(f"Expected nonempty features [B,{self.window_size},{self.feature_dim}]")
        features = features.to(device=self.q_low.device, dtype=torch.float32)
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite")
        if noise is not None:
            if generator is not None or noise.shape != (len(steps), *features.shape):
                raise ValueError("noise must have shape [K,B,W,F] and cannot accompany generator")
            noise = noise.to(device=features.device, dtype=torch.float32)
            if not torch.isfinite(noise).all():
                raise ValueError("Noise must be finite")
        x_0 = 2.0 * (features - self.q_low) / (self.q_high - self.q_low + 1e-8) - 1.0
        total_raw = torch.zeros(features.shape[0], device=features.device)
        total_normalized = torch.zeros_like(total_raw)
        for i, t_scalar in enumerate(steps):
            t = torch.full((features.shape[0],), t_scalar, dtype=torch.long, device=features.device)
            eps = (noise[i] if noise is not None else torch.randn(
                features.shape, dtype=torch.float32, device=features.device, generator=generator))
            prediction = self.model(self.scheduler.add_noise(x_0, eps, t), t)
            error = ((prediction - eps) ** 2).mean(dim=(-1, -2))
            total_raw += error
            if normalize and update_normalizer:
                self.normalizer.update(t_scalar, error)
            total_normalized += self.normalizer.normalize(t_scalar, error)
        raw = total_raw / len(steps)
        normalized = total_normalized / len(steps)
        return SMPScore(torch.exp(-(normalized if normalize else raw) * ws), raw, normalized)
