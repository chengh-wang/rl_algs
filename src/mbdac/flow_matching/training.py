"""Task-independent offline conditional Flow Matching on feature windows."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import FlowVelocityModel
from .prior import FlowMotionPrior, linear_path


def _windows(value):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 3 or any(d <= 0 for d in value.shape)
            or not torch.isfinite(value).all()):
        raise ValueError('windows must be finite floating nonempty [N,W,F]')


def fit_quantiles(windows, span_floor=1e-4, *, low_quantile=.01, high_quantile=.99):
    """Fit feature scaling on training windows only, across samples and time."""
    _windows(windows)
    if not math.isfinite(span_floor) or span_floor <= 0:
        raise ValueError('span_floor must be finite and positive')
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError('Expected 0 <= low_quantile < high_quantile <= 1')
    raw = windows.detach().to(device='cpu', dtype=torch.float32)
    if not torch.isfinite(raw).all():
        raise ValueError('windows overflow float32')
    low, high = torch.quantile(raw.reshape(-1, raw.shape[-1]),
        torch.tensor([low_quantile, high_quantile]), dim=0)
    high = torch.maximum(high, low + span_floor)
    if not torch.isfinite(high-low).all() or not (high > low).all():
        raise ValueError('Quantiles must have finite positive spans')
    return low, high


def flow_matching_loss(model, x, num_noise_samples=10, *, generator=None):
    """Conditional velocity MSE on normalized windows, using native tau [0,1]."""
    _windows(x)
    if isinstance(num_noise_samples, bool) or not isinstance(num_noise_samples, int) or num_noise_samples <= 0:
        raise ValueError('num_noise_samples must be a positive integer')
    if generator is not None and (not isinstance(generator, torch.Generator) or torch.device(generator.device) != x.device):
        raise ValueError('generator must match windows device')
    expanded = x.repeat_interleave(num_noise_samples, dim=0)
    tau = torch.rand(len(expanded), device=x.device, dtype=x.dtype, generator=generator)
    noise = torch.randn(expanded.shape, device=x.device, dtype=x.dtype, generator=generator)
    point, target = linear_path(expanded, noise, tau)
    return torch.nn.functional.mse_loss(model(point, tau), target)


@dataclass(frozen=True)
class FlowTrainingConfig:
    epochs: int = 2000
    batch_size: int = 1024
    num_noise_samples: int = 10
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    time_embedding_scale: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    span_floor: float = 1e-4
    low_quantile: float = .01
    high_quantile: float = .99
    seed: int = 42

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'num_noise_samples', 'd_model', 'nhead', 'num_layers'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError('seed must be an integer')
        if self.d_model % 2 or self.d_model % self.nhead:
            raise ValueError('d_model must be even and divisible by nhead')
        for name in ('learning_rate', 'max_grad_norm', 'span_floor', 'time_embedding_scale'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not math.isfinite(self.dropout) or not 0 <= self.dropout <= 1:
            raise ValueError('dropout must be in [0,1]')
        if not 0 <= self.low_quantile < self.high_quantile <= 1:
            raise ValueError('Expected 0 <= low_quantile < high_quantile <= 1')


def train_flow_prior(windows, output_dir, config=None, *, feature_schema=None,
                     device='cpu', validation_windows=None):
    """Train features of any task; return the saved inference prior path.

    Caller owns feature meaning, episode boundaries and train/validation split.
    This function seeds global Torch RNG. report.json records training;
    prior.pt contains only inference state. No RL agent is owned here.
    """
    config = FlowTrainingConfig() if config is None else config
    if not isinstance(config, FlowTrainingConfig):
        raise ValueError('config must be FlowTrainingConfig')
    _windows(windows)
    if feature_schema is not None and not isinstance(feature_schema, Mapping):
        raise ValueError('feature_schema must be a mapping or None')
    schema = deepcopy(dict(feature_schema)) if feature_schema is not None else None
    output = Path(output_dir)
    target = output/'prior.pt'
    if target.exists() or (output/'report.json').exists():
        raise FileExistsError(f'Refusing to overwrite training output {output}')
    raw = windows.detach().to(device='cpu', dtype=torch.float32)
    low, high = fit_quantiles(raw, config.span_floor,
        low_quantile=config.low_quantile, high_quantile=config.high_quantile)
    normalized = 2*(raw-low)/(high-low+1e-8)-1
    validation = None
    if validation_windows is not None:
        _windows(validation_windows)
        if validation_windows.shape[1:] != raw.shape[1:]:
            raise ValueError('Training and validation window layouts differ')
        validation = 2*(validation_windows.detach().to('cpu', torch.float32)-low)/(high-low+1e-8)-1
    if not torch.isfinite(normalized).all() or (validation is not None and not torch.isfinite(validation).all()):
        raise ValueError('Feature normalization overflow')
    torch.manual_seed(config.seed)
    model = FlowVelocityModel(raw.shape[-1], raw.shape[1], config.d_model,
        config.nhead, config.num_layers, config.dropout,
        time_embedding_scale=config.time_embedding_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loader = DataLoader(TensorDataset(normalized), batch_size=config.batch_size, shuffle=True)
    val_loader = None if validation is None else DataLoader(
        TensorDataset(validation), batch_size=config.batch_size,
        generator=torch.Generator().manual_seed(config.seed))
    val_rng = torch.Generator(device=device).manual_seed(config.seed)
    report = dict(shape=list(raw.shape), config=asdict(config),
        validation_shape=None if validation is None else list(validation.shape),
        optimizer_updates=0, epochs=[])
    for epoch in range(1, config.epochs+1):
        model.train()
        total = 0.
        for (batch,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, batch.to(device), config.num_noise_samples)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite flow loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(batch)
            report['optimizer_updates'] += 1
        metrics = dict(epoch=epoch, train_velocity_mse=total/len(raw))
        if val_loader is not None:
            model.eval()
            val_rng.manual_seed(config.seed)
            with torch.no_grad():
                val_total = sum(flow_matching_loss(model, b.to(device), config.num_noise_samples,
                    generator=val_rng).item()*len(b) for (b,) in val_loader)
            metrics['validation_velocity_mse'] = val_total/len(validation)
        report['epochs'].append(metrics)
    prior = FlowMotionPrior(model, low, high, feature_schema=schema)
    output.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        prior.save_checkpoint(stream)
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    return target
