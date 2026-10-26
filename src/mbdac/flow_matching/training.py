"""Tensor-only offline training. Checkpoints support inference, not resume."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from mbdac.smp.pretrain import fit_quantiles
from .model import FlowVelocityModel
from .prior import PATH_CONTRACT, linear_path


def flow_matching_loss(model, x, num_noise_samples=10, *, generator=None):
    """Independent continuous uniform tau/noise pairs in original B,K order."""
    if isinstance(num_noise_samples,bool) or not isinstance(num_noise_samples,int) or num_noise_samples <= 0:
        raise ValueError("num_noise_samples must be positive integer")
    if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim != 3 or any(d <= 0 for d in x.shape) or not torch.isfinite(x).all():
        raise ValueError("x must be finite floating nonempty [B,W,F]")
    expanded = x[:,None].expand(len(x),num_noise_samples,*x.shape[1:]).reshape(-1,*x.shape[1:])
    tau = torch.rand(len(expanded),device=x.device,generator=generator)
    noise = torch.randn(expanded.shape,device=x.device,dtype=x.dtype,generator=generator)
    point,target = linear_path(expanded,noise,tau)
    return torch.nn.functional.mse_loss(model(point,tau*50),target)

def checkpoint_payload(model, optimizer, cfg, low, high, schema, epoch, manifest, metrics):
    return dict(checkpoint_version=1,
                checkpoint_purpose="inference; training metadata only, no resume contract",
                prior_type="linear_flow",path_contract=PATH_CONTRACT.copy(),cfg=dict(cfg),
                model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                q_low=low.cpu(),q_high=high.cpu(),feature_schema=schema,epoch=epoch,
                reference_manifest=manifest,metrics=metrics)

@dataclass(frozen=True)
class FlowTrainingConfig:
    epochs: int = 2000
    batch_size: int = 1024
    num_noise_samples: int = 10
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    span_floor: float = 1e-4
    seed: int = 42

    def __post_init__(self):
        for name in ('epochs','batch_size','num_noise_samples','d_model','nhead','num_layers'):
            value = getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int):
            raise ValueError('seed must be an integer')
        if self.d_model % 2 or self.d_model % self.nhead:
            raise ValueError('d_model must be even and divisible by nhead')
        for name in ('learning_rate','max_grad_norm','span_floor'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not math.isfinite(self.dropout) or not 0 <= self.dropout <= 1:
            raise ValueError('Invalid weight_decay or dropout')


def train_flow_prior(windows, output_dir, config=None, *, feature_schema=None, device='cpu'):
    """Train raw floating windows [N,W,F]; return new v1-compatible prior.pt path.

    Offline q01/q99 scaling uses all supplied windows without clipping. Saved
    optimizer/RNG metadata describes the completed run; no resume API is promised.
    """
    config = FlowTrainingConfig() if config is None else config
    if not isinstance(config,FlowTrainingConfig):
        raise ValueError('config must be FlowTrainingConfig')
    if not isinstance(windows,torch.Tensor) or not windows.is_floating_point() or windows.ndim != 3 or any(d <= 0 for d in windows.shape) or not torch.isfinite(windows).all():
        raise ValueError('windows must be finite floating nonempty [N,W,F]')
    if feature_schema is not None and not isinstance(feature_schema,Mapping):
        raise ValueError('feature_schema must be a mapping or None')
    schema = deepcopy(dict(feature_schema)) if feature_schema is not None else None
    output = Path(output_dir)
    target = output/'prior.pt'
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    raw = windows.detach().to('cpu',torch.float32)
    if not torch.isfinite(raw).all():
        raise ValueError('windows overflow float32')
    low,high = fit_quantiles(raw,config.span_floor)
    if not torch.isfinite(high-low).all() or not (high > low).all():
        raise ValueError('Quantiles must have finite positive spans')
    torch.manual_seed(config.seed)
    model = FlowVelocityModel(raw.shape[-1],raw.shape[1],config.d_model,config.nhead,config.num_layers,config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    normalized = 2*(raw-low)/(high-low+1e-8)-1
    loader = DataLoader(TensorDataset(normalized),batch_size=config.batch_size,shuffle=True)
    for epoch in range(1,config.epochs+1):
        model.train()
        total = 0.
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model,batch,config.num_noise_samples)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite flow loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.max_grad_norm,error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*len(batch)
    cfg = dict(asdict(config),feature_dim=raw.shape[-1],window_size=raw.shape[1],prior_type='linear_flow',use_ema=False,
               loss='velocity_mse',conditioning_scale=50.,sampling_policy='uniform',augmentation=False)
    payload = checkpoint_payload(model,optimizer,cfg,low,high,schema,epoch,
                                 {'source':'supplied_tensor','nwindows':len(raw)},
                                 {'epoch':epoch,'train_velocity_mse':total/len(raw)})
    payload.update(checkpoint_version=1,checkpoint_purpose='inference; training metadata only, no resume contract')
    output.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:
        torch.save(payload,stream)
    return target


__all__ = ['FlowTrainingConfig','train_flow_prior','flow_matching_loss','fit_quantiles']
