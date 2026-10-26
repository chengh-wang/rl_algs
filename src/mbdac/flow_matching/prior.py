"""Frozen, task-independent linear-flow prior and strict inference checkpoints."""
from collections.abc import Mapping
from copy import deepcopy
import torch
from torch import nn
from .model import FlowVelocityModel, FlowModelConfig

PATH_CONTRACT = {
    'prior_type': 'linear_flow', 'version': 2,
    'path': 'x_tau=(1-tau)*epsilon+tau*x', 'target': 'x-epsilon',
    'tau_distribution': 'Uniform[0,1)', 'conditioning': 'continuous_tau',
    'loss': 'velocity_mse', 'use_ema': False,
}


def linear_path(x, noise, tau):
    """Return affine path and conditional velocity for floating [B,W,F], [B]."""
    if any(not isinstance(v, torch.Tensor) or not v.is_floating_point() or not torch.isfinite(v).all() for v in (x, noise, tau)):
        raise ValueError('Path inputs must be finite floating tensors')
    if x.ndim != 3 or any(d <= 0 for d in x.shape) or noise.shape != x.shape or tau.shape != (len(x),):
        raise ValueError('Expected nonempty x/noise [B,W,F] and tau [B]')
    if noise.device != x.device or tau.device != x.device or noise.dtype != x.dtype:
        raise ValueError('Path tensors must share device; x/noise must share dtype')
    if ((tau < 0) | (tau > 1)).any():
        raise ValueError('tau must be in [0,1]')
    t = tau.to(x.dtype)[:, None, None]
    point, target = (1-t)*noise+t*x, x-noise
    if not torch.isfinite(point).all() or not torch.isfinite(target).all():
        raise ValueError('Path arithmetic overflow')
    return point, target


class FlowMotionPrior(nn.Module):
    """Frozen velocity model and feature quantiles, without scoring statistics."""
    def __init__(self, model, q_low, q_high, cfg=None, feature_schema=None):
        super().__init__()
        if not isinstance(model, FlowVelocityModel):
            raise ValueError('model must be FlowVelocityModel')
        config = model.config.to_dict()
        if cfg is not None and dict(cfg) != config:
            raise ValueError('cfg must exactly match model.config')
        self.feature_dim, self.window_size = model.feature_dim, model.window_size
        low, high = [torch.as_tensor(q, dtype=torch.float32).detach().clone() for q in (q_low, q_high)]
        if low.shape != (self.feature_dim,) or high.shape != low.shape or not all(torch.isfinite(q).all() for q in (low, high, high-low)) or not (high > low).all():
            raise ValueError('Quantiles must have finite positive feature spans')
        if feature_schema is not None and not isinstance(feature_schema, Mapping):
            raise ValueError('feature_schema must be a mapping or None')
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite model parameters')
        self.model = model.float().eval().requires_grad_(False)
        self.register_buffer('q_low', low.to(next(model.parameters()).device))
        self.register_buffer('q_high', high.to(next(model.parameters()).device))
        self.checkpoint_cfg = config
        self.feature_schema = deepcopy(dict(feature_schema)) if feature_schema is not None else None
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def get_extra_state(self):
        """Persist interpretation of weights when nested in a scorer state."""
        return deepcopy(dict(version=1, model_config=self.model.config.to_dict(),
                             path_contract=PATH_CONTRACT,
                             feature_schema=self.feature_schema))

    def set_extra_state(self, state):
        if not isinstance(state, Mapping) or state != self.get_extra_state():
            raise ValueError('Prior configuration differs from checkpoint')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate the complete prior subtree before Torch copies any prior tensors.
        # This hook also runs when an enclosing scorer recursively restores state.
        selected = {key[len(prefix):]:value for key,value in state_dict.items()
                    if key.startswith(prefix)}
        expected_model = self.model.state_dict()
        expected_keys = {'q_low','q_high','_extra_state'} | {
            'model.'+key for key in expected_model}
        if set(selected) != expected_keys:
            raise ValueError('Invalid prior state keys')
        self.set_extra_state(selected['_extra_state'])
        for key, expected in (('q_low',self.q_low),('q_high',self.q_high)):
            value = selected[key]
            if (not isinstance(value,torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype or not torch.isfinite(value).all()):
                raise ValueError('Invalid prior quantile tensor')
        low, high = selected['q_low'], selected['q_high']
        if low.device != high.device:
            raise ValueError('Prior quantiles must share device')
        span = high-low
        if not torch.isfinite(span).all() or not (span > 0).all():
            raise ValueError('Prior quantiles must have finite positive feature spans')
        for key, expected in expected_model.items():
            value = selected['model.'+key]
            if (not isinstance(value,torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype or not torch.isfinite(value).all()):
                raise ValueError('Invalid prior model tensor: '+key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def checkpoint_payload(self):
        return dict(checkpoint_version=2, prior_type='linear_flow', path_contract=deepcopy(PATH_CONTRACT),
                    cfg=deepcopy(self.checkpoint_cfg), model={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                    q_low=self.q_low.detach().cpu().clone(), q_high=self.q_high.detach().cpu().clone(),
                    feature_schema=deepcopy(self.feature_schema))

    def save_checkpoint(self, path):
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def load_checkpoint(cls, path, device='cpu', expected_feature_dim=None,
                        expected_window_size=None, expected_feature_schema=None):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        required = {'checkpoint_version','prior_type','path_contract','cfg','model','q_low','q_high','feature_schema'}
        if not isinstance(checkpoint, Mapping) or not required <= set(checkpoint):
            raise ValueError('Incomplete flow checkpoint')
        if checkpoint['checkpoint_version'] != 2 or checkpoint['prior_type'] != 'linear_flow' or checkpoint['path_contract'] != PATH_CONTRACT:
            raise ValueError('Expected version 2 linear-flow checkpoint with matching path contract')
        config = FlowModelConfig.from_dict(checkpoint['cfg'])
        for actual, expected in ((config.feature_dim, expected_feature_dim),(config.window_size, expected_window_size)):
            if expected is not None and (isinstance(expected, bool) or not isinstance(expected, int) or expected <= 0 or actual != expected):
                raise ValueError('Checkpoint dimensions differ from expected dimensions')
        schema = checkpoint['feature_schema']
        if expected_feature_schema is not None and schema != expected_feature_schema:
            raise ValueError('Checkpoint feature_schema differs from expected feature_schema')
        with torch.random.fork_rng(devices=[]):
            model = FlowVelocityModel(**config.to_dict())
        model.load_state_dict(checkpoint['model'], strict=True)
        return cls(model, checkpoint['q_low'], checkpoint['q_high'], config.to_dict(), schema).to(device)
