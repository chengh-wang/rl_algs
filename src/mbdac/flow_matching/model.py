"""Pure Torch conditional-velocity DiT with original checkpoint state keys."""
import math
from mbdac.smp.vendor.model import DiffusionDenoiser


class FlowVelocityModel(DiffusionDenoiser):
    """Predict x - epsilon from [B,W,F] path points and continuous tau * 50."""
    def __init__(self, feature_dim, window_size, d_model=256, nhead=4,
                 num_layers=2, dropout=0.0, head_dim=None):
        for name, value in locals().copy().items():
            if name in ('feature_dim', 'window_size', 'd_model', 'nhead', 'num_layers', 'head_dim') and value is not None:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f'{name} must be a positive integer')
        if d_model % 2 or not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError('Invalid architecture or dropout')
        super().__init__(feature_dim, window_size, d_model, nhead, num_layers, dropout, head_dim)
