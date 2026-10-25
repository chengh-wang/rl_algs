"""Pure SMP ancestral sampling and window decoding; no simulator state writes.

Sampling follows upstream smp/rl/events.py::_ddpm_sample and decoding follows
_prime_sim_and_buffer and sampling/feature_to_state.py at commit
28d1dfd1b7b4447977fbdb2062167d9f87a2faaa. Joint and end-effector layouts are
provided explicitly by MotionFeatureSchema instead of upstream G1 constants.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from .math import quat_apply, quat_from_matrix, quat_mul

from .features import MotionFeatureSchema
from .prior import FrozenSMPPrior


@dataclass(frozen=True)
class DecodedMotionWindow:
    """Decoded poses/twists; joint columns follow joint_names exactly.

    ``last_root_state`` is [position(3), wxyz(4), world linear/angular velocity(6)].
    World positions include env_origins; buffer_frames positions exclude them.
    End-effector positions are decoded observations, not forward kinematics.
    """
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    ee_pos_w: torch.Tensor
    root_pos_env: torch.Tensor
    ee_pos_env: torch.Tensor
    last_root_state: torch.Tensor
    joint_names: tuple[str, ...]
    end_effector_names: tuple[str, ...]

    @property
    def buffer_frames(self) -> dict[str, torch.Tensor]:
        """Full windows in the environment-relative feature-buffer convention."""
        return dict(root_pos_w=self.root_pos_env, root_quat_w=self.root_quat_w,
                    root_lin_vel_w=self.root_lin_vel_w, root_ang_vel_w=self.root_ang_vel_w,
                    joint_pos=self.joint_pos, joint_vel=self.joint_vel, ee_pos_w=self.ee_pos_env)


@torch.no_grad()
def sample_feature_windows(prior: FrozenSMPPrior, num_samples: int) -> torch.Tensor:
    """Run the checkpoint's full DDPM schedule and inverse quantile transform.

    Uses the ordinary torch RNG exactly like upstream (including ancestral
    noise). Save/restore torch RNG state for reproducible samples. No clipping,
    rejection sampling, or stability correction is applied; nonfinite output
    raises an error instead of silently repairing a failed model sample.
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")
    x_t = torch.randn(num_samples, prior.window_size, prior.feature_dim, device=prior.q_low.device)
    for t_int in reversed(range(prior.scheduler.num_timesteps)):
        t = torch.full((num_samples,), t_int, dtype=torch.long, device=x_t.device)
        eps = prior.model(x_t, t)
        x_t = prior.scheduler.step(eps, x_t, t_int)
    windows = (x_t + 1.0) / 2.0 * (prior.q_high - prior.q_low) + prior.q_low
    if not torch.isfinite(windows).all():
        raise RuntimeError("SMP ancestral sampling produced nonfinite features; output was not clipped")
    return windows


def rot6d_to_quat(rot6d: torch.Tensor) -> torch.Tensor:
    """Decode first/third rotation columns by right-handed Gram-Schmidt."""
    if rot6d.shape[-1] != 6 or not torch.isfinite(rot6d).all():
        raise ValueError("Rotation features must be finite [...,6] arrays")
    col0 = rot6d[..., :3]
    norm0 = col0.norm(dim=-1, keepdim=True)
    if (norm0 <= 1e-12).any():
        raise ValueError("Degenerate first rotation column; no orientation can be decoded")
    col0 = col0 / norm0
    col2 = rot6d[..., 3:]
    col2 = col2 - (col0 * col2).sum(dim=-1, keepdim=True) * col0
    norm2 = col2.norm(dim=-1, keepdim=True)
    if (norm2 <= 1e-12).any():
        raise ValueError("Degenerate third rotation column; no orientation can be decoded")
    col2 = col2 / norm2
    col1 = torch.cross(col2, col0, dim=-1)
    return quat_from_matrix(torch.stack((col0, col1, col2), dim=-1))


@torch.no_grad()
def decode_feature_windows(windows: torch.Tensor, schema: MotionFeatureSchema, *,
                           env_origins: torch.Tensor | None = None,
                           chosen_yaw: float | torch.Tensor = 0.0,
                           anchor_xy: torch.Tensor | None = None) -> DecodedMotionWindow:
    """Decode denormalized [N,W,F] windows using named joints and sample_dt.

    chosen_yaw is in radians (scalar or [N]); anchor_xy is [N,2] relative to
    each environment origin. Feature Z is preserved relative to that origin.
    Like upstream, arbitrary generated windows are decoded as-is: their final
    XY and heading may differ from the feature convention's exact anchor.
    """
    if windows.ndim != 3 or windows.shape[-1] != schema.feature_dim or min(windows.shape[:2]) <= 0:
        raise ValueError(f"Expected nonempty [N,W,{schema.feature_dim}] feature windows")
    windows = windows.to(dtype=torch.float32)
    if not torch.isfinite(windows).all():
        raise ValueError("Feature windows must be finite; no clipping is applied")
    n, w, _ = windows.shape
    def placement(value, shape, name):
        value = torch.as_tensor(value, dtype=windows.dtype, device=windows.device)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite with shape {shape}")
        return value
    origins = (torch.zeros(n,3,device=windows.device) if env_origins is None
               else placement(env_origins,(n,3),"env_origins"))
    xy = (torch.zeros(n,2,device=windows.device) if anchor_xy is None
          else placement(anchor_xy,(n,2),"anchor_xy"))
    yaw = torch.as_tensor(chosen_yaw,dtype=windows.dtype,device=windows.device)
    if yaw.ndim == 0:
        yaw = yaw.expand(n)
    yaw = placement(yaw,(n,),"chosen_yaw")
    yaw_quat = torch.stack((torch.cos(yaw/2), torch.zeros_like(yaw),
                            torch.zeros_like(yaw), torch.sin(yaw/2)),dim=-1)
    yaw_w = yaw_quat[:,None].expand(n,w,4).reshape(-1,4)
    j = len(schema.joint_names)
    e = len(schema.end_effector_names)
    joints = windows[...,9:9+j].clone()
    ee_end = 9+j+3*e
    local_pos = windows[...,:3].clone()
    local_pos[...,2] = 0
    root_env = quat_apply(yaw_w,local_pos.reshape(-1,3)).reshape(n,w,3)
    root_env[...,:2] += xy[:,None]
    root_env[...,2] = windows[...,2]
    root_quat = quat_mul(yaw_w,rot6d_to_quat(windows[...,3:9].reshape(-1,6))).reshape(n,w,4)
    linear = quat_apply(yaw_w,windows[...,ee_end:ee_end+3].reshape(-1,3)).reshape(n,w,3)
    angular = quat_apply(yaw_w,windows[...,ee_end+3:ee_end+6].reshape(-1,3)).reshape(n,w,3)
    yaw_ee = yaw_quat[:,None,None].expand(n,w,e,4).reshape(-1,4)
    ee_offset = quat_apply(yaw_ee,windows[...,9+j:ee_end].reshape(-1,3)).reshape(n,w,e,3)
    ee_env = ee_offset + root_env[:,:,None]
    joint_vel = torch.zeros_like(joints)
    if w > 1:
        joint_vel[:,:-1] = (joints[:,1:] - joints[:,:-1]) / schema.sample_dt
        joint_vel[:,-1] = joint_vel[:,-2]
    root_world = root_env + origins[:,None]
    ee_world = ee_env + origins[:,None,None]
    last = torch.cat((root_world[:,-1],root_quat[:,-1],linear[:,-1],angular[:,-1]),dim=-1)
    result = DecodedMotionWindow(root_world,root_quat,linear,angular,joints,joint_vel,
                                ee_world,root_env,ee_env,last,schema.joint_names,schema.end_effector_names)
    if any(not torch.isfinite(value).all() for value in result.buffer_frames.values()):
        raise RuntimeError("Decoded motion contains nonfinite state; no clipping is applied")
    if not torch.isfinite(last).all() or not torch.isfinite(ee_world).all():
        raise RuntimeError("World placement produced nonfinite state; no clipping is applied")
    return result
