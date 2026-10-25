"""Pure torch SMP motion features for named schemas.

MotionFeatureBuffer feature transforms adapted from paper-plane SMP
src/smp/rl/utils.py, commit 28d1dfd1b7b4447977fbdb2062167d9f87a2faaa.
No task state, simulation state, or reset distributions are modified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from .math import (
  matrix_from_quat, quat_apply_inverse, quat_conjugate, quat_mul, yaw_quat,
)


@dataclass(frozen=True)
class MotionFeatureSchema:
  joint_names: tuple[str, ...]
  end_effector_names: tuple[str, ...]
  sample_dt: float
  root_body_name: str = "pelvis"
  entity_name: str = "robot"
  quaternion_order: str = "wxyz"

  def __post_init__(self):
    for field in ("joint_names", "end_effector_names"):
      names = getattr(self, field)
      if not isinstance(names, tuple) or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError(f"{field} must be a tuple of nonempty names")
      if len(set(names)) != len(names):
        raise ValueError(f"{field} contains duplicate names")
    for field in ("root_body_name", "entity_name"):
      if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
        raise ValueError(f"{field} must be a nonempty name")
    if isinstance(self.sample_dt, bool) or not math.isfinite(self.sample_dt) or self.sample_dt <= 0:
      raise ValueError("sample_dt must be finite and positive")
    if self.quaternion_order != "wxyz":
      raise ValueError("Only wxyz quaternions are supported")

  @property
  def feature_dim(self):
    return 15 + len(self.joint_names) + 3 * len(self.end_effector_names)


class MotionFeatureBuffer:
  """Rolling per-env buffer of the last ``window_size`` kinematic samples;
  ``compute_features()`` returns a window anchored at the LAST frame's yaw-only
  local frame, layout (matching ``scripts/csv_to_npz.py``):

      ``[root_pos(3), root_rot(6), joint_pos(J), ee_pos(E*3),
         root_lin_vel(3), root_ang_vel(3)]``

  ``joint_vel`` is stored for symmetry but excluded from the output.  Positions
  use the caller's frame (SMP RL feeds env-origin-relative)."""

  def __init__(
    self,
    num_envs: int,
    window_size: int,
    num_joints: int,
    num_ee: int,
    device: torch.device | str,
  ) -> None:
    for name, value in (("num_envs", num_envs), ("window_size", window_size)):
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    for name, value in (("num_joints", num_joints), ("num_ee", num_ee)):
      if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    self.num_envs = num_envs
    self.window_size = window_size
    self.num_joints = num_joints
    self.num_ee = num_ee
    self.device = torch.device(device)

    self.root_pos_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_quat_w = torch.zeros(num_envs, window_size, 4, device=self.device)
    self.root_quat_w[..., 0] = 1.0
    self.root_lin_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.root_ang_vel_w = torch.zeros(num_envs, window_size, 3, device=self.device)
    self.ee_pos_w = torch.zeros(num_envs, window_size, num_ee, 3, device=self.device)
    self.joint_pos = torch.zeros(num_envs, window_size, num_joints, device=self.device)
    self.joint_vel = torch.zeros(num_envs, window_size, num_joints, device=self.device)

  @property
  def feature_dim(self):
    return 15 + self.num_joints + 3 * self.num_ee

  def _frames(self, root_pos_w, root_quat_w, root_lin_vel_w,
              root_ang_vel_w, ee_pos_w, joint_pos, joint_vel=None):
    values = dict(root_pos_w=root_pos_w, root_quat_w=root_quat_w,
                  root_lin_vel_w=root_lin_vel_w, root_ang_vel_w=root_ang_vel_w,
                  ee_pos_w=ee_pos_w, joint_pos=joint_pos,
                  joint_vel=torch.zeros_like(joint_pos) if joint_vel is None else joint_vel)
    for name, value in values.items():
      expected = (self.num_envs, *getattr(self, name).shape[2:])
      if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}")
      if value.device != self.root_pos_w.device or not value.is_floating_point():
        raise ValueError(f"{name} must be floating point on {self.device}")
    return values

  def reset_current(self, env_ids, **frames):
    """Repeat each selected environment's current frame through its window."""
    values = self._frames(**frames)
    ids = torch.as_tensor(env_ids, device=self.device)
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
      raise ValueError("env_ids must be a one-dimensional integer index array")
    if bool(((ids < 0) | (ids >= self.num_envs)).any()):
      raise ValueError("env_ids out of bounds")
    for name, value in values.items():
      getattr(self, name)[ids] = value[ids, None]

  def reset(self, env_ids, root_pos_w, root_quat_w, root_lin_vel_w,
            root_ang_vel_w, ee_pos_w, joint_pos, joint_vel=None):
    """Initialize selected histories from current full-batch frames."""
    self.reset_current(env_ids, root_pos_w=root_pos_w, root_quat_w=root_quat_w,
                       root_lin_vel_w=root_lin_vel_w, root_ang_vel_w=root_ang_vel_w,
                       ee_pos_w=ee_pos_w, joint_pos=joint_pos, joint_vel=joint_vel)

  def update(self, root_pos_w, root_quat_w, root_lin_vel_w,
             root_ang_vel_w, ee_pos_w, joint_pos, joint_vel=None):
    """Append current frames after validating the entire batch."""
    values = self._frames(root_pos_w, root_quat_w, root_lin_vel_w,
                          root_ang_vel_w, ee_pos_w, joint_pos, joint_vel)
    for name, value in values.items():
      history = getattr(self, name)
      history[:, :-1] = history[:, 1:].clone()
      history[:, -1] = value

  def compute_features(self) -> torch.Tensor:
    """Return features ``(num_envs, W, 3+6+J+E*3+3+3)``, all anchored to the LAST
    frame's yaw-only local frame (layout in the class docstring)."""
    N = self.num_envs
    W = self.window_size
    E = self.num_ee

    anchor_pos_T = self.root_pos_w[:, -1]
    anchor_quat_T = self.root_quat_w[:, -1]
    yaw_T = yaw_quat(anchor_quat_T)
    heading_inv_T = quat_conjugate(yaw_T)
    heading_inv_T_W = heading_inv_T[:, None, :].expand(N, W, 4)
    yaw_T_W = yaw_T[:, None, :].expand(N, W, 4).reshape(-1, 4)

    root_offset = self.root_pos_w - anchor_pos_T[:, None, :]
    root_pos_local = quat_apply_inverse(yaw_T_W, root_offset.reshape(-1, 3)).reshape(
      N, W, 3
    )
    root_pos_local = root_pos_local.clone()
    root_pos_local[..., 2] = self.root_pos_w[..., 2]

    # 6D rot is stacked [col0, col2] = [rotated-x-axis, rotated-z-axis].
    root_rot_local_quat = quat_mul(
      heading_inv_T_W.reshape(-1, 4),
      self.root_quat_w.reshape(-1, 4),
    ).reshape(N, W, 4)
    root_rot_mat = matrix_from_quat(root_rot_local_quat.reshape(-1, 4)).reshape(
      N, W, 3, 3
    )
    root_rot_6d = torch.cat([root_rot_mat[..., :, 0], root_rot_mat[..., :, 2]], dim=-1)

    ee_offset_w = self.ee_pos_w - self.root_pos_w[:, :, None, :]
    yaw_T_E = yaw_T[:, None, None, :].expand(N, W, E, 4).reshape(-1, 4)
    ee_pos_local = quat_apply_inverse(yaw_T_E, ee_offset_w.reshape(-1, 3)).reshape(
      N, W, E * 3
    )

    lin_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_lin_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)
    ang_vel_local = quat_apply_inverse(
      yaw_T_W, self.root_ang_vel_w.reshape(-1, 3)
    ).reshape(N, W, 3)

    return torch.cat(
      [
        root_pos_local,
        root_rot_6d,
        self.joint_pos,
        ee_pos_local,
        lin_vel_local,
        ang_vel_local,
      ],
      dim=-1,
    )
