"""State representations and geometry, independent of simulator SDKs."""
import torch
from torch import nn
from mbdac.utils.rotations import mat_to_6d, mat_to_quat, quat_to_mat, so3_exp, so3_log, yaw_mat


class EuclideanCodec(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.state_dim = self.feature_dim = self.delta_dim = dim
        self.register_buffer('error_floor', torch.full((dim,), 1e-3))

    def features(self, state):
        return state

    def targets(self, state, next_state):
        return next_state - state

    def integrate(self, state, delta):
        return state + delta

    def error(self, predicted, target):
        return predicted - target


class ArticulatedCodec(nn.Module):
    """Floating root (wxyz quaternion) and any number of scalar joints.

    Local increment convention follows SuperMimic's MIT implementation.
    """
    def __init__(self, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.state_dim = self.feature_dim = 13 + 2 * num_joints
        self.delta_dim = 12 + 2 * num_joints
        self.register_buffer('error_floor', torch.cat((
            torch.full((3,), .003), torch.full((3,), .005),
            torch.full((3,), .02), torch.full((3,), .05),
            torch.full((num_joints,), .003), torch.full((num_joints,), .05))))

    def unpack(self, s):
        n = self.num_joints
        return s[..., :3], quat_to_mat(s[..., 3:7]), s[..., 7:10], s[..., 10:13], s[..., 13:13+n], s[..., 13+n:]

    def pack(self, S):
        p, R, v, w, q, qd = S
        return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)

    def features(self, s):
        p, R, v, w, q, qd = self.unpack(s)
        Rt = R.transpose(-1, -2)
        return torch.cat((p[..., 2:3], mat_to_6d(yaw_mat(R).transpose(-1, -2) @ R),
                          (Rt @ v.unsqueeze(-1)).squeeze(-1),
                          (Rt @ w.unsqueeze(-1)).squeeze(-1), q, qd), -1)

    def targets(self, state, next_state):
        p, R, v, w, q, qd = self.unpack(state)
        p1, R1, v1, w1, q1, qd1 = self.unpack(next_state)
        Rt = R.transpose(-1, -2)
        return torch.cat(((yaw_mat(R).transpose(-1, -2) @ (p1-p).unsqueeze(-1)).squeeze(-1),
                          so3_log(Rt @ R1), (Rt @ (v1-v).unsqueeze(-1)).squeeze(-1),
                          (Rt @ (w1-w).unsqueeze(-1)).squeeze(-1), q1-q, qd1-qd), -1)

    def integrate(self, state, delta):
        p, R, v, w, q, qd = self.unpack(state)
        dp, dr, dv, dw, dq, dqd = delta.split((3,3,3,3,self.num_joints,self.num_joints), -1)
        return self.pack((p+(yaw_mat(R) @ dp.unsqueeze(-1)).squeeze(-1), R @ so3_exp(dr),
                          v+(R @ dv.unsqueeze(-1)).squeeze(-1), w+(R @ dw.unsqueeze(-1)).squeeze(-1), q+dq, qd+dqd))

    def error(self, predicted, target):
        return self.targets(target, predicted)
