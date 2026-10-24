"""Physical-frame information needed by the native velocity task's force heads."""
import torch
from examples.mjlab_velocity.model_task import VelocityCodec
from mbdac.utils.rotations import so3_exp


def test_world_force_features_retain_yaw_while_local_dynamics_targets_are_equivariant():
    codec=VelocityCodec(2,2)
    R=torch.eye(3)[None];Z=so3_exp(torch.tensor([[0.,0.,1.2]]))
    p=torch.tensor([[0.,0.,.3]]);zero=torch.zeros(1,3);q=torch.zeros(1,2)
    memory=torch.zeros(1,4)
    pack=lambda rotation,position:torch.cat((codec.pack((position,rotation,zero,zero,q,q)),memory),-1)
    a=pack(R,p);b=pack(Z,p)
    assert codec.features(a).shape[-1]==codec.feature_dim
    assert not torch.allclose(codec.features(a),codec.features(b))
    displacement=torch.tensor([[.01,.02,0.]])
    a1=pack(R,p+displacement);b1=pack(Z,p+(Z@displacement[...,None]).squeeze(-1))
    torch.testing.assert_close(codec.targets(a,a1),codec.targets(b,b1),atol=1e-6,rtol=1e-6)
