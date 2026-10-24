"""Compare the optional prior against the retained fork's actual modules."""
import copy
import pytest
import torch

pytest.importorskip('rsl_rl.modules.discriminator')
pytest.importorskip('mujoco')
from mbdac.algorithms import MBDACAMP,MBDACConfig
from mbdac.modules.codecs import EuclideanCodec
from examples.mjlab_amp.upstream_prior import UpstreamAmpPrior


def test_upstream_prior_optimizer_normalization_reward_and_checkpoint(tmp_path):
    torch.manual_seed(41)
    p0,p1,e0,e1=[torch.randn(16,2) for _ in range(4)]
    e0[0,0]=100.  # Exercise the upstream feature clipping path.
    class Task:
        codec=EuclideanCodec(2)
        action_dim=1; context_dim=1; obs_dim=2; goal_dim=1; critic_dim=4; amp_feature_dim=2
        def amp_features(self,state): return state
        def sample_expert_features(self,count): assert count==16; return e0,e1
        def make_amp_prior(self,device,options): return UpstreamAmpPrior(2,(16,),device)
    cfg=MBDACConfig(actor_hidden=(8,),wm_hidden=(8,),critic_hidden=(8,),
                    amp=dict(batch_size=16,max_grad_norm=None,lr=1e-3),precision='fp32')
    alg=MBDACAMP(Task(),4,cfg)
    class Replay:
        def sample_starts(self,count,history):
            assert count==16 and history==1
            return dict(s=p0,s1=p1),{}
    alg.replay=Replay()
    alg.prior.norm.update(torch.randn(64,2)*torch.tensor([.5,2.])+torch.tensor([1.,-1.]))
    reference=copy.deepcopy(alg.prior.discriminator)
    norm=copy.deepcopy(alg.prior.norm.original)
    # Same forward reward, but the wrapper additionally retains actor input gradients.
    expected,_=reference.predict_amp_reward(e0,e1,torch.zeros(16),normalizer=norm)
    actual=.1*alg.prior.reward(e0,e1)
    assert torch.allclose(expected,actual,atol=1e-7)
    differentiable=p0.clone().requires_grad_()
    gradient=torch.autograd.grad(alg.prior.reward(differentiable,p1).sum(),differentiable)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    optimizer=torch.optim.Adam([
        dict(params=reference.trunk.parameters(),weight_decay=1e-3),
        dict(params=reference.amp_linear.parameters(),weight_decay=.1)],lr=cfg.amp['lr'])
    pn0,pn1,en0,en1=[norm.normalize_torch(x,'cpu') for x in (p0,p1,e0,e1)]
    dp=reference(torch.cat((pn0,pn1),-1)).squeeze(-1)
    de=reference(torch.cat((en0,en1),-1)).squeeze(-1)
    loss=.5*((dp+1).square().mean()+(de-1).square().mean())+reference.compute_grad_pen(e0,e1,lambda_=10)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    norm.update(pn0.numpy()); norm.update(en0.numpy())
    logs=alg.update_prior()
    assert abs(logs['amp_loss']-float(loss.detach()))<1e-6
    for a,b in zip(alg.prior.discriminator.parameters(),reference.parameters()):
        assert torch.allclose(a,b,atol=1e-7,rtol=0)
    assert (abs(alg.prior.norm.original.mean-norm.mean)<1e-12).all()
    assert (abs(alg.prior.norm.original.var-norm.var)<1e-12).all()
    # Checkpoint contains portable numerical normalizer state, not SDK objects.
    path=tmp_path/'prior.pt'; torch.save(alg.save(),path)
    restored=MBDACAMP(Task(),4,cfg); restored.load(torch.load(path,weights_only=False))
    assert torch.equal(restored.prior.score(p0,p1),alg.prior.score(p0,p1))
