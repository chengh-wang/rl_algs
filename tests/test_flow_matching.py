"""Standalone continuous-flow contract and original v1 compatibility checks."""
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mbdac.flow_matching import (
    FlowMotionPrior, FlowMotionScorer, FlowTrainingConfig, FlowVelocityModel,
    FrozenFlowPrior, PATH_CONTRACT, flow_matching_loss, legacy_score_times,
    linear_path, train_flow_prior,
)


def small_model(f=3, w=2):
    return FlowVelocityModel(f, w, d_model=8, nhead=2, num_layers=1)


def small_prior(f=3, w=2):
    return FlowMotionPrior(small_model(f,w), -torch.ones(f), torch.ones(f))


def test_linear_path_endpoints_and_conditional_velocity():
    x, eps = torch.randn(2,3,4), torch.randn(2,3,4)
    point, target = linear_path(x,eps,torch.tensor([0.,1.]))
    torch.testing.assert_close(point[0],eps[0])
    torch.testing.assert_close(point[1],x[1])
    torch.testing.assert_close(target,x-eps)
    with pytest.raises(ValueError):
        linear_path(x,eps,torch.tensor([-.1,.5]))


def test_loss_matches_independent_construction_and_gradients():
    model = small_model()
    expected_model = deepcopy(model)
    x = torch.randn(2,2,3)
    generator = torch.Generator().manual_seed(123)
    actual = flow_matching_loss(model,x,3,generator=generator)
    expanded = x.repeat_interleave(3,dim=0)
    generator.manual_seed(123)
    tau = torch.rand(6,generator=generator)
    noise = torch.randn(expanded.shape,generator=generator)
    t = tau[:,None,None]
    expected = ((expected_model((1-t)*noise+t*expanded,tau*50)-(expanded-noise))**2).mean()
    torch.testing.assert_close(actual,expected)
    actual.backward(); expected.backward()
    for p,q in zip(model.parameters(),expected_model.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    for bad in (torch.zeros(0,2,3),torch.ones(2,2,3,dtype=torch.long),torch.full((1,2,3),float('nan'))):
        with pytest.raises(ValueError):
            flow_matching_loss(model,bad)


def test_fixed_noise_matches_manual_continuous_residual():
    prior = small_prior()
    times = (.2,.7)
    scorer = FlowMotionScorer(prior,times,ws=2.)
    features = torch.randn(4,2,3)
    noise = torch.randn(2,4,2,3)
    result = scorer.score(features,noise=noise)
    x = 2*(features-prior.q_low)/(prior.q_high-prior.q_low+1e-8)-1
    errors = []
    for t,eps in zip(times,noise):
        tau = torch.full((len(x),),t)
        errors.append((prior.model(t*x+(1-t)*eps,tau*50)-(x-eps)).square().mean((-1,-2)))
    expected = torch.stack(errors)
    torch.testing.assert_close(result.per_time_error,expected)
    torch.testing.assert_close(result.raw_error,expected.mean(0))
    torch.testing.assert_close(result.normalized_error,expected.mean(0))
    torch.testing.assert_close(result.reward,torch.exp(-2*expected.mean(0)))
    assert not scorer.normalizer.count.any()
    assert not hasattr(prior,'scheduler') and not hasattr(prior,'normalizer')


def test_legacy_parity_with_mapped_times_and_online_statistics():
    model = small_model()
    old = FrozenFlowPrior(deepcopy(model),-torch.ones(3),torch.ones(3))
    new = FlowMotionScorer(FlowMotionPrior(model,-torch.ones(3),torch.ones(3)),legacy_score_times())
    for _ in range(2):
        features,noise = torch.randn(4,2,3),torch.randn(3,4,2,3)
        legacy = old.score(features,noise=noise)
        current = new.score(features,noise=noise,update_normalizer=True)
        for name in ('reward','raw_error','normalized_error'):
            torch.testing.assert_close(getattr(current,name),getattr(legacy,name))
        torch.testing.assert_close(new.normalizer.mean,old.normalizer.mean[[8,15,22]])
        torch.testing.assert_close(new.normalizer.count,old.normalizer.count[[8,15,22]])


def test_frozen_prior_and_opt_in_normalizer_roundtrip_configuration_guard():
    prior = small_prior()
    prior.train()
    assert not prior.training and not prior.model.training
    assert all(not p.requires_grad for p in prior.parameters())
    scorer = FlowMotionScorer(prior,(.2,.8))
    features,noise = torch.randn(4,2,3),torch.randn(2,4,2,3)
    weights = deepcopy(prior.state_dict())
    scorer.score(features,noise=noise)
    assert not scorer.normalizer.count.any()
    scorer.score(features,noise=noise,update_normalizer=True)
    assert scorer.normalizer.count.tolist() == [4,4]
    for key,value in prior.state_dict().items():
        torch.testing.assert_close(value,weights[key])
    state = scorer.normalizer_state_dict()
    restored = FlowMotionScorer(small_prior(),(.2,.8))
    restored.load_normalizer_state_dict(state)
    torch.testing.assert_close(restored.normalizer.mean,scorer.normalizer.mean)
    torch.testing.assert_close(restored.normalizer.count,scorer.normalizer.count)
    restored.load_state_dict(deepcopy(scorer.state_dict()))
    for incompatible in (FlowMotionScorer(prior,(.3,.8)),FlowMotionScorer(prior,(.2,.8),ws=2)):
        with pytest.raises(ValueError,match='configuration'):
            incompatible.load_normalizer_state_dict(state)
        with pytest.raises(ValueError,match='configuration'):
            incompatible.load_state_dict(deepcopy(scorer.state_dict()))


def test_explicit_noise_leaves_rng_unchanged_and_generators_reproduce():
    scorer = FlowMotionScorer(small_prior(),(.1,.9))
    features,noise = torch.randn(2,2,3),torch.randn(2,2,2,3)
    state = torch.get_rng_state().clone()
    scorer.score(features,noise=noise)
    torch.testing.assert_close(torch.get_rng_state(),state)
    a = scorer.score(features,generator=torch.Generator().manual_seed(12))
    b = scorer.score(features,generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(a.per_time_error,b.per_time_error)
    with pytest.raises(ValueError):
        scorer.score(features,noise=noise,generator=torch.Generator())


def test_tensor_training_checkpoint_load_schema_dimensions_and_overwrite(tmp_path):
    windows = torch.randn(5,3,7)
    config = FlowTrainingConfig(epochs=1,batch_size=3,num_noise_samples=2,d_model=8,nhead=2,num_layers=1)
    schema = {'layout':['opaque','custom'],'version':7}
    target = train_flow_prior(windows,tmp_path,config,feature_schema=schema)
    assert target == tmp_path/'prior.pt'
    checkpoint = torch.load(target,weights_only=True)
    assert checkpoint['checkpoint_version'] == 1
    assert checkpoint['path_contract'] == PATH_CONTRACT
    assert checkpoint['epoch'] == 1
    assert checkpoint['metrics']['train_velocity_mse'] > 0
    expected = torch.quantile(windows.reshape(-1,7),torch.tensor([.01,.99]),dim=0)
    torch.testing.assert_close(checkpoint['q_low'],expected[0])
    torch.testing.assert_close(checkpoint['q_high'],torch.maximum(expected[1],expected[0]+config.span_floor))
    prior = FlowMotionPrior.load_checkpoint(target,expected_feature_dim=7,expected_window_size=3,expected_feature_schema=schema)
    score = FlowMotionScorer(prior,(.25,)).score(windows,noise=torch.zeros(1,*windows.shape))
    assert torch.isfinite(score.reward).all()
    # Original v1 loader remains able to consume new checkpoints.
    FrozenFlowPrior.load_checkpoint(target)
    for kwargs in ({'expected_feature_dim':8},{'expected_window_size':4},{'expected_feature_schema':{'layout':'other'}}):
        with pytest.raises(ValueError):
            FlowMotionPrior.load_checkpoint(target,**kwargs)
    with pytest.raises(FileExistsError):
        train_flow_prior(windows,tmp_path,config)
    checkpoint['checkpoint_version'] = 2
    torch.save(checkpoint,tmp_path/'future.pt')
    with pytest.raises(ValueError,match='version'):
        FlowMotionPrior.load_checkpoint(tmp_path/'future.pt')
    checkpoint.pop('checkpoint_version')
    torch.save(checkpoint,tmp_path/'original.pt')
    FlowMotionPrior.load_checkpoint(tmp_path/'original.pt')


@pytest.mark.parametrize('bad', [torch.ones(1,2,3,dtype=torch.long),torch.zeros(0,2,3),torch.full((1,2,3),float('nan')),torch.zeros(1,0,3)])
def test_training_rejects_invalid_raw_windows(tmp_path,bad):
    with pytest.raises(ValueError):
        train_flow_prior(bad,tmp_path,FlowTrainingConfig(epochs=1))
    assert not (tmp_path/'prior.pt').exists()


def test_configuration_and_scoring_reject_invalid_inputs():
    for kwargs in ({'epochs':0},{'d_model':3},{'batch_size':True},{'learning_rate':float('nan')},{'weight_decay':-1}):
        with pytest.raises(ValueError):
            FlowTrainingConfig(**kwargs)
    with pytest.raises(ValueError):
        FlowVelocityModel(0,2)
    prior = small_prior()
    for times in ((),(0.,), (1.,), (float('nan'),)):
        with pytest.raises(ValueError):
            FlowMotionScorer(prior,times)
    scorer = FlowMotionScorer(prior,(.2,))
    for features in (torch.ones(1,2,3,dtype=torch.long),torch.zeros(0,2,3),torch.full((1,2,3),float('inf')),torch.zeros(1,3,3)):
        with pytest.raises(ValueError):
            scorer.score(features)


def test_import_and_small_training_without_flashsac_or_simulators(tmp_path):
    code = '''
import importlib.abc, sys, torch
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(s in fullname.lower() for s in ('flashsac','mjlab','isaaclab','mujoco','gymnasium')):
            raise AssertionError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0,Block())
from mbdac.flow_matching import FlowTrainingConfig, train_flow_prior, FlowMotionPrior, FlowMotionScorer
path = train_flow_prior(torch.randn(2,2,3),sys.argv[1],FlowTrainingConfig(epochs=1,batch_size=2,num_noise_samples=1,d_model=8,nhead=2,num_layers=1))
prior = FlowMotionPrior.load_checkpoint(path)
assert FlowMotionScorer(prior,(.4,)).score(torch.randn(1,2,3)).reward.shape == (1,)
'''
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path)],capture_output=True,text=True,env={**os.environ,'OMP_NUM_THREADS':'2'},timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr


def test_score_rejects_float32_overflow():
    scorer = FlowMotionScorer(small_prior(),(.2,))
    with pytest.raises(ValueError,match='overflow'):
        scorer.score(torch.full((1,2,3),1e300,dtype=torch.float64))
    with pytest.raises(ValueError,match='overflow'):
        scorer.score(torch.ones(1,2,3),noise=torch.full((1,1,2,3),1e300,dtype=torch.float64))
