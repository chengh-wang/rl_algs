"""Simulator-independent continuous velocity matching contracts."""
from copy import deepcopy
import os
import json
import re
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from mbdac.flow_matching import (
    FlowMotionPrior, FlowMotionScorer, FlowTrainingConfig, FlowVelocityModel,
    PATH_CONTRACT, flow_matching_loss,
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
    expected = ((expected_model((1-t)*noise+t*expanded,tau)-(expanded-noise))**2).mean()
    torch.testing.assert_close(actual,expected)
    actual.backward(); expected.backward()
    for p,q in zip(model.parameters(),expected_model.parameters()):
        torch.testing.assert_close(p.grad,q.grad)
    for bad in (torch.zeros(0,2,3),torch.ones(2,2,3,dtype=torch.long),torch.full((1,2,3),float('nan'))):
        with pytest.raises(ValueError):
            flow_matching_loss(model,bad)


def test_model_accepts_only_native_continuous_tau():
    model = small_model()
    x = torch.randn(2,2,3)
    assert model(x,torch.tensor([0.,1.])).shape == x.shape
    for tau in (torch.tensor([-.1,.2]), torch.tensor([.2,1.1]),
                torch.tensor([.2,float('nan')]), torch.tensor([0,1]), torch.zeros(2,1)):
        with pytest.raises(ValueError):
            model(x,tau)


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
        errors.append((prior.model(t*x+(1-t)*eps,tau)-(x-eps)).square().mean((-1,-2)))
    expected = torch.stack(errors)
    torch.testing.assert_close(result.per_time_error,expected)
    torch.testing.assert_close(result.raw_error,expected.mean(0))
    torch.testing.assert_close(result.normalized_error,expected.mean(0))
    torch.testing.assert_close(result.reward,torch.exp(-2*expected.mean(0)))
    assert not scorer.normalizer.count.any()
    assert not hasattr(prior,'scheduler') and not hasattr(prior,'normalizer')


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
    first = scorer.normalizer.mean.clone()
    second_result = scorer.score(features[:2], noise=noise[:,:2], update_normalizer=True)
    assert scorer.normalizer.count.tolist() == [6,6]
    torch.testing.assert_close(scorer.normalizer.mean, (first*4 + second_result.per_time_error.sum(1))/6)
    for key,value in prior.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value,weights[key])
        else:
            assert value == weights[key]
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
    validation = torch.full((2,3,7), 100.)
    target = train_flow_prior(windows,tmp_path,config,feature_schema=schema,validation_windows=validation)
    assert target == tmp_path/'prior.pt'
    checkpoint = torch.load(target,weights_only=True)
    assert checkpoint['checkpoint_version'] == 2
    assert checkpoint['path_contract'] == PATH_CONTRACT
    report = json.loads((tmp_path/'report.json').read_text())
    assert report['epochs'][-1]['train_velocity_mse'] > 0
    assert report['epochs'][-1]['validation_velocity_mse'] > 0
    assert report['validation_shape'] == [2,3,7]
    assert report['optimizer_updates'] == 2
    expected = torch.quantile(windows.reshape(-1,7),torch.tensor([.01,.99]),dim=0)
    torch.testing.assert_close(checkpoint['q_low'],expected[0])
    torch.testing.assert_close(checkpoint['q_high'],torch.maximum(expected[1],expected[0]+config.span_floor))
    prior = FlowMotionPrior.load_checkpoint(target,expected_feature_dim=7,expected_window_size=3,expected_feature_schema=schema)
    score = FlowMotionScorer(prior,(.25,)).score(windows,noise=torch.zeros(1,*windows.shape))
    assert torch.isfinite(score.reward).all()
    prior.save_checkpoint(tmp_path/'copy.pt')
    restored = FlowMotionPrior.load_checkpoint(tmp_path/'copy.pt')
    restored_score = FlowMotionScorer(restored,(.25,)).score(windows,noise=torch.zeros(1,*windows.shape))
    torch.testing.assert_close(restored_score.per_time_error,score.per_time_error)
    for kwargs in ({'expected_feature_dim':8},{'expected_window_size':4},{'expected_feature_schema':{'layout':'other'}}):
        with pytest.raises(ValueError):
            FlowMotionPrior.load_checkpoint(target,**kwargs)
    with pytest.raises(FileExistsError):
        train_flow_prior(windows,tmp_path,config)
    for name, mutate in (
        ('version', lambda c: c.update(checkpoint_version=1)),
        ('future_version', lambda c: c.update(checkpoint_version=3)),
        ('missing_schema', lambda c: c.pop('feature_schema')),
        ('missing_version', lambda c: c.pop('checkpoint_version')),
        ('type', lambda c: c.update(prior_type='unknown')),
        ('schema', lambda c: c.update(feature_schema=['invalid'])),
        ('dimensions', lambda c: c['cfg'].update(feature_dim=8)),
        ('weights', lambda c: c['model'].pop(next(iter(c['model'])))),
        ('contract', lambda c: c.update(path_contract={})),
    ):
        broken = deepcopy(checkpoint)
        mutate(broken)
        bad_path = tmp_path / (name + '.pt')
        torch.save(broken, bad_path)
        with pytest.raises((ValueError, RuntimeError)):
            FlowMotionPrior.load_checkpoint(bad_path)



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
        if fullname == 'mbdac.smp' or fullname.startswith('mbdac.smp.') or any(s in fullname.lower() for s in ('flashsac','mjlab','isaaclab','mujoco','gymnasium')):
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


@pytest.mark.parametrize('feature_dim,window_size', [(2,1),(7,4)])
def test_generic_feature_and_window_dimensions(feature_dim, window_size):
    model = small_model(feature_dim, window_size)
    windows = torch.randn(3, window_size, feature_dim)
    loss = flow_matching_loss(model, windows, 2)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
    prior = FlowMotionPrior(model, -torch.ones(feature_dim), torch.ones(feature_dim))
    result = FlowMotionScorer(prior, (.13,.81)).score(windows)
    assert result.per_time_error.shape == (2,3)
    assert torch.isfinite(result.reward).all()


def test_flow_source_has_no_foreign_algorithm_or_simulator_dependency():
    source = Path(__file__).resolve().parents[1] / 'src' / 'mbdac' / 'flow_matching'
    forbidden = re.compile(r'ddpm|diffusion|legacy|mbdac\.smp|flashsac|mjlab|isaaclab|mujoco|gymnasium', re.I)
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.md') and path.name != 'PROVENANCE.md':
            assert forbidden.search(path.read_text()) is None, str(path)


def test_tensor_npz_cli_with_generic_schema(tmp_path):
    import numpy as np
    windows = np.random.default_rng(17).normal(size=(4,1,5)).astype('float32')
    source = tmp_path/'input.npz'
    np.savez(source, features=windows)
    schema = {'columns':['a','b','c','d','e'], 'units':'arbitrary', 'revision':3}
    schema_path = tmp_path/'schema.json'
    schema_path.write_text(json.dumps(schema))
    output = tmp_path/'trained'
    result = subprocess.run([
        sys.executable, '-m', 'mbdac.flow_matching.pretrain',
        '--windows', str(source), '--output', str(output),
        '--feature-schema', str(schema_path), '--epochs', '1', '--batch-size', '2',
        '--d-model', '8', '--nhead', '2', '--num-layers', '1',
        '--num-noise-samples', '1', '--device', 'cpu',
    ], capture_output=True, text=True, env={**os.environ,'OMP_NUM_THREADS':'2'}, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    prior = FlowMotionPrior.load_checkpoint(output/'prior.pt', expected_feature_dim=5,
                                            expected_window_size=1, expected_feature_schema=schema)
    features = torch.from_numpy(windows)
    score = FlowMotionScorer(prior,(.37,)).score(features, noise=torch.zeros(1,*features.shape))
    assert torch.isfinite(score.reward).all()


def test_validation_does_not_change_training_and_loading_preserves_rng(tmp_path):
    windows = torch.randn(4, 2, 3)
    cfg = FlowTrainingConfig(epochs=2, batch_size=2, num_noise_samples=1,
                             d_model=8, nhead=2, num_layers=1)
    a = train_flow_prior(windows, tmp_path/'a', cfg)
    b = train_flow_prior(windows, tmp_path/'b', cfg,
                         validation_windows=torch.ones(2, 2, 3))
    first, second = torch.load(a, weights_only=True), torch.load(b, weights_only=True)
    for name, tensor in first['model'].items():
        torch.testing.assert_close(tensor, second['model'][name], rtol=0, atol=0)
    before = torch.get_rng_state().clone()
    FlowMotionPrior.load_checkpoint(a)
    assert torch.equal(before, torch.get_rng_state())


def test_full_scorer_state_rejects_prior_contract_and_invalid_tensors():
    prior = small_prior()
    scorer = FlowMotionScorer(prior, (.2, .8))
    state = deepcopy(scorer.state_dict())
    different_scale = FlowMotionPrior(
        FlowVelocityModel(3, 2, d_model=8, nhead=2, num_layers=1,
                          time_embedding_scale=100.), -torch.ones(3), torch.ones(3))
    different_schema = FlowMotionPrior(small_model(), -torch.ones(3), torch.ones(3),
                                      feature_schema={'layout': 'other'})
    for incompatible in (different_scale, different_schema):
        with pytest.raises(ValueError):
            FlowMotionScorer(incompatible, (.2, .8)).load_state_dict(state)
    for key in ('prior.q_low', 'prior.model.proj_in.weight'):
        invalid = deepcopy(state)
        invalid[key].fill_(float('nan'))
        with pytest.raises(ValueError):
            scorer.load_state_dict(invalid)
