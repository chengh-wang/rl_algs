"""Generic reward contracts and C51 diagnostics; no simulator or GPU required."""
from dataclasses import replace
import torch
import pytest
from mbdac.flashsac.diagnostics import categorical_support_diagnostics
from mbdac.flashsac.config import RunnerConfig
from mbdac.flashsac.runner import FlashSACRunner as CoreRunner, raw_module
from mbdac.runners.flash_sac_runner import FlashSACRunner, SMPFlashSACRunner
from mbdac.rewards.potential import PotentialRewardProvider, PotentialShapingConfig
from test_flashsac_runner import ToyEnv, config


def test_c51_probability_weighted_clipping_and_projected_mean():
    log_probs=torch.tensor([[.25,.5,.25]]).log()
    prediction=log_probs[None].expand(2,-1,-1)
    out=categorical_support_diagnostics(log_probs,torch.ones(1),torch.zeros(1),torch.zeros(1),
        1.,-1.,1.,prediction,torch.tensor([[0.,.25,.75]]))
    for name,expected in [('target_below_support_mass',0.),('target_above_support_mass',.25),
            ('target_clipped_mass',.25),('target_unclipped_mean',1.),('target_clamped_mean',.75),
            ('projected_mean',.75),('predicted_mean',0.),('predicted_lower_edge_mass',.25),
            ('predicted_upper_edge_mass',.25)]:
        torch.testing.assert_close(out['c51/'+name],torch.tensor(expected))
    terminal=categorical_support_diagnostics(log_probs,torch.tensor([-2.]),torch.ones(1),torch.tensor([999.]),
        .99,-1.,1.,prediction,torch.tensor([[1.,0.,0.]]))
    assert terminal['c51/target_below_support_mass']==1
    assert terminal['c51/target_clamped_mean']==-1


def test_diagnostics_do_not_change_rng_weights_or_original_metrics():
    torch.set_num_threads(1)
    cfg=RunnerConfig(device='cpu',capacity=16,warmup=4,batch_size=4,use_amp=False,use_compile=False)
    def build(diagnose):
        torch.manual_seed(42)
        learner=CoreRunner(3,5,2,4,replace(cfg,critic_diagnostics=diagnose),{'task':'diagnostic-parity'})
        torch.manual_seed(3)
        learner.observe(dict(observation=torch.randn(4,8),next_observation=torch.randn(4,8),action=torch.randn(4,2).tanh(),
            reward=torch.randn(4),terminated=torch.zeros(4,dtype=torch.bool),truncated=torch.zeros(4,dtype=torch.bool)))
        return learner
    plain,diagnostic=build(False),build(True)
    torch.manual_seed(88);left=plain.update();left_rng=torch.get_rng_state().clone()
    torch.manual_seed(88);right=diagnostic.update();right_rng=torch.get_rng_state().clone()
    assert torch.equal(left_rng,right_rng)
    assert not any('/c51/' in k for k in left)
    assert any('/c51/' in k for k in right)
    assert all(torch.isfinite(v).all() for v in right.values())
    for key,value in left.items():torch.testing.assert_close(value,right[key],rtol=0,atol=0)
    for name,network in plain.networks.items():
        for key,value in raw_module(network).state_dict().items():
            assert torch.equal(value,raw_module(diagnostic.networks[name]).state_dict()[key])


class PotentialEnv(ToyEnv):
    def __init__(self,gamma=.99):
        self.reward_provider=PotentialRewardProvider(PotentialShapingConfig('toy_phi','v1',gamma=gamma,beta=10.),
            next_potential_is_pre_reset=True)
        super().__init__()
    def step(self,actions):
        obs,reward,done,extras=super().step(actions)
        extras['reward_potential']={'current':torch.full((4,),.1*(self.t-1)),
            'next':torch.full((4,),.1*self.t),'version':'v1'}
        return obs,reward,done,extras


def provider_config():
    cfg=config();cfg['reward']={'source':'provider'}
    cfg['algorithm'].update(reward_normalization='fixed',reward_scale=2.,gamma=.99)
    return cfg


def test_generic_provider_uses_task_stats_combined_replay_and_cold_resume(tmp_path):
    torch.set_num_threads(1)
    env=PotentialEnv();runner=FlashSACRunner(env,provider_config())
    observed=[];original=runner.alg.observe
    def record(transition):
        observed.append({k:v.clone() for k,v in transition.items()})
        return original(transition)
    runner.alg.observe=record
    runner.learn(2)
    assert 'reward_provider' in runner.contract and 'smp' not in runner.contract
    assert not torch.equal(observed[0]['reward'],observed[0]['task_reward'])
    torch.testing.assert_close(observed[0]['reward']-observed[0]['task_reward'],torch.full((4,),.99))
    # Both components share the same divisor; replay stores unscaled combined reward.
    torch.testing.assert_close(runner.alg.replay._rewards[:4].reshape(-1),observed[0]['reward'])
    assert runner.alg.reward_scale==2
    path=tmp_path/'provider.pt';runner.save(path,include_replay=True)
    restored_env=PotentialEnv();restored=FlashSACRunner(restored_env,provider_config())
    restored.load(path)
    assert restored_env.reward_provider.calls==2
    assert restored_env.reward_provider._next is None and restored_env.reward_provider.last_task_reward is None
    restored.learn(1)
    assert restored_env.reward_provider.calls==3


def test_generic_provider_rejects_discount_endpoints_and_double_composition():
    cfg=provider_config()
    with pytest.raises(ValueError,match='algorithm_discount'):
        FlashSACRunner(PotentialEnv(.9),cfg)
    env=PotentialEnv();env.smp_contract={'already':'smp'}
    smp=config(True);smp['reward']={'source':'provider'}
    with pytest.raises(ValueError,match='combined'):
        SMPFlashSACRunner(env,smp)
    env=PotentialEnv();step=env.step
    def missing(actions):
        obs,reward,done,extras=step(actions)
        del extras['reward_potential']['next']
        return obs,reward,done,extras
    env.step=missing
    with pytest.raises(ValueError,match='endpoints'):
        FlashSACRunner(env,cfg).learn(1)


def test_provider_contract_strict_restore_and_legacy_contract_unchanged(tmp_path):
    torch.set_num_threads(1)
    plain=FlashSACRunner(ToyEnv(),config())
    assert set(plain.contract)=={'task','obs_groups','runner'}
    runner=FlashSACRunner(PotentialEnv(),provider_config());runner.learn(1)
    path=tmp_path/'provider.pt';runner.save(path)
    changed_env=PotentialEnv()
    changed_env.reward_provider=PotentialRewardProvider(PotentialShapingConfig('toy_phi','changed'),next_potential_is_pre_reset=True)
    with pytest.raises(ValueError,match='contract'):
        FlashSACRunner(changed_env,provider_config()).load(path)


@pytest.mark.parametrize('mode',['legacy','episodic'])
def test_potential_shaping_rejects_changing_reward_denominator(mode):
    cfg=provider_config()
    cfg['algorithm'].update(reward_normalization=mode,reward_scale=1.)
    with pytest.raises(ValueError,match="requires reward_normalization='fixed'.*telescoping"):
        FlashSACRunner(PotentialEnv(),cfg)


def test_potential_shaping_accepts_explicit_fixed_divisor():
    torch.set_num_threads(1)
    runner=FlashSACRunner(PotentialEnv(),provider_config())
    assert runner.learner_cfg.reward_normalization=='fixed'
    assert runner.alg.reward_scale==2.
    runner.learn(1)
