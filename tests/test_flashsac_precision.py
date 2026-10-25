"""Opt-in reward units/critic precision; no simulator or source checkout needed."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import os
import tempfile
import unittest
import torch
from mbdac.flashsac.config import RunnerConfig
from mbdac.flashsac.runner import FlashSACRunner, raw_module
from mbdac.flashsac.normalization import make_reward_normalizer
from mbdac.flashsac.learner import initialize_actor_only, initialize_value_from_actor_source, prepare_actor_initialization

NEW_KEYS=('gamma','critic_min_v','critic_max_v','critic_num_bins','reward_normalization','reward_scale','critic_diagnostics')


class PrecisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def runner(self,**kw):
        torch.manual_seed(9)
        config=dict(device='cpu',capacity=16,warmup=4,batch_size=4,use_amp=False,use_compile=False)
        config.update(kw)
        return FlashSACRunner(3,4,2,2,RunnerConfig(**config),{'units':'test'})

    def transition(self,reward,task=None,term=None,trunc=None):
        d=dict(observation=torch.zeros(2,7),next_observation=torch.ones(2,7),
            action=torch.zeros(2,2),reward=torch.tensor(reward,dtype=torch.float32),
            terminated=torch.tensor(term or [False,False]),truncated=torch.tensor(trunc or [False,False]))
        if task is not None:d['task_reward']=torch.tensor(task,dtype=torch.float32)
        return d

    def test_episodic_terminal_partial_timeout(self):
        n=make_reward_normalizer(RunnerConfig(gamma=.5,reward_normalization='episodic'),torch.device('cpu'),2)
        seen=[];original=n.G_rms.update
        def record(value):seen.append(value.clone());original(value)
        n.G_rms.update=record
        for d in (self.transition([1,10]),self.transition([2,20],term=[True,False]),
                  self.transition([3,30],trunc=[False,True])):
            n.update_reward_stats(d['reward'],d['terminated'],d['truncated'])
        for actual,expected in zip(seen,([1,10],[2.5,25],[3,42.5])):
            torch.testing.assert_close(actual,torch.tensor(expected,dtype=torch.float32),rtol=0,atol=0)
        torch.testing.assert_close(n.G_r,torch.tensor([3.,0.]),rtol=0,atol=0)
        self.assertEqual(float(n.G_r_max),42.5)
        # Legacy terminal carry remains intentionally available and unchanged.
        old=make_reward_normalizer(RunnerConfig(gamma=.5),torch.device('cpu'),1)
        for r,done in ((1,False),(2,True),(3,False)):
            old.update_reward_stats(torch.tensor([float(r)]),torch.tensor([done]),torch.tensor([False]))
        self.assertEqual(float(old.G_r),4.)

    def test_task_channel_common_scale_and_fixed_invariance(self):
        a=self.runner(reward_normalization='episodic');b=self.runner(reward_normalization='episodic')
        a.observe(self.transition([11,22],task=[1,2]));b.observe(self.transition([101,202],task=[1,2]))
        torch.testing.assert_close(a.normalizer.G_r,b.normalizer.G_r,rtol=0,atol=0)
        torch.testing.assert_close(a.reward_scale,b.reward_scale,rtol=0,atol=0)
        self.assertTrue(torch.equal(a.replay._rewards[:2],torch.tensor([11.,22.])))
        # Warm buffer and check one denominator applies to the combined reward.
        a.observe(self.transition([11,22],task=[1,2]))
        batch=a.batch(torch.tensor([0,1]))
        torch.testing.assert_close(batch['reward'],batch['raw_reward']/a.reward_scale,rtol=0,atol=0)
        f=self.runner(reward_normalization='fixed',reward_scale=4.)
        before={k:getattr(f.normalizer.G_rms,k).clone() for k in ('mean','var','count')}
        for _ in range(3):f.observe(self.transition([100,200],task=[3,4],term=[True,False]))
        self.assertEqual(float(f.reward_scale),4.)
        for k,v in before.items():torch.testing.assert_close(getattr(f.normalizer.G_rms,k),v,rtol=0,atol=0)
        torch.testing.assert_close(f.normalizer.G_r,torch.zeros(2),rtol=0,atol=0)
        batch=f.batch(torch.tensor([0,1]));torch.testing.assert_close(batch['reward'],torch.tensor([25.,50.]),rtol=0,atol=0)

    def test_new_modes_checkpoint_next_update_and_old_defaults(self):
        for mode in ('episodic','fixed'):
            opts=dict(reward_normalization=mode,gamma=.9,critic_min_v=-9.,critic_max_v=7.,critic_num_bins=81)
            if mode=='fixed':opts['reward_scale']=2.
            r=self.runner(**opts)
            for _ in range(2):r.observe(self.transition([1,2],task=[.1,.2]))
            r.update()
            self.assertEqual(r.replay._gamma,.9);self.assertEqual(r.upstream_cfg.critic_num_bins,81)
            with tempfile.TemporaryDirectory()as d:
                p=Path(d)/'x.pt';r.save(p,include_replay=True)
                fresh=self.runner(**opts);fresh.load(p,restore_rng=True,continue_same_environments=True)
                torch.manual_seed(81);one=r.update()
                torch.manual_seed(81);two=fresh.update()
                for k in one:torch.testing.assert_close(one[k],two[k],rtol=0,atol=0)
                cold=self.runner(**opts);cold.load(p)
                torch.testing.assert_close(cold.normalizer.G_r,torch.zeros(2),rtol=0,atol=0)
                with self.assertRaises(ValueError):self.runner().load(p)
        r=self.runner()
        for _ in range(2):r.observe(self.transition([1,2]))
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'old.pt';r.save(p);state=torch.load(p,weights_only=True)
            for key in NEW_KEYS:state['config'].pop(key)
            torch.save(state,p);self.runner().load(p)
            diagnostics=self.runner(critic_diagnostics=True);diagnostics.load(p)
            for name in r.networks:
                expected=raw_module(r.networks[name]).state_dict()
                actual=raw_module(diagnostics.networks[name]).state_dict()
                for key in expected:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
            self.assertTrue(diagnostics.upstream_cfg.critic_diagnostics)
            # Logging may change; numerical units may not.
            for opts in (dict(gamma=.9),dict(critic_min_v=-8.),dict(critic_max_v=8.),
                         dict(critic_num_bins=81),dict(reward_normalization='fixed',reward_scale=2.)):
                with self.assertRaises(ValueError):self.runner(**opts).load(p)
            with self.assertRaises(ValueError):self.runner(reward_normalization='episodic').load(p)

    def test_actor_transfer_allows_new_value_units_but_value_transfer_rejects(self):
        source=self.runner()
        for _ in range(2):source.observe(self.transition([1,2]))
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'s.pt';source.save(p);state=torch.load(p,weights_only=True)
            for opts in (dict(reward_normalization='episodic'),dict(reward_normalization='fixed',reward_scale=2.),
                         dict(gamma=.9),dict(critic_min_v=-8.),dict(critic_num_bins=51)):
                dest=self.runner(**opts)
                metadata,validated,_=prepare_actor_initialization(p,dest.contract)
                before_value={k:v.clone()for k,v in raw_module(dest.critic).state_dict().items()}
                initialize_actor_only(dest,validated)
                for k,v in state['networks']['actor']['model'].items():
                    torch.testing.assert_close(raw_module(dest.actor).state_dict()[k],v,rtol=0,atol=0)
                with self.assertRaises(ValueError):initialize_value_from_actor_source(dest,state)
                for k,v in before_value.items():torch.testing.assert_close(raw_module(dest.critic).state_dict()[k],v,rtol=0,atol=0)
                self.assertEqual(dest.update_step,0);self.assertEqual(len(dest.replay),0)
            with self.assertRaises(ValueError):prepare_actor_initialization(p,{'units':'wrong'})
            wrong_dims=self.runner()
            changed=dict(state,dimensions=[99,4,2,2])
            with self.assertRaises(ValueError):initialize_actor_only(wrong_dims,changed)

    def test_value_metadata_legacy_shape_and_explicit_new_modes(self):
        for mode in ('legacy','episodic','fixed'):
            source=self.runner(reward_normalization=mode)
            for _ in range(2):source.observe(self.transition([1,2]))
            with tempfile.TemporaryDirectory()as d:
                path=Path(d)/'source.pt';source.save(path)
                metadata,_,_=prepare_actor_initialization(path,source.contract,include_value=True)
                units=metadata['value_initialization']['reward_units']
                legacy_keys={'normalize_reward','normalized_G_max','gamma','n_step','critic_num_bins','critic_min_v','critic_max_v'}
                if mode=='legacy':self.assertEqual(set(units),legacy_keys)
                else:
                    self.assertEqual(set(units),legacy_keys|{'reward_normalization','reward_scale'})
                    self.assertEqual(units['reward_normalization'],mode)
                    self.assertEqual(units['reward_scale'],1.)

    @unittest.skipUnless(os.environ.get('MBDAC_HEAD_LEARNER_SOURCE'), 'Optional original HEAD helper fixture not supplied')
    def test_legacy_metadata_exact_head_helper(self):
        # A read-only source snapshot permits direct old/new API comparison.
        name='mbdac.flashsac._head_initialization_reference'
        spec=importlib.util.spec_from_file_location(name,os.environ['MBDAC_HEAD_LEARNER_SOURCE'])
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        source=self.runner()
        for _ in range(2):source.observe(self.transition([1,2]))
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'source.pt';source.save(path)
            expected,_,_=module.prepare_actor_initialization(path,source.contract,include_value=True)
            actual,_,_=prepare_actor_initialization(path,source.contract,include_value=True)
            self.assertEqual(actual,expected)

    def test_diagnostic_reward_units_no_rng_or_extra_forward(self):
        from unittest.mock import patch
        from mbdac.flashsac.learner import SMPFlashSACRunner
        for delay in (None,10):
            if delay is None:r=self.runner(reward_normalization='fixed',reward_scale=2.,critic_diagnostics=True)
            else:
                config=RunnerConfig(device='cpu',capacity=16,warmup=4,batch_size=4,use_amp=False,
                    use_compile=False,reward_normalization='fixed',reward_scale=2.,critic_diagnostics=True)
                r=SMPFlashSACRunner(3,4,2,2,config,{},actor_delay_updates=delay)
            for _ in range(2):r.observe(self.transition([1,2]))
            for _ in range(2):
                metrics=r.update()
                torch.testing.assert_close(metrics['reward/scale'],torch.tensor(2.),rtol=0,atol=0)
                if 'temperature/value' in metrics:alpha=metrics['temperature/value']
                else:alpha=raw_module(r.temperature).log_temp.detach().exp()
                torch.testing.assert_close(metrics['temperature/raw_unit_value'],2*alpha,rtol=0,atol=0)
            for initial in ({},{'temperature/value':torch.tensor([.003])}):
                rng=torch.get_rng_state().clone()
                with patch.object(raw_module(r.temperature),'forward',side_effect=AssertionError('extra forward')):
                    r._add_reward_unit_diagnostics(initial)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        base=self.runner()
        for _ in range(2):base.observe(self.transition([1,2]))
        result=base.update()
        self.assertNotIn('reward/scale',result)
        self.assertNotIn('temperature/raw_unit_value',result)

    def test_invalid_configuration(self):
        for kw in (dict(gamma=0),dict(gamma=1.1),dict(gamma=float('nan')),dict(critic_min_v=5),
                   dict(critic_max_v=float('inf')),dict(critic_num_bins=1),dict(critic_num_bins=3.2),
                   dict(reward_normalization='bad'),dict(reward_normalization='fixed',reward_scale=0),
                   dict(reward_scale=2),dict(critic_diagnostics=1)):
            with self.assertRaises(ValueError):RunnerConfig(**kw)


if __name__=='__main__':unittest.main()
