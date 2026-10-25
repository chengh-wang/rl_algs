"""Standalone FlashSAC CPU numerical/checkpoint/initialization regression tests."""
import copy
import os
import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
import torch
from mbdac.flashsac.config import RunnerConfig
from mbdac.flashsac.runner import FlashSACRunner, raw_module
from mbdac.flashsac.learner import SMPFlashSACRunner, initialize_actor_only, prepare_actor_initialization, checkpoint_sha, initialize_value_from_actor_source


def weights(net):
    return {k:v.detach().clone() for k,v in raw_module(net).state_dict().items()}


def equal(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)


class LearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def runner(self,delay=None):
        torch.manual_seed(12)
        cfg=RunnerConfig(device='cpu',capacity=16,warmup=4,batch_size=4,use_amp=False,use_compile=False)
        cls=FlashSACRunner if delay is None else SMPFlashSACRunner
        kw={} if delay is None else dict(actor_delay_updates=delay)
        return cls(6,7,2,2,cfg,dict(units='caller-v1',feature_schema={'channels':['a']},coordinates={'type':'affine'}),**kw)

    def fill(self,r):
        torch.manual_seed(30)
        for _ in range(2):
            r.observe(dict(observation=torch.randn(2,13),next_observation=torch.randn(2,13),
                action=torch.rand(2,2)*2-1,reward=torch.randn(2),terminated=torch.zeros(2,dtype=torch.bool),
                truncated=torch.zeros(2,dtype=torch.bool)))

    def test_delay_and_default_parity(self):
        a=self.runner(); b=self.runner(0);self.fill(a);self.fill(b)
        torch.manual_seed(40);ma=a.update()
        torch.manual_seed(40);mb=b.update()
        self.assertEqual(ma.keys(),mb.keys())
        for k in ma:torch.testing.assert_close(ma[k],mb[k],rtol=0,atol=0)
        for k in a.networks:self.assertTrue(equal(weights(a.networks[k]),weights(b.networks[k])))
        r=self.runner(2);self.fill(r); actor=weights(r.actor);temp=weights(r.temperature);critic=weights(r.critic)
        for _ in range(2):
            metric=r.update();self.assertNotIn('actor/loss',metric)
        self.assertTrue(equal(actor,weights(r.actor)));self.assertTrue(equal(temp,weights(r.temperature)))
        self.assertFalse(equal(critic,weights(r.critic)));self.assertEqual(r.update_step,2)
        self.assertIn('actor/loss',r.update());self.assertFalse(equal(actor,weights(r.actor)))
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'r.pt';r.save(p);fresh=self.runner(2);fresh.load(p)
            self.assertEqual(fresh.update_step,3)

    def test_initialization_isolation_and_sidecar(self):
        source=self.runner();self.fill(source);source.update()
        dest=self.runner(); before={k:weights(v)for k,v in dest.networks.items()if k!='actor'}
        norm=torch.nn.Module();norm.register_buffer('mean',torch.ones(3));norm.register_buffer('count',torch.zeros(3,dtype=torch.long))
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'source.pt';source.save(p)
            ns={'mean':torch.ones(3)*2,'count':torch.ones(3,dtype=torch.long)*5}
            torch.save(dict(learner_sha256=checkpoint_sha(p),contract=source.contract,
                wrapper=dict(schema=source.contract['feature_schema'],prior_normalizer=ns)),p.with_suffix('.smp.pt'))
            meta,state,normalizer=prepare_actor_initialization(p,dest.contract,include_normalizer=True)
            initialize_actor_only(dest,state,prior_normalizer=norm,normalizer_state=normalizer)
            self.assertTrue(equal(weights(source.actor),weights(dest.actor)))
            for k,v in before.items():self.assertTrue(equal(v,weights(dest.networks[k])))
            self.assertEqual(dest.update_step,0);self.assertEqual(dest.vector_steps,0);self.assertEqual(len(dest.replay),0)
            for net in dest.networks.values():
                if net.optimizer is not None:self.assertEqual(len(net.optimizer.state),0)
            self.assertTrue(torch.equal(norm.mean,ns['mean']));self.assertIn('normalizer_sidecar_sha256',meta)
            bad=copy.deepcopy(state);bad['dimensions'][0]+=1
            with self.assertRaises(ValueError):initialize_actor_only(dest,bad)
            badnorm={'mean':torch.ones(4),'count':torch.zeros(4,dtype=torch.long)}
            with self.assertRaises(ValueError):initialize_actor_only(dest,state,prior_normalizer=norm,normalizer_state=badnorm)
            with self.assertRaises(ValueError):prepare_actor_initialization(p,{**dest.contract,'units':'bad'})
            side=torch.load(p.with_suffix('.smp.pt'),weights_only=True);side['learner_sha256']='bad';torch.save(side,p.with_suffix('.smp.pt'))
            with self.assertRaises(ValueError):prepare_actor_initialization(p,dest.contract,include_normalizer=True)


    def test_matched_value_initialization(self):
        source=self.runner();self.fill(source);source.update()
        source.normalizer.G_r.fill_(7.)
        dest=self.runner()
        before_actor=weights(dest.actor);before_temp=weights(dest.temperature)
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'value.pt';source.save(p)
            metadata,state,_=prepare_actor_initialization(p,dest.contract,include_value=True)
            self.assertIn('value_initialization',metadata)
            self.assertEqual(metadata['behavior'],'actor_critic_target_and_reward_units_initialization_for_new_objective_not_resume')
            actor_metadata,_,_=prepare_actor_initialization(p,dest.contract)
            self.assertEqual(actor_metadata['behavior'],'actor_weights_and_batchnorm_only_fresh_other_learner_state')
            self.assertNotIn('value_initialization',actor_metadata)
            initialize_value_from_actor_source(dest,state)
            for name in ('critic','target'):
                self.assertTrue(equal(weights(getattr(source,name)),weights(getattr(dest,name))))
            self.assertTrue(torch.equal(dest.normalizer.G_r,torch.zeros(dest.num_envs)))
            self.assertTrue(torch.equal(dest.normalizer.G_r_max,source.normalizer.G_r_max))
            for name in ('mean','var','count'):
                self.assertTrue(torch.equal(getattr(dest.normalizer.G_rms,name),getattr(source.normalizer.G_rms,name)))
            self.assertTrue(equal(before_actor,weights(dest.actor)))
            self.assertTrue(equal(before_temp,weights(dest.temperature)))
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(dest.update_step,0);self.assertEqual(dest.vector_steps,0);self.assertEqual(dest.transitions,0)
            self.assertEqual(len(dest.replay),0)
            for net in dest.networks.values():
                if net.optimizer is not None:self.assertEqual(len(net.optimizer.state),0)
            bad=copy.deepcopy(state);bad['rms']['var']=torch.ones(5)
            with self.assertRaises(ValueError):initialize_value_from_actor_source(dest,bad)
            bad=copy.deepcopy(state);bad['networks']['critic']['model'][next(iter(bad['networks']['critic']['model']))]=torch.zeros(1)
            with self.assertRaises(ValueError):initialize_value_from_actor_source(dest,bad)
            old=dest.upstream_cfg.normalized_G_max;dest.upstream_cfg.normalized_G_max=3.
            with self.assertRaises(ValueError):initialize_value_from_actor_source(dest,state)
            dest.upstream_cfg.normalized_G_max=old
            bad=copy.deepcopy(state);bad['rms']['var'].fill_(float('nan'))
            with self.assertRaises(ValueError):initialize_value_from_actor_source(dest,bad)


    def test_replay_checkpoint_numerical_repeat(self):
        r=self.runner();self.fill(r);r.update()
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'resume.pt';r.save(p,include_replay=True)
            restored=self.runner();restored.load(p,restore_rng=True,continue_same_environments=True)
            self.assertEqual(len(restored.replay),len(r.replay))
            torch.manual_seed(918);first=r.update()
            torch.manual_seed(918);second=restored.update()
            self.assertEqual(first.keys(),second.keys())
            for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
            for name in r.networks:
                self.assertTrue(equal(weights(r.networks[name]),weights(restored.networks[name])))
            torch.testing.assert_close(r.normalizer.G_r,restored.normalizer.G_r,rtol=0,atol=0)
            # Legacy checkpoint configs did not contain optional calibration keys.
            old=torch.load(p,weights_only=True)
            for key in ('target_entropy','initial_temperature','target_sigma'):old['config'].pop(key)
            legacy=Path(d)/'legacy.pt';torch.save(old,legacy)
            self.runner().load(legacy)

    def test_config_calibration_and_source_independence(self):
        from dataclasses import replace
        from mbdac.vendor.flashsac.spaces import ShapeSpace
        c=RunnerConfig(device='cpu',target_entropy=-7.,initial_temperature=.003,target_sigma=.2)
        u=c.upstream(2)
        self.assertEqual(u.temp_target_entropy,-7.)
        self.assertEqual(u.temp_initial_value,.003)
        self.assertEqual(u.temp_target_sigma,.2)
        class Calibrated(RunnerConfig):
            def upstream(self,action_dim):
                return replace(super().upstream(action_dim),temp_target_entropy=-9.)
        self.assertEqual(Calibrated().upstream(2).temp_target_entropy,-9.)
        # A descriptor intentionally has no Gym behavior; replay only needs these.
        self.assertEqual(ShapeSpace((3,)).shape,(3,))


    @unittest.skipUnless(os.environ.get('FLASHSAC_PARITY_CHECKOUT'), 'Optional external source parity fixture not supplied')
    def test_pinned_source_exact_parity(self):
        package=Path(os.environ['FLASHSAC_PARITY_CHECKOUT'])/'src/rl/flashsac'
        spec=importlib.util.spec_from_file_location('_reference_flashsac',package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        from _reference_flashsac.config import RunnerConfig as OriginalConfig
        from _reference_flashsac.runner import FlashSACRunner as OriginalRunner
        kwargs=dict(device='cpu',capacity=16,warmup=4,batch_size=4,use_amp=False,use_compile=False)
        torch.manual_seed(12);source=OriginalRunner(6,7,2,2,OriginalConfig(**kwargs),{})
        torch.manual_seed(12);new=FlashSACRunner(6,7,2,2,RunnerConfig(**kwargs),{})
        self.fill(source);self.fill(new)
        for name in source.networks:self.assertTrue(equal(weights(source.networks[name]),weights(new.networks[name])))
        torch.manual_seed(880);a=source.batch()
        torch.manual_seed(880);b=new.batch()
        self.assertEqual(a.keys(),b.keys())
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        obs=torch.randn(2,6)
        for _ in range(3):
            torch.manual_seed(773);ma=source.update()
            torch.manual_seed(773);mb=new.update()
            self.assertEqual(ma.keys(),mb.keys())
            for key in ma:torch.testing.assert_close(ma[key],mb[key],rtol=0,atol=0)
            for name in source.networks:self.assertTrue(equal(weights(source.networks[name]),weights(new.networks[name])))
            torch.manual_seed(665);aa=source.actions(obs)
            torch.manual_seed(665);ab=new.actions(obs)
            torch.testing.assert_close(aa,ab,rtol=0,atol=0)
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/'source.pt';source.save(p,include_replay=True)
            fresh=FlashSACRunner(6,7,2,2,RunnerConfig(**kwargs),{})
            fresh.load(p,continue_same_environments=True,restore_rng=True)
            torch.manual_seed(993);ma=source.update()
            torch.manual_seed(993);mb=fresh.update()
            for key in ma:torch.testing.assert_close(ma[key],mb[key],rtol=0,atol=0)
            for name in source.networks:self.assertTrue(equal(weights(source.networks[name]),weights(fresh.networks[name])))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
