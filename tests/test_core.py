import copy
import torch
from tensordict import TensorDict
from mbdac.algorithms import MBDAC,MBDACAMP,MBDACConfig
from mbdac.algorithms.mbdac import lambda_returns,frozen
from mbdac.modules.codecs import EuclideanCodec,ArticulatedCodec
from mbdac.modules.primitives import PrimitiveHeads,PrimitiveSpec
from mbdac.storage.replay import ReplayBuffer
from mbdac.runners import ModelBasedRunner


class ToyTask:
    codec=EuclideanCodec(2)
    action_dim=1; context_dim=1; obs_dim=2; goal_dim=1; critic_dim=4; amp_feature_dim=2
    def normalize_context(self,x): return x
    def observation(self,s,a,ctx,c,primitives=None): return s
    def add_noise(self,o): return o
    def advance(self,c): return c
    def goal(self,c): return c
    def valid_start(self,c,h): return torch.ones(len(c),dtype=torch.bool)
    def reward(self,s,c,a,prev,primitives=None): return -(s[:,0]-c[:,0]).square()-.01*a.square().sum(-1)
    def termination_probability(self,s,c,primitives=None): return torch.sigmoid((s[:,0].abs()-5)*10)
    def critic_features(self,s,c,ctx,primitives=None): return torch.cat((s,c,ctx),-1)
    def amp_features(self,s): return s
    def sample_expert_features(self,n): return torch.ones(n,2),torch.ones(n,2)


class ToyEnv:
    num_envs=4; num_actions=1; device='cpu'; model_task=ToyTask()
    def __init__(self):
        self.state=torch.zeros(4,2); self.count=torch.zeros(4,dtype=torch.long); self.last=torch.zeros(4,1)
    def get_observations(self): return TensorDict({'actor':self.state.clone(),'goal':torch.ones(4,1)},batch_size=[4])
    def step(self,a):
        s=self.state.clone(); s1=s+torch.cat((a,.5*a),-1)*.1; done=self.count==3
        tr=dict(s=s,s1=s1.clone(),a=a.clone(),a_prev=self.last.clone(),context=torch.zeros(4,1),condition=torch.ones(4,1),ep_step=self.count.clone(),terminated=done,truncated=torch.zeros_like(done))
        self.state=s1; self.state[done]=0; self.count+=1; self.count[done]=0; self.last=a.clone(); self.last[done]=0
        return self.get_observations(),-s1.square().sum(-1),done.long(),{'model_transition':tr}
    def close(self): pass


def config(): return MBDACConfig(history=2,horizon=3,wm_window=2,wm_batches=1,wm_batch=8,pi_batch=8,buffer_steps=8,warmup_steps=4,wm_only_iters=0,actor_hidden=(16,),wm_hidden=(16,),critic_hidden=(16,),critic_epochs=1,critic_minibatches=1,precision='fp32')


def fill(alg,env):
    for _ in range(6):
        a=alg.act(env.get_observations()); obs,r,d,e=env.step(a); alg.process_env_step(obs,r,d,e)


def test_lambda_return_terminal_and_bootstrap():
    out=lambda_returns(torch.tensor([[1.],[2.]]),torch.tensor([[0.],[1.]]),torch.tensor([[3.],[4.],[100.]]),.9,.95)
    assert torch.allclose(out,torch.tensor([[1+.9*(.05*4+.95*2)],[2.]]))


def test_actor_gradient_crosses_frozen_model_without_updating_it():
    torch.manual_seed(1); env=ToyEnv(); alg=MBDAC(env.model_task,4,config()); fill(alg,env)
    alg.update_world_model(); before=copy.deepcopy(alg.world_model.state_dict()); actor=copy.deepcopy(alg.actor.state_dict())
    logs=alg.update_actor_critic()
    assert torch.isfinite(torch.tensor(list(logs.values()))).all()
    assert all(torch.equal(v,alg.world_model.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in alg.world_model.parameters())
    assert any(not torch.equal(v,alg.actor.state_dict()[k]) for k,v in actor.items())


def test_replay_windows_do_not_train_across_reset():
    env=ToyEnv(); alg=MBDAC(env.model_task,4,config()); fill(alg,env)
    w=alg.replay.sample_windows(256,4)
    for i in range(1,4): assert not torch.any((w['mask'][:,i]>0)&w['terminated'][:,:i].any(1))
    s,h=alg.replay.sample_starts(128,4)
    # Reconstructed historical transitions must all be from the same episode.
    assert torch.all(h['ep_step'][:,-1]==s['ep_step'])


def test_checkpoint_resume_empty_replay_collects_before_updates(tmp_path):
    from dataclasses import asdict
    cfg={'algorithm':{'class_name':'MBDAC',**asdict(config())},'log_interval':1}
    runner=ModelBasedRunner(ToyEnv(),cfg); runner.learn(1); runner.save(tmp_path/'model.pt')
    resumed=ModelBasedRunner(ToyEnv(),cfg); resumed.load(tmp_path/'model.pt'); assert resumed.alg.replay.size==0
    resumed.learn(1); assert resumed.current_learning_iteration==2 and resumed.alg.replay.size>=2


def test_inference_does_not_change_training_mode_or_statistics():
    from dataclasses import asdict
    runner=ModelBasedRunner(ToyEnv(),{'algorithm':{'class_name':'MBDAC',**asdict(config())}})
    runner.alg.train_mode(); before=copy.deepcopy(runner.alg.actor.state_dict())
    policy=runner.get_inference_policy('cpu'); policy(runner.env.get_observations())
    assert runner.alg.actor.training
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in before.items())


def test_amp_update_only_changes_discriminator():
    env=ToyEnv(); cfg=config(); cfg.amp={'hidden':(16,)}; alg=MBDACAMP(env.model_task,4,cfg); fill(alg,env)
    actor=copy.deepcopy(alg.actor.state_dict()); wm=copy.deepcopy(alg.world_model.state_dict())
    logs=alg.update_prior(); assert logs['amp_loss']>=0
    assert all(torch.equal(v,alg.actor.state_dict()[k]) for k,v in actor.items())
    assert all(torch.equal(v,alg.world_model.state_dict()[k]) for k,v in wm.items())
    alg.update_actor_critic()


def test_articulated_codec_roundtrip_and_finite_gradients():
    codec=ArticulatedCodec(3); state=torch.zeros(2,19); state[:,3]=1; state[:,2]=.8
    delta=torch.randn(2,18)*.01; delta.requires_grad_(True)
    s1=codec.integrate(state,delta); recovered=codec.targets(state,s1)
    assert torch.allclose(delta,recovered,atol=2e-5)
    recovered.square().sum().backward(); assert torch.isfinite(delta.grad).all()


def test_primitives_can_be_reused_with_different_reward_scales():
    head=PrimitiveHeads(3,[PrimitiveSpec('velocity',2,'m/s','body')],hidden=(8,))
    x=torch.randn(4,3,requires_grad=True); v=head(x)['velocity']; target=torch.ones_like(v)
    r1=torch.exp(-(v-target).square().sum(-1)/.5**2)
    r2=3*torch.exp(-(v-target).square().sum(-1)/1.5**2)
    assert not torch.allclose(r1,r2)
    (r1+r2).sum().backward(); assert x.grad is not None and torch.isfinite(x.grad).all()


def test_duplicate_state_primitive_is_rejected():
    import pytest
    task=ToyTask(); task.analytic_primitives=('velocity',)
    task.primitive_specs=(PrimitiveSpec('velocity',2,'m/s','body'),)
    with pytest.raises(ValueError,match='already available'): MBDAC(task,4,config())


def test_export_uses_explicit_history_and_matches_inference(tmp_path):
    from dataclasses import asdict
    runner=ModelBasedRunner(ToyEnv(),{'algorithm':{'class_name':'MBDAC',**asdict(config())}})
    exported=runner.export_jit(tmp_path/'policy.pt'); restored=torch.jit.load(str(tmp_path/'policy.pt'))
    history=torch.randn(5,2,2); goal=torch.randn(5,1)
    with torch.no_grad():
        expected=runner.alg.actor(history,goal)
        assert torch.allclose(exported(history,goal),expected,atol=1e-6)
        assert torch.allclose(restored(history,goal),expected,atol=1e-6)
    assert runner.alg.actor.training


def test_periodic_checkpoint_records_active_training_time(tmp_path):
    from dataclasses import asdict
    cfg={'algorithm':{'class_name':'MBDAC',**asdict(config())},'log_interval':1,'save_interval':1}
    runner=ModelBasedRunner(ToyEnv(),cfg,log_dir=tmp_path); runner.learn(2); runner.close()
    first=torch.load(tmp_path/'model_1.pt',weights_only=False)
    second=torch.load(tmp_path/'model_2.pt',weights_only=False)
    assert 0<first['elapsed_s']<second['elapsed_s']<=runner.elapsed_s


def test_actor_only_warm_start_keeps_world_model_and_iteration(tmp_path):
    from dataclasses import asdict
    cfg={'algorithm':{'class_name':'MBDAC',**asdict(config())},'log_interval':1}
    source=ModelBasedRunner(ToyEnv(),cfg); source.learn(1); source.save(tmp_path/'model.pt')
    target=ModelBasedRunner(ToyEnv(),cfg); before=copy.deepcopy(target.alg.world_model.state_dict())
    target.load(tmp_path/'model.pt',load_cfg={'actor':True})
    assert target.current_learning_iteration==0 and target.elapsed_s==0
    assert all(torch.equal(v,target.alg.world_model.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,target.alg.actor.state_dict()[k]) for k,v in source.alg.actor.state_dict().items())


def test_loading_without_saved_replay_clears_existing_replay():
    env=ToyEnv(); alg=MBDAC(env.model_task,4,config()); fill(alg,env)
    checkpoint=copy.deepcopy(alg.save()); assert alg.replay.size>0
    alg.load(checkpoint); assert alg.replay.size==0


def test_physical_diagnostics_mask_resets_for_a_known_model():
    from mbdac.utils.diagnostics import world_model_diagnostics
    env=ToyEnv(); alg=MBDAC(env.model_task,4,config()); fill(alg,env)
    class ExactModel:
        def __call__(self,s,a,ctx): return s+torch.cat((a,.5*a),-1)*.1
    result=world_model_diagnostics(ExactModel(),env.model_task,alg.replay,window=3,batch=128)
    assert result['model/step_3_samples']>0
    assert result['model/one_step_state_rmse']==0 and result['model/step_3_state_rmse']<1e-6


def test_optimizer_schedule_matches_reference_warmup_and_10_1_20_ratio():
    env=ToyEnv(); cfg=config(); cfg.wm_batches=10; cfg.wm_only_iters=2
    cfg.critic_epochs=5; cfg.critic_minibatches=4
    alg=MBDAC(env.model_task,4,cfg); fill(alg,env)
    calls=dict(wm=0,actor=0,critic=0); order=[]
    for name,opt in [('wm',alg.opt_wm),('actor',alg.opt_actor),('critic',alg.opt_critic)]:
        original=opt.step
        def counted(*args,_name=name,_step=original,**kwargs):
            calls[_name]+=1; order.append(_name); return _step(*args,**kwargs)
        opt.step=counted
    alg.update(0); alg.update(1)
    assert calls==dict(wm=20,actor=0,critic=0)
    order.clear(); alg.update(2)
    assert calls==dict(wm=30,actor=1,critic=20)
    assert order==['wm']*10+['actor']+['critic']*20
