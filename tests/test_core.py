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
    def advance(self,c,state=None,context=None): return c
    def goal(self,c): return c
    def valid_start(self,c,h): return torch.ones(len(c),dtype=torch.bool)
    def reward(self,s,c,a,prev,primitives=None,previous_state=None): return -(s[:,0]-c[:,0]).square()-.01*a.square().sum(-1)
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
    policy=runner.get_inference_policy('cpu'); action=policy(runner.env.get_observations())
    assert not action.requires_grad
    assert runner.alg.actor.training
    assert all(torch.equal(v,runner.alg.actor.state_dict()[k]) for k,v in before.items())


def test_cpu_inference_accepts_gpu_environment_reset_mask():
    import pytest
    from dataclasses import asdict
    if not torch.cuda.is_available(): pytest.skip('Cross-device reset requires CUDA')
    runner=ModelBasedRunner(ToyEnv(),{'algorithm':{'class_name':'MBDAC',**asdict(config())}})
    for policy_device,mask_device in [('cpu','cuda'),('cuda','cpu')]:
        policy=runner.get_inference_policy(policy_device)
        obs=lambda value:TensorDict(dict(actor=torch.full((4,2),value,device=policy_device),
                                        goal=torch.ones(4,1,device=policy_device)),batch_size=[4])
        policy(obs(1.)); policy(obs(2.))
        policy.reset(torch.tensor([True,False,True,False],device=mask_device))
        action=policy(obs(3.))
        expected=torch.tensor([[[3.,3.],[3.,3.]],[[2.,2.],[3.,3.]],
                               [[3.,3.],[3.,3.]],[[2.,2.],[3.,3.]]],device=policy_device)
        assert torch.equal(policy.history,expected)
        assert action.device.type==policy_device and not action.requires_grad


def test_amp_update_only_changes_discriminator():
    env=ToyEnv(); cfg=config(); cfg.amp={'hidden':(16,)}; alg=MBDACAMP(env.model_task,4,cfg); fill(alg,env)
    actor=copy.deepcopy(alg.actor.state_dict()); wm=copy.deepcopy(alg.world_model.state_dict())
    logs=alg.update_prior(); assert logs['amp_loss']>=0
    assert all(torch.equal(v,alg.actor.state_dict()[k]) for k,v in actor.items())
    assert all(torch.equal(v,alg.world_model.state_dict()[k]) for k,v in wm.items())
    alg.update_actor_critic()


def test_amp_starts_after_wm_warmup_and_fits_prior_before_actor():
    env=ToyEnv(); cfg=config(); cfg.wm_only_iters=2; cfg.amp={'hidden':(16,)}
    alg=MBDACAMP(env.model_task,4,cfg); fill(alg,env)
    initial=copy.deepcopy(alg.prior.state_dict()); order=[]
    for name,opt in [('wm',alg.opt_wm),('prior',alg.opt_prior),('actor',alg.opt_actor),('critic',alg.opt_critic)]:
        original=opt.step
        def counted(*args,_name=name,_step=original,**kwargs):
            order.append(_name); return _step(*args,**kwargs)
        opt.step=counted
    alg.update(0); alg.update(1)
    assert order==['wm','wm']
    assert all(torch.equal(v,alg.prior.state_dict()[k]) for k,v in initial.items())
    order.clear(); alg.update(2)
    assert order==['wm','prior','actor','critic']
    assert alg.opt_prior.param_groups[0]['lr']==alg.learning_rate


def test_amp_style_reward_reaches_actor_through_frozen_dynamics_and_prior():
    class StyleTask(ToyTask):
        def reward(self,s,c,a,prev,primitives=None,previous_state=None): return torch.zeros_like(s[:,0])
        def termination_probability(self,s,c,primitives=None): return torch.zeros_like(s[:,0])
    env=ToyEnv(); env.model_task=StyleTask(); cfg=config(); cfg.amp={'hidden':()}
    alg=MBDACAMP(env.model_task,4,cfg); fill(alg,env); alg.update_world_model()
    with torch.no_grad():
        for p in alg.critic.parameters(): p.zero_()
        alg.prior.net[-1].weight.fill_(.1); alg.prior.net[-1].bias.zero_()
    before=copy.deepcopy(alg.prior.state_dict()); wm=copy.deepcopy(alg.world_model.state_dict())
    logs=alg.update_actor_critic()
    assert logs['actor_grad_norm']>0
    assert all(torch.equal(v,alg.prior.state_dict()[k]) for k,v in before.items())
    assert all(torch.equal(v,alg.world_model.state_dict()[k]) for k,v in wm.items())
    assert all(p.grad is None for p in alg.prior.parameters())


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


def test_restored_replay_is_cut_before_new_simulator_collection(tmp_path):
    from dataclasses import asdict
    cfg={'algorithm':{'class_name':'MBDAC',**asdict(config())},'log_interval':1}
    runner=ModelBasedRunner(ToyEnv(),cfg); runner.learn(1)
    runner.save(tmp_path/'model.pt',include_replay=True)
    resumed=ModelBasedRunner(ToyEnv(),cfg); resumed.load(tmp_path/'model.pt')
    replay=resumed.alg.replay
    last=replay._row(replay.size-1)
    assert replay.data['truncated'][last].all()
    resumed._collect(1)
    windows=replay.sample_windows(512,3)
    for i in range(1,3):
        assert not torch.any((windows['mask'][:,i]>0)&windows['truncated'][:,:i].any(1))


def test_history_honors_episode_boundary_with_a_stale_counter():
    replay=ReplayBuffer(8,1,'cpu')
    for index in range(6):
        replay.add(dict(s=torch.tensor([[float(index)]]),ep_step=torch.tensor([100+index]),
                        terminated=torch.tensor([False]),truncated=torch.tensor([index==3])))
    starts,history=replay.sample_starts(512,4)
    after=starts['s'][:,0]>=4
    assert after.any()
    assert (history['s'][after,:,0]>=4).all()


def test_physical_diagnostics_mask_resets_for_a_known_model():
    from mbdac.utils.diagnostics import world_model_diagnostics
    env=ToyEnv(); alg=MBDAC(env.model_task,4,config()); fill(alg,env)
    class ExactModel:
        def __call__(self,s,a,ctx): return s+torch.cat((a,.5*a),-1)*.1
        def primitives(self,s,ctx): return {'position':s[...,:1]}
        def transition_primitives(self,s,a,s1,ctx): return {'force':a}
    result=world_model_diagnostics(ExactModel(),env.model_task,alg.replay,window=3,batch=128)
    assert result['model/step_3_samples']>0
    assert result['model/one_step_state_rmse']==0 and result['model/step_3_state_rmse']<1e-6
    replay=ReplayBuffer(8,4,'cpu')
    for _ in range(6):
        _,_,_,extra=env.step(torch.ones(4,1)*.1); tr=extra['model_transition']
        replay.add({**tr,'primitive/position':tr['s1'][...,:1],'primitive/force':tr['a']})
    task=copy.deepcopy(env.model_task)
    task.primitive_specs=(PrimitiveSpec('position',1,'m','world'),PrimitiveSpec('force',1,'N','world',transition=True))
    result=world_model_diagnostics(ExactModel(),task,replay,window=3,batch=128)
    for kind in ('head_oracle','head_rollout'):
        for quantity in ('position','force'): assert result[f'model/{kind}/{quantity}_rmse']<1e-6


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


def test_transition_primitive_has_action_gradients_without_reward_supervision():
    from mbdac.modules.world_model import WorldModel
    model=WorldModel(EuclideanCodec(2),1,1,hidden=(8,),primitive_specs=[PrimitiveSpec('contact_force',1,'N','world',transition=True)])
    state=torch.randn(4,2); action=torch.randn(4,1,requires_grad=True); context=torch.zeros(4,1)
    nxt=model(state,action,context)
    assert model.primitives(nxt,context)=={}
    force=model.transition_primitives(state,action,nxt,context)['contact_force']
    cost1=force.square().sum(); cost2=3*force.square().sum()
    g1=torch.autograd.grad(cost1,action,retain_graph=True)[0]
    g2=torch.autograd.grad(cost2,action)[0]
    assert torch.isfinite(g1).all() and g1.abs().sum()>0
    assert torch.allclose(g2,3*g1,atol=1e-6)
    with torch.no_grad():
        features=model.transition_inputs(state,action,nxt,context)
        model.transition_heads.update_normalization({'contact_force':torch.randn(4,1)},features)
    loss=model.transition_heads.loss(features,{'contact_force':torch.randn(4,1)},torch.ones(4))
    assert torch.isfinite(loss)


def test_legacy_checkpoint_without_unused_transition_heads_loads_strictly():
    alg=MBDAC(ToyTask(),4,config())
    old={k:v for k,v in alg.world_model.state_dict().items() if not k.startswith('transition_heads.')}
    alg.world_model.load_state_dict(old,strict=True)


def test_exogenous_boundary_masks_world_model_windows():
    env=ToyEnv(); replay=ReplayBuffer(8,4,'cpu')
    for i in range(6):
        _,_,_,extras=env.step(torch.ones(4,1)*.1); tr=extras['model_transition']
        tr['model_boundary']=torch.full((4,),i==1,dtype=torch.bool); replay.add(tr)
    window=replay.sample_windows(256,4)
    assert window['model_boundary'].any()
    for i in range(1,4):
        assert not torch.any((window['mask'][:,i]>0)&window['model_boundary'][:,:i].any(1))


def test_random_episode_length_initialization_is_owned_by_adapter():
    from dataclasses import asdict
    class RandomizedEnv(ToyEnv):
        randomized=False
        def randomize_episode_lengths(self): self.randomized=True
    env=RandomizedEnv(); runner=ModelBasedRunner(env,{'algorithm':{'class_name':'MBDAC',**asdict(config())},'log_interval':1})
    runner.learn(1,init_at_random_ep_len=True)
    assert env.randomized
