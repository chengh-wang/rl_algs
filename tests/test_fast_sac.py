"""Source parity and model-gradient contracts, with no simulator SDK imports."""
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import torch
from mbdac.algorithms import MBDACFastSAC,MBDACFastSACConfig
from mbdac.algorithms.mbdac import frozen
from mbdac.modules.holosoma import DifferentiableNormalization
from mbdac.vendor.holosoma.normalization import EmpiricalNormalization
from mbdac.runners import ModelBasedRunner
from test_core import ToyEnv,fill


def config(horizon=2):
    return MBDACFastSACConfig(history=2,horizon=horizon,wm_window=2,wm_batches=1,
        wm_batch=8,pi_batch=8,buffer_steps=8,warmup_steps=4,wm_only_iters=0,
        wm_hidden=(16,),precision='fp32',fast_sac=dict(actor_hidden_dim=16,
            critic_hidden_dim=16,num_atoms=21,critic_batch=8,critic_updates=2))


def test_vendored_actor_critic_are_exact_pinned_source():
    from mbdac.vendor.holosoma import fast_sac
    path=Path(fast_sac.__file__)
    manifest=json.loads(path.with_name('source.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['files']['fast_sac.py']['sha256']


def test_source_normalization_statistics_match_but_input_gradient_survives():
    source=EmpiricalNormalization((3,),'cpu'); adapted=DifferentiableNormalization(3)
    for n in [7,11,3]:
        x=torch.randn(n,3)
        source(x,update=True);adapted.update(x)
    x=torch.randn(5,3,requires_grad=True)
    assert torch.equal(source(x,update=False),adapted(x))
    assert all(torch.equal(v,adapted.state_dict()[k]) for k,v in source.state_dict().items())
    adapted(x).sum().backward()
    assert torch.allclose(x.grad,(1/(source._std+source.eps)).expand_as(x))


def test_horizon_zero_loss_and_all_actor_gradients_match_holosoma():
    torch.manual_seed(21);env=ToyEnv();alg=MBDACFastSAC(env.model_task,4,config(0));fill(alg,env)
    start,h=alg._sample_starts();hist=alg._history(h)
    rng=torch.get_rng_state()
    with frozen(alg.world_model,alg.critic):
        objective,_=alg.actor_objective(start,hist)
        expected_inputs=alg.actor.inputs(hist,alg.task.goal(start['condition']))
        torch.set_rng_state(rng)
        actions,logp=alg.actor.net.get_actions_and_log_probs(expected_inputs)
        x=alg._critic_features(start['s'],start['condition'],start['context'])
        q=alg.critic.net.get_value(torch.softmax(alg.critic.net(alg.critic.norm(x),actions),-1)).mean(0)
        expected=(alg.log_alpha.exp().detach()*logp-q).mean()
        assert torch.equal(-objective,expected)
        actual_grad=torch.autograd.grad(-objective,tuple(alg.actor.parameters()),retain_graph=True)
        expected_grad=torch.autograd.grad(expected,tuple(alg.actor.parameters()))
        for actual,wanted in zip(actual_grad,expected_grad): assert torch.equal(actual,wanted)


def test_actor_uses_model_gradients_without_training_wm_q_or_temperature():
    torch.manual_seed(22);env=ToyEnv();alg=MBDACFastSAC(env.model_task,4,config());fill(alg,env)
    alg.update_world_model()
    before={name:copy.deepcopy(getattr(alg,name).state_dict()) for name in ['actor','critic','world_model','target_critic']}
    alpha=alg.log_alpha.detach().clone();seen=[]
    def inputs(module,args,output):
        if torch.is_grad_enabled():
            output.register_hook(lambda grad:seen.append(float(grad.abs().sum())))
    handle=alg.world_model.net.register_forward_hook(inputs)
    logs=alg.update_actor();handle.remove()
    assert logs['actor_grad_norm']>0 and any(g>0 for g in seen)
    for name in ['critic','world_model','target_critic']:
        assert all(torch.equal(v,getattr(alg,name).state_dict()[k]) for k,v in before[name].items())
        assert all(p.grad is None for p in getattr(alg,name).parameters())
    assert torch.equal(alpha,alg.log_alpha) and alg.log_alpha.grad is None
    assert any(not torch.equal(v,alg.actor.state_dict()[k]) for k,v in before['actor'].items())


def test_real_c51_update_matches_source_projection_and_terminal_bootstrap():
    torch.manual_seed(23);env=ToyEnv();alg=MBDACFastSAC(env.model_task,4,config());fill(alg,env)
    batch=alg._real_batch(8)
    alg._real_batch=lambda _:batch
    tr,_,next_hist,c1=batch
    tr['terminated'][:4]=True;tr['terminated'][4:]=False;tr['truncated'][4:]=True
    next_x=alg._critic_features(tr['s1'],c1,tr['context'])
    x=alg._critic_features(tr['s'],tr['condition'],tr['context'])
    reference=copy.deepcopy(alg.critic)
    optimizer=torch.optim.AdamW(reference.parameters(),lr=alg.cfg.fast_sac['critic_lr'],betas=(.9,.95),weight_decay=alg.cfg.fast_sac['weight_decay'])
    rng=torch.get_rng_state()
    with torch.no_grad():
        a1,logp=alg.actor.sample(next_hist,alg.task.goal(c1))
        boot=(~tr['terminated']).float()
        targets=alg.target_critic.projection(next_x,a1,tr['reward']-alg.cfg.gamma*boot*alg.log_alpha.exp()*logp,boot,alg.cfg.gamma)
        assert torch.allclose(targets.sum(-1),torch.ones_like(targets[...,0]),atol=1e-6)
    reference_loss=-(targets*torch.log_softmax(reference(x,tr['a']),-1)).sum(-1).mean(1).sum()
    reference_loss.backward();optimizer.step()
    torch.set_rng_state(rng);logs=alg.update_real_critic()
    assert abs(logs['critic_loss']-float(reference_loss.detach()))<1e-6
    for actual,wanted in zip(alg.critic.parameters(),reference.parameters()): assert torch.allclose(actual,wanted,atol=1e-7)


def test_schedule_and_full_runner_resume_export(tmp_path):
    cfg=config();cfg.wm_only_iters=1
    options={'algorithm':{'class_name':'MBDACFastSAC',**asdict(cfg)},'log_interval':1}
    runner=ModelBasedRunner(ToyEnv(),options)
    runner.learn(1);assert runner.alg.critic_updates==0 and runner.alg.actor_updates==0
    runner.learn(1);assert runner.alg.critic_updates==2 and runner.alg.actor_updates==1
    runner.save(tmp_path/'model.pt',include_replay=True)
    resumed=ModelBasedRunner(ToyEnv(),options);resumed.load(tmp_path/'model.pt')
    assert resumed.alg.critic_updates==2 and resumed.alg.actor_updates==1
    assert torch.equal(resumed.alg.log_alpha,runner.alg.log_alpha)
    for key,val in runner.alg.target_critic.state_dict().items():
        assert torch.equal(resumed.alg.target_critic.state_dict()[key],val)
    resumed.learn(1);assert resumed.current_learning_iteration==3 and resumed.alg.critic_updates==4
    exported=resumed.export_jit(tmp_path/'policy.pt')
    history=torch.randn(5,2,2);goal=torch.ones(5,1)
    assert torch.allclose(exported(history,goal),resumed.alg.actor(history,goal),atol=1e-6)
