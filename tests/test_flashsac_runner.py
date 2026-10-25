"""Real CPU FlashSAC updates through a simulator-free RSL four-return toy env."""
import copy
import torch
import pytest
from mbdac.runners.flash_sac_runner import FlashSACRunner, SMPFlashSACRunner
from mbdac.adapters.model_free import observation_groups, real_transition
from mbdac.flashsac.runner import raw_module


class ToyEnv:
    num_envs, num_actions, device = 4, 2, 'cpu'
    def __init__(self): self.reset()
    def reset(self):
        self.t = 0
        self.obs = {'policy':torch.zeros(4,3), 'privileged':torch.ones(4,5)}
        return self.obs, {}
    def get_observations(self): return self.obs
    def step(self, actions):
        self.t += 1
        terminal = {'policy':torch.full((4,3),float(self.t)),
                    'privileged':torch.full((4,5),float(self.t+10))}
        done = torch.zeros(4,dtype=torch.bool)
        done[0] = self.t % 2 == 0
        self.obs = {k:v.clone() for k,v in terminal.items()}
        for value in self.obs.values():value[done] = -100
        reward = -actions.square().sum(-1) + torch.arange(4).float()/4
        return self.obs, reward, done, {'terminated':torch.zeros_like(done),
            'truncated':done, 'terminal_observation':terminal, 'applied_action':actions.clamp(-.5,.5)}


def config(smp=False):
    cfg = dict(algorithm=dict(class_name='SMPFlashSAC' if smp else 'FlashSAC',
               capacity=64,warmup=4,batch_size=8,updates_per_step=1,
               use_compile=False,use_amp=False,schedule_updates=30),
               obs_groups={'actor':['policy'],'critic':['privileged']},
               task_contract={'test':'opaque_action_and_observation_schema'})
    if smp:cfg['smp']={'reward_source':'environment','actor_delay_updates':1}
    return cfg


def test_real_finite_updates_and_actor_critic_separation():
    torch.set_num_threads(1)
    runner=FlashSACRunner(ToyEnv(),config())
    initial={k:{n:v.clone() for n,v in raw_module(net).state_dict().items()}
             for k,net in runner.alg.networks.items()}
    metrics=runner.learn(6)
    assert metrics and all(torch.isfinite(v).all() for v in metrics.values())
    for key in ('actor','critic'):
        assert any(not torch.equal(v,initial[key][n]) for n,v in raw_module(runner.alg.networks[key]).state_dict().items())
    assert runner.alg.actor_dim==3 and runner.alg.critic_dim==5
    assert runner.real_transitions==24 and runner.current_learning_iteration==6
    assert not hasattr(runner.env,'model_task')


def test_terminal_snapshot_timeout_and_applied_action():
    env=ToyEnv();groups=config()['obs_groups']
    before=observation_groups(env.get_observations(),groups,4,'cpu')
    env.t=1
    action=torch.ones(4,2)
    obs,reward,done,extra=env.step(action)
    after=observation_groups(obs,groups,4,'cpu')
    tr=real_transition(before,after,action,reward,done,extra,groups,'cpu')
    assert torch.equal(tr['next_observation'][0,:3],torch.full((3,),2.))
    assert after['actor'][0,0]==-100
    assert tr['truncated'][0] and not tr['terminated'][0]
    assert torch.equal(tr['action'],torch.full((4,2),.5))
    with pytest.raises(ValueError,match='terminal_observation'):
        real_transition(before,after,action,reward,done,{},groups,'cpu')
    extra={'time_outs':done,'terminal_observation':extra['terminal_observation']}
    assert not real_transition(before,after,action,reward,done,extra,groups,'cpu')['terminated'].any()


def test_checkpoint_cold_resume_and_inference_isolated(tmp_path):
    torch.set_num_threads(1)
    runner=FlashSACRunner(ToyEnv(),config());runner.learn(4)
    path=tmp_path/'runner.pt';runner.save(path,infos={'test':1},include_replay=True)
    policy=runner.get_inference_policy()
    training=raw_module(runner.alg.actor).training
    out=policy(runner.env.get_observations())
    assert out.shape==(4,2) and (out.abs()<=1).all() and not out.requires_grad
    assert raw_module(runner.alg.actor).training==training
    restored=FlashSACRunner(ToyEnv(),config())
    assert restored.load(path)=={'test':1}
    assert restored.current_learning_iteration==4
    assert restored.last_load_info['simulator_state_restored'] is False
    assert len(restored.alg.replay)==len(runner.alg.replay)
    assert restored.env.t==0
    restored.learn(2)
    bad=config();bad['task_contract']['test']='changed'
    with pytest.raises(ValueError,match='contract'):
        FlashSACRunner(ToyEnv(),bad).load(path)


def test_smp_requires_explicit_composition_and_restores_persistent_state(tmp_path):
    torch.set_num_threads(1)
    with pytest.raises(ValueError,match='smp_contract'):
        SMPFlashSACRunner(ToyEnv(),config(True))
    class SMPEnv(ToyEnv):
        smp_contract={'features':'toy','reward':'already_composed'}
        persistent=3
        def smp_state_dict(self):return {'normalizer':self.persistent}
        def load_smp_state_dict(self,state):self.persistent=state['normalizer']
    env=SMPEnv();runner=SMPFlashSACRunner(env,config(True));runner.learn(3)
    path=tmp_path/'smp.pt';runner.save(path)
    env2=SMPEnv();env2.persistent=9
    restored=SMPFlashSACRunner(env2,config(True));restored.load(path)
    assert env2.persistent==3 and env2.t==0
    assert restored.contract['smp_reward_source']=='environment'


def test_provider_composes_rewards_and_uses_done_subset_snapshots(tmp_path):
    torch.set_num_threads(1)
    class Provider:
        contract={'layout':'toy_full_valid_windows','composition':'task_plus_prior'}
        def __init__(self): self.calls=0;self.resets=0;self.history=[]
        def __call__(self,rewards,extras):
            self.calls+=1
            self.history.append(1)
            return rewards+2
        def state_dict(self):return {'calls':self.calls,'history':list(self.history)}
        def load_state_dict(self,state):self.calls=state['calls'];self.history=list(state['history'])
        def reset(self):self.resets+=1;self.history=[]
    env=ToyEnv();env.smp_reward_provider=Provider()
    cfg=config(True);cfg['smp']['reward_source']='reward_provider'
    runner=SMPFlashSACRunner(env,cfg);runner.learn(2)
    assert env.smp_reward_provider.calls==2
    assert runner.alg.replay._rewards[:8].mean()>1
    path=tmp_path/'provider.pt';runner.save(path)
    restored_env=ToyEnv();restored_env.smp_reward_provider=Provider()
    restored=SMPFlashSACRunner(restored_env,cfg);restored.load(path)
    assert restored_env.smp_reward_provider.calls==2 and restored_env.smp_reward_provider.resets==1
    assert restored_env.smp_reward_provider.history==[]
    env=ToyEnv();before=observation_groups(env.obs,cfg['obs_groups'],4,'cpu');env.t=1
    action=torch.zeros(4,2);obs,r,d,e=env.step(action)
    e['terminal_observation']={k:v[d] for k,v in e['terminal_observation'].items()}
    tr=real_transition(before,observation_groups(obs,cfg['obs_groups'],4,'cpu'),action,r,d,e,cfg['obs_groups'],'cpu')
    assert tr['next_observation'][0,0]==2


def test_jit_export_normalized_actor_only(tmp_path):
    torch.set_num_threads(1)
    runner=FlashSACRunner(ToyEnv(),config())
    path=tmp_path/'actor.pt';runner.export_jit(path)
    loaded=torch.jit.load(str(path));x=torch.randn(4,3)
    assert torch.allclose(loaded(x),runner.get_inference_policy()(x))


def test_seed_logging_final_save_and_strict_delay(tmp_path):
    import json
    cfg=config();cfg['algorithm']['seed']=231
    first=FlashSACRunner(ToyEnv(),cfg,log_dir=tmp_path/'first')
    random_seed_weights={k:v.clone() for k,v in raw_module(first.alg.actor).state_dict().items()}
    second=FlashSACRunner(ToyEnv(),cfg)
    assert all(torch.equal(v,random_seed_weights[k]) for k,v in raw_module(second.alg.actor).state_dict().items())
    first.learn(3);first.close()
    assert (tmp_path/'first/config.json').exists()
    assert (tmp_path/'first/model_3.pt').exists()
    row=json.loads((tmp_path/'first/metrics.jsonl').read_text().splitlines()[-1])
    assert row['vector_steps']==3 and row['real_transitions']==12 and row['update_steps']>0
    assert row['wall_s']>=0 and row['time/update_wall_s']>=0
    cfg=config(True);cfg['smp']['actor_delay_updates']=1.5
    with pytest.raises(ValueError,match='integer'):
        SMPFlashSACRunner(ToyEnv(),cfg)


def test_no_simulator_sdk_imports_in_subprocess():
    import os
    from pathlib import Path
    import subprocess
    import sys
    script = r"""
import importlib.abc,sys,torch
class RejectSDK(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'mjlab','mujoco','warp','isaaclab','isaacgym'}:
            raise AssertionError('Simulator SDK imported: '+fullname)
sys.meta_path.insert(0,RejectSDK())
from mbdac.runners.flash_sac_runner import FlashSACRunner
class Env:
    num_envs=2;num_actions=2;device='cpu'
    def __init__(self): self.reset()
    def reset(self):self.obs={'actor':torch.zeros(2,3),'critic':torch.ones(2,4)}
    def get_observations(self):return self.obs
    def step(self,a):return self.obs,torch.ones(2),torch.zeros(2,dtype=torch.bool),{}
torch.set_num_threads(1)
r=FlashSACRunner(Env(),{'algorithm':{'class_name':'FlashSAC','capacity':8,'warmup':2,'batch_size':2,'use_compile':False,'use_amp':False,'updates_per_step':1}})
r.learn(2)
assert r.alg.update_step==2
print('noSDK real update OK')
"""
    result=subprocess.run([sys.executable,'-c',script],env=os.environ.copy(),capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stderr
    assert 'noSDK real update OK' in result.stdout


def test_compiled_bound_method_snapshot_does_not_reference_live_actor(tmp_path):
    torch.set_num_threads(1)
    runner=FlashSACRunner(ToyEnv(),config())
    live=raw_module(runner.alg.actor)
    # Exercise upstream's instance-method replacement, without costly inductor.
    live.get_mean_and_std=torch.compile(live.get_mean_and_std,backend='eager')
    x=torch.randn(4,3)
    with torch.no_grad(): expected=torch.tanh(live.get_mean_and_std(observations=x,training=False)[0]).clone()
    rng=torch.get_rng_state().clone()
    policy=runner.get_inference_policy('cpu')
    assert torch.equal(torch.get_rng_state(),rng)
    assert 'get_mean_and_std' not in vars(policy.actor)
    assert policy.actor.get_mean_and_std.__self__ is policy.actor
    path=tmp_path/'compiled_actor_snapshot.pt'
    runner.export_jit(path)
    exported=torch.jit.load(str(path))
    assert torch.allclose(policy(x),expected)
    assert torch.allclose(exported(x),expected)
    with torch.no_grad():
        for parameter in live.parameters():parameter.zero_()
    changed=torch.tanh(live.get_mean_and_std(observations=x,training=False)[0])
    assert not torch.allclose(changed,expected)
    assert torch.allclose(policy(x),expected)
    assert torch.allclose(exported(x),expected)


def test_reward_provider_requires_explicit_episode_reset_hook():
    class MissingReset:
        contract={'features':'toy'}
        def __call__(self,rewards,extras):return rewards
        def state_dict(self):return {'history':torch.ones(2,3)}
        def load_state_dict(self,state):self.history=state['history']
    env=ToyEnv();env.smp_reward_provider=MissingReset()
    cfg=config(True);cfg['smp']['reward_source']='reward_provider'
    with pytest.raises(ValueError,match='reset'):
        SMPFlashSACRunner(env,cfg)
