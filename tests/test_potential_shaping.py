"""Precision/adversarial-prior and boundary tests for potential shaping."""
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from mbdac.rewards import PotentialShapingConfig,PotentialRewardProvider,compose_potential_reward


def config(**kw):
    return PotentialShapingConfig('adversarial_style','frozen-v1',**kw)


def provider(**kw):
    return PotentialRewardProvider(config(**kw),next_potential_is_pre_reset=True)


def extras(current,nxt,term=None,trunc=None,version='frozen-v1'):
    n=len(current)
    return dict(terminated=torch.zeros(n,dtype=torch.bool) if term is None else term,
        truncated=torch.zeros(n,dtype=torch.bool) if trunc is None else trunc,
        reward_potential=dict(current=current,next=nxt,version=version))


def test_precision_optimum_and_adversarial_prior():
    a=torch.linspace(-1,1,20001,dtype=torch.float64)
    task=-a.square();style=(a+1)/2
    assert abs(float(a[(task+.4*style).argmax()])-.1)<1e-6
    positive_task=torch.exp(-a.square())
    assert abs(float(a[(positive_task*style).argmax()])-(math.sqrt(3)-1)/2)<1e-4
    c=config(gamma=.93,beta=50)
    start=torch.full_like(a,.2)
    first=compose_potential_reward(task,start,style,torch.zeros_like(a,dtype=torch.bool),c)
    last=compose_potential_reward(torch.zeros_like(a),style,style,torch.ones_like(a,dtype=torch.bool),c)
    total=first+c.gamma*last
    torch.testing.assert_close(total,task-c.beta*start,rtol=0,atol=3e-14)
    assert abs(float(a[total.argmax()]))<1e-12


def test_mdp_value_and_soft_policy_invariance():
    gamma=.83;beta=3.;phi=torch.tensor([.7,-.4,0.],dtype=torch.float64)
    # Two nonterminal states, two actions, and an absorbing terminal.
    successor=torch.tensor([[0,1],[2,0],[2,2]])
    reward=torch.tensor([[-.3,.2],[1.,-.1],[0.,0.]],dtype=torch.float64)
    terminal=successor==2
    shaped=reward+beta*(gamma*torch.where(terminal,0.,phi[successor])-phi[:,None])
    shaped[2]=0
    for alpha in (0.,.17):
        def solve(r):
            v=torch.zeros(3,dtype=torch.float64)
            for _ in range(1000):
                q=r+gamma*torch.where(terminal,0.,v[successor])
                v=q.max(-1).values if alpha==0 else alpha*torch.logsumexp(q/alpha,-1)
                v[2]=0
            return q,v
        q,v=solve(reward);qs,vs=solve(shaped)
        torch.testing.assert_close(vs,v-beta*phi,atol=1e-12,rtol=0)
        assert torch.equal(q[:2].argmax(-1),qs[:2].argmax(-1))
        if alpha:
            torch.testing.assert_close(torch.softmax(q[:2]/alpha,-1),torch.softmax(qs[:2]/alpha,-1),atol=1e-12,rtol=0)


def test_telescoping_invalid_history_timeout_and_terminal():
    c=config(gamma=.9,beta=2.)
    # Invalid history Phi=0 becomes valid; timeout retains endpoint tail.
    p=torch.tensor([0.,0.,.4,-.2,.8],dtype=torch.float64)
    total=torch.zeros(1,dtype=torch.float64)
    for t in range(4):
        f=compose_potential_reward(torch.zeros(1,dtype=torch.float64),p[t:t+1],p[t+1:t+2],torch.tensor([False]),c)
        total+=c.gamma**t*f
    torch.testing.assert_close(total,c.beta*(-p[:1]+c.gamma**4*p[-1:]),atol=1e-15,rtol=0)
    f=compose_potential_reward(torch.zeros(1,dtype=torch.float64),p[-1:],torch.tensor([1.],dtype=torch.float64),torch.tensor([True]),c)
    torch.testing.assert_close(total+c.gamma**4*f,-c.beta*p[:1],atol=1e-15,rtol=0)


def test_autograd_terminal_and_bounds():
    c=config(gamma=.8,beta=2.)
    task=torch.tensor([1.,2.],requires_grad=True)
    current=torch.tensor([.1,.2],requires_grad=True)
    nxt=torch.tensor([.3,.4],requires_grad=True)
    reward=compose_potential_reward(task,current,nxt,torch.tensor([True,False]),c)
    reward.sum().backward()
    assert torch.equal(task.grad,torch.ones(2))
    assert torch.equal(current.grad,torch.full((2,),-2.))
    torch.testing.assert_close(nxt.grad,torch.tensor([0.,1.6]))
    with pytest.raises(ValueError):compose_potential_reward(task,current,nxt+2,torch.tensor([False,False]),c)
    with pytest.raises(ValueError):compose_potential_reward(task,current,nxt,torch.zeros(2),c)
    for kw in (dict(gamma=1.1),dict(beta=-1),dict(max_abs_potential=0),dict(beta=float('nan'))):
        with pytest.raises(ValueError):config(**kw)


def test_provider_partial_resets_continuity_and_snapshot():
    p=provider()
    task=torch.tensor([1.,2.,3.])
    e=extras(torch.tensor([0.,.2,.3]),torch.tensor([.4,.5,.6]),
             term=torch.tensor([False,True,False]),trunc=torch.tensor([False,False,True]))
    result=p(task,e)
    torch.testing.assert_close(result,task+torch.tensor([.99*.4,-.2,.99*.6-.3]))
    assert torch.equal(e['task_reward'],task) and torch.equal(p.last_task_reward,task)
    snapshot=p.state_dict();snapshot['next_potential'][0]=-.9
    assert p.state_dict()['next_potential'][0]==.4
    e['reward_potential']['next'][0]=-.8;task[0]=99
    assert p.state_dict()['next_potential'][0]==.4 and p.last_task_reward[0]==1
    # Ended rows may start arbitrary new chains; continuing row must exactly match.
    p(torch.zeros(3),extras(torch.tensor([.4,-.9,.9]),torch.tensor([.2,.1,-.1])))
    state=copy.deepcopy(p.state_dict())
    with pytest.raises(ValueError,match='continuity'):
        p(torch.zeros(3),extras(torch.tensor([.20001,.1,-.1]),torch.zeros(3)))
    assert p.calls==2 and torch.equal(p.state_dict()['next_potential'],state['next_potential'])
    q=provider();q.load_state_dict(state);state['next_potential'].zero_()
    q(torch.zeros(3),extras(torch.tensor([.2,.1,-.1]),torch.zeros(3)))
    q.reset();assert q.calls==3 and q.state_dict()['next_potential'] is None
    q(torch.zeros(1),extras(torch.tensor([-.8]),torch.tensor([.8])))
    assert q.calls==4


def test_contract_version_flags_and_checkpoint_rejection():
    with pytest.raises(ValueError):PotentialRewardProvider(config(),next_potential_is_pre_reset=False)
    p=provider();assert p.contract['algorithm_discount']==.99
    assert p.contract['next_potential_is_pre_reset'] is True
    for bad in ({},dict(terminated=torch.tensor([False])),
                extras(torch.zeros(1),torch.zeros(1),version='new-version')):
        with pytest.raises(ValueError):p(torch.zeros(1),bad)
    s=p.state_dict();s['contract']['potential_version']='wrong'
    with pytest.raises(ValueError):p.load_state_dict(s)
    s=p.state_dict();s['calls']=-1
    with pytest.raises(ValueError):p.load_state_dict(s)


def test_no_sdk_imports():
    code='''
import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mjlab','mujoco','warp','isaaclab'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Deny())
from mbdac.rewards import PotentialRewardProvider,PotentialShapingConfig
p=PotentialRewardProvider(PotentialShapingConfig('test','v1'),next_potential_is_pre_reset=True)
assert p.calls==0
'''
    env=dict(os.environ,PYTHONPATH=os.environ.get('MBDAC_TEST_PACKAGE_PATH',str(Path(__file__).resolve().parents[1]/'src')))
    subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True)


def test_fixed_divisor_preserves_telescoping_changing_divisor_does_not():
    c=config(gamma=.9,beta=1.)
    # Equal task-return paths differ only in intermediate potential. Both end
    # at the same true terminal, with Phi(start)=0 and terminal Phi forced 0.
    middle=torch.tensor([0.,1.],dtype=torch.float64)
    zeros=torch.zeros(2,dtype=torch.float64)
    first=compose_potential_reward(torch.ones(2,dtype=torch.float64),zeros,middle,
                                   torch.zeros(2,dtype=torch.bool),c)
    last=compose_potential_reward(zeros,middle,zeros,torch.ones(2,dtype=torch.bool),c)
    task_return=torch.ones(2,dtype=torch.float64)
    fixed=first/2+c.gamma*(last/2)
    torch.testing.assert_close(fixed,task_return/2,rtol=0,atol=1e-15)
    # Dividing step 0 by 1 and step 1 by 2 leaves an action/path-dependent
    # gamma*Phi(middle)/2 residual despite the identical underlying task return.
    changing=first+c.gamma*(last/2)
    torch.testing.assert_close(changing,torch.tensor([1.,1.45],dtype=torch.float64),rtol=0,atol=1e-15)
    assert changing[0]!=changing[1]
