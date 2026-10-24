from types import SimpleNamespace
import pytest
import torch
from mbdac.modules.codecs import EuclideanCodec
from mbdac.modules.primitives import PhysicalSignals,PrimitiveSpec
from mbdac.modules.world_model import WorldModel


def test_contact_reward_routes_physics_and_reuses_heads_across_reward_scales():
    signals=PhysicalSignals()
    signals.register(PrimitiveSpec('contact',1,'probability','hand/ball',binary=True),measure=lambda e:e.contact)
    signals.register(PrimitiveSpec('relative_velocity',3,'m/s','world'),
                     measure=lambda e:pytest.fail('Do not supervise an available analytic quantity'),
                     analytic=lambda s,c,**_:s[...,:3]-s[...,3:6])
    required=('contact','relative_velocity')
    assert [spec.name for spec in signals.learned_specs(required)]==['contact']
    assert signals.analytic_names(required)==('relative_velocity',)
    env=SimpleNamespace(contact=torch.tensor([True,False])); labels=signals.capture(env,required)
    env.contact.zero_(); assert labels['primitive/contact'][0,0]==1
    state=torch.randn(2,6,requires_grad=True); context=torch.zeros(2,1)
    model=WorldModel(EuclideanCodec(6),1,1,hidden=(8,),primitive_specs=signals.learned_specs(required))
    physical=signals.resolve(state,context,model.primitives(state,context),required)
    def mdp_reward(physical,std):
        return physical['contact'].squeeze(-1)*torch.exp(-physical['relative_velocity'].square().sum(-1)/std**2)
    r1=mdp_reward(physical,.5); r2=3*mdp_reward(physical,2.)
    assert not torch.allclose(r1,r2)
    (r1+r2).sum().backward()
    assert torch.isfinite(state.grad).all() and state.grad.abs().sum()>0


def test_unknown_or_malformed_physics_is_rejected_instead_of_learning_reward_total():
    signals=PhysicalSignals()
    with pytest.raises(KeyError,match='Unknown physical dependency'): signals.learned_specs(('unknown',))
    with pytest.raises(ValueError,match='analytic function or a real measurement'):
        signals.register(PrimitiveSpec('force',3,'N','world'))
    signals.register(PrimitiveSpec('force',3,'N','world'),measure=lambda e:torch.zeros(2,2))
    with pytest.raises(ValueError,match='expected 3 physical values'): signals.capture(None,('force',))
