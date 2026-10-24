import math
import torch
from mbdac.utils.metrics import mean_metrics


def test_mean_metrics_handles_missing_keys_vectors_and_mixed_scalars():
    x=torch.tensor([1.,3.],requires_grad=True)
    actual=mean_metrics([{'a':x,'b':2.,'c':torch.tensor(4.)},{'a':torch.tensor(4.),'b':6.,'d':-2}])
    assert actual=={'a':3.,'b':4.,'c':4.,'d':-2.}
    assert all(type(v) is float for v in actual.values())
    assert x.grad is None


def test_mean_metrics_does_not_silently_drop_nonfinite_diagnostics():
    actual=mean_metrics([{'a':torch.tensor(float('nan')),'b':float('inf')}])
    assert math.isnan(actual['a']) and math.isinf(actual['b'])


def test_batched_collection_logging_preserves_real_data_and_normalizers():
    from dataclasses import asdict
    from mbdac.runners import ModelBasedRunner
    from test_core import ToyEnv,config
    def env():
        e=ToyEnv();step=e.step
        def wrapped(a):
            obs,r,d,extra=step(a)
            extra['log']={'vector':r.clone(),'scalar':r.mean(),'number':1.1}
            return obs,r,d,extra
        e.step=wrapped;return e
    torch.manual_seed(91)
    cfg={'algorithm':{'class_name':'MBDAC',**asdict(config())},'batch_scalar_logging':False}
    a=ModelBasedRunner(env(),cfg)
    torch.manual_seed(91);cfg['batch_scalar_logging']=True;b=ModelBasedRunner(env(),cfg)
    rng=torch.get_rng_state();old=a._collect(6)
    torch.set_rng_state(rng);new=b._collect(6)
    assert old==new and a.real_transitions==b.real_transitions
    for key,value in a.alg.world_model.state_dict().items():assert torch.equal(value,b.alg.world_model.state_dict()[key])
    rows=a.alg.replay._row(torch.arange(a.alg.replay.size))
    for key,value in a.alg.replay.data.items():assert torch.equal(value[rows],b.alg.replay.data[key][rows])


def test_scalar_snapshot_owns_mutable_environment_storage():
    from mbdac.utils.metrics import snapshot_metrics
    x=torch.tensor(2.,requires_grad=True)
    row=snapshot_metrics({'value':x,'number':3})
    with torch.no_grad():x.fill_(7.)
    assert mean_metrics([row])=={'value':2.,'number':3.}
