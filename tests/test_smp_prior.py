"""SMP numerical contracts without simulator imports or robot dimensions."""
import ast
from dataclasses import asdict
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from mbdac.smp import (MotionFeatureSchema,MotionFeatureBuffer,FrozenSMPPrior,
    DiffusionDenoiser,DDPMScheduler,decode_feature_windows,sample_feature_windows)
from mbdac.smp.math import matrix_from_quat,quat_apply,quat_mul,quat_from_matrix
from mbdac.smp.pretrain import build_motion_windows,diffusion_loss,fit_quantiles


def prior(f=22,w=4):
    return FrozenSMPPrior(DiffusionDenoiser(f,w,16,2,1),DDPMScheduler(50),-torch.ones(f),torch.ones(f))


def test_no_simulator_sdk_imports():
    code='''
import importlib.abc,sys,torch
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'mjlab','mujoco','warp','isaaclab'}:
   raise AssertionError('Simulator dependency: '+fullname)
sys.meta_path.insert(0,Deny())
from mbdac.smp import *
from mbdac.smp.pretrain import diffusion_loss
p=FrozenSMPPrior(DiffusionDenoiser(18,2,16,2,1),DDPMScheduler(),-torch.ones(18),torch.ones(18))
p.score(torch.zeros(1,2,18),update_normalizer=False)
assert not any(n.split('.')[0] in {'mjlab','mujoco','warp','isaaclab'} for n in sys.modules)
'''
    env=dict(os.environ,PYTHONPATH=os.environ.get('MBDAC_TEST_PACKAGE_PATH',str(Path(__file__).resolve().parents[1]/'src')))
    subprocess.run([sys.executable,'-c',code],env=env,check=True,capture_output=True,text=True)


def test_checkpoint_numpy_path_ema_strict_and_normalizer_roundtrip(tmp_path):
    torch.manual_seed(8)
    p=prior()
    schema=MotionFeatureSchema(('a',),('tip0','tip1'),.017)
    model=p.model.state_dict()
    bad={k:torch.zeros_like(v) for k,v in model.items()}
    path=tmp_path/'prior.pt'
    torch.save(dict(cfg=dict(feature_dim=22,window_size=4,d_model=16,nhead=2,num_layers=1,path=tmp_path),
                    model=bad,model_ema=model,q_low=np.full(22,-1,np.float32),q_high=np.ones(22,np.float32),
                    feature_schema=asdict(schema)),path)
    loaded=FrozenSMPPrior.load_checkpoint(path,expected_feature_dim=22,expected_window_size=4)
    assert loaded.feature_schema==asdict(schema)
    assert all(torch.equal(v,loaded.model.state_dict()[k]) for k,v in model.items())
    with pytest.raises(ValueError):
        FrozenSMPPrior.load_checkpoint(path,expected_feature_dim=23)
    x=torch.randn(3,4,22);noise=torch.randn(3,3,4,22)
    first=loaded.score(x,noise=noise)
    assert loaded.normalizer.count[[8,15,22]].tolist()==[3,3,3]
    q=prior();q.load_state_dict(loaded.state_dict())
    assert torch.equal(first.reward,q.score(x,noise=noise,update_normalizer=False).reward)
    before={k:v.clone() for k,v in loaded.normalizer.state_dict().items()}
    loaded.score(x,noise=noise,update_normalizer=False)
    assert all(torch.equal(before[k],v) for k,v in loaded.normalizer.state_dict().items())
    assert not first.reward.requires_grad
    loaded.train();assert not loaded.model.training
    assert all(not p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError):loaded.score(x,fixed_timesteps=())
    with pytest.raises(ValueError):prior().score(x,ws=float('nan'))
    with pytest.raises(ValueError):FrozenSMPPrior(p.model,p.scheduler,torch.zeros(22),torch.zeros(22))
    broken=dict(model);broken.pop(next(iter(broken)))
    torch.save(dict(cfg=loaded.checkpoint_cfg,model=broken,q_low=np.zeros(22,np.float32),q_high=np.ones(22,np.float32)),path)
    with pytest.raises(RuntimeError):FrozenSMPPrior.load_checkpoint(path)


def test_fixed_noise_reward_reference_and_no_clipping():
    torch.manual_seed(12);p=prior();x=torch.randn(4,4,22)*7;noise=torch.randn(3,4,4,22)
    # The quantile range is [-1,1], hence x_0=x (up to upstream epsilon).
    normalized=2*(x-p.q_low)/(p.q_high-p.q_low+1e-8)-1
    errors=[]
    for i,t in enumerate((8,15,22)):
        ts=torch.full((4,),t,dtype=torch.long)
        errors.append(((p.model(p.scheduler.add_noise(normalized,noise[i],ts),ts)-noise[i])**2).mean((-1,-2)))
    expected=torch.stack(errors).mean(0)
    score=p.score(x,noise=noise,normalize=False,update_normalizer=False)
    torch.testing.assert_close(score.raw_error,expected,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(score.reward,torch.exp(-4*expected),rtol=1e-6,atol=1e-6)
    assert p.normalizer.count.sum()==0
    clipped=p.score(x.clamp(-1,1),noise=noise,normalize=False,update_normalizer=False)
    assert not torch.allclose(score.raw_error,clipped.raw_error)


def test_dynamic_schema_decode_feature_roundtrip_and_partial_reset():
    torch.manual_seed(29)
    schema=MotionFeatureSchema(('shoulder','finger','knee'),('tip','heel'),.017,root_body_name='root')
    b=MotionFeatureBuffer(2,5,3,2,'cpu')
    fields=('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','joint_pos','joint_vel','ee_pos_w')
    for name in fields:
        value=torch.randn_like(getattr(b,name))
        if name=='root_quat_w':value/=value.norm(dim=-1,keepdim=True)
        getattr(b,name).copy_(value)
    f=b.compute_features();origins=torch.tensor([[3.,-4.,2.],[-2.,7.,-1.]])
    decoded=decode_feature_windows(f,schema,env_origins=origins,chosen_yaw=torch.tensor([.7,-1.2]))
    assert decoded.joint_names==schema.joint_names
    recovered=MotionFeatureBuffer(2,5,3,2,'cpu')
    for name,value in decoded.buffer_frames.items():getattr(recovered,name).copy_(value)
    torch.testing.assert_close(recovered.compute_features(),f,rtol=2e-5,atol=3e-6)
    assert torch.equal(decoded.root_pos_env[...,2],f[...,2])
    torch.testing.assert_close(decoded.root_pos_w,decoded.root_pos_env+origins[:,None])
    torch.testing.assert_close(decoded.last_root_state[:,:3],decoded.root_pos_w[:,-1])
    expected=(f[:,1:,9:12]-f[:,:-1,9:12])/.017
    assert torch.equal(decoded.joint_vel[:,:-1],expected)
    assert torch.equal(decoded.joint_vel[:,-1],expected[:,-1])
    other=recovered.root_pos_w[1].clone()
    recovered.reset_current(torch.tensor([0]),**{name:getattr(b,name)[:,-1] for name in fields})
    assert torch.equal(recovered.root_pos_w[1],other)
    assert torch.equal(recovered.root_pos_w[0],b.root_pos_w[0,-1].expand(5,3))
    # Independent analytic z-axis quarter turn and col0+col2 convention.
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
    torch.testing.assert_close(quat_apply(q,torch.tensor([[1.,0.,0.]])),torch.tensor([[0.,1.,0.]]),atol=2e-7,rtol=0)
    mat=matrix_from_quat(q)
    torch.testing.assert_close(matrix_from_quat(quat_from_matrix(mat)),mat,atol=2e-7,rtol=0)


def test_ddpm_generation_rng_and_reference_loop():
    torch.manual_seed(71);p=prior();state=torch.get_rng_state()
    x=torch.randn(2,p.window_size,p.feature_dim)
    for t in reversed(range(50)):
        ts=torch.full((2,),t,dtype=torch.long)
        x=p.scheduler.step(p.model(x,ts),x,t)
    expected=(x+1)/2*(p.q_high-p.q_low)+p.q_low
    torch.set_rng_state(state);actual=sample_feature_windows(p,2)
    assert torch.equal(actual,expected)
    torch.set_rng_state(state);assert torch.equal(sample_feature_windows(p,2),actual)
    assert torch.isfinite(actual).all()


def test_generic_npz_pretraining_schema_and_l1_loss(tmp_path):
    schema=MotionFeatureSchema(('finger','hip'),('tool',),.04,root_body_name='base')
    count=7
    np.savez(tmp_path/'clip.npz',joint_names=np.asarray(['hip','finger','extra']),
        body_names=np.asarray(['base','tool']),fps=np.asarray(25.),
        body_pos_w=np.zeros((count,2,3),np.float32),
        body_quat_w=np.tile(np.array([1,0,0,0],np.float32),(count,2,1)),
        body_lin_vel_w=np.zeros((count,2,3),np.float32),body_ang_vel_w=np.zeros((count,2,3),np.float32),
        joint_pos=np.tile(np.array([1,2,3],np.float32),(count,1)),joint_vel=np.zeros((count,3),np.float32))
    windows,out_schema,manifest=build_motion_windows(tmp_path,3,schema=schema)
    assert out_schema==schema and windows.shape==(5,3,20)
    assert manifest['nwindows']==5
    assert torch.equal(windows[...,9:11],torch.tensor([2.,1.]).expand(5,3,2))
    low,high=fit_quantiles(windows);assert (high>low).all()
    with pytest.raises(ValueError):build_motion_windows(tmp_path,3)
    class Zero(torch.nn.Module):
        def forward(self,x,t):return torch.zeros_like(x)
    scheduler=DDPMScheduler();torch.manual_seed(7);state=torch.get_rng_state()
    x=torch.randn(2,3,20);state=torch.get_rng_state();loss=diffusion_loss(Zero(),scheduler,x,3)
    torch.set_rng_state(state);scheduler.sample_timesteps(6,x.device);noise=torch.randn(6,3,20)
    assert torch.equal(loss,noise.abs().mean()) # preserve actual upstream L1, not MSE



def _original_package(owner, filename, definitions):
    """Resolve an independent pre-extraction fixture, never a thin shim."""
    reference=os.environ.get('MBDAC_SMP_REFERENCE')
    if reference:
        package=Path(reference)
    elif owner:
        package=Path(owner)/'src/rl/smp'
    else:
        pytest.skip('Original comparison needs MBDAC_SMP_REFERENCE (pre-extraction package directory)')
    path=package/filename
    if not path.is_file():
        pytest.skip(f'Original comparison unavailable: {path}; set MBDAC_SMP_REFERENCE')
    tree=ast.parse(path.read_text())
    found={node.name for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef))}
    if not set(definitions)<=found:
        pytest.skip(f'Original comparison skipped: {path} is a compatibility shim, not an independent implementation; set MBDAC_SMP_REFERENCE')
    return package


def test_optional_owning_source_checkpoint_parity():
    source=os.environ.get('MBDAC_SMP_SOURCE')
    if not source:pytest.skip('Set MBDAC_SMP_SOURCE for owning-checkout checkpoint/features parity')
    root=Path(source)
    checkpoint=root/'logs/smp_prior/bones_six_fullbody_official_20261024/prior.pt'
    actual=FrozenSMPPrior.load_checkpoint(checkpoint)
    with np.load(checkpoint.parent/'windows.npz') as data:
        all_features=torch.from_numpy(data['features'].copy())
    features=all_features[:3]
    values=dict(actual.feature_schema)
    for name in ('joint_names','end_effector_names'):values[name]=tuple(values[name])
    schema=MotionFeatureSchema(**values)
    extracted,_,_=build_motion_windows(checkpoint.parent/'window_manifest.json',10,schema=schema)
    assert torch.equal(extracted,all_features) # compare recorded pre-migration extraction, all six clips

    reference=_original_package(source,'prior.py',('FrozenSMPPrior',))
    package=types.ModuleType('_smp_parity_source');package.__path__=[str(reference)]
    sys.modules[package.__name__]=package
    spec=importlib.util.spec_from_file_location(package.__name__+'.prior',reference/'prior.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    expected=module.FrozenSMPPrior.load_checkpoint(checkpoint)
    torch.manual_seed(4);noise=torch.randn(3,*features.shape)
    a=actual.score(features,noise=noise);b=expected.score(features,noise=noise)
    assert torch.equal(a.reward,b.reward) and torch.equal(a.raw_error,b.raw_error)
    # Extract upstream feature class only; no reader or SDK import is executed.
    tree=ast.parse((reference/'features.py').read_text()) if (reference/'features.py').exists() else ast.Module(body=[],type_ignores=[])
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionFeatureBuffer']
    if not classes:
        upstream=Path(os.environ.get('MBDAC_SMP_UPSTREAM','/tmp/paper_plane_smp_upstream_20261024'))
        tree=ast.parse((upstream/'src/smp/rl/utils.py').read_text())
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionFeatureBuffer']
    assert len(classes)==1
    cls=classes[0]
    from mbdac.smp import math as pure_math
    namespace=dict(torch=torch,**{n:getattr(pure_math,n) for n in
        ('matrix_from_quat','quat_apply_inverse','quat_conjugate','quat_mul','yaw_quat')})
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'source_features','exec'),namespace)
    ref=namespace['MotionFeatureBuffer'](2,5,3,2,'cpu');ours=MotionFeatureBuffer(2,5,3,2,'cpu')
    for name in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','ee_pos_w','joint_pos','joint_vel'):
        x=torch.randn_like(getattr(ours,name))
        if name=='root_quat_w':x/=x.norm(dim=-1,keepdim=True)
        getattr(ours,name).copy_(x);getattr(ref,name).copy_(x)
    assert torch.equal(ref.compute_features(),ours.compute_features())


def test_optional_native_rotation_and_feature_parity():
    if not os.environ.get('MBDAC_SMP_SDK_PARITY'):
        pytest.skip('Set MBDAC_SMP_SDK_PARITY=1 to compare optional installed MJLab math')
    native=pytest.importorskip('mjlab.utils.lab_api.math')
    from mbdac.smp import math as pure
    torch.manual_seed(441)
    q=torch.randn(600,4);q/=q.norm(dim=-1,keepdim=True)
    q2=torch.randn_like(q);q2/=q2.norm(dim=-1,keepdim=True)
    v=torch.randn(600,3)
    for name,args in [('matrix_from_quat',(q,)),('quat_conjugate',(q,)),('quat_mul',(q,q2)),
                      ('yaw_quat',(q,)),('quat_apply',(q,v)),('quat_apply_inverse',(q,v)),
                      ('quat_from_matrix',(native.matrix_from_quat(q),))]:
        assert torch.equal(getattr(native,name)(*args),getattr(pure,name)(*args)),name
    upstream=Path(os.environ.get('MBDAC_SMP_UPSTREAM','/tmp/paper_plane_smp_upstream_20261024'))
    source=upstream/'src/smp/rl/utils.py'
    if not source.exists():pytest.skip('Original SMP checkout needed for feature/math combined parity')
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MotionFeatureBuffer')
    namespace=dict(torch=torch,**{n:getattr(native,n) for n in
        ('matrix_from_quat','quat_apply_inverse','quat_conjugate','quat_mul','yaw_quat')})
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'source_features','exec'),namespace)
    reference=namespace['MotionFeatureBuffer'](4,10,24,5,'cpu')
    actual=MotionFeatureBuffer(4,10,24,5,'cpu')
    for name in ('root_pos_w','root_quat_w','root_lin_vel_w','root_ang_vel_w','ee_pos_w','joint_pos','joint_vel'):
        x=torch.randn_like(getattr(actual,name))
        if name=='root_quat_w':x/=x.norm(dim=-1,keepdim=True)
        getattr(actual,name).copy_(x);getattr(reference,name).copy_(x)
    assert torch.equal(reference.compute_features(),actual.compute_features())

    owner=os.environ.get('MBDAC_SMP_SOURCE')
    if owner or os.environ.get('MBDAC_SMP_REFERENCE'):
        reference=_original_package(owner,'generation.py',('sample_feature_windows','decode_feature_windows'))
        package=types.ModuleType('_smp_original_generation');package.__path__=[str(reference)]
        sys.modules[package.__name__]=package
        # Inject only schema/prior type imports; original generation still uses
        # actual installed MJLab quaternion kernels, compared to pure kernels.
        feature_module=types.ModuleType(package.__name__+'.features')
        feature_module.MotionFeatureSchema=MotionFeatureSchema
        sys.modules[feature_module.__name__]=feature_module
        prior_module=types.ModuleType(package.__name__+'.prior')
        prior_module.FrozenSMPPrior=FrozenSMPPrior
        sys.modules[prior_module.__name__]=prior_module
        spec=importlib.util.spec_from_file_location(package.__name__+'.generation',reference/'generation.py')
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        schema=MotionFeatureSchema(tuple(f'j{i}' for i in range(24)),tuple(f'ee{i}' for i in range(5)),.02)
        features=actual.compute_features()
        origins=torch.randn(4,3);yaw=torch.randn(4);xy=torch.randn(4,2)
        a=old.decode_feature_windows(features,schema,env_origins=origins,chosen_yaw=yaw,anchor_xy=xy)
        b=decode_feature_windows(features,schema,env_origins=origins,chosen_yaw=yaw,anchor_xy=xy)
        for name in a.__dataclass_fields__:
            left,right=getattr(a,name),getattr(b,name)
            assert torch.equal(left,right) if isinstance(left,torch.Tensor) else left==right
        p=prior();state=torch.get_rng_state();a=old.sample_feature_windows(p,2)
        torch.set_rng_state(state);b=sample_feature_windows(p,2)
        assert torch.equal(a,b)
