"""Check value/gradient equivalence and time FK pruning on either native robot."""
import argparse
import json
import time
from pathlib import Path
import torch
from .environment import NativeVelocityEnv
from .model_task import NativeVelocityTask


def main():
    p=argparse.ArgumentParser();p.add_argument('--robot',choices=('go1','g1'),default='g1')
    p.add_argument('--out',type=Path,default=Path('logs/native_fk_profile.json'));args=p.parse_args()
    torch.set_num_threads(4);e=NativeVelocityEnv(args.robot,16);task=NativeVelocityTask(e)
    try:
        s=e.physical_state()[:1].repeat(256,1).requires_grad_();kin=task.kin;ids=(task.imu_linear,task.imu_angular)
        def full(s):
            outputs=kin.sites(kin.forward(*task.codec.unpack(s)))
            return tuple(o[...,list(ids),:,:] if o.ndim==4 else o[...,list(ids),:] for o in outputs)
        def pruned(s):return kin.site_motion(ids,*task.codec.unpack(s))
        full_o=full(s);fast_o=pruned(s)
        value_error=max(float((a-b).abs().max()) for a,b in zip(full_o,fast_o))
        weights=[torch.randn_like(o) for o in full_o]
        loss=lambda outputs:sum((o*w).sum() for o,w in zip(outputs,weights))
        g0=torch.autograd.grad(loss(full_o),s)[0];g1=torch.autograd.grad(loss(fast_o),s)[0]
        gradient_error=float((g0-g1).abs().max());times={}
        for name,fn in [('full',full),('pruned',pruned)]:
            for _ in range(3):torch.autograd.grad(loss(fn(s)),s)
            torch.cuda.synchronize();t=time.perf_counter()
            for _ in range(20):torch.autograd.grad(loss(fn(s)),s)
            torch.cuda.synchronize();times[name]=(time.perf_counter()-t)/20
        result=dict(robot=args.robot,batch=256,value_error=value_error,gradient_error=gradient_error,
            forward_backward_s=times,scope='Only analytic sensor FK; not end-to-end training speedup')
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result));assert value_error<1e-6 and gradient_error<1e-5
    finally:e.close()


if __name__=='__main__':main()
