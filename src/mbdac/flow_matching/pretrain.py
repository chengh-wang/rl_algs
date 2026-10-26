"""Pretrain linear conditional velocity on unchanged full-body motion windows."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from mbdac.smp.features import MotionFeatureSchema
from mbdac.smp.pretrain import build_motion_windows, fit_quantiles
from mbdac.smp.vendor.model import DiffusionDenoiser
from .prior import PATH_CONTRACT, linear_path

from .training import flow_matching_loss, checkpoint_payload

def main(argv=None, *, schema_override=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--references',required=True); p.add_argument('--output',required=True)
    p.add_argument('--feature-schema',required=schema_override is None)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--device',default='cpu')
    for name,default in [('epochs',2000),('batch-size',1024),('seed',42),('window-size',10),('window-batch-size',4096),('d-model',256),('nhead',4),('num-layers',2),('num-noise-samples',10),('save-every',100),('log-every',10)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('learning-rate',3e-4),('weight-decay',1e-4),('max-grad-norm',1.),('span-floor',1e-4)]:
        p.add_argument('--'+name,type=float,default=default)
    args=p.parse_args(argv)
    import math
    if any(getattr(args,k)<=0 for k in ('epochs','batch_size','window_size','window_batch_size','d_model','nhead','num_layers','num_noise_samples','save_every','log_every')):
        p.error('Counts and architecture dimensions must be positive')
    if args.d_model%2 or args.d_model%args.nhead:
        p.error('d-model must be even and divisible by nhead')
    if any(not math.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in ('learning_rate','max_grad_norm','span_floor')) or not math.isfinite(args.weight_decay) or args.weight_decay<0:
        p.error('Invalid optimizer/quantile settings')
    torch.manual_seed(args.seed)
    output=Path(args.output)
    if (output/'prior.pt').exists():
        raise FileExistsError('Refusing to overwrite existing prior.pt')
    if schema_override is not None:
        if args.feature_schema is not None or not isinstance(schema_override,MotionFeatureSchema):
            raise ValueError('Choose explicit schema file or MotionFeatureSchema override')
        schema=schema_override
    else:
        values=json.loads(Path(args.feature_schema).read_text())
        for key in ('joint_names','end_effector_names'): values[key]=tuple(values[key])
        schema=MotionFeatureSchema(**values)
    train,schema,manifest=build_motion_windows(args.references,args.window_size,args.window_batch_size,schema=schema)
    low,high=fit_quantiles(train,args.span_floor)
    cfg={**vars(args),'feature_dim':int(train.shape[-1]),'dropout':0.,'prior_type':'linear_flow','use_ema':False,
         'loss':'velocity_mse','conditioning_scale':50.,'feature_view':'full','sampling_policy':'uniform','augmentation':False}
    output.mkdir(parents=True,exist_ok=True)
    for name,value in [('config',cfg),('feature_schema',asdict(schema)),('window_manifest',manifest),('path_contract',PATH_CONTRACT)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    import numpy as np
    np.savez_compressed(output/'windows.npz',features=train.numpy(),q_low=low.numpy(),q_high=high.numpy())
    report={'shape':list(train.shape),'nwindows':len(train),'nclips':manifest['nclips'],
            'normalization':'original training q01/q99; span floor; no clipping',
            'optimizer_updates_per_epoch':(len(train)+args.batch_size-1)//args.batch_size,'epochs':[],'cfg':cfg}
    def write_report(): (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    write_report()
    if args.prepare_only:
        print(json.dumps(report)); return
    model=DiffusionDenoiser(cfg['feature_dim'],args.window_size,args.d_model,args.nhead,args.num_layers).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    normalized=2*(train-low)/(high-low+1e-8)-1
    loader=DataLoader(TensorDataset(normalized),batch_size=args.batch_size,shuffle=True)
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.
        for (batch,) in loader:
            batch=batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss=flow_matching_loss(model,batch,args.num_noise_samples)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite flow loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm,error_if_nonfinite=True)
            optimizer.step(); total+=loss.item()*len(batch)
        metrics={'epoch':epoch,'train_velocity_mse':total/len(train)}
        report['epochs'].append(metrics)
        if epoch%args.save_every==0 or epoch==args.epochs:
            temporary=output/'prior.pt.tmp'
            torch.save(checkpoint_payload(model,optimizer,cfg,low,high,asdict(schema),epoch,manifest,metrics),temporary)
            temporary.replace(output/'prior.pt')
        if epoch%args.log_every==0 or epoch==args.epochs:
            write_report(); print(json.dumps(metrics),flush=True)

if __name__=='__main__': main()
