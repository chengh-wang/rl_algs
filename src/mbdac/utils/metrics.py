"""Reduce real-step log values with one device-to-host transfer per device.

Only detached diagnostics are batched. Neither data collection nor any optimizer
computation changes. Scalar statistics retain the public Python-float interface.
"""
import torch


def snapshot_metrics(row):
    """Own scalar log storage before the next simulator step reuses buffers."""
    result={};groups={}
    for key,value in row.items():
        if isinstance(value,torch.Tensor):
            value=value.detach().float()
            value=value.reshape(()) if value.numel()==1 else value.mean()
            groups.setdefault(value.device,[]).append((key,value))
        else:result[key]=float(value)
    for device,items in groups.items():
        packed=torch.stack([v for _,v in items])
        result.update((key,value) for (key,_),value in zip(items,packed.unbind()))
    return result


def mean_metrics(rows):
    values={}
    for row in rows:
        for key,value in row.items():values.setdefault(key,[]).append(value)
    groups={}
    for key,items in values.items():
        for index,value in enumerate(items):
            if isinstance(value,torch.Tensor):
                value=value.detach().float()
                value=value.reshape(()) if value.numel()==1 else value.mean()
                groups.setdefault(value.device,[]).append((key,index,value))
            else:items[index]=float(value)
    for device,items in groups.items():
        packed=torch.stack([v for _,_,v in items]).cpu().tolist()
        for (key,index,_),value in zip(items,packed):values[key][index]=float(value)
    return {key:sum(items)/len(items) for key,items in values.items()}
