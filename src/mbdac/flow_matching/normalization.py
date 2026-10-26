"""Independent count-weighted per-time flow error normalization."""
from copy import deepcopy
import math
import torch
from torch import nn


class FlowErrorNormalizer(nn.Module):
    """Mean error by scoring time. An optional cap freezes after its final batch.

    max_count=None uses every sample. A batch is accepted only while count is
    below max_count; the complete batch contributes, so the final count may exceed it.
    """
    def __init__(self, num_times, min_value=1e-4, max_count=None):
        super().__init__()
        if isinstance(num_times,bool) or not isinstance(num_times,int) or num_times <= 0:
            raise ValueError('num_times must be a positive integer')
        if isinstance(min_value,bool) or not isinstance(min_value,(int,float)) or not math.isfinite(min_value) or min_value <= 0:
            raise ValueError('min_value must be finite and positive')
        if max_count is not None and (isinstance(max_count,bool) or not isinstance(max_count,int) or max_count < 0):
            raise ValueError('max_count must be nonnegative integer or None')
        self.num_times, self.min_value, self.max_count = num_times, float(min_value), max_count
        self.register_buffer('mean',torch.ones(num_times))
        self.register_buffer('count',torch.zeros(num_times,dtype=torch.long))

    def get_extra_state(self):
        return dict(version=1,num_times=self.num_times,min_value=self.min_value,max_count=self.max_count)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Normalizer configuration differs from checkpoint')

    def _validate_values(self, index, values):
        if isinstance(index,bool) or not isinstance(index,int) or not 0 <= index < self.num_times:
            raise ValueError('Invalid scoring time index')
        if not isinstance(values,torch.Tensor) or not values.is_floating_point() or values.numel() == 0 or values.device != self.mean.device or not torch.isfinite(values).all() or (values < 0).any():
            raise ValueError('Errors must be nonempty finite nonnegative floating tensors on normalizer device')

    @torch.no_grad()
    def update(self, index, values):
        self._validate_values(index,values)
        old = self.count[index].item()
        if self.max_count is not None and old >= self.max_count:
            return
        total = old + values.numel()
        if total > torch.iinfo(torch.long).max:
            raise ValueError('Normalizer count overflow')
        mean = values.double().mean()
        updated = mean if old == 0 else self.mean[index].double()*(old/total)+mean*(values.numel()/total)
        updated = updated.to(self.mean.dtype)
        if not torch.isfinite(updated):
            raise ValueError('Normalizer mean overflow')
        self.mean[index] = updated
        self.count[index] = total

    def normalize(self,index,values):
        self._validate_values(index,values)
        return values / self.mean[index].clamp(min=self.min_value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = deepcopy(state_dict)
        if set(state) != {'mean','count','_extra_state'}:
            raise ValueError('Invalid normalizer state keys')
        self.set_extra_state(state['_extra_state'])
        mean, count = state['mean'],state['count']
        if not isinstance(mean,torch.Tensor) or mean.dtype != self.mean.dtype or mean.shape != self.mean.shape or not torch.isfinite(mean).all() or (mean < 0).any():
            raise ValueError('Invalid normalizer means')
        if not isinstance(count,torch.Tensor) or count.dtype != torch.long or count.shape != self.count.shape or (count < 0).any():
            raise ValueError('Invalid normalizer counts')
        if ((count == 0) & (mean != 1)).any():
            raise ValueError('Unobserved normalizer means must equal one')
        return super().load_state_dict(state,strict=strict,assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Recursive module loading must enforce the same invariant as direct loading.
        selected = {key[len(prefix):]:value for key,value in state_dict.items() if key.startswith(prefix)}
        if set(selected) != {'mean','count','_extra_state'}:
            raise ValueError('Invalid normalizer state keys')
        self.set_extra_state(selected['_extra_state'])
        mean, count = selected['mean'],selected['count']
        if not isinstance(mean,torch.Tensor) or mean.dtype != self.mean.dtype or mean.shape != self.mean.shape or not torch.isfinite(mean).all() or (mean < 0).any():
            raise ValueError('Invalid normalizer means')
        if not isinstance(count,torch.Tensor) or count.dtype != torch.long or count.shape != self.count.shape or (count < 0).any():
            raise ValueError('Invalid normalizer counts')
        if ((count == 0) & (mean != 1)).any():
            raise ValueError('Unobserved normalizer means must equal one')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)
