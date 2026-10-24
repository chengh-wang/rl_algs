"""Time/env replay; window masking and history sampling adapted from SuperMimic (MIT)."""
import torch


class ReplayBuffer:
    def __init__(self, capacity, num_envs, device):
        self.capacity,self.num_envs,self.device=capacity,num_envs,device
        self.data={}; self.ptr=self.size=0

    @torch.no_grad()
    def add(self, transition):
        if not self.data:
            self.data={k:torch.empty((self.capacity,self.num_envs,*v.shape[1:]),device=self.device,dtype=v.dtype) for k,v in transition.items()}
        if transition.keys()!=self.data.keys():
            raise ValueError('Replay transition schema changed')
        for k,v in transition.items():
            self.data[k][self.ptr].copy_(v)
        self.ptr=(self.ptr+1)%self.capacity; self.size=min(self.size+1,self.capacity)

    def _row(self, age):
        return (self.ptr-self.size+age)%self.capacity

    def sample_windows(self, batch, window):
        if self.size<window:
            raise RuntimeError(f'Replay has {self.size} steps; need {window}. Collect before updating.')
        start=torch.randint(self.size-window+1,(batch,),device=self.device)
        env=torch.randint(self.num_envs,(batch,),device=self.device)
        rows=self._row(start[:,None]+torch.arange(window,device=self.device))
        out={k:v[rows,env[:,None]] for k,v in self.data.items()}
        done=out['terminated']|out['truncated']
        out['mask']=torch.cumprod(torch.cat((torch.ones(batch,1,device=self.device),1-done[:,:-1].float()),1),1)
        return out

    def sample_starts(self, batch, history):
        if not self.size:
            raise RuntimeError('Empty replay')
        age=torch.randint(self.size,(batch,),device=self.device)
        env=torch.randint(self.num_envs,(batch,),device=self.device); row=self._row(age)
        back=torch.arange(history-1,-1,-1,device=self.device)[None]
        back=torch.minimum(torch.minimum(back,self.data['ep_step'][row,env,None].long()),age[:,None])
        hr=self._row(age[:,None]-back)
        return ({k:v[row,env] for k,v in self.data.items()},
                {k:v[hr,env[:,None]] for k,v in self.data.items()})

    def state_dict(self):
        # Store only initialized rows. Pointer preserves circular-buffer order.
        return dict(capacity=self.capacity,num_envs=self.num_envs,ptr=self.ptr,size=self.size,
                    data={k:v[:self.size].cpu() for k,v in self.data.items()})

    def load_state_dict(self, state):
        if state['capacity']!=self.capacity or state['num_envs']!=self.num_envs:
            raise ValueError('Replay dimensions differ; explicitly warm-start without replay instead')
        self.data={}
        for k,v in state['data'].items():
            self.data[k]=torch.empty((self.capacity,*v.shape[1:]),device=self.device,dtype=v.dtype)
            self.data[k][:state['size']].copy_(v.to(self.device))
        self.ptr,self.size=state['ptr'],state['size']
