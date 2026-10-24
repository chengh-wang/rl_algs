import torch
from torch import nn
from .networks import mlp, run_network, RunningNorm
from .primitives import PrimitiveHeads


class WorldModel(nn.Module):
    def __init__(self, codec, action_dim, context_dim, hidden=(2048,2048,2048), precision='fp32', primitive_specs=()):
        super().__init__()
        self.codec=codec; self.precision=precision
        self.input_norm=RunningNorm(codec.feature_dim+action_dim+context_dim)
        self.output_norm=RunningNorm(codec.delta_dim,eps=1e-4)
        self.net=mlp(codec.feature_dim+action_dim+context_dim,hidden,codec.delta_dim,nn.SiLU,True)
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.primitive_heads=PrimitiveHeads(codec.feature_dim+context_dim,primitive_specs)

    def inputs(self, state, action, context):
        return torch.cat((self.codec.features(state),action,context),-1)

    def forward(self, state, action, context):
        y=run_network(self.net,self.input_norm(self.inputs(state,action,context)),self.precision)
        return self.codec.integrate(state,y*self.output_norm.std()+self.output_norm.mean)

    def primitive_inputs(self,state,context):
        return torch.cat((self.codec.features(state),context),-1)

    def primitives(self, state, context):
        return self.primitive_heads(self.primitive_inputs(state,context))
