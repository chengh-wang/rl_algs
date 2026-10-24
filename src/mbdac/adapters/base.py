"""Adapters preserve the RSL-RL four-return step interface.

extras['model_transition'] must be an immutable snapshot of the applied action
and physical next state BEFORE automatic reset. A normal VecEnv alone cannot
reconstruct a terminal state that the environment has already overwritten.
"""
from typing import Protocol
from tensordict import TensorDict
import torch


class ModelVecEnv(Protocol):
    model_task: object
    num_envs: int
    num_actions: int
    device: str | torch.device

    def get_observations(self) -> TensorDict: ...
    def step(self, actions: torch.Tensor) -> tuple[TensorDict,torch.Tensor,torch.Tensor,dict]: ...


class RslModelAdapter:
    """Thin bridge for an existing RSL VecEnv whose task exposes model transitions."""
    def __init__(self, env, model_task):
        self.env=env; self.model_task=model_task
        self.num_envs,self.num_actions,self.device=env.num_envs,env.num_actions,env.device

    def __getattr__(self, name): return getattr(self.env,name)
    def get_observations(self): return self.env.get_observations()
    def step(self, actions):
        result=self.env.step(actions)
        if 'model_transition' not in result[3]:
            raise KeyError('Task must capture model_transition before reset; see ModelVecEnv contract')
        return result
