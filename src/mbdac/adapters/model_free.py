"""Simulator-free RSL four-return contract for real off-policy transitions.

Returned observations are the next policy inputs (possibly already reset).
Every done row MUST have a pre-reset ``extras['terminal_observation']`` snapshot
of all configured observation groups, either N rows or exactly done.sum rows.
``terminated``/``truncated`` are explicit masks, or ``time_outs`` distinguishes
ordinary time limits; without any timeout information all dones are failures.
Actions are normalized learner coordinates. An adapter that clips them reports
``applied_action`` in the same coordinates, before its physical control mapping.
No model_task, state codec, physical model_transition or reference data is needed.
"""
from collections.abc import Mapping
from typing import Protocol
import torch


class ModelFreeVecEnv(Protocol):
    num_envs: int
    num_actions: int
    device: str | torch.device
    def reset(self) -> object: ...
    def get_observations(self) -> Mapping[str, torch.Tensor]: ...
    def step(self, actions: torch.Tensor) -> tuple[Mapping, torch.Tensor, torch.Tensor, dict]: ...


def observation_groups(obs, groups, n, device):
    result = {}
    for role, names in groups.items():
        parts = [obs[name].to(device) for name in names]
        if any(p.ndim != 2 or p.shape[0] != n for p in parts):
            raise ValueError(f'{role} observations must be [num_envs, features]')
        result[role] = torch.cat(parts, -1).detach().clone()
    return result


def real_transition(before, after, action, reward, dones, extras, groups, device):
    n = action.shape[0]
    def mask(value, name):
        value = torch.as_tensor(value, device=device)
        if value.shape != (n,):
            raise ValueError(f'{name} must have shape [num_envs]')
        if not ((value == 0) | (value == 1)).all():
            raise ValueError(f'{name} must be binary')
        return value.bool()
    done = mask(dones, 'dones')
    if ('terminated' in extras) != ('truncated' in extras):
        raise ValueError('Supply terminated and truncated together')
    if 'terminated' in extras:
        terminated = mask(extras['terminated'], 'terminated')
        truncated = mask(extras['truncated'], 'truncated')
    else:
        truncated = mask(extras.get('time_outs', torch.zeros_like(done)), 'time_outs')
        terminated = done & ~truncated
    if not torch.equal(terminated | truncated, done):
        raise ValueError('Done flags disagree with termination/timeout masks')
    next_obs = {k: v.clone() for k, v in after.items()}
    if done.any():
        terminal = extras.get('terminal_observation')
        if terminal is None:
            raise ValueError('Every done requires terminal_observation before reset')
        first = terminal[groups['actor'][0]]
        count = int(done.sum())
        rows = first.shape[0]
        if rows not in (n, count):
            raise ValueError('terminal_observation must have N or done-count rows')
        measured = observation_groups(terminal, groups, rows, device)
        for key in next_obs:
            if measured[key].shape[-1] != next_obs[key].shape[-1]:
                raise ValueError('Terminal observation schema differs')
            next_obs[key][done] = measured[key][done] if rows == n else measured[key]
    applied = torch.as_tensor(extras.get('applied_action', action), device=device).detach().clone()
    if applied.shape != action.shape or not torch.isfinite(applied).all():
        raise ValueError('applied_action must match finite normalized policy actions')
    reward = torch.as_tensor(reward, device=device).detach().clone()
    if reward.shape != (n,) or not torch.isfinite(reward).all():
        raise ValueError('Reward must be finite [num_envs]')
    return dict(observation=torch.cat((before['actor'], before['critic']), -1),
                next_observation=torch.cat((next_obs['actor'], next_obs['critic']), -1),
                action=applied, reward=reward, terminated=terminated.clone(), truncated=truncated.clone())
