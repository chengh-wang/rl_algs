"""Compose upstream SAC networks/losses/replay without its NumPy environment loop.

Packed replay keeps actor and critic information distinct. No environment reward,
PD, simulator, upstream network, or upstream loss implementation is changed.
"""
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .config import RunnerConfig, UPSTREAM_COMMIT

from mbdac.vendor.flashsac.spaces import ShapeSpace
from mbdac.vendor.flashsac.agent import (
    _init_flashsac_networks, _sample_flashsac_actions, _build_truncated_zeta_cdf, _update_networks,
)
from mbdac.vendor.flashsac.utils.reward_normalization import RewardNormalizer
from mbdac.vendor.flashsac.buffers.torch_buffer import TorchUniformBuffer


def raw_module(network):
    return getattr(network.network, '_orig_mod', network.network)


class FlashSACRunner:
    def __init__(self, actor_dim, critic_dim, action_dim, num_envs, cfg: RunnerConfig, contract: dict):
        self.cfg = cfg
        if cfg.float32_matmul_precision not in ('highest', 'high'):
            raise ValueError('Use explicit highest or high float32 matmul precision')
        torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
        self.upstream_cfg = cfg.upstream(action_dim)
        self.device = torch.device(cfg.device)
        self.actor_dim, self.critic_dim, self.action_dim = actor_dim, critic_dim, action_dim
        self.num_envs = num_envs
        self.contract = contract
        if cfg.capacity < num_envs:
            raise ValueError('Replay capacity must fit at least one vector step')
        self.actor, self.critic, self.target, self.temperature = _init_flashsac_networks(
            actor_dim, critic_dim, action_dim, self.upstream_cfg, self.device)
        self.networks = dict(actor=self.actor, critic=self.critic, target=self.target, temperature=self.temperature)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=cfg.use_amp)
        self.noise = torch.zeros(num_envs, action_dim, device=self.device)
        self.count = torch.tensor(0, dtype=torch.int32, device=self.device)
        self.repeat = torch.tensor(1, dtype=torch.int32, device=self.device)
        self.cdf = _build_truncated_zeta_cdf(2., 16).to(self.device)
        self.normalizer = RewardNormalizer(.99, 5., True, self.device)
        self.replay = TorchUniformBuffer(
            ShapeSpace((actor_dim+critic_dim,), np.float32),
            ShapeSpace((action_dim,), np.float32),
            n_step=1, gamma=.99, max_length=cfg.capacity, min_length=cfg.warmup,
            sample_batch_size=cfg.batch_size, device_type=cfg.device)
        self.update_step = self.vector_steps = self.transitions = 0

    @torch.no_grad()
    def actions(self, actor_observation, deterministic=False):
        if actor_observation.shape[-1] != self.actor_dim:
            raise ValueError('Actor received the wrong observation group')
        torch.compiler.cudagraph_mark_step_begin()
        self.noise, action, self.count, self.repeat = _sample_flashsac_actions(
            self.actor, self.noise, actor_observation, 0. if deterministic else 1.,
            self.count, self.repeat, self.cdf)
        # Own outputs across the next compiled invocation.
        self.noise, self.count, self.repeat = self.noise.clone(), self.count.clone(), self.repeat.clone()
        return action.clone()

    def observe(self, transition):
        self.replay.add(transition)
        self.normalizer.update_reward_stats(transition['reward'], transition['terminated'], transition['truncated'])
        self.vector_steps += 1
        self.transitions += len(transition['reward'])

    def batch(self, indices=None):
        batch = self.replay.sample(indices)
        for key, actor_key in [('observation', 'actor_observation'), ('next_observation', 'actor_next_observation')]:
            packed = batch[key]
            batch[actor_key] = packed[:, :self.actor_dim].contiguous()
            batch[key] = packed[:, self.actor_dim:].contiguous()
        batch['raw_reward'] = batch['reward']
        batch['reward'] = self.normalizer.normalize_rewards(batch['reward'])
        return batch

    def update(self):
        if not self.replay.can_sample():
            raise RuntimeError('Replay warmup is incomplete')
        torch.compiler.cudagraph_mark_step_begin()
        batch = self.batch()
        metrics = _update_networks(batch, self.actor, self.critic, self.target, self.temperature,
            self.upstream_cfg, self.update_step % self.upstream_cfg.actor_update_period == 0,
            self.device, self.scaler)
        self.update_step += 1
        metrics['reward/raw'] = batch['raw_reward'].mean()
        metrics['reward/normalized'] = batch['reward'].mean()
        return {k: v.detach().clone() for k, v in metrics.items()}

    def save(self, path, include_replay=False):
        """Save learner state. Simulator state is NOT saved: not an exact env resume."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(version=1, upstream_commit=UPSTREAM_COMMIT, config=asdict(self.cfg),
            contract=self.contract, dimensions=[self.actor_dim,self.critic_dim,self.action_dim,self.num_envs],
            update_step=self.update_step, vector_steps=self.vector_steps, transitions=self.transitions,
            scaler=self.scaler.state_dict(), noise=self.noise, count=self.count, repeat=self.repeat,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == 'cuda' and torch.cuda.is_available() else [], python_rng=random.getstate(),
            numpy_rng=(lambda s: (s[0],s[1].tolist(),s[2],s[3],s[4]))(np.random.get_state()),
            networks={}, normalizer={name: getattr(self.normalizer,name) for name in ['G_r','G_r_max']},
            rms={name:getattr(self.normalizer.G_rms,name) for name in ['mean','var','count']},
            replay=None, simulator_state_saved=False)
        for name, net in self.networks.items():
            state['networks'][name] = dict(model=raw_module(net).state_dict(),
                optimizer=net.optimizer.state_dict() if net.optimizer else None,
                scheduler=net.scheduler.state_dict() if net.scheduler else None)
        if include_replay:
            n = len(self.replay)
            state['replay'] = dict(length=n, index=self.replay._current_idx,
                **{k:getattr(self.replay,k)[:n].detach().cpu().clone() for k in
                   ['_observations','_next_observations','_actions','_rewards','_terminateds','_truncateds']})
        temp = path.with_suffix(path.suffix+'.tmp')
        torch.save(state, temp)
        temp.replace(path)

    def load(self, path, *, restore_rng=False, continue_same_environments=False,
             allow_cold_replay_resize=False):
        """Continue learner; default starts fresh episodes, clearing per-episode state.

        continue_same_environments is only for in-process validation with unchanged
        simulator state. Cold replay resumes are explicitly returned to the caller.
        Replay capacity may change only by explicit opt-in when no replay was
        saved. This does not permit changing any other algorithm/task setting.
        """
        state = torch.load(path, map_location='cpu', weights_only=True)
        if (state['version'] != 1 or state['upstream_commit'] != UPSTREAM_COMMIT or
            state['contract'] != self.contract or state['dimensions'] !=
            [self.actor_dim,self.critic_dim,self.action_dim,self.num_envs]):
            raise ValueError('Checkpoint environment/action/observation contract mismatch')
        # Compilation is a runtime choice, not a change to network state keys.
        allowed = {'use_compile', 'compile_mode'}
        old_capacity = state['config']['capacity']
        capacity_changed = old_capacity != self.cfg.capacity
        if capacity_changed and allow_cold_replay_resize:
            if state['replay'] is not None or continue_same_environments:
                raise ValueError('Replay resize requires a cold, fresh-environment resume')
            allowed.add('capacity')
        if any(state['config'].get(k, 'highest' if k == 'float32_matmul_precision' else None) != v
               for k,v in asdict(self.cfg).items() if k not in allowed):
            raise ValueError('Checkpoint algorithm/replay configuration mismatch')
        for name, net in self.networks.items():
            saved = state['networks'][name]
            raw_module(net).load_state_dict(saved['model'], strict=True)
            if net.optimizer: net.optimizer.load_state_dict(saved['optimizer'])
            if net.scheduler: net.scheduler.load_state_dict(saved['scheduler'])
        self.scaler.load_state_dict(state['scaler'])
        for k in ['update_step','vector_steps','transitions']: setattr(self,k,state[k])
        for k in ['noise','count','repeat']: setattr(self,k,state[k].to(self.device))
        for k,v in state['normalizer'].items(): setattr(self.normalizer,k,v.to(self.device))
        for k,v in state['rms'].items(): setattr(self.normalizer.G_rms,k,v.to(self.device))
        self.replay._num_in_buffer = self.replay._current_idx = 0
        self.replay._n_step_transitions.clear()
        if state['replay'] is not None:
            saved = state['replay']; n = saved['length']
            for k,v in saved.items():
                if k.startswith('_'): getattr(self.replay,k)[:n].copy_(v)
            self.replay._num_in_buffer, self.replay._current_idx = n,saved['index']
        if not continue_same_environments:
            self.normalizer.G_r = torch.zeros(self.num_envs, device=self.device)
            self.noise.zero_(); self.count.zero_(); self.repeat.fill_(1)
        if restore_rng:
            torch.set_rng_state(state['torch_rng'])
            if self.device.type == 'cuda' and torch.cuda.is_available() and state['cuda_rng']:
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            random.setstate(state['python_rng'])
            s = state['numpy_rng']; np.random.set_state((s[0],np.asarray(s[1],dtype=np.uint32),*s[2:]))
        return dict(replay_restored=state['replay'] is not None, simulator_state_restored=False,
                    replay_capacity_before=old_capacity, replay_capacity_after=self.cfg.capacity,
                    replay_capacity_changed=capacity_changed)


# Explicit low-level name for integrations exposing a separate public runner.
FlashSACLearner = FlashSACRunner
