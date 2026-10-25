"""RSL-style model-free FlashSAC runners; no simulator or WM dependency.

SMP rewards are explicitly either already composed by the environment, or supplied
by a task-owned stateful reward provider. This runner never infers physics or
motion features. Potential-based shaping requires a fixed reward denominator:
changing the denominator over time destroys its telescoping identity.
Checkpoints contain learner/persistent SMP state, not simulator
state or live policy/SMP episode histories; load starts fresh collection.
"""
from __future__ import annotations
import copy
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
import os
import json
import random
import time
import numpy as np
import torch
from torch import nn
from mbdac.adapters.model_free import observation_groups, real_transition


class InferencePolicy(nn.Module):
    def __init__(self, actor, actor_groups):
        super().__init__()
        self.actor = actor
        self.actor_groups = actor_groups

    def reset(self, dones=None):
        """Stateless actor; native environment owns any observation history."""

    @torch.no_grad()
    def forward(self, obs):
        value = torch.cat([obs[k] for k in self.actor_groups], -1) if isinstance(obs, dict) or hasattr(obs, 'keys') else obs
        mean, _ = self.actor.get_mean_and_std(observations=value, training=False)
        return torch.tanh(mean)


class FlashSACRunner:
    """Public runner compatible with the existing RSL constructor/method shape.

    ``algorithm`` (besides class_name) or ``learner`` contains RunnerConfig fields. ``obs_groups`` maps actor/critic
    to ordered lists of 2-D TensorDict/dict keys. ``task_contract`` is a portable
    opaque task/action/observation schema, validated exactly on restore.
    One learning iteration collects one vector step and performs updates_per_step
    updates after replay warmup. Inference returns normalized policy actions.
    """
    is_smp = False

    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        from mbdac.flashsac.config import RunnerConfig
        from mbdac.flashsac.runner import FlashSACRunner as Learner
        self.env, self.device = env, torch.device(device)
        self.cfg = copy.deepcopy(train_cfg)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir: self.log_dir.mkdir(parents=True, exist_ok=True)
        self.groups = self.cfg.get('obs_groups', {'actor': ['actor'], 'critic': ['critic']})
        if set(self.groups) != {'actor', 'critic'} or any(not isinstance(v, (list, tuple)) or not v or any(not isinstance(k, str) for k in v) for v in self.groups.values()):
            raise ValueError('obs_groups requires nonempty ordered actor and critic key lists')
        self.groups = {k:list(v) for k,v in self.groups.items()}
        self._obs = observation_groups(env.get_observations(), self.groups, env.num_envs, self.device)
        algorithm = dict(self.cfg.get('algorithm', {}))
        expected = 'SMPFlashSAC' if self.is_smp else 'FlashSAC'
        name = algorithm.pop('class_name', expected).split(':')[-1].split('.')[-1]
        if name not in (expected, expected + 'Runner'):
            raise ValueError(f'Expected algorithm.class_name={expected}')
        delay = algorithm.pop('actor_delay_updates', 0)
        options = dict(algorithm)
        for key, value in self.cfg.get('learner', {}).items():
            if key in options and options[key] != value:
                raise ValueError(f'Conflicting learner configuration: {key}')
            options[key] = value
        if delay and not self.is_smp:
            raise ValueError('actor_delay_updates belongs to SMPFlashSAC')
        allowed = {f.name for f in fields(RunnerConfig)}
        if set(options)-allowed: raise ValueError(f'Unknown learner options: {set(options)-allowed}')
        options['device'] = str(self.device)
        self.learner_cfg = RunnerConfig(**options)
        # Apply the configured seed before constructing any numerical modules.
        torch.manual_seed(self.learner_cfg.seed)
        random.seed(self.learner_cfg.seed)
        np.random.seed(self.learner_cfg.seed)
        self.log_interval = self.cfg.get('log_interval', 10)
        self.save_interval = self.cfg.get('save_interval', 500)
        for name, value in (('log_interval',self.log_interval),('save_interval',self.save_interval)):
            if isinstance(value,bool) or not isinstance(value,int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        self.writer = self.wandb = None
        self.elapsed_s = 0.
        self._active_start = None
        logger = self.cfg.get('logger', 'json')
        if logger not in ('json','tensorboard','wandb'):
            raise ValueError('logger must be json, tensorboard or wandb')
        if self.log_dir:
            (self.log_dir/'config.json').write_text(json.dumps(self.cfg,indent=2,default=str)+'\n')
            if logger == 'tensorboard':
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.log_dir))
            elif logger == 'wandb':
                import wandb
                self.wandb = wandb.init(project=self.cfg.get('wandb_project','flashsac'),
                    name=self.cfg.get('run_name'),dir=str(self.log_dir),config=self.cfg)
        self.contract = dict(task=copy.deepcopy(self.cfg.get('task_contract', {})), obs_groups=self.groups,
                             runner='SMPFlashSAC' if self.is_smp else 'FlashSAC')
        self.reward_provider = None
        self.smp_source = None
        self.actor_delay_updates = 0
        if self.is_smp:
            smp = self.cfg.get('smp', {})
            self.smp_source = smp.get('reward_source')
            self.actor_delay_updates = smp.get('actor_delay_updates', delay)
            if (isinstance(self.actor_delay_updates,bool) or not isinstance(self.actor_delay_updates,int)
                    or self.actor_delay_updates < 0):
                raise ValueError('actor_delay_updates must be a nonnegative integer')
            if self.smp_source == 'environment':
                if not isinstance(getattr(env, 'smp_contract', None), dict):
                    raise ValueError('Environment-composed SMP requires env.smp_contract')
                if callable(getattr(env, 'smp_state_dict', None)) != callable(getattr(env, 'load_smp_state_dict', None)):
                    raise ValueError('Supply smp_state_dict and load_smp_state_dict together')
                self.contract['smp'] = copy.deepcopy(env.smp_contract)
            elif self.smp_source == 'reward_provider':
                self.reward_provider = getattr(env, 'smp_reward_provider', None)
                if not callable(self.reward_provider) or not all(callable(getattr(self.reward_provider, k, None)) for k in ('state_dict', 'load_state_dict', 'reset')):
                    raise ValueError('Reward provider must be callable, checkpointable and expose reset()')
                if not isinstance(getattr(self.reward_provider, 'contract', None), dict):
                    raise ValueError('Reward provider requires a portable contract dict')
                self.contract['smp'] = copy.deepcopy(self.reward_provider.contract)
            else:
                raise ValueError('SMP requires explicit environment or reward_provider reward_source')
            self.contract.update(smp_reward_source=self.smp_source, actor_delay_updates=self.actor_delay_updates)
            from mbdac.flashsac.learner import SMPFlashSACRunner as Learner
        # Generic composition is opt-in and separate from the legacy SMP path.
        # Absent mode leaves historical public checkpoint contracts unchanged.
        self.generic_reward_provider = None
        reward_cfg = self.cfg.get('reward')
        if reward_cfg is not None:
            if not isinstance(reward_cfg, dict) or reward_cfg.get('source') != 'provider':
                raise ValueError("Generic reward requires reward={'source':'provider'}")
            if set(reward_cfg) != {'source'}:
                raise ValueError('Provider settings belong to its explicit contract, not runner reward config')
            if self.is_smp:
                raise ValueError('Generic reward provider cannot be combined with SMP reward composition')
            provider = getattr(env, 'reward_provider', None)
            if not callable(provider) or not all(callable(getattr(provider,k,None))
                    for k in ('state_dict','load_state_dict','reset')):
                raise ValueError('Generic reward_provider must be callable, checkpointable and expose reset()')
            contract = getattr(provider,'contract',None)
            if not isinstance(contract,dict):
                raise ValueError('Generic reward_provider requires a portable contract dict')
            if (contract.get('kind') == 'potential_based_shaping'
                    and self.learner_cfg.reward_normalization != 'fixed'):
                raise ValueError("potential_based_shaping requires reward_normalization='fixed'; "
                                 "a changing denominator destroys telescoping")
            discount = contract.get('algorithm_discount')
            if (isinstance(discount,bool) or not isinstance(discount,(int,float))
                    or not np.isfinite(discount) or discount != self.learner_cfg.gamma):
                raise ValueError('Reward provider algorithm_discount must match learner gamma')
            if contract.get('next_potential_semantics') != 'pre_reset':
                raise ValueError('Reward provider must declare pre_reset next potential semantics')
            if contract.get('normalization_base') != 'task_reward':
                raise ValueError('Reward provider must declare task_reward normalization base')
            self.generic_reward_provider = provider
            self.contract['reward_provider'] = copy.deepcopy(contract)
        args = (self._obs['actor'].shape[-1], self._obs['critic'].shape[-1], env.num_actions,
                env.num_envs, self.learner_cfg, self.contract)
        self.alg = Learner(*args, **({'actor_delay_updates':self.actor_delay_updates} if self.is_smp else {}))
        self.current_learning_iteration = 0
        self.last_load_info = None

    @property
    def real_transitions(self): return self.alg.transitions

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if isinstance(num_learning_iterations, bool) or not isinstance(num_learning_iterations, int) or num_learning_iterations < 0:
            raise ValueError('num_learning_iterations must be a nonnegative integer')
        if init_at_random_ep_len:
            randomize = getattr(self.env, 'randomize_episode_lengths', None)
            if randomize is None: raise ValueError('Adapter must explicitly support random episode lengths')
            randomize()
            self._obs = observation_groups(self.env.get_observations(), self.groups, self.env.num_envs, self.device)
        metrics = {}
        self._active_start = time.perf_counter()
        try:
            for iteration_index in range(num_learning_iterations):
                started = time.perf_counter()
                action = self.alg.actions(self._obs['actor'])
                obs, rewards, dones, extras = self.env.step(action)
                after = observation_groups(obs, self.groups, self.env.num_envs, self.device)
                task_reward = None
                if self.generic_reward_provider is not None:
                    task_reward = torch.as_tensor(rewards,device=self.device).detach().clone()
                    payload = extras.get('reward_potential')
                    if not isinstance(payload,Mapping) or not {'current','next','version'} <= payload.keys():
                        raise ValueError('Reward provider requires current and pre-reset next reward_potential endpoints')
                    rewards = self.generic_reward_provider(rewards,extras)
                if self.reward_provider is not None:
                    # Provider owns valid-history masking and returns composed rewards.
                    rewards = self.reward_provider(rewards, extras)
                transition = real_transition(self._obs, after, action, rewards, dones, extras, self.groups, self.device)
                if self.generic_reward_provider is not None:
                    transition['task_reward'] = task_reward
                self.alg.observe(transition)
                self._obs = after
                collected = time.perf_counter()
                metrics = {}
                if self.alg.replay.can_sample():
                    for _ in range(self.learner_cfg.updates_per_step): metrics.update(self.alg.update())
                    if any(not torch.isfinite(v).all() for v in metrics.values()): raise RuntimeError('Nonfinite learner metrics')
                self.current_learning_iteration += 1
                if self.current_learning_iteration % self.log_interval == 0 or iteration_index == num_learning_iterations-1:
                    logs = {k:float(v.detach().float().mean()) for k,v in metrics.items()}
                    logs.update(iteration=self.current_learning_iteration, real_transitions=self.alg.transitions,
                        vector_steps=self.alg.vector_steps, update_steps=self.alg.update_step,
                        sim_reward=float(transition['reward'].mean()), wall_s=self._elapsed(),
                        **{'time/collect_wall_s':collected-started,
                           'time/update_wall_s':time.perf_counter()-collected})
                    if self.generic_reward_provider is not None:
                        logs.update(task_reward=float(task_reward.mean()),
                                    learner_reward=float(transition['reward'].mean()),
                                    sim_reward=float(task_reward.mean()))
                    self._log(logs)
                if self.log_dir and self.current_learning_iteration % self.save_interval == 0:
                    self.save(self.log_dir/f'model_{self.current_learning_iteration}.pt')
        finally:
            self.elapsed_s = self._elapsed()
            self._active_start = None
            if self.writer: self.writer.flush()
        if self.log_dir and (num_learning_iterations == 0 or self.current_learning_iteration % self.save_interval):
            self.save(self.log_dir/f'model_{self.current_learning_iteration}.pt')
        return metrics

    def _elapsed(self):
        return self.elapsed_s + (time.perf_counter()-self._active_start if self._active_start is not None else 0.)

    def _log(self, logs):
        if self.log_dir:
            with (self.log_dir/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(logs,sort_keys=True)+'\n')
        if self.writer:
            for name,value in logs.items(): self.writer.add_scalar(name,value,self.current_learning_iteration)
        if self.wandb: self.wandb.log(logs,step=self.current_learning_iteration)

    def _smp_state(self):
        if self.reward_provider is not None: return self.reward_provider.state_dict()
        method = getattr(self.env, 'smp_state_dict', None)
        return method() if self.is_smp and method is not None else None

    def save(self, path, infos=None, include_replay=False):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name+'.public.tmp')
        self.alg.save(temporary, include_replay=include_replay)
        state = torch.load(temporary, map_location='cpu', weights_only=True)
        state['public_runner'] = dict(version=1, iteration=self.current_learning_iteration, elapsed_s=self._elapsed(), infos=infos,
                                      smp_state=self._smp_state(), simulator_state_saved=False)
        if self.generic_reward_provider is not None:
            state['public_runner']['reward_provider_state'] = self.generic_reward_provider.state_dict()
        torch.save(state, temporary)
        os.replace(temporary, path)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None and set(load_cfg) - {'rng'}:
            raise ValueError('Only load_cfg.rng is supported; full learner restore is otherwise required')
        if not strict: raise ValueError('Non-strict learner contract restore is unsupported')
        state = torch.load(path, map_location=map_location or 'cpu', weights_only=True)
        public = state.get('public_runner', {})
        if public.get('version') != 1 or state.get('contract') != self.contract:
            raise ValueError('Public runner checkpoint contract mismatch')
        # Live env is not checkpointed. Import persistent state, then reset
        # collection and all task-owned episode histories.
        if self.generic_reward_provider is not None and 'reward_provider_state' not in public:
            raise ValueError('Checkpoint missing generic reward provider state')
        self.last_load_info = self.alg.load(path, restore_rng=(load_cfg or {}).get('rng', True))
        saved_smp = public.get('smp_state')
        if saved_smp is not None:
            if self.reward_provider is not None: self.reward_provider.load_state_dict(saved_smp)
            else:
                loader = getattr(self.env, 'load_smp_state_dict', None)
                if loader is None: raise ValueError('SMP checkpoint requires env.load_smp_state_dict')
                loader(saved_smp)
        if self.generic_reward_provider is not None:
            if 'reward_provider_state' not in public:
                raise ValueError('Checkpoint missing generic reward provider state')
            self.generic_reward_provider.load_state_dict(public['reward_provider_state'])
        reset = getattr(self.env, 'reset', None)
        if reset is None: raise ValueError('Fresh-environment restore requires env.reset()')
        reset()
        if self.generic_reward_provider is not None:
            self.generic_reward_provider.reset()
        if self.reward_provider is not None:
            self.reward_provider.reset()
        self._obs = observation_groups(self.env.get_observations(), self.groups, self.env.num_envs, self.device)
        self.current_learning_iteration = public['iteration']
        self.elapsed_s = float(public.get('elapsed_s',0.))
        return public.get('infos')

    def get_inference_policy(self, device=None):
        from mbdac.flashsac.runner import raw_module
        actor = copy.deepcopy(raw_module(self.alg.actor))
        # Upstream installs torch.compile(bound_method) on the actor instance.
        # deepcopy keeps that function's closure bound to the live training
        # actor. Drop runtime method overrides on the COPY before device moves,
        # so Python binds class methods to the independent snapshot instead.
        for module in actor.modules():
            for name, value in tuple(vars(module).items()):
                if (callable(value) and callable(getattr(type(module), name, None))
                        and (name == 'get_mean_and_std'
                             or hasattr(value, '_torchdynamo_orig_callable'))):
                    delattr(module, name)
            # nn.Module.compile can also cache a compiled bound __call__.
            module._compiled_call_impl = None
        actor = actor.to(device or self.device).eval()
        return InferencePolicy(actor, self.groups['actor'])

    def export_jit(self, path):
        policy = self.get_inference_policy('cpu')
        inputs = torch.zeros(1, self._obs['actor'].shape[-1])
        exported = torch.jit.trace(policy, inputs)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); exported.save(str(path))
        return exported

    def close(self):
        """Close loggers; caller owns the environment lifecycle."""
        if self.writer: self.writer.close()
        if self.wandb: self.wandb.finish()


class SMPFlashSACRunner(FlashSACRunner):
    """FlashSAC with explicit task-owned SMP composition and optional actor delay."""
    is_smp = True
