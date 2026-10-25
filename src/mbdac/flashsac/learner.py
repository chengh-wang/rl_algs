"""Opt-in learner initialization and actor delay for controlled SMP experiments."""
from pathlib import Path
import hashlib
import torch
from .runner import FlashSACRunner, raw_module, _update_networks
from .config import RunnerConfig, UPSTREAM_COMMIT


def checkpoint_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_actor_initialization(path, contract, *, include_normalizer=False, include_value=False, compatibility_keys=None, config_factory=RunnerConfig):
    """Read and validate only the transferable environment/coordinate contract."""
    path = Path(path).resolve()
    state = torch.load(path, map_location='cpu', weights_only=True)
    source = state['contract']
    # Caller owns contract semantics. By default compare every target key; an
    # explicit subset permits objective changes while preserving chosen units.
    for key in (contract.keys() if compatibility_keys is None else compatibility_keys):
        if source.get(key) != contract.get(key):
            raise ValueError(f'Actor initialization contract mismatch: {key}')
    metadata = dict(source_path=str(path), source_sha256=checkpoint_sha(path),
        source_vector_steps=state['vector_steps'], source_update_step=state['update_step'],
        behavior='actor_weights_and_batchnorm_only_fresh_other_learner_state')
    if include_value:
        metadata['behavior'] = 'actor_critic_target_and_reward_units_initialization_for_new_objective_not_resume'
        if state.get('version') != 1 or state.get('upstream_commit') != UPSTREAM_COMMIT:
            raise ValueError('Value source checkpoint version/upstream mismatch')
        units = config_factory(**state['config']).upstream(state['dimensions'][2])
        metadata['value_initialization'] = dict(kind='matched_critic_target_and_reward_units_for_new_objective_not_resume',
            models=['critic', 'target'], reward_statistics=['G_r_max', 'mean', 'var', 'count'],
            reward_units={k:getattr(units,k) for k in ('normalize_reward','normalized_G_max','gamma','n_step','critic_num_bins','critic_min_v','critic_max_v')},
            fresh_episode_return=True, fresh_optimizers_replay_counters_temperature_rng=True)
    normalizer = None
    if include_normalizer:
        side = path.with_suffix('.smp.pt')
        saved = torch.load(side, map_location='cpu', weights_only=True)
        if saved['learner_sha256'] != metadata['source_sha256'] or saved['contract'] != source:
            raise ValueError('Initialization SMP sidecar does not match source learner')
        wrapper = saved['wrapper']
        if 'feature_schema' in contract and wrapper['schema'] != contract['feature_schema']:
            raise ValueError('Initialization normalizer feature schema mismatch')
        normalizer = wrapper['prior_normalizer']
        metadata['normalizer_sidecar_sha256'] = checkpoint_sha(side)
        metadata['normalizer_behavior'] = 'prior_normalizer_only_no_wrapper_or_gsi_state'
    return metadata, state, normalizer


def _check_tensors(target, source, name):
    if target.keys() != source.keys():
        raise ValueError(f'{name} state keys differ')
    for key, value in source.items():
        if not isinstance(value, torch.Tensor) or value.shape != target[key].shape or value.dtype != target[key].dtype:
            raise ValueError(f'{name} tensor shape/dtype mismatch: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'{name} tensor is nonfinite: {key}')


def initialize_actor_only(learner, state, *, prior_normalizer=None, normalizer_state=None):
    """Leave critic, target, temperature, optimizers, replay, RNG and counters fresh."""
    if list(state['dimensions'][:3]) != [learner.actor_dim, learner.critic_dim, learner.action_dim]:
        raise ValueError('Actor initialization dimensions mismatch')
    actor = raw_module(learner.actor)
    weights = state['networks']['actor']['model']
    _check_tensors(actor.state_dict(), weights, 'actor')
    if normalizer_state is not None:
        if prior_normalizer is None:
            raise ValueError('Normalizer destination is required')
        _check_tensors(prior_normalizer.state_dict(), normalizer_state, 'prior normalizer')
        if any((v < 0).any() for v in normalizer_state.values()):
            raise ValueError('Prior normalizer means/counts must be nonnegative')
    actor.load_state_dict(weights, strict=True)
    if normalizer_state is not None:
        prior_normalizer.load_state_dict(normalizer_state, strict=True)


def initialize_value_from_actor_source(learner, state, *, config_factory=RunnerConfig):
    """Transfer value models and their reward units, without resuming learning.

    FlashSACRunner.save stores normalizer={G_r,G_r_max} and
    rms={mean,var,count}. Only G_r_max/RMS transfer; G_r is reset for new envs.
    """
    if state.get('version') != 1 or state.get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Value source checkpoint version/upstream mismatch')
    if list(state['dimensions'][:3]) != [learner.actor_dim, learner.critic_dim, learner.action_dim]:
        raise ValueError('Value source dimensions mismatch')
    source_cfg = config_factory(**state['config']).upstream(learner.action_dim)
    for key in ('normalize_reward', 'normalized_G_max', 'gamma', 'n_step',
                'critic_num_bins', 'critic_min_v', 'critic_max_v'):
        if getattr(source_cfg, key) != getattr(learner.upstream_cfg, key):
            raise ValueError(f'Value source reward normalization config mismatch: {key}')
    norm = learner.normalizer
    if (norm.gamma != source_cfg.gamma or norm.G_max != source_cfg.normalized_G_max or
            norm.epsilon != 1e-8 or norm.load_rms is not True):
        raise ValueError('Destination reward normalizer differs from pinned source implementation')
    models = {}
    for name in ('critic', 'target'):
        model = raw_module(getattr(learner, name))
        weights = state['networks'][name]['model']
        _check_tensors(model.state_dict(), weights, name)
        models[name] = (model, weights)
    if set(state['normalizer']) != {'G_r', 'G_r_max'}:
        raise ValueError('Source reward normalizer keys mismatch')
    source_return = state['normalizer']['G_r']
    if (not isinstance(source_return, torch.Tensor) or source_return.shape != (state['dimensions'][3],)
            or source_return.dtype != norm.G_r.dtype or not torch.isfinite(source_return).all()):
        raise ValueError('Source episode-return tensor mismatch')
    _check_tensors({'G_r_max': norm.G_r_max}, {'G_r_max': state['normalizer']['G_r_max']}, 'reward normalizer')
    rms_target = {name:getattr(norm.G_rms,name) for name in ('mean','var','count')}
    _check_tensors(rms_target, state['rms'], 'reward RMS')
    if ((state['normalizer']['G_r_max'] < 0).any() or (state['rms']['var'] < 0).any()
            or (state['rms']['count'] < 0).any()):
        raise ValueError('Reward normalization max/variance/count must be nonnegative')
    # Validate everything before changing any destination state.
    for model, weights in models.values():
        model.load_state_dict(weights, strict=True)
    norm.G_r_max = state['normalizer']['G_r_max'].to(learner.device).clone()
    for name, value in state['rms'].items():
        setattr(norm.G_rms, name, value.to(learner.device).clone())
    norm.G_r = torch.zeros(learner.num_envs, dtype=norm.G_r.dtype, device=learner.device)


class SMPFlashSACRunner(FlashSACRunner):
    """Base FlashSAC update semantics with an explicit actor/temperature delay."""
    def __init__(self, *args, actor_delay_updates=0, **kwargs):
        if isinstance(actor_delay_updates, bool) or not isinstance(actor_delay_updates, int) or actor_delay_updates < 0:
            raise ValueError('Actor delay must be a nonnegative integer')
        self.actor_delay_updates = actor_delay_updates
        super().__init__(*args, **kwargs)

    def update(self):
        if not self.replay.can_sample():
            raise RuntimeError('Replay warmup is incomplete')
        torch.compiler.cudagraph_mark_step_begin()
        batch = self.batch()
        do_actor = (self.update_step >= self.actor_delay_updates and
                    self.update_step % self.upstream_cfg.actor_update_period == 0)
        metrics = _update_networks(batch, self.actor, self.critic, self.target, self.temperature,
            self.upstream_cfg, do_actor, self.device, self.scaler)
        self.update_step += 1
        metrics['reward/raw'] = batch['raw_reward'].mean()
        metrics['reward/normalized'] = batch['reward'].mean()
        return {k: v.detach().clone() for k, v in metrics.items()}
