"""Dependency pin and explicit experiment configuration (not PPO parameters)."""
from dataclasses import dataclass, replace
import math

UPSTREAM_COMMIT = '87edc9061150ae9e962dd84e6544e27a1554b3ab'
@dataclass(frozen=True)
class RunnerConfig:
    seed: int = 42
    capacity: int = 200_000
    warmup: int = 100_000
    batch_size: int = 2048
    updates_per_step: int = 2
    schedule_updates: int = 100_000
    use_compile: bool = True
    compile_mode: str = 'max-autotune'
    use_amp: bool = True
    float32_matmul_precision: str = 'highest'
    device: str = 'cuda:0'
    target_entropy: float | None = None
    initial_temperature: float | None = None
    target_sigma: float | None = None

    gamma: float = .99
    critic_min_v: float = -5.
    critic_max_v: float = 5.
    critic_num_bins: int = 101
    reward_normalization: str = 'legacy'
    reward_scale: float = 1.
    critic_diagnostics: bool = False

    def __post_init__(self):
        if isinstance(self.gamma, bool) or not math.isfinite(self.gamma) or not 0 < self.gamma <= 1:
            raise ValueError('gamma must be finite in (0,1]')
        if (not math.isfinite(self.critic_min_v) or not math.isfinite(self.critic_max_v)
                or self.critic_min_v >= self.critic_max_v):
            raise ValueError('Critic support must have finite increasing endpoints')
        if isinstance(self.critic_num_bins, bool) or not isinstance(self.critic_num_bins, int) or self.critic_num_bins < 2:
            raise ValueError('critic_num_bins must be an integer >= 2')
        if self.reward_normalization not in ('legacy', 'episodic', 'fixed'):
            raise ValueError('Unknown reward normalization mode')
        if isinstance(self.reward_scale, bool) or not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError('reward_scale must be finite and positive')
        if self.reward_normalization != 'fixed' and self.reward_scale != 1.:
            raise ValueError('reward_scale is a fixed-mode divisor; online modes require 1')
        if not isinstance(self.critic_diagnostics, bool):
            raise ValueError('critic_diagnostics must be bool')

    def upstream(self, action_dim):
        from mbdac.vendor.flashsac.agent import FlashSACConfig
        if not (0 < self.warmup <= self.capacity and self.batch_size > 1
                and self.updates_per_step > 0 and self.schedule_updates > 0):
            raise ValueError('Invalid replay/schedule configuration')
        for name in ('target_entropy', 'initial_temperature', 'target_sigma'):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or (name != 'target_entropy' and value <= 0)):
                raise ValueError(f'Invalid {name}')
        result = FlashSACConfig(
            seed=self.seed, normalize_reward=True, normalized_G_max=5.,
            asymmetric_observation=True, device_type=self.device,
            buffer_max_length=self.capacity, buffer_min_length=self.warmup,
            buffer_device_type=self.device, sample_batch_size=self.batch_size,
            learning_rate_init=3e-4, learning_rate_peak=3e-4, learning_rate_end=1.5e-4,
            learning_rate_warmup_rate=1e-6, learning_rate_warmup_step=int(1e-6*self.schedule_updates),
            learning_rate_decay_rate=1., learning_rate_decay_step=self.schedule_updates,
            actor_num_blocks=2, actor_hidden_dim=128, actor_bc_alpha=0.,
            actor_noise_zeta_mu=2., actor_noise_zeta_max=16, actor_update_period=2,
            critic_num_blocks=2, critic_hidden_dim=256, critic_num_bins=self.critic_num_bins,
            critic_min_v=self.critic_min_v, critic_max_v=self.critic_max_v, critic_target_update_tau=.01,
            temp_initial_value=.01, temp_target_sigma=.15,
            temp_target_entropy=.5*action_dim*math.log(2*math.pi*math.e*.15**2),
            gamma=self.gamma, n_step=1,  # Avoid upstream fixed-gamma^n early-truncation mismatch.
            use_compile=self.use_compile, compile_mode=self.compile_mode, use_amp=self.use_amp,
            load_optimizer=True, load_reward_normalizer=True,
        )

        overrides = {}
        if self.critic_diagnostics:
            overrides['critic_diagnostics'] = True
        if self.target_sigma is not None:
            overrides['temp_target_sigma'] = self.target_sigma
            overrides['temp_target_entropy'] = .5*action_dim*math.log(2*math.pi*math.e*self.target_sigma**2)
        if self.target_entropy is not None:
            overrides['temp_target_entropy'] = self.target_entropy
        if self.initial_temperature is not None:
            overrides['temp_initial_value'] = self.initial_temperature
        return replace(result, **overrides)
