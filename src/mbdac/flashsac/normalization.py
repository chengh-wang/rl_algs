"""Opt-in reward units; the vendored legacy numerical path stays untouched."""
import torch
from mbdac.vendor.flashsac.utils.reward_normalization import RewardNormalizer


class EpisodicRewardNormalizer(RewardNormalizer):
    """Accumulate and record the terminal return, then clear only done rows."""
    @torch.no_grad()
    def update_reward_stats(self, reward, terminated, truncated):
        accumulated = self.gamma * self.G_r + reward
        self.G_r_max = torch.maximum(self.G_r_max, accumulated.abs().max())
        self.G_rms.update(accumulated)
        done = terminated | truncated
        self.G_r = torch.where(done, torch.zeros_like(accumulated), accumulated)


class FixedRewardNormalizer(RewardNormalizer):
    """A fixed positive divisor with compatible, intentionally inert RMS state."""
    def __init__(self, *args, scale=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = float(scale)

    @torch.no_grad()
    def update_reward_stats(self, reward, terminated, truncated):
        # No online statistics, episode returns or reward-scale drift.
        pass

    def normalize_rewards(self, rewards):
        return rewards / self.scale


def make_reward_normalizer(cfg, device, num_envs):
    if cfg.reward_normalization == 'legacy':
        return RewardNormalizer(cfg.gamma, 5., True, device)
    cls = EpisodicRewardNormalizer if cfg.reward_normalization == 'episodic' else FixedRewardNormalizer
    kwargs = {'scale': cfg.reward_scale} if cls is FixedRewardNormalizer else {}
    normalizer = cls(cfg.gamma, 5., True, device, **kwargs)
    normalizer.G_r = torch.zeros(num_envs, dtype=torch.float32, device=device)
    return normalizer


def reward_denominator(normalizer, cfg):
    """Common positive divisor used for combined task+prior rewards."""
    if cfg.reward_normalization == 'fixed':
        return torch.as_tensor(cfg.reward_scale, dtype=torch.float32, device=normalizer.device)
    return torch.maximum(torch.sqrt(normalizer.G_rms.var + normalizer.epsilon),
                         normalizer.G_r_max / normalizer.G_max)
