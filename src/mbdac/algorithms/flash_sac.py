"""Model-free learners; the public VecEnv runners live in ``mbdac.runners``.

These aliases expose the numerical learner without imposing MBDAC's world-model
task contract. Construction takes dimensions, a RunnerConfig and an opaque task
contract; collection and simulator lifecycle belong to the caller.
"""
from mbdac.flashsac.runner import FlashSACLearner as FlashSAC
from mbdac.flashsac.learner import SMPFlashSACRunner as SMPFlashSAC

__all__ = ['FlashSAC', 'SMPFlashSAC']
