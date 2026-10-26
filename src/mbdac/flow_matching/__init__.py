"""Simulator-independent continuous Flow Matching SMP and legacy v1 bridge."""
from .prior import FrozenFlowPrior, PATH_CONTRACT, linear_path
from .model import FlowVelocityModel
from .scoring import FlowMotionPrior, FlowMotionScorer, FlowScore, legacy_score_times
from .training import FlowTrainingConfig, train_flow_prior, flow_matching_loss, fit_quantiles

__all__ = ['FlowVelocityModel', 'FlowMotionPrior', 'FlowMotionScorer', 'FlowScore',
           'FlowTrainingConfig', 'train_flow_prior', 'flow_matching_loss', 'fit_quantiles',
           'legacy_score_times', 'FrozenFlowPrior', 'PATH_CONTRACT', 'linear_path']
