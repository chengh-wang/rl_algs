"""Read-only C51 support diagnostics in the critic's existing reward units."""
import torch


@torch.no_grad()
def categorical_support_diagnostics(target_log_probs, reward, terminated, entropy,
                                    gamma, min_v, max_v, pred_log_probs, projected_probs):
    """Measure probability mass, not unweighted atom counts.

    Inputs are tensors already evaluated by the critic update. ``entropy`` is
    its existing alpha*log_prob term (despite that upstream name). No sampling,
    forward pass, normalization update, clipping repair or training loss occurs.
    Strict inequalities identify mass changed by support clipping; mass exactly
    at either support endpoint is valid in-support mass.
    """
    probabilities = target_log_probs.detach().float().exp()
    bins = torch.linspace(min_v, max_v, probabilities.shape[-1],
                          device=probabilities.device, dtype=probabilities.dtype)
    values = (reward.detach().float().reshape(-1, 1)
              + gamma * (bins - entropy.detach().float().reshape(-1, 1))
              * (1 - terminated.detach().float().reshape(-1, 1)))
    low = (probabilities * (values < min_v)).sum(-1)
    high = (probabilities * (values > max_v)).sum(-1)
    prediction = pred_log_probs.detach().float().exp()
    projection = projected_probs.detach().float()
    return {
        'c51/target_below_support_mass':low.mean(),
        'c51/target_above_support_mass':high.mean(),
        'c51/target_clipped_mass':(low+high).mean(),
        'c51/target_unclipped_mean':(probabilities*values).sum(-1).mean(),
        'c51/target_clamped_mean':(probabilities*values.clamp(min_v,max_v)).sum(-1).mean(),
        'c51/projected_lower_edge_mass':projection[...,0].mean(),
        'c51/projected_upper_edge_mass':projection[...,-1].mean(),
        'c51/projected_mean':(projection*bins).sum(-1).mean(),
        'c51/predicted_lower_edge_mass':prediction[...,0].mean(),
        'c51/predicted_upper_edge_mass':prediction[...,-1].mean(),
        'c51/predicted_mean':(prediction*bins).sum(-1).mean(),
    }
