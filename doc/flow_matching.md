# Flow Matching Scored Motion Prior

`mbdac.flow_matching` implements a task-independent **Flow Matching version of
Scored Motion Prior (SMP)**. It learns a conditional velocity field from reference
feature windows, then uses velocity prediction residuals to score new windows.
The module owns the prior algorithm, offline training, scoring and persistent
scoring statistics. Its runtime uses PyTorch; the array CLI also uses NumPy.

## Algorithm

Let a reference feature window be `F` with shape `[W,D]`. Training-only feature
quantiles define normalized data

```text
X = 2 * (F - q_low) / (q_high - q_low + 1e-8) - 1
```

The quantile probabilities and minimum span are configurable; defaults are
q01/q99 and 1e-4. Values are not clipped. For each window, sample Gaussian noise
and a continuous time:

```text
epsilon ~ Normal(0,I), tau ~ Uniform[0,1)
x_tau = (1-tau) * epsilon + tau * X
u = X - epsilon
L(theta) = mean((v_theta(x_tau, tau) - u)**2)
```

The noise endpoint is 0 and the data endpoint is 1. `FlowVelocityModel` is a
continuous-time DiT. Its public input time is always `tau in [0,1]`; optional
`time_embedding_scale` rescales the internal sinusoidal embedding and is part
of the checkpointed model configuration (default 1).

After training, freeze the model. At each explicitly configured scoring time,
sample independent noise and calculate

```text
e_k(X) = mean_W,D((v_theta(x_tau_k, tau_k) - (X-epsilon_k))**2)
s(X) = exp(-ws * mean_k(e_k(X) / max(mu_k, min_value)))
```

Here `mu_k` is a saved count-weighted error mean. `normalize=False` selects
raw mean error for the exponential. The result is a residual-derived style
score. It is not a probability density or task success probability. Sampling a
trajectory with an ODE solver is outside the current public API.

## Generic task boundary

All inputs are finite floating tensors `[N,W,D]`. Window length and feature
width are model configuration, with `W=1` supported. There are no robot names,
joint counts, units, coordinate frames, simulator handles or task rewards in
the algorithm. A task may supply locomotion features, hand/object trajectories
or another reference representation through the same interface.

The caller defines feature ordering, units, frames, sample interval and complete
window validity. `feature_schema` is an optional mapping that records this
layout; `expected_feature_schema` enforces exact metadata equality at load.
The caller also owns reference processing, episode/clip boundaries, history,
terminal handling and reset masks. Supply only complete valid windows. Each
feature layout or motion distribution supplies its own reference windows and
trained prior.

A learner consumes the resulting score through the caller's explicit reward
composition. The algorithm has no policy optimizer, environment adapter,
simulator import or reward time-step convention. It can be used with different
RL learners or evaluated on saved trajectories without a learner.

## Public interfaces

| API | Input | Output |
| --- | --- | --- |
| `FlowVelocityModel` | normalized path point `[B,W,D]`, `tau [B]` | velocity `[B,W,D]` |
| `flow_matching_loss` | model, normalized windows, noise sample count | scalar differentiable MSE |
| `fit_quantiles` | raw training windows, quantile settings | per-feature low/high vectors |
| `train_flow_prior` | raw windows, `FlowTrainingConfig`, optional heldout windows | path to `prior.pt` |
| `FlowMotionPrior.load_checkpoint` | checkpoint, expected dimensions/schema | frozen model and quantiles |
| `FlowMotionScorer.score` | raw windows, noise or generator, explicit statistics update flag | `FlowScore` |

`FlowScore.reward`, `raw_error` and `normalized_error` are `[B]` tensors;
`per_time_error` is raw `[K,B]`, in the configured scoring-time order.

### Train the prior

```python
from mbdac.flow_matching import FlowTrainingConfig, train_flow_prior

# reference_windows: caller-prepared [N,W,D] finite floating tensor.
schema = {"columns": ["feature_a", "feature_b"], "sample_dt": 0.02}
checkpoint = train_flow_prior(
    reference_windows, "logs/my_flow_prior",
    FlowTrainingConfig(epochs=2000, batch_size=1024, num_noise_samples=10),
    feature_schema=schema,
    validation_windows=heldout_windows,  # optional, same W/D layout
    device="cuda:0",
)
```

The convenience trainer fits quantiles on training windows only and uses AdamW.
Defaults are width 256, four heads, two layers, dropout 0, lr 3e-4, weight decay 1e-4,
gradient norm cap 1 and seed 42. Each epoch visits all windows once in shuffled
batches, so optimizer updates are `epochs * ceil(N/batch_size)`.
`report.json` records configuration, actual updates and per-epoch velocity MSE.
Optional validation uses the training quantiles and fixed evaluation noise/time
samples. Split source clips/episodes before preparing heldout windows; the
trainer cannot infer independence from feature tensors.

For custom data pipelines, use `flow_matching_loss` directly with your model,
optimizer and normalized batches. It samples independent continuous times and
noise, with an optional device-matched generator.

### Array training CLI

Save generic feature windows with
`numpy.savez("windows.npz", features=reference_windows_numpy)`:

```bash
python -m mbdac.flow_matching.pretrain \
  --windows /path/to/windows.npz --output logs/my_flow_prior \
  --feature-schema /path/to/layout.json \
  --epochs 2000 --batch-size 1024 --device cuda:0
```

`--feature-schema` is optional JSON metadata. `--validation-windows` accepts a
separate NPZ with the same `features` layout. The CLI exposes all training
configuration fields, including quantile settings and time embedding scale.
No kinematic archive convention, robot schema or retargeting tool is required.

### Score new windows

```python
import torch
from mbdac.flow_matching import FlowMotionPrior, FlowMotionScorer

prior = FlowMotionPrior.load_checkpoint(
    checkpoint, device="cuda:0",
    expected_feature_dim=reference_windows.shape[-1],
    expected_window_size=reference_windows.shape[1],
    expected_feature_schema=schema,
)
scorer = FlowMotionScorer(prior, times=(0.2, 0.5, 0.8), ws=4.0)
rng = torch.Generator(device="cuda:0").manual_seed(123)
training_score = scorer.score(valid_windows, generator=rng, update_normalizer=True)
style_reward = training_score.reward

# Evaluation uses fixed saved scoring statistics.
evaluation_score = scorer.score(eval_windows, generator=rng)
```

Scoring times are arbitrary explicit continuous values in `(0,1)`; their choice
is a research/configuration decision. All model weights stay frozen, and scoring
runs without autograd. The consuming learner learns from the provided reward
through its own update rule.

## Scoring state, randomness and checkpoints

- Statistics updates are opt-in; `update_normalizer=False` is the default.
  Each time index starts with mean 1 / count 0. If requested, its whole batch updates
  the count-weighted mean before scoring. `min_value` defaults to 1e-4.
- `max_count=None` includes all samples. An optional nonnegative integer cap
  accepts a whole batch while the previous count is below the cap; the final
  batch can cross it. A cap of 0 freezes statistics from initialization.
- Save `scorer.normalizer_state_dict()` and restore it with
  `load_normalizer_state_dict`. Times, score weight, normalization mode, floor
  and cap must match. Invalid means/counts and incompatible configurations are
  rejected. A full scorer `state_dict()` also includes the frozen prior.
- Explicit `noise=[K,B,W,D]` permits deterministic comparisons and consumes no
  global RNG. Otherwise, noise is drawn once per time in configured order. An
  optional generator must match the prior device. Save its state separately
  when needed. Offline convenience training seeds global PyTorch RNG.
- Version 2 prior checkpoints contain model configuration/weights, quantiles,
  feature metadata and the linear-path contract. Loading uses
  `weights_only=True` and rejects other formats/versions. The inference
  checkpoint does not save optimizer or simulator state. The convenience
  trainer refuses to overwrite an existing training output.

## Verification

```bash
python -m pytest tests/test_flow_matching.py
python -m examples.flow_matching.standalone --output /tmp/flow_prior_smoke
```

The tests verify native continuous-time path/loss math and gradients, manual
velocity residuals, scoring state, checkpoint contracts, actual small CPU
training, training-only quantiles with heldout data, generic dimensions including
single-frame windows, CLI training and dependency isolation. Tests block the
older motion-prior package, policy learners and simulator imports while training
and scoring this module. Interface tests establish reusability of the algorithm;
policy quality for a new application requires its own evaluation.

The DiT architecture retains its [source attribution and applicable notices](../src/mbdac/flow_matching/PROVENANCE.md).
