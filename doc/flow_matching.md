# Flow Matching motion prior

`mbdac.flow_matching` is a standalone, tensor-only motion prior and residual
scorer. It imports no simulator, robot task, FlashSAC learner or runner. Any
learner can consume its scalar scores. This is a local Flow Matching variant of
the SMP scoring idea, not the original diffusion SMP algorithm or an official
upstream release.

## Responsibility and public interface

| Component | Input | Output / responsibility |
| --- | --- | --- |
| `FlowVelocityModel` | normalized path point `[B,W,F]`, continuous time `50*tau` `[B]` | conditional velocity `[B,W,F]` |
| `flow_matching_loss` | model, normalized windows `[B,W,F]` | scalar velocity MSE with gradients |
| `train_flow_prior` | raw windows `[N,W,F]`, `FlowTrainingConfig`, output directory | path to a new `prior.pt` |
| `FlowMotionPrior.load_checkpoint` | checkpoint, expected dimensions/schema | frozen network and training quantiles |
| `FlowMotionScorer` | prior, explicit continuous times, scoring settings | stateful scorer independent of any learner |
| `scorer.score` | raw windows `[B,W,F]`, optional noise or generator | `FlowScore` tensors described below |

`B` is batch size, `W` is window length, `F` is feature dimension; none is fixed
to R1, a hand or a particular simulator. Inputs must be finite floating tensors.
Feature order, units, frames and sample interval must match training. Store them
in `feature_schema` and pass `expected_feature_schema` when loading. This metadata
is an equality contract; the prior cannot infer the physical meaning of a tensor.

The task owns state reading, kinematics/retargeting, history, reset masks,
terminal observations, reward composition, action mapping and physics. Supply
only complete valid windows; do not score reset padding or concatenate different
episodes. Score a terminal window before clearing its history. The module never
combines task rewards, multiplies by a simulator time step or initializes a robot.

## Offline training

For raw training windows `F`, training-only per-feature q01/q99 scaling is

```text
X = 2 * (F - q_low) / (q_high - q_low + 1e-8) - 1
```

The upper quantile is raised to at least `q_low + span_floor`; values are not
clipped. Every supplied window is a training sample. Split reference clips before
window extraction if validation is needed; overlapping windows are not an
independent validation set.

For each normalized window, independently draw `epsilon ~ N(0,I)` and
`tau ~ Uniform[0,1)`:

```text
x_tau = (1 - tau) * epsilon + tau * X
u = X - epsilon
loss = mean((v_theta(x_tau, 50*tau) - u)**2)
```

Zero is the noise endpoint; one is the data endpoint. Time stays continuous.
The model reuses the existing SMP DiT architecture and state keys, trained with
a new velocity target. Diffusion epsilon weights cannot be substituted for a
trained Flow checkpoint.

```python
from mbdac.flow_matching import FlowTrainingConfig, train_flow_prior

# windows: your prepared finite float tensor [N,W,F], in a documented layout.
checkpoint = train_flow_prior(
    windows, "logs/my_flow_prior",
    FlowTrainingConfig(epochs=2000, batch_size=1024, num_noise_samples=10),
    feature_schema={"layout": "my_motion_v1", "sample_dt": 0.02},
    device="cuda:0",
)
```

Defaults: DiT width256, four heads, two layers, dropout0; AdamW lr3e-4,
weight decay1e-4, gradient norm cap1, seed42, span floor1e-4. One epoch visits all
supplied windows in shuffled batches. Updates per epoch are `ceil(N/batch_size)`;
the default2000 epochs do not generally mean2000 updates. This offline function
seeds and uses PyTorch's global RNG; it is not a background learner callback.

For already retargeted kinematic NPZ clips and an explicit `MotionFeatureSchema`,
the separate convenience CLI builds clip-local stride-one windows:

```bash
python -m mbdac.flow_matching.pretrain \
  --references /path/to/reference_manifest.json \
  --feature-schema /path/to/feature_schema.json \
  --output logs/my_flow_prior --device cuda:0 --epochs 2000
```

The CLI reads arrays and names; no simulator runs. The
[existing motion preparation guide](smp_flashsac.md#smp-的明确组合方式) describes
the kinematic input/schema boundary. Raw motion retargeting belongs outside the
algorithm module.

## Frozen online scoring

```python
import torch
from mbdac.flow_matching import FlowMotionPrior, FlowMotionScorer, legacy_score_times

prior = FlowMotionPrior.load_checkpoint(
    checkpoint, device="cuda:0",
    expected_feature_dim=windows.shape[-1],
    expected_window_size=windows.shape[1],
    expected_feature_schema={"layout": "my_motion_v1", "sample_dt": 0.02},
)
scorer = FlowMotionScorer(prior, times=legacy_score_times(), ws=4.0)
noise_rng = torch.Generator(device="cuda:0").manual_seed(123)
result = scorer.score(valid_windows, generator=noise_rng, update_normalizer=True)
style_reward = result.reward  # [B]; combine with task rewards in your adapter
```

You may choose continuous `times=(0.2, 0.5, 0.8)` explicitly. To reproduce the
trial's historical score locations, `legacy_score_times((8,15,22))` maps cosine50
DDPM labels by `tau=sqrt(alpha_bar)/(sqrt(alpha_bar)+sqrt(1-alpha_bar))`, yielding
approximately `(0.768424, 0.640753, 0.528123)`. This preserves signal/noise ratio,
not equivalence of DDPM and Flow errors. These are score locations, not ODE steps.

At each configured time, draw independent noise, construct the linear path and
compute

```text
e_k = mean_W,F((v_theta(x_tau_k, 50*tau_k) - (X-epsilon_k))**2)
s = exp(-ws * mean_k(e_k / max(mu_k, min_value)))
```

`FlowScore.reward`, `raw_error` and `normalized_error` have shape `[B]`;
`per_time_error` is raw `[K,B]`. `normalize=False` selects raw mean error for
the exponential. The scalar is a residual-derived style reward, **not a density,
likelihood, catching probability or guarantee of natural motion**. The module
does not currently implement Flow ODE sampling/GSI.

Model weights remain frozen, and scoring runs under `no_grad`: the RL policy
learns from the reward through its own algorithm, not by backpropagating the
Flow network into actions. Only explicitly requested normalizer updates mutate
scoring state. `update_normalizer=False` is the default, including evaluation.
If requested, update each time's count-weighted mean before dividing. Counts
stop updating once the previous count is strictly greater than50,000,000;
the last whole batch may cross this cap. The denominator floor defaults to1e-4.

## Randomness, persistence and compatibility

- For deterministic comparison supply `noise=[K,B,W,F]`. This consumes no global
  RNG; it cannot accompany `generator`. Otherwise a generator must match the
  prior device; noise is drawn in configured time order, one `[B,W,F]` draw per
  time. Save its state separately if your experiment needs continuation.
- Save `scorer.normalizer_state_dict()` alongside learner state. Restore with
  `load_normalizer_state_dict`; mismatched times, weight, normalization mode,
  floor or cap are rejected. Save the feature-history/reset state in the task.
  The scorer's full `state_dict()` also contains the frozen model.
- New prior checkpoints declare `checkpoint_version=1`. The original unversioned
  linear-flow v1 checkpoints remain loadable. Loading uses `weights_only=True`,
  checks the path/type/architecture and rejects unsupported versions. Optimizer
  and RNG fields are provenance/training metadata; no resumable-training API or
  simulator-state restoration is promised. Training refuses to overwrite an
  existing `prior.pt`.
- `FrozenFlowPrior` remains the legacy v1 compatibility API. It retains
  `fixed_timesteps=(8,15,22)`, the50-slot normalizer and `scheduler.num_timesteps`
  expected by the old experiment adapter. New integrations should use
  `FlowMotionPrior` + `FlowMotionScorer`, whose public interface has no scheduler,
  learner or environment dependency. Normalizer indices differ; do not load
  a legacy50-slot normalizer directly into the new K-slot scorer.

The frozen Flow prior and the optional reset generator are separate capabilities.
Using the Flow scorer does not replace an experiment's DDPM GSI or task reset
implementation.

## Verification and research status

Run from a source checkout with the dev extra:

```bash
python -m pytest tests/test_flow_matching.py
python -m examples.flow_matching.standalone --output /tmp/flow_prior_smoke
```

Tests cover path/loss math, explicit-noise residuals, legacy score parity,
normalizer persistence/configuration, checkpoint validation, real small CPU
training and import independence. They verify the interface; they do not certify
policy quality. The single-seed R1/Dex3 V9 trial that motivated this module did
not improve the final52k policy over its matched diffusion baseline. This module
is provided for controlled research, without a general performance claim.

The model and shared SMP utilities retain their existing
[source attribution and licensing limits](../src/mbdac/smp/PROVENANCE.md).
Our linear-path objective, standalone API and scoring/training orchestration are
local additions; the upstream SMP snapshot is not relicensed by this addition.
