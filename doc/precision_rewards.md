# Precision tasks: reward objectives and FlashSAC controls

These opt-in controls are independent of robot, simulator, observation dimension
and action meaning. They do not change an MJLab task, its physics, native
rewards or reset timing. The existing `SMPFlashSACRunner` additive composition
remains available with its historical defaults and checkpoint contracts.

## What is being optimized?

A fixed weighted sum maximizes the sum, not task precision first. For the scalar
example

\[
r_{task}(a)=-a^2,\quad S(a)=(a+1)/2,\quad
r(a)=r_{task}(a)+0.4S(a),
\]

the optimum is `a=0.1`, although the task optimum is `a=0`. Multiplication does
not generally solve this either: `exp(-a²)*(a+1)/2` has its maximum at
`a=(sqrt(3)-1)/2`, about `0.366`. Multiplying signed penalties can additionally
weaken those penalties. Reward composition is an objective choice, not simply
a numerical tuning choice.

The [SMP reproduction](https://github.com/SUZ-tsinghua/smp) explicitly describes
its multiplicative reward as a difference from the original additive approach.
Neither formula is a general guarantee of task priority.

## Optional potential-based shaping

The new `mbdac.rewards` module supplies a pure Torch differentiable composition:

\[
r'_t=r^{task}_t+\beta[\gamma(1-d^{term}_t)\Phi(h_{t+1})-\Phi(h_t)].
\]

Here `h` may include a motion-feature history. It must be a consistent state
representation for the intended control problem. The task reward includes all
task costs and penalties; they are never multiplied by the motion score.
For a bounded potential and matching discount, the discounted shaping return
telescopes to `-beta*Phi(h0)` plus a vanishing boundary term. With finite
episodes the terminal potential is zero. Thus the ideal task optimum is
unchanged under these conditions, following
[Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

For a fixed entropy coefficient the same state-dependent value shift also
preserves the ideal soft policy: `Q'=Q-beta*Phi(h)` leaves action differences
at a state unchanged. This is an extension by direct substitution in the soft
Bellman equation, not an empirical claim from that paper. SAC still optimizes
return and entropy together, as described by
[Haarnoja et al.](https://arxiv.org/abs/1812.05905). Neither deterministic
inference nor automatic temperature tuning implies zero task error.

**This shaping does not retain a permanent preference for motion style.** A
motion prior used as a potential can guide finite learning, but its total
contribution cancels as above. A persistent style objective subject to a task
error budget requires a separately specified constrained or lexicographic
learner. This implementation does not claim to implement that learner, ensure
safety, or prove precision/contact-task generalization with neural networks.

### Public runner contract

Use `FlashSACRunner` with an explicit provider; do not also enable the legacy
SMP reward composer:

```python
from mbdac.runners import FlashSACRunner
from mbdac.rewards import PotentialRewardProvider, PotentialShapingConfig

env.reward_provider = PotentialRewardProvider(
    PotentialShapingConfig(
        potential_id="frozen_motion_score", version="checkpoint-noise-stats-v1",
        gamma=0.99, beta=0.5, max_abs_potential=1.0,
    ),
    next_potential_is_pre_reset=True,
)
cfg = {
    "algorithm": {
        "class_name": "FlashSAC", "gamma": 0.99,
        "reward_normalization": "fixed", "reward_scale": 1.0,
        "critic_min_v": -5.0, "critic_max_v": 5.0,
        "critic_num_bins": 101, "critic_diagnostics": True,
    },
    "reward": {"source": "provider"},
}
runner = FlashSACRunner(env, cfg, device="cuda:0")
```

Each `env.step()` supplies floating reward `[N]`, explicit boolean
`extras['terminated']` and `extras['truncated']`, and:

```python
extras["reward_potential"] = {
    "current": phi_current,   # float [N], same dtype/device as reward
    "next": phi_next,         # TRUE pre-reset endpoint, including timeouts
    "version": "checkpoint-noise-stats-v1",
}
```

The adapter remains responsible for physical state, frames, feature history,
and pre-reset actor/critic terminal observations. The provider checks finite
bounds, versions and exact endpoint continuity on continuing rows. It cannot
verify simulator provenance. Invalid histories use `Phi=0`; masking the whole
shaping difference at history boundaries would break cancellation. True
termination uses zero next potential. Timeouts retain the pre-reset potential
because the learner bootstraps; a reset state's potential is not a replacement.
`gamma=1` requires the finite-episode terminal conditions.

A frozen SMP scorer can supply `Phi`: freeze the denoiser and its score
normalizer, use an explicit fixed noise bank, and identify all of them in the
potential version. `FrozenSMPPrior.score(..., update_normalizer=False,
noise=fixed_noise)` supports this. Cache the computed next endpoint for use as
the following current endpoint. Resampling or updating statistics when scoring
the same boundary independently is rejected by continuity checks. SMP feature
construction and scoring are caller-owned; this API does not silently switch
the existing R1 adapter to shaping.

The runner requires a **fixed positive common reward divisor** for the
potential-based provider. A changing divisor destroys cancellation even if
both components share a divisor at each update. The original task reward is
stored as a separate statistics channel and the unscaled combined reward goes
to replay. Provider state and definition are checkpointed; cold restore retains
persistent accounting and clears episode caches. This integration applies to
the FlashSAC runner family; the differentiable composer can be reused elsewhere
but is not automatically wired into `ModelBasedRunner`.

## Reward scale, entropy and critic support

`RunnerConfig` exposes these fields without changing legacy defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `gamma` | `0.99` | Learner/replay discount |
| `reward_normalization` | `legacy` | `legacy`, `episodic`, or `fixed` |
| `reward_scale` | `1.0` | Positive divisor in fixed mode only |
| `critic_min_v`, `critic_max_v` | `-5`, `5` | Categorical value support, in learner reward units |
| `critic_num_bins` | `101` | Number of support atoms |
| `critic_diagnostics` | `False` | Support, scale and entropy diagnostics |

`legacy` retains the vendored terminal carry equation for old runs. `episodic`
corrects that boundary: accumulate and record the terminal reward, then clear
done rows. For rewards `[1,2(done),3]` with discount `.99`, legacy statistics
see `[1,2,4.98]`; episodic statistics see `[1,2.99,3]`. Both termination and
timeout clear episode accumulation. Fixed mode has no online RMS drift.

If reward is divided by `D`, the entropy coefficient in original reward units
is `D*alpha`. Automatic temperature fitting does not directly measure this
denominator. With diagnostics enabled the learner reports `reward/scale` and
`temperature/raw_unit_value`, making those units inspectable. This is not an
automatic reward-scale/temperature correction.

The distributional critic clips TD atoms to its support. Diagnostics report
probability-weighted mass below/above support **before** clipping, projected
and predicted edge mass, and corresponding means. They reuse already computed
tensors, without extra action sampling, forward passes, batch-normalization
updates or loss changes. Atom spacing is **not** a task precision floor:
linear categorical projection can retain differences in the within-bin mean.
Reward-only normalization does not guarantee that entropy-augmented values fit
the support. Potential shaping also shifts values, so an inadequate support
can invalidate ideal equivalence in the approximate learner.

Checkpoint load rejects changing discount, support, normalization mode or
fixed divisor; compile mode and diagnostics are runtime choices. Older
checkpoints missing the new fields use legacy defaults. Actor-only transfer
keeps coordinate/dimension checks while leaving other learner state fresh;
value transfer additionally requires matching reward/value units.

## Reproducible diagnostic benchmark

Run from the source checkout:

```bash
python -m examples.precision_control.benchmark --output logs/precision --plot
```

It trains the actual public FlashSAC runner on a one-step, two-dimensional
normalized position task, comparing task-only, additive prior and potential
shaping. The additive optimum is displaced by `+0.1` on x; task-only and
potential shaping have the same task optimum. All runs use the same budget,
isolated matched goal streams and a grid frozen before training. Reports
include error relative to both the task optimum and each method's own reward
optimum, plus configurations, curves, checkpoints and evaluation actions.

In this deliberately small test the potential reward is a constant terminal
offset. It checks the reward path and objective bias; it is not a demonstration
of improved exploration. Its held-out grid tests interpolation in the same
goal domain, with two seeds by default. It provides no evidence about unseen
robots, contact dynamics, catching, physical units or natural arm swing.
Detailed run results belong in experiment reports, not the comparison plots
for other algorithms in the README.

The [completed diagnostic report](precision_control_results.md) includes the
two-seed results, validation evidence and limits of the conclusions.
