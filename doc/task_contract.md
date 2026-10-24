# Task and physical primitive contract

The learning core imports no simulator. A task supplies a state codec and Torch functions that work on batched real or predicted states. A simulator adapter supplies **real pre-reset transitions**. The simulator itself never participates in autograd.

## Required interface

The task exposes `codec`, `action_dim`, `context_dim`, `obs_dim`, `goal_dim`, and `critic_dim`, plus:

```python
normalize_context(context)
observation(state, previous_action, context, condition, primitives=None)
add_noise(observation)
goal(condition)
advance(condition, state=None, context=None)
valid_start(condition, horizon)
reward(state1, condition1, action, previous_action, primitives=None, previous_state=None)
termination_probability(state1, condition1, primitives=None)
critic_features(state, condition, context, primitives=None)
```

`condition` can be a motion frame index, velocity command, or another task-specific tensor. Keep exogenous commands separate from physical state. Analytic functions must preserve gradients through their physical inputs. Imagined termination is a smooth probability; the real simulator can use hard thresholds. The task owns reward units and timestep scaling.

The optional [physical-signal registry](general_task_interface.md) generates learned-head declarations and captures pre-reset measurement labels from registered dependencies. Analytic providers take precedence, and changing reward scales does not change these physical targets.

The adapter accepts the **policy action** and owns its conversion to actual controls. `model_transition['a']` must be that same applied policy action, after any policy-action clipping. The task and world model must see a consistent action coordinate system. For G1, `q_target = q_default + calibration_bias + action_scale * a`; no reference joint targets are injected into this control path.

## Learn the physical quantity, not the reward

For a speed reward:

```python
# Obtain velocity directly from predicted state/FK when available.
v_body = R.transpose(-1, -2) @ v_world.unsqueeze(-1)
reward = weight * torch.exp(
    -(v_body.squeeze(-1)[..., :2] - command_xy).square().sum(-1) / std**2
)
```

The world model predicts physical state; the task composes this expression. Changing `command_xy`, `std`, or `weight` does not change the prediction target. In this example a velocity head is unnecessary because root velocity is already represented in articulated state.

If a physical quantity is unavailable analytically, declare it:

```python
primitive_specs = (PrimitiveSpec('contact_probability', 2, 'probability', 'left/right foot', binary=True),)
```

Each spec declares its units and frame. `analytic_primitives` lists quantities already available from state/FK; overlapping learned names are rejected. A head receives physical state features and physical environment context, **not commands, reward weights, tolerances, or the reward total**.

`primitive/<name>` in replay is the real measurement at **s1 before reset**, with the declared units/frame. Head input and output normalization update only from real data. Heads train against real physical states, rather than compensating for world-model state errors. During imagination their outputs stay differentiable with respect to predicted state, while their parameters are frozen during actor optimization. Observation, reward, termination and critic functions can all consume the same primitive predictions.

Some measurements, such as a contact impulse or an instantaneous motor/contact acceleration, depend on the applied action and the transition. Declare those with `PrimitiveSpec(..., transition=True)`. Their heads receive `(state, action, state1, physical_context)` and train on real pre-reset transition measurements. This avoids treating them as state-only quantities. Transition predictions are available in the reward's `primitives` dictionary; state heads remain available to observation, termination and critic functions. When a quantity is analytically available from the transition, compute it directly using `previous_state`; for example, interval-average acceleration is `(v1-v0)/dt`. This average is different from a simulator's instantaneous acceleration at its last physics substep.

## Real transition schema

All values have leading dimension `num_envs`:

| Key | Meaning |
| --- | --- |
| `s`, `s1` | Physical state before action and after physics, before reset |
| `a`, `a_prev` | Applied policy action and previous applied policy action |
| `context` | Physical parameters, calibration, friction, etc. for this transition |
| `condition` | Task condition before stepping |
| `ep_step` | Steps since the most recent reset, zero at episode start |
| `terminated`, `truncated` | Real hard failure and timeout masks |
| `primitive/<name>` | Optional real physical measurement at s1 |

Reward labels are useful diagnostics but are not reward-head supervision. Replay windows include the terminal transition and mask everything after it. History samples stay within their episode and available replay range.

`model_boundary` optionally marks an exogenous change after the recorded physics/reward snapshot, such as a velocity push applied by an interval event. WM windows stop at that boundary; the next real transition starts from the actual perturbed state. Observation histories continue because the environment did not reset. Task advancement may use predicted `state` and `context` for state-dependent goal/termination bookkeeping.

Restoring a checkpoint with replay marks its latest stored streams as truncated before new collection: checkpoint files do not contain simulator state. Historical physical transitions remain valid, but windows must not bridge the restored buffer to different live physics. Actor histories honor these cuts even when a caller's episode counter has not reset. `model_boundary` alone does not cut actor history.

## Framework adapters

`MjlabModelAdapter` wraps native `ManagerBasedRlEnv` five-return stepping into the RSL-style four-return interface. The G1 example subclasses the native environment only to capture raw `qpos/qvel` before reset. MJLab still owns physics substeps, action/reward/observation/termination managers, parallel world allocation, and resets.

The core also exposes a generic `RslModelAdapter` protocol bridge. Other simulator backends are deferred from the current MJLab training and acceptance scope; a bridge alone does not supply the task's physical functions or terminal-state capture.

## Reusing an existing AMP module

An AMP task may implement `make_amp_prior(device, amp_options)` to return a
task-owned `nn.Module`; the core imports no framework or RL library to construct
it. The module exposes `norm` (callable with `update`), `net` (normalized paired
features to score) and differentiable `reward(phi0, phi1)` before external reward
scaling. Optional hooks are `optimizer_parameters()` for parameter groups,
`gradient_penalty(expert0, expert1, strength)`, and
`after_update(normalized_policy_pairs, normalized_expert_pairs)`.
`normalize_before_update=False` delegates normalization updates to that last
hook. Without a provider the package's independent AMP prior is unchanged.
`amp.max_grad_norm=None` disables discriminator clipping while still rejecting
non-finite gradients. Actor, model and critic clipping are unaffected.

The native velocity example's `UpstreamAmpPrior` calls the pinned fork's actual
discriminator, normalizer and gradient-penalty code. Its reward wrapper removes
only the source prediction helper's `no_grad` context, preserving the same
forward formula with input gradients. Checkpoint state includes portable numeric
normalizer statistics; it does not pickle the external modules. Actor-only
inference still requires neither the prior nor its source library.

`world_model_diagnostics` reports state prediction errors in physical units. For declared primitive heads it separately measures error on real input states/transitions (`head_oracle`) and error through the multi-step predicted rollout (`head_rollout`). Head RMSE uses each spec's physical units, or probability units for binary contact, and excludes samples after reset or an exogenous replay boundary.
