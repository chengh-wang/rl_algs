# Task and physical primitive contract

The learning core imports no simulator. A task supplies a state codec and Torch functions that work on batched real or predicted states. A simulator adapter supplies **real pre-reset transitions**. The simulator itself never participates in autograd.

## Required interface

The task exposes `codec`, `action_dim`, `context_dim`, `obs_dim`, `goal_dim`, and `critic_dim`, plus:

```python
normalize_context(context)
observation(state, previous_action, context, condition, primitives=None)
add_noise(observation)
goal(condition)
advance(condition)
valid_start(condition, horizon)
reward(state1, condition1, action, previous_action, primitives=None)
termination_probability(state1, condition1, primitives=None)
critic_features(state, condition, context, primitives=None)
```

`condition` can be a motion frame index, velocity command, or another task-specific tensor. Keep exogenous commands separate from physical state. Analytic functions must preserve gradients through their physical inputs. Imagined termination is a smooth probability; the real simulator can use hard thresholds. The task owns reward units and timestep scaling.

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

Some measurements, such as an impulse over a time interval, are transition-dependent. This state-head interface should not silently treat those as state-only quantities; represent the needed physical memory in state or add an explicit transition-conditioned primitive adapter.

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

## Framework adapters

`MjlabModelAdapter` wraps native `ManagerBasedRlEnv` five-return stepping into the RSL-style four-return interface. The G1 example subclasses the native environment only to capture raw `qpos/qvel` before reset. MJLab still owns physics substeps, action/reward/observation/termination managers, parallel world allocation, and resets.

`IsaacLabModelAdapter` is a framework-independent bridge around an Isaac Lab RSL wrapper. It requires the task to capture the same pre-reset transition schema. This bridge has not yet been validated in a live Isaac Lab runtime.
