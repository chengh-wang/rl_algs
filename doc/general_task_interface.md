# Reusable MDP rewards and physical signals

New tasks reuse the learner. Their adapter exposes physical state (including relevant objects), physical measurements and task conditions. Reward functions remain task code. The package creates and fits missing physical heads from declarations; it does not learn a weighted reward total or infer arbitrary Python semantics.

`PhysicalSignals` is the simulator-independent registry used by the MJLab velocity example. It selects analytic providers first, creates `PrimitiveSpec` declarations only for missing dependencies, captures detached pre-reset measurement labels, and resolves analytic/predicted values for imagined reward composition. Unknown dependencies and measurement shape mismatches raise explicit errors.

```python
from types import SimpleNamespace
import torch
from mbdac.modules.primitives import PhysicalSignals, PrimitiveSpec

signals = PhysicalSignals()
signals.register(
    PrimitiveSpec("hand_ball_contact", 1, "probability", "hand/ball", binary=True),
    measure=lambda env: env.hand_ball_contact,  # actual batched 0/1 contact
)
signals.register(
    PrimitiveSpec("relative_velocity", 3, "m/s", "world"),
    analytic=lambda state, context, **transition: hand_velocity(state) - ball_velocity(state),
)
dependencies = ("hand_ball_contact", "relative_velocity")
task.primitive_specs = signals.learned_specs(dependencies)
task.analytic_primitives = signals.analytic_names(dependencies)

def catch_reward(env, std=0.5):
    p = env.physical["hand_ball_contact"].squeeze(-1)
    dv = env.physical["relative_velocity"]
    return p * torch.exp(-dv.square().sum(-1) / std**2)

# In the native pre-reset capture hook:
transition.update(signals.capture(native_env, dependencies))

# In the model task, using the WM's physical primitive predictions:
physical = signals.resolve(state1, context, primitives, dependencies,
                           previous_state=state0, action=action)
reward = catch_reward(SimpleNamespace(physical=physical), std=current_std)
```

The same MDP formula can consume actual contact during real reward computation and predicted contact probability during imagination. Weights, targets and tolerances remain outside the heads and can change independently. State/FK quantities receive no redundant head. The positive regression example tests contact plus relative velocity, changed reward scales and gradients, without a simulator SDK.

An arbitrary black-box reward containing `.item()`, Python branches or hard comparisons cannot automatically provide useful pathwise gradients. Common events can be expressed as registered physical probabilities or smooth analytic gates. Unsupported dependencies require an explicit provider; they never silently become a reward-value head. A smooth contact surrogate must ultimately be validated against actual contact success in native physics.

This registry does not infer unobserved object state, implement a catching simulator or automatically rewrite arbitrary existing environment code. Native adapters still own state extraction and the pre-reset capture hook. The current physical training scope remains MJLab G1, followed by its AMP velocity task.
