# Validation on 2026-10-23

Hardware: RTX 4070 Laptop, 8 GiB. Isolated environment: Python 3.11.16, PyTorch 2.10.0, MJLab 1.2.0, MuJoCo/MuJoCo-Warp 3.8.1, Warp 1.12.0. Exact resolved dependencies are in `requirements-mjlab.lock.txt`.

## Numerical and interface checks

- Fifteen CPU tests passed: terminal lambda-return targets; gradient through frozen world model without changing its weights; no replay windows across reset; empty-replay checkpoint resume; inference isolation; discriminator update isolation; articulated SO(3) roundtrip gradients; reusable physical primitive outputs under different reward scales; duplicate state/head rejection; exported actor equivalence; periodic checkpoint training-time accounting; actor-only warm start; clearing stale replay on load; physical prediction diagnostics with terminal masks; and actual optimizer call counts/order during warmup and joint learning.
- Native G1 `ManagerBasedRlEnv`: 16 randomized test states, compared against MuJoCo forward kinematics, including nontrivial root rotations and angular velocities.

| Quantity | Maximum absolute error |
| --- | ---: |
| Root state write/read roundtrip | 2.38e-7 |
| Tracked link position | 5.36e-7 m |
| Tracked link rotation matrix | 2.45e-7 |
| Tracked link linear velocity | 8.44e-7 m/s |
| Tracked link angular velocity | 5.38e-7 rad/s |
| Per-joint action scale | 1.89e-8 rad/action-unit |
| PD gains | 0 |
| Native reward vs. shared analytic formula | 0 |

`mjOBJ_XBODY` is used for link-origin velocity comparison; `mjOBJ_BODY` measures the inertial/COM origin and must not be compared directly with link-origin FK velocity.

A forced hard failure confirmed that replay retains the physical next state before auto-reset, rather than the reset reference pose. All 29 joint names/order and all PD gains were checked. Re-run with `python -m examples.mjlab_g1.validate`.

1024 native MJLab worlds completed real sampling plus full-width MBDAC actor/world-model/critic updates, 32-step imagination, and W&B upload: [full architecture interface run](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-g1/runs/cwsojsy3). This short run is not evidence of a trained tracking controller.

The MBDACAMP extension also completed native 1024-world updates with small smoke-test networks: [AMP integration run](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-g1/runs/t979wwr5). This tests the interface and gradients; it does not establish AMP policy quality.

The wheel builds successfully and importing core algorithms/runners does not import MJLab, MuJoCo or Isaac Lab.

## Active first learning run

[G1 MBDAC 1024 v1](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-g1/runs/nf103cri) uses all 40 original G1 reference clips, full network widths, 500 world-model-only iterations and then joint world-model/actor/critic learning. Initial run length: 2000 iterations. It has entered actual actor updates. No tracking-success or convergence claim is made yet.

The v1 process started before the additional per-reward and custom episode-metric logging fix. Its existing logs retain simulator reward, native episode reward/termination counts, model losses, actor/critic gradients and timing. The added detailed terms will appear in newly launched processes; the change does not alter reward formulas.

## Still required

- Quantitative physical evaluation of trained checkpoints and full reference clips, including videos.
- Live Isaac Lab task integration. The current adapter is only a protocol bridge.
- A matched velocity task, AMP extension training, and a fair measured comparison against the retained AMP+PPO baseline.
- Broader installation/runtime verification beyond the tested Linux/NVIDIA environment.
