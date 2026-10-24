# Native MJLab AMP velocity comparison (2026-10-24)

Status: MBDACAMP training is running. No speed or quality advantage is established.

## Fixed evaluation protocol

Both actors use the same native `Unitree-G1-AMP-Flat` environment, seed 1042, standing initialization, 20-second requested horizon, 5 ms physics and 20 ms control. Noise, domain randomization, pushes and automatic reset are disabled. Steady errors exclude the first two seconds and post-failure samples. Training retains the original randomization, motion resets and command curriculum.

## Retained AMP+PPO checkpoint 1500

The preserved baseline used 4096 environments and 147,554,304 real transitions. Launch-to-checkpoint time was 8196.95 s (136.62 min). It was stopped at the user-requested 1.5k checkpoint; this is not a fully converged all-command policy.

| Command | Survives 20 s | XY RMSE (m/s) | Yaw RMSE (rad/s) | Foot liftoffs L/R |
| --- | --- | ---: | ---: | --- |
| forward | True | 0.1592 | 0.3346 | [31, 27] |
| backward | True | 0.4009 | 0.1205 | [0, 0] |
| lateral_left | True | 0.4005 | 0.1196 | [0, 0] |
| turn_left | True | 0.0023 | 0.7264 | [0, 0] |
| stand | True | 0.0025 | 0.1260 | [0, 0] |

The baseline walks forward. It largely stands for the backward/lateral/turn commands, so survival alone is insufficient. A successful comparison must also show commanded movement and suitable gait.

Evidence: `logs/velocity_eval_ppo_1500/evaluation.json`, `traces.npz`, `forward.mp4`; original baseline checkpoint and analysis remain in the preserved upstream experiment.

## MBDACAMP experiment

Uses 1024 environments, WM batch 128 and actor batch 256. After 500 WM-only iterations, the schedule is 10 WM / 1 discriminator / 1 actor / 20 critic steps. The actor and critic use widths 512/256/128; the WM uses 2048/2048/2048. The actor maintains four observation frames plus a goal encoder. These are documented differences from the baseline, not a strict equal-architecture/equal-batch experiment.

The first full-task actor compilation failed in Triton after warmup. The continuation resumes the stable iteration-500 checkpoint, compiles dynamics only, and uses ordinary autograd for analytic heads. The failed 391.72 s process cost is included in the launch metadata for the comparison. Do not equate cumulative trainer time with complete process cost.

[MBDACAMP W&B](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-velocity/runs/0qn25b2s)

Candidate physical evaluation and matched comparison remain pending. No additional MBDAC-only long training is scheduled.
