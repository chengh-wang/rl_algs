# MBDAC delivery and measured results — 2026-10-24

The independent installable package, native training examples, source-prior integration and actual comparison experiments are delivered locally. **The velocity result is mixed; an overall speedup to equal policy quality is not established.** No code or artifacts have been pushed or published.

## Package and interface

- `mbdac` provides MBDAC and MBDACAMP with RSL-style runner/config/learn/save/load/inference/export. The learning core requires no simulator SDK or RSL library.
- A task supplies physical state, analytic Torch functions and pre-reset transitions. `PhysicalSignals` prefers analytic providers and declares missing physical heads with units/frames. Reward weights, targets and tolerances remain analytic; learned heads never target the reward sum.
- Real replay fits the WM. AMP separates real policy/expert data; actor learning differentiates through frozen WM, critic and prior inputs. No differentiable simulator is used.
- The native velocity example directly calls the pinned AMP fork's discriminator, normalizer, gradient penalty and expert loader. Its reward wrapper retains input gradients. Core supports a task-owned prior factory without importing the fork.
- MJLab owns native managers, 5 ms physics and 20 ms controls. Laptop learning scale is 1024 worlds / WM batch 128 / actor batch 256. Native velocity actor history is 4; widths 512/256/128. WM widths 2048³. Other simulator training is deferred; the earlier genuine Isaac Lab adapter smoke remains retained.

## Verified schedule and portability

The source-prior 5000 checkpoint contains exactly 50,000 WM optimizer steps, 4,500 actor steps, 4,500 discriminator steps and 90,000 critic steps. The first 500 outer iterations perform 5000 WM-only optimizer steps; subsequent iterations use 10 WM → 1 D → 1 actor → 20 critic. Actual actor/WM normalization counts equal its 12,996,608 real transitions. The source normalizer count confirms D state survived resume.

- Full native-dependency regression suite: **28 passed**.
- Installed wheel with no MJLab, MuJoCo, Isaac Lab or RSL: **26 passed, 2 optional tests skipped**.
- Native 32-world observation check: gyro max 1.67e-6, gravity 2.38e-7, command/q/qd/previous action 0. Native reward terms below 2e-9; FK and forced pre-reset terminal capture pass.
- Trained source-prior 5000 actor loads/exports without simulator/RSL SDKs; CPU eager/export max absolute difference 0. Export is stateless `(history,goal) → action`; the native action manager still applies clipping/PD scales.

Evidence: `logs_final_native_tests.console.log`, `logs_final_core_wheel_tests.console.log`, `logs/velocity_upstream_5000_optimizer_audit.json`, `logs/velocity_validation_full_observations.json`, `logs/velocity_upstream_5000_export/export.json`.

## First milestone: G1 motion tracking

MBDAC trained in 1024 native worlds. The shown 4000 checkpoint took 3435.28 s (57 min 15 s), with 20 s prefix mean return 0.7633 and 34/40 prefixes completed. Full-duration evaluation at 5000 completed 26/40 entire clips, mean normalized return 0.58493; every clip was evaluated to its full nominal horizon. Difficult fall/get-up and several other clips still fail. Prefix and full-clip returns are different measurements. The user-approved walking video and full evaluation are retained; no additional long MBDAC-only training was run.

Evidence: `logs/eval_4000/reference_left_policy_right.mp4`, `logs/eval_5000_full/metrics.json`.

## AMP velocity: same task and physical evaluation

Five fixed nominal commands use seed 1042, standing initialization, 20 s, no auto-reset/noise/randomization/pushes. All five survive 20 s for PPO 1500 and both source-prior 4000/5000. The retained PPO run used 4096 worlds and 147,554,304 real transitions. Iteration numbers, batch counts and architectures are not equal across algorithms.

| Checkpoint | Real transitions | Measured process cost to checkpoint | Forward body XY RMSE (m/s) | Forward body yaw RMSE (rad/s) | Forward native torso XY RMSE (m/s) | Forward native world-Z yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AMP+PPO 1500 | 147,554,304 | 136.62 min | 0.1581 | 0.3379 | 0.1395 | 0.1484 |
| MBDACAMP 4000 | 10,436,608 | 53.34 min | 0.1544 | 0.2978 | 0.1722 | 0.2528 |
| MBDACAMP 5000 | 12,996,608 | 66.43 min | 0.2272 | 0.7596 | 0.1887 | 0.3112 |

The 4000 body-frame forward point is close to PPO at 53.34 min, but torso errors, slip and joint acceleration are worse. It does not establish equal overall quality. The 5000 gait lifts its feet more, but forward body vx is 0.919 m/s for 0.8 m/s and overshoots. Backward tracking is better than the retained PPO; lateral/turning remain weak in both. No single tested source checkpoint uniformly dominates the baseline.

### All commands, checkpoint 5000

| Command | PPO body XY RMSE (m/s) | MBDACAMP body XY RMSE (m/s) | PPO native torso XY RMSE (m/s) | MBDACAMP native torso XY RMSE (m/s) | PPO native yaw RMSE (rad/s) | MBDACAMP native yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward | 0.1581 | 0.2272 | 0.1395 | 0.1887 | 0.1484 | 0.3112 |
| backward | 0.3994 | 0.1408 | 0.3989 | 0.1553 | 0.0135 | 0.0852 |
| lateral_left | 0.3987 | 0.3978 | 0.3956 | 0.3989 | 0.0640 | 0.0547 |
| turn_left | 0.0023 | 0.0097 | 0.0046 | 0.0109 | 0.5810 | 0.5245 |
| stand | 0.0021 | 0.0294 | 0.0040 | 0.0298 | 0.0663 | 0.0651 |

Forward 5000 foot-slip RMS is 0.199 m/s versus PPO 0.131; joint-acceleration RMS is 37.2 versus 27.4 rad/s². Foot liftoffs are [38, 40] versus [30, 25] over 18 s; these count contact-loss events, not gait cycles.

Selected-lineage process cost to 5000 is 3985.74 s (66.43 min), including initial failed compilation, launches and replay refill. Other independent-prior prototype processes cost an additional 2798.22 s. The combined recorded cost is 113.07 min. Engineering, evaluation and smoke checks are excluded. Cumulative trainer time alone is 62.82 min and excludes some launch/failure costs. Lower elapsed duration at these different quality levels is not proof of faster learning to equal quality.

WM fresh-data one/eight-step root linear-velocity RMSE is 0.0167/0.0200 m/s and joint-position RMSE 0.00351/0.00863 rad. Acceleration-head RMSE is 49.0 rad/s² on true states and 98.0 through predicted rollout; contact probability RMSE 0.0752/0.1348. These are physical predictions, not learned reward totals. Accurate values do not guarantee accurate contact gradients.

## Reproduce and inspect

Start with [README](../README.md) and the [velocity setup/benchmark guide](velocity_benchmark.md). The isolated AMP fork has a documented RSL dependency-metadata exception; the core wheel does not depend on that fork. Provenance, pinned revisions, data-source notices and licenses are retained. The repo remains local on `feat/initial-mbdac`; no remote/push was performed.

Local outputs below are experiment evidence, not distributed wheel/Git assets:

- Comparison: `logs/velocity_comparison_upstream_5000/comparison.json` / `comparison.md`.
- Actual physics video: `logs/velocity_comparison_upstream_5000/ppo_left_mbdacamp_right.mp4` (left PPO 1500, right MBDACAMP 5000).
- Full learning curves: `logs/velocity_upstream_progress_5000.png`.
- Raw physical signals: `logs/velocity_comparison_upstream_5000/forward_signals.png`.
- Actor export: `logs/velocity_upstream_5000_export/policy.pt`.
- Full source-prior checkpoints: `logs/velocity_mbdacamp_upstream_1024_v2/model_4000.pt` / `model_5000.pt`.

[Source-prior run 1](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-velocity/runs/yls5tbj3), [continuation and final evaluation](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-velocity/runs/qrjibfxn).

The implementation and measured comparison requirements are verified. General-purpose reliable velocity control, arbitrary zero-shot task conversion and the future catching task are not established by these experiments.
