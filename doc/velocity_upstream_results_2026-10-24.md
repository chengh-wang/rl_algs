# Upstream-prior AMP velocity results (2026-10-24)

Status: 3000 evaluated; same configuration continues toward 5000. The package and numerical integration work, but command-control quality is **not accepted** and matched-quality speedup is **not established**.

## Method and evaluation

MBDACAMP reuses the pinned fork's actual discriminator, normalizer, gradient penalty and optimizer groups. Only the reward wrapper retains input gradients for model-based actor learning. Source normalization update ordering is preserved. Core uses 10 WM / 1 D / 1 actor / 20 critic steps after 500 WM-only iterations. Discriminator lr is fixed 1e-3; WM/actor/critic use the 2e-4 model-based schedule. PPO's KL-dependent schedule, update batch sizes and policy architectures differ as documented in `velocity_benchmark.md`.

Both actors use the same native task, hardware, evaluator, seed1042, standing initialization and five fixed 20 s commands. No auto-reset, observation noise, randomization or pushes. Errors exclude the first 2 s and post-failure samples. Two velocity frames are kept separate: pelvis/body (historical metric) and torso-yaw XY plus torso world-z angular velocity (native reward target).

All five cases survive 20 s for both PPO1500 and source-prior MBDACAMP3000. Survival does not mean command tracking.

## Body-frame errors

| Command | PPO XY RMSE (m/s) | AMP3000 XY RMSE (m/s) | PPO yaw RMSE (rad/s) | AMP3000 yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| forward | 0.1581 | 0.3072 | 0.3379 | 0.3550 |
| backward | 0.3994 | 0.3074 | 0.1307 | 0.1369 |
| lateral_left | 0.3987 | 0.4006 | 0.1236 | 0.1215 |
| turn_left | 0.0023 | 0.0895 | 0.7292 | 0.5972 |
| stand | 0.0021 | 0.0344 | 0.1260 | 0.1327 |

## Native reward-frame errors

| Command | PPO torso XY RMSE (m/s) | AMP3000 torso XY RMSE (m/s) | PPO world-z yaw RMSE (rad/s) | AMP3000 world-z yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| forward | 0.1395 | 0.3431 | 0.1484 | 0.3845 |
| backward | 0.3989 | 0.3069 | 0.0135 | 0.1952 |
| lateral_left | 0.3956 | 0.3993 | 0.0640 | 0.0655 |
| turn_left | 0.0046 | 0.0938 | 0.5810 | 0.6002 |
| stand | 0.0040 | 0.0364 | 0.0663 | 0.0568 |

Forward body vx is 0.536 m/s for a 0.8 request. XY RMSE improves from 0.804 at 2000 to 0.307 at 3000. It begins to respond to backward commands but still under-tracks; lateral and turning remain weak. Forward foot-slip RMS is 0.265 m/s and joint-acceleration RMS 40.1 rad/s². The video has short steps/contact chatter; liftoffs ([70, 80]) are contact-loss events, not a direct estimate of gait frequency.

## Measured cost and validation

3000 contains 7,811,072 real transitions and 2190.21 s cumulative trainer time. Total checkpoint-lineage process cost to this saved checkpoint is 2387.74 s (39.80 min), including the initial failed compilation process. PPO1500 cost 8196.95 s (136.62 min) and used 147,554,304 transitions. Other independent-prior development trials consumed a separate 2798.22 s. Costs to these different quality levels do not establish faster learning at equal quality.

The strengthened 32-world native validator passes: gyro reconstruction max absolute error 1.67e-6, gravity 2.38e-7, command/joint position/joint velocity/previous-action errors 0. Reward-term errors are below 2e-9 and terminal pre-reset capture is verified. The source-prior checkpoint loads without simulator or RSL SDKs; its stateless exported CPU actor matches eager outputs with maximum absolute difference 0. These checks validate wiring/export, not policy quality.

Evidence: `logs/velocity_eval_upstream_{2000,3000}/evaluation.json`, `logs/velocity_eval_ppo_1500_frames/evaluation.json`, `logs/velocity_comparison_upstream_3000/comparison.json`, `logs/velocity_validation_full_observations.json`, `logs/velocity_upstream_3000_export/export.json`.

[Training run](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-velocity/runs/yls5tbj3). Continuation restores all trained networks, optimizers and prior normalizer state; no prior reset, new reward or extra MBDAC-only training.
