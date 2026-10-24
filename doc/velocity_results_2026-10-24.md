# Native MJLab AMP velocity comparison (2026-10-24)

Status: the independent-prior trial completed 4000 MBDACAMP iterations and five matched physical evaluations. A follow-up now reuses the upstream prior modules directly. Velocity tracking and gait quality remain below the retained PPO forward-walking baseline; no speed advantage at matched quality is established.

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

No additional MBDAC-only long training is scheduled.

## First MBDACAMP learning results

All checks below use actual native physics, not motion playback or world-model imagination. At iteration 1000, every case failed around 1.1 s. At 1500 and 2000, every case survived the full 20 s. However, commanded direction and yaw are not yet learned satisfactorily.

| Command | 1500 XY RMSE (m/s) | 2000 XY RMSE (m/s) | 2000 yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: |

| forward | 0.2505 | 0.2256 | 1.4697 |
| backward | 0.9383 | 0.9957 | 1.2857 |
| lateral_left | 0.7144 | 0.7983 | 1.3934 |
| turn_left | 0.6424 | 0.6274 | 1.0139 |
| stand | 0.5832 | 0.6762 | 1.5019 |

The 2000 actor moves forward and turns across all five commands. Its forward mean body vx is 0.713 m/s for a 0.8 m/s request, but XY RMSE is 0.226 m/s and yaw RMSE 1.470 rad/s, worse than PPO's 0.159 / 0.335. It makes 92/105 foot liftoffs in 18 steady seconds, with 0.469 m/s foot-slip RMS and 82.1 rad/s² joint acceleration RMS (PPO: 0.134 / 27.6). This is stable motion, not accepted smooth command tracking. Continued AMP learning is needed; survival alone does not establish success.

The saved 2000 checkpoint contains 5,251,072 real transitions and 1386.34 s cumulative trainer time. Including the earlier failed compilation process and both launches gives 1584.04 s (26.40 min) to the saved checkpoint, versus 8196.95 s for PPO 1500. These are costs to different quality levels, not a time-to-equivalent-quality result. Full process termination took slightly longer due to checkpoint/log synchronization.

Fresh nominal physics data gives WM one/eight-step linear-velocity RMSE 0.0405/0.0571 m/s, angular-velocity RMSE 0.294/0.358 rad/s and joint-position RMSE 0.00854/0.0247 rad. The acceleration head RMSE is 73.4 rad/s² with true state inputs and 122.7 with predicted states; foot-contact RMSE is 0.166/0.320. These identify remaining model/physical-head approximation error. Offline evaluation of 4995 real nominal transitions shows mean AMP style reward 0.687 and only 0.76% at zero, so the style reward is not uniformly clipped away.

The evaluator initially missed assigning its diagnostic task to the physical capture hook. That caused a missing-label error after simulation; it has been fixed and the complete evaluation rerun. Training captured the labels correctly throughout. Both failed and successful evaluation artifacts are retained.

Independent discriminator optimizer details also differ from the fork's PPO implementation: MBDACAMP follows the model-based learner's 2e-4 schedule and does not apply the fork's separate 0.001 trunk / 0.1 head weight decays. Both use LS-GAN and gradient penalty 10, expert features, width 1024/512/256 and the same task/style mixture. This is an algorithm comparison, not a claim that all optimizer settings are identical.

Additional code-level differences are explicit. The independent prior uses ELU
and unbounded normalized inputs; the retained fork uses ReLU and clips normalized
features to ±10. Our gradient penalty is evaluated on normalized expert features;
the retained fork calls `compute_grad_pen(*sample_amp_expert)` on its raw expert
tuple, although its classification inputs are normalized separately. Our prior
has its own clipped optimizer step; the fork's joint optimizer clips policy
gradients but not discriminator gradients. These choices are part of the current
MBDACAMP extension. The comparison holds task, expert files and evaluation
conditions fixed, but does not isolate the causal effect of model-based actor
updates alone.

Evidence: `logs/velocity_comparison_2000/comparison.json`, `logs/velocity_progress_2000.png`, `logs/velocity_eval_mbdacamp_{1000,1500,2000}/evaluation.json`, `logs/velocity_eval_mbdacamp_2000/forward.mp4` and `prior_audit.json`.

### Command-path diagnostic during continuation

Using the exact nominal initial observation, CPU actor outputs reproduce the
first physical-evaluation action at 1000, 1500 and 2000 with maximum absolute
difference 0. Changing only command values changes actor actions, confirming
that commands reach the policy. At 2000, the forward-versus-stand policy-action
RMS difference is 0.0487; at 2500 it is 0.0879. These are action coordinates before
the native per-joint scale, not radians or a physical tracking error. This checks
the input route and changing policy sensitivity; it does not establish successful
velocity control. Evidence: `logs/velocity_initial_command_sensitivity.json`.


## Independent-prior continuation to 4000

The 3000 and 4000 checkpoints both survive every nominal 20 s case. At 4000,
forward vx falls to 0.226 m/s for a 0.8 m/s request. Yaw and slipping improve,
but the robot shuffles rather than meeting the requested walking speed.

| Command | 3000 XY RMSE (m/s) | 4000 XY RMSE (m/s) | 4000 yaw RMSE (rad/s) |
| --- | ---: | ---: | ---: |
| forward | 0.5568 | 0.5762 | 0.2642 |
| backward | 0.2962 | 0.4022 | 0.1346 |
| lateral_left | 0.4139 | 0.3730 | 0.3369 |
| turn_left | 0.1287 | 0.0262 | 0.6063 |
| stand | 0.1375 | 0.0416 | 0.1341 |

Checkpoint 4000 contains 10,436,608 real transitions, with 2964.17 s cumulative
trainer time. Total process cost to the saved checkpoint, including all three
launches and the failed first compilation, is 3181.42 s (53.02 min). This remains
below PPO's 136.62 min in duration, but **does not reach matched control quality**.
Forward foot-slip RMS falls from 0.469 to 0.217 m/s, and joint-acceleration RMS
from 82.1 to 43.8 rad/s². Contact liftoffs remain excessive (123/110 over 18 s).
Fresh WM one/eight-step linear-velocity RMSE is 0.0222/0.0220 m/s and joint-position
RMSE 0.00366/0.0102 rad. Improved model errors do not prove correct actor gradients
or a successful velocity policy.

The independent-prior experiment is retained as measured research evidence; no
extra iterations are appended to it. The next AMP experiment directly imports
the original discriminator/normalizer/gradient penalty, with an input-differentiable
wrapper. Its reward, full discriminator optimizer step, normalization update and
checkpoint restoration pass numerical comparison against the actual fork. Its
1024-world smoke test passes; policy quality for this variant is still unproven.
See `doc/velocity_benchmark.md` for exact upstream semantics and remaining
algorithm/rate/batch differences. Reuse the genuinely WM-only iteration-500
checkpoint with a newly initialized upstream prior, without carrying over the
independent discriminator or the later independently trained actor.

Evidence: `logs/velocity_comparison_4000/comparison.json`,
`logs/velocity_eval_mbdacamp_{3000,4000}/evaluation.json`,
`logs/velocity_eval_mbdacamp_4000/forward.mp4`,
`logs_upstream_prior_tests.console.log` and
`logs/velocity_upstream_prior_smoke/metrics.jsonl`.
