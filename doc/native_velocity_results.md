# Native MJLab velocity benchmark — 2026-10-24

The tested WM + FastSAC hybrid learns Go1 velocity tracking from scratch, but
**does not beat upstream Holosoma FastSAC in wall time**. It reaches the declared
acceptance rule sooner than the package's original MBDAC recipe. A matching
WM-disabled control is also faster than the hybrid in this pilot. G1 transfers
without a robot-specific learner change, but does not pass every command within
its fixed budget. These are experimental, single-seed results.

Protocol and full recipe differences: [native_velocity_benchmark.md](native_velocity_benchmark.md).
Hardware: RTX 4070 Laptop 8 GB; Python 3.11, PyTorch 2.10, native MJLab 1.2.0,
MuJoCo/MuJoCo Warp 3.8.1, Warp 1.12.0. Each training run uses 1024 native
`ManagerBasedRlEnv` worlds. Go1 replaces Go2 with the user's agreement.
All networks, normalizers, optimizers and replay buffers start fresh, seed 42.
No development checkpoint initializes a formal run.

## Original online measurements

| Go1 method | First observed strict pass: active process | Wall time through result availability | XY RMSE (m/s) | Yaw RMSE (rad/s) | Real transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Holosoma FastSAC | 309.97 s | **330.53 s (5.51 min)** | 0.04912 | 0.04988 | 3,083,264 |
| Original package MBDAC | 2451.07 s | **2544.83 s (42.41 min)** | 0.05246 | 0.04611 | 9,076,736 |
| WM + FastSAC hybrid | 914.48 s | **955.46 s (15.92 min)** | 0.07986 | 0.07416 | 2,212,864 |
| Same integration, real SAC only | 611.74 s | **641.65 s (10.69 min)** | 0.06132 | 0.05532 | 2,909,184 |

The primary comparison is the first three rows. The fourth isolates the added
model contribution within the new integration. All four pass again at the next
online evaluation, with return change below 5%, triggering the declared stopping
rule. `converged` in their manifests means that rule; it does not establish an
optimal reward or statistical convergence.

Acceptance requires XY velocity RMSE ≤ 0.20 m/s and yaw-rate RMSE ≤ 0.30 rad/s,
plus survival 1.0. We report both the original aggregate gate and the stricter
requirement that **each command** passes both errors. All four Go1 runs pass
both. Evaluation uses 5 commands × 8 deterministic repeatability copies, 20 s
without resets; RMSE excludes the first 2 s. Failed samples are retained.
Return is the fixed task's sum of per-step rewards, with zero contribution after
failure. Forty worlds are not forty independent randomized starts.

Online evaluations are approximately 300 active seconds apart. The table gives
first *observed* passes, not exact threshold-crossing times. Active process time
excludes evaluation but includes setup, imports, compilation, collection,
updates, diagnostics and checkpoint I/O. Wall time includes evaluation through
the result becoming available. It is not the total duration of the stopped run.
Different algorithms' iterations are not equivalent units of work.
Collection and aggregate update timers are retained per iteration. These trials
did not independently instrument every WM/Q/actor inner phase or isolate cache
compilation time, so no unmeasured phase breakdown is claimed.

## G1 transfer test

The same hybrid recipe, fresh weights, 1024 worlds and native G1 task ran for
3000 updates, with no task reward tuning. It used 3,083,264 real transitions,
1664.82 active process seconds, 1763.01 s before cleanup and 1772.45 s for the
complete subprocess including cleanup/synchronization.

| G1 command [vx, vy, yaw rate] | XY RMSE (m/s) | Yaw RMSE (rad/s) | Survival | Strict pass |
| --- | ---: | ---: | ---: | --- |
| [0, 0, 0] | 0.00078 | 0.00242 | 1.0 | Yes |
| [0.8, 0, 0] | **0.21281** | 0.17037 | 1.0 | **No** |
| [-0.5, 0, 0] | 0.15575 | 0.09905 | 1.0 | Yes |
| [0, 0.5, 0] | 0.11144 | 0.20455 | 1.0 | Yes |
| [0, 0, 0.5] | 0.06410 | 0.28879 | 1.0 | Yes |
| Aggregate | 0.13121 | 0.18115 | 1.0 | Aggregate gate only |

No strict all-command pass was observed. This run ends at its fixed iteration
budget, not by convergence; the budget was not extended to obtain a positive
result. The video uses the final measured checkpoint and displays this status.

## Retrospective checkpoint measurements

A separate systematic evaluation of ordinary saved checkpoints preserves the
original online rows and uses exact real-transition counters. All earlier
ordinary saved checkpoints were evaluated in order until the first strict pass.
TensorBoard timestamps estimate when each training iteration was logged; they
are **not** the original time at which an evaluation result was available.

| Method | Adjacent failed / passing saved updates | Logged process minutes: failed → pass | Real transitions: failed → pass | Passing XY / yaw RMSE |
| --- | --- | ---: | ---: | --- |
| Original Holosoma FastSAC | 1750 / 2000 | 3.269 → **3.679** | 1,793,024 → **2,049,024** | 0.08553 / 0.07176 |
| Original package MBDAC | 3250 / 3500 | 38.901 → **41.993** | 8,385,536 → **9,025,536** | 0.04122 / 0.04492 |
| WM + FastSAC hybrid | 1750 / 2000 | 12.612 → **14.573** | 1,803,264 → **2,059,264** | 0.09246 / 0.06234 |
| Same integration, real SAC only | 1750 / 2000 | 6.724 → **7.603** | 1,803,264 → **2,059,264** | 0.06267 / 0.08898 |

At this checkpoint resolution all three SAC routes require approximately the
same real samples. **The apparent sample advantage of WM in the sparse online
table does not survive this check.** WM has not demonstrated either a wall-time
or a sample-efficiency advantage over source FastSAC or the matching control.
It still reaches acceptance earlier than the original package MBDAC recipe.
Adjacent failed/passing samples do not guarantee a first-ever crossing between
them: unsaved transient passes are unknown and no monotonicity or interpolation
is assumed. The differing timestamps are intentionally kept separate from the
original online result-availability times above.

This post-hoc pass performs 38 new physical evaluations plus four reused online
initialization results. Its separately measured cost is **396.16 s (6.60 min)**,
including checkpoint loading, hashing, parsing, evaluation and plots. Waiting
for the earlier finalizer is excluded and not counted twice.

## Why this is not an isolated algorithmic speed claim

The environment's physics, rewards, commands, observations, terminations and
randomization are shared. All Go1 task fingerprints match. Complete learning
recipes differ, as the protocol documents:

- The original baseline inherits the unchanged Holosoma FastSAC learning loop,
  pinned to `d18d6cc50f872c15e904a22ceac22313cec955c8`.
- Its actor and Q normalization use sampled replay statistics; the new
  integration uses collected observations. The new actor duplicates the native
  velocity command in its goal input, and its Q includes previous-action and
  actor-history features. Replay sampling and actor update positions also differ.
- Source FastSAC compiles the full learner; the hybrid's real SAC path is eager
  and its WM is compiled. The timing comparison includes that integration cost.
- The hybrid mixes 0.9 real SAC and 0.1 H=8 model-return actor gradients after
  500 warmup updates. The real-only control disables WM fitting and model
  gradients with the same integration. It reaches online acceptance earlier,
  so these data do not demonstrate a useful speedup from the model contribution.

Analytic reward formulas match native rewards when supplied the same physical
measurements. This is not proof of perfect imagined rewards. Native kinematic
caches can lag raw state by one 5 ms substep; learned contact/force heads,
smooth imagined contacts and nominal mass/COM are explicit approximations.
Only real transitions fit the WM; real Q training retains measured features,
applied actions and pre-reset successors. All task adapters remain outside the
simulator-independent learning core.

## Costs, reproducibility and evidence

All failed or corrected development trials are retained. The machine-readable
`logs/native_velocity_cost_ledger_audit.json` distinguishes exact full subprocess
costs from before-cleanup lower bounds; unmeasured cleanup/rendering is not
estimated from file modification times. Post-hoc evaluation cost is recorded
separately. No pilot cost is folded into a formal checkpoint's training time.

| Full-size trial | Before cleanup (s) | Full subprocess (s), if recorded | Disposition |
| --- | ---: | ---: | --- |
| Source FastSAC v1 | 660.64 | Unknown | Successful development pilot |
| Source FastSAC v2 | 641.41 | Unknown | Formal source baseline |
| MBDAC v2 | 480.29 | 497.44 | Stopped after physical-head heading issue |
| MBDAC v3 | 2862.40 | 2900.46 | Formal MBDAC; stable acceptance |
| WMFastSAC v1 | 526.12 | Unknown | Excluded: real Q used predicted contacts |
| WMFastSAC v2 | 1000.64 | Unknown | Stopped after heading issue; mainly standing |
| WMFastSAC v3 | 1265.79 | 1293.24 | Formal hybrid; stable acceptance |
| Real-only v3 | 954.07 | 962.38 | Matched control; stable acceptance |
| G1 hybrid v3 | 1763.01 | 1772.45 | Budget completed; strict gate not reached |

The nine full-size trials plus ten smokes consumed **at least 171.88 process
minutes**, using full subprocess costs where available and before-cleanup
timers otherwise. This excludes offline evaluation/rendering, tests and
engineering. The development full-size trials alone account for a 44.75-minute
lower bound; their costs and defects are not hidden by the successful v3 runs.
Including checkpoint refinement and the 71.02 s of separately measured final
evaluation bodies gives a measured workflow lower bound of **179.66 minutes**.
Rendering, unmeasured finalizer overhead and engineering still remain outside
that lower bound.

The path-free tested environment snapshot is
`requirements-native-velocity.lock.txt`; installation and original-source setup
are in the [README](../README.md). Scripts reproduce the sequential matrix,
physical video generation, checkpoint refinement and W&B publication. The
standalone package is `dist/mbdac-0.3.3-py3-none-any.whl`; the full source archive
includes examples, scripts, dependency pins, protocol and license notices.
The wheel's core imports and exported inference policies work without MJLab,
MuJoCo, Isaac Lab, RSL-RL or an external Holosoma installation.

Final main-environment tests: **54 passed, 1 skipped**, including CUDA BF16
actor-loss/gradient and C51 comparisons. The skipped module is the optional
AMP fork discriminator test, unavailable in this standard RSL installation;
it is not part of these velocity learners. The isolated SDK-free installed-wheel
suite has **50 passed, 5 optional-dependency skips**. All 30 source/data files
and five license/provenance files match the tested wheel byte for byte; its
SHA256 is `1cca78c45fa242e0f7963c85bdcf1c2ea080aea15ba85ae93ce1311eadaf0bb9`.
The selected MBDAC/hybrid inference exports exactly match eager output in a
fresh torch-only process with package and simulator imports blocked.

Raw configs, evaluation JSONL, task fingerprints and source snapshots are kept
per run. The historical source baseline lacks a `runtime.json`; its exact source
pin/config are retained, and current pinned learner/module bytes were verified.
This provenance limitation is preserved rather than reconstructed as an
original runtime measurement.

Training logs:
[Holosoma FastSAC](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/ujc6scu7),
[MBDAC](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/bvt8gqf1),
[WM + FastSAC](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/coxfbec2),
[real-only control](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/7qoi4est),
[G1 hybrid](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/fovnaujm).

Final local evidence is under `logs/native_velocity_v3_final/`: primary reward,
RMSE and survival figures versus wall time and real transitions; the three-way
Go1 video; each selected policy's raw trajectory, physical metrics and video;
and separate control/G1 figures. Videos render actual native GPU state, without
joint smoothing or extra physics.

The five independently repeated physical evaluations retain the same acceptance
classifications as their online samples, with small numerical differences kept
in `logs/native_velocity_physical_repeat_audit.json`. CPU decoding and visual
inspection confirmed all six videos contain 500 changing scene frames at 25 fps
for 20 seconds; the G1 label explicitly says its strict threshold was not met.
