# Native velocity experiment results — in progress

Date: 2026-10-24. Protocol: [native_velocity_benchmark.md](native_velocity_benchmark.md).
Native MJLab 1.2.0, RTX 4070 Laptop 8 GB, 1024 real worlds, seed 42.
Go1 replaces Go2 with the user's explicit agreement; G1 is the second task.
All networks, normalizers, optimizers and replay buffers start fresh.

## Completed primary results

| Method | First strict pass, excluding evaluation | Elapsed through result availability | XY RMSE (m/s) | Yaw RMSE (rad/s) | Real transitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Holosoma FastSAC | 309.97 s | 330.53 s | 0.04912 | 0.04988 | 3,083,264 |
| Original MBDAC | 2451.07 s | 2544.83 s | 0.05246 | 0.04611 | 9,076,736 |
| WM + FastSAC hybrid | Running; not yet passed | — | — | — | — |

All thresholds must pass separately for every command and all 40
nominal worlds must survive. Both completed methods pass again at the next
evaluation. The code's `converged` label means this operational stopping rule,
not a claim of statistical convergence or optimal reward.

Online physical evaluations occur approximately every 300 seconds. These are
first *observed* passing times, not exact threshold-crossing times. The initial
failed sample and preceding failed sample bracket each crossing. Elapsed time
excluding evaluation includes imports, setup, compilation, diagnostics and
checkpoint I/O. Full elapsed at result availability is recorded separately.
The source baseline is pinned to Holosoma commit `d18d6cc50f872c15e904a22ceac22313cec955c8`.

The hybrid's first training evaluation (311.65 active seconds, update870) has
XY RMSE0.47737 m/s, yaw RMSE0.21896 rad/s and survival1.0. It is standing and has
not learned the commanded motion. Its return90.98 alone is not evidence of
velocity tracking. No claim of WM speedup is supported at this stage.

## Costs and retained failures

| Trial | Elapsed before cleanup (s) | Observed full subprocess (s) | Real transitions | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Source FastSAC v1 | 660.638 | Not recorded | 6,145,024 | Successful development pilot |
| Source FastSAC v2 | 641.408 | Not recorded | 6,206,464 | Primary baseline; stable acceptance |
| MBDAC v2 | 480.295 | 497.442 | 2,287,616 | Stopped after physical-head frame issue |
| MBDAC v3 | 2862.403 | 2900.457 | 10,151,936 | Primary; stable acceptance |
| WMFastSAC v1 | 526.119 | Not recorded | 1,980,416 | Excluded: predicted contacts used for real Q |
| WMFastSAC v2 | 1000.640 | Not recorded | 3,062,784 | Model-only pilot; stopped after heading issue |

Completed full-size trials used at least6226.70 process seconds (103.78 min),
including exact subprocess cleanup where recorded. Ten additional smokes add
57.76 s, making a lower bound of104.74 min. This excludes the running hybrid,
tests, engineering, offline evaluations and rendering. Missing cleanup timers
remain unknown; file modification times are not substituted for measurements.
No pilot checkpoints initialize the formal runs.

Detailed machine-readable audit: generated `logs/native_velocity_cost_ledger_audit.json`.
Raw configs, runtime metadata and source archives are saved inside each trial.
Physical videos and final figures are pending the matrix completion. The artifact
builder is queued after the live matrix, with no GPU overlap with training.

## Limits on interpretation

This is a one-seed nominal laptop benchmark of complete recipes. The native
physics, rewards, commands, noise, termination and evaluation protocol match;
network widths, optimizers and update schedules differ and are documented.
The separately queued real-only control uses the hybrid's integration without
WM updates to help separate those effects. Forty deterministic replicas are
repeatability copies, not 40 independent randomized starts. G1 reuses the same
configuration in fresh training; it does not reuse learned weights.
