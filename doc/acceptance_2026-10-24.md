# Delivery audit: evidence and measured limitations

Status: implementation, portability checks and actual measured comparisons are complete.
The velocity result is mixed; reliable all-command control and an overall
matched-quality speedup are not established.

| Requirement | Current evidence | Assessment |
| --- | --- | --- |
| Independent installable repository | Local `feat/initial-mbdac`; wheel built and installed into `.venv-core`; `logs/current_wheel_import_audit.json`; no Git remote | Implemented locally; no push/publication |
| RSL-style runner/config/save/load/inference | `src/mbdac/runners/model_based_runner.py`; runner resume, selective warm start, active-time checkpoint and export tests | Implemented and exercised |
| Simulator-independent learner | Installed wheel imports without MJLab, MuJoCo, Isaac Lab or RSL; `logs/core_wheel_validation.json` and SDK-free AMP/inference tests | Core verified; task adapters still supply physical semantics |
| Real replay, pre-reset state/applied action | Native capture hooks, forced-terminal checks in `logs/velocity_validation.json`, replay boundary/history/resume tests | Verified; no cross-reset WM windows |
| Analytic functions first, physical heads otherwise | `PhysicalSignals`, duplicate-provider rejection, state/transition primitive tests, native acceleration/contact/force heads | Implemented; weights/commands/reward totals are not head targets |
| No differentiable simulator | Collection runs without autograd; BPTT crosses only learned WM and Torch task functions; gradient-isolation tests | Verified implementation |
| Reference MBDAC update rules | `doc/update_schedule_audit.md`, optimizer-call test, actual source5000 optimizer audit: WM50,000 / actor4,500 / D4,500 / critic90,000 | Verified at documented laptop batches, including restored optimizer state |
| Native manager-based MJLab | Native task factories and `ManagerBasedRlEnv` step/managers retained; real 1024-world runs and videos | Verified |
| Other framework adapter | `IsaacLabModelAdapter` delegates the RSL wrapper; earlier real Cartpole terminal/resume/export smoke in `logs/isaac_cartpole_validation/validation.json` | Bridge/smoke evidence retained; further Isaac Lab training deferred by user |
| G1 motion-tracking learning, physics and video | 4000 video/time (3435.28 s); 5000 full-clip evaluation completes 26/40 clips; `logs/eval_4000`, `logs/eval_5000_full` and W&B | Actual policy learned; no claim of solving all clips |
| Action scale, FK, reward and observation order | G1 native validation; `logs/velocity_validation_full_observations.json`: gyro error 1.67e-6, gravity 2.38e-7, remaining actor channels 0; reward/FK/terminal checks pass | Verified on actual native states; no IMU/body-frame mismatch found |
| AMP discriminator and actor style gradient | Generic isolation/style-gradient tests; native smoke updates; task-owned source-prior factory; source reward/optimizer/normalizer/checkpoint parity test | Implemented; module parity is not policy success |
| AMP real training and matched PPO comparison | Independent-prior checkpoints 1000–4000 evaluated with same five 20 s commands/seed/task; `logs/velocity_comparison_4000` | Measured negative result: no matched-quality speed advantage |
| Source-prior real training/comparison | 2000/3000/4000/5000 evaluated against retained PPO in both frames; `logs/velocity_comparison_upstream_5000` and final report | Completed comparison with mixed results; no uniform matched-quality speedup claimed |
| Checkpoint portability and export | Trained G1 4000 / AMP 2000 / source-prior AMP4000 and5000 export max difference0; `logs/velocity_upstream_5000_export/export.json` loads without simulator/RSL; numeric prior checkpoint and cross-device partial reset tests | Verified through evaluated5000; full native suite28 passed, installed SDK-free wheel26 passed/2optional skips |
| Fresh-start instructions, versions and provenance | README, pinned lock/data fetch, isolated AMP setup, `licenses/PROVENANCE.md`, dependency-metadata exception documented | Provided; source examples run from repository root |
| Preserve previous work | Old G1/PPO experiments, failed compilation, all independent-prior checkpoints/videos remain; no remote/push | Preserved |

The requested later training is **AMP only**. No new long MBDAC-only or Isaac Lab
training is scheduled. Contact/catching is a future task: the package provides
physical-signal routing, not a claim that arbitrary Python rewards can be
automatically differentiated without state/measurement definitions.

Completed evidence covers the requested implementation, native learning,
portability, examples and measured comparison. Final artifacts and quantitative
limitations are listed in [the delivery report](final_report_2026-10-24.md).
No training process remains running. The known policy limitations are reported
outcomes: several G1 clips fail, AMP lateral/turning remain weak, and no source
checkpoint uniformly dominates PPO. Finite losses or shorter duration are not
accepted as proof of equivalent policy quality.
