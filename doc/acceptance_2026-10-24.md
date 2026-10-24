# Delivery audit: evidence and remaining work

Status: **active**, while the source-prior AMP experiment is training. This is an
evidence inventory, not a declaration that all policies are successful.

| Requirement | Current evidence | Assessment |
| --- | --- | --- |
| Independent installable repository | Local `feat/initial-mbdac`; wheel built and installed into `.venv-core`; `logs/current_wheel_import_audit.json`; no Git remote | Implemented locally; no push/publication |
| RSL-style runner/config/save/load/inference | `src/mbdac/runners/model_based_runner.py`; runner resume, selective warm start, active-time checkpoint and export tests | Implemented and exercised |
| Simulator-independent learner | Installed wheel imports without MJLab, MuJoCo, Isaac Lab or RSL; `logs/core_wheel_validation.json` and SDK-free AMP/inference tests | Core verified; task adapters still supply physical semantics |
| Real replay, pre-reset state/applied action | Native capture hooks, forced-terminal checks in `logs/velocity_validation.json`, replay boundary/history/resume tests | Verified; no cross-reset WM windows |
| Analytic functions first, physical heads otherwise | `PhysicalSignals`, duplicate-provider rejection, state/transition primitive tests, native acceleration/contact/force heads | Implemented; weights/commands/reward totals are not head targets |
| No differentiable simulator | Collection runs without autograd; BPTT crosses only learned WM and Torch task functions; gradient-isolation tests | Verified implementation |
| Reference MBDAC update rules | `doc/update_schedule_audit.md`, optimizer-call test, 500 WM-only iterations then 10 WM / 1 actor / 20 critic | Verified at documented laptop batches |
| Native manager-based MJLab | Native task factories and `ManagerBasedRlEnv` step/managers retained; real 1024-world runs and videos | Verified |
| Other framework adapter | `IsaacLabModelAdapter` delegates the RSL wrapper; earlier real Cartpole terminal/resume/export smoke in `logs/isaac_cartpole_validation/validation.json` | Bridge/smoke evidence retained; further Isaac Lab training deferred by user |
| G1 motion-tracking learning, physics and video | 4000 video/time (3435.28 s); 5000 full-clip evaluation completes 26/40 clips; `logs/eval_4000`, `logs/eval_5000_full` and W&B | Actual policy learned; no claim of solving all clips |
| Action scale, FK, reward and observation order | G1 native validation and velocity reward/FK/AMP-feature validation; native actor values copied exactly | Verified original checks; stronger analytic-versus-native observation check is added but awaits its serial GPU run |
| AMP discriminator and actor style gradient | Generic isolation/style-gradient tests; native smoke updates; task-owned source-prior factory; source reward/optimizer/normalizer/checkpoint parity test | Implemented; module parity is not policy success |
| AMP real training and matched PPO comparison | Independent-prior checkpoints 1000–4000 evaluated with same five 20 s commands/seed/task; `logs/velocity_comparison_4000` | Measured negative result: no matched-quality speed advantage |
| Source-prior real training/comparison | `logs/velocity_mbdacamp_upstream_1024_v1`, source prior from pinned fork, actual finite updates | **Pending** final physical metrics, video and complete cost report |
| Checkpoint portability and export | Trained G1 4000 / AMP 2000 actor export max absolute difference 0; numeric source-normalizer checkpoint state; CPU/GPU partial reset smoke | Verified existing artifacts; export latest source-prior actor after its evaluation |
| Fresh-start instructions, versions and provenance | README, pinned lock/data fetch, isolated AMP setup, `licenses/PROVENANCE.md`, dependency-metadata exception documented | Provided; source examples run from repository root |
| Preserve previous work | Old G1/PPO experiments, failed compilation, all independent-prior checkpoints/videos remain; no remote/push | Preserved |

The requested later training is **AMP only**. No new long MBDAC-only or Isaac Lab
training is scheduled. Contact/catching is a future task: the package provides
physical-signal routing, not a claim that arbitrary Python rewards can be
automatically differentiated without state/measurement definitions.

Before completion: finish and physically evaluate the current source-prior run;
execute the strengthened observation validator after releasing the training GPU;
export that trained actor without simulator/RSL SDKs; upload measurements/video
and update the comparison/progress documents. Interpret negative results openly;
neither duration alone nor finite losses prove successful command control.
