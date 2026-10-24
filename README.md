# MBDAC

Independent PyTorch package for **MBDAC** (model-based differentiable actor-critic) and **MBDACAMP** (the same learner with an AMP motion prior). The public runner follows RSL-RL conventions. Simulator gradients are not required.

Version 0.3 adds **MBDACFastSAC**: the pinned Holosoma FastSAC numerical actor/Q modules, real-only distributional Q fitting, and a short entropy-regularized world-model actor rollout. It is a model-based extension of **Holosoma FastSAC**, not the separate FlashSAC paper implementation. Existing MBDAC/MBDACAMP algorithms and checkpoints remain independent. Early 0.2 FastSAC prototypes omitted Q controller memory and are retained as debugging evidence, not valid method comparisons.

The first corrected FastSAC+WM pilot completed 2000 updates but **did not learn full motion tracking**: the final policy triggers failure at 2.12 seconds of a 6-second reference. Numerical/interface validation below does not establish policy quality or a training speedup.

Version 0.3.1 continues debugging toward full physical tracking. It removes endpoint-only reference reset bins, gives finite reference completion explicit non-bootstrap semantics, adds an optional Q-only warmup gate, and batches model reward evaluation without changing its objective/gradients. The new goal and diagnostic evidence are recorded in [progress](doc/progress_2026-10-24.md#fastsac-world-model-debug-goal-active).

Experimental implementation with completed native **MJLab** training and measured comparisons. G1 motion tracking and AMP velocity learning run in 1024 `ManagerBasedRlEnv` worlds; full updates, physical evaluations, resume and SDK-free actor export are verified. Velocity quality is mixed and does not establish an overall speedup at equal quality. Other simulator backends have adapter/smoke evidence only and were not used for the current training.

## Fresh installation (Linux, NVIDIA GPU)

Use Python 3.11 and a working NVIDIA driver. The tested environment uses PyTorch 2.10.0 with its CUDA 12.8 runtime wheels; a separate CUDA toolkit installation is not required for this example. This creates an isolated environment and does not change existing Conda environments.

If `uv` is absent, install it using the [official installer](https://docs.astral.sh/uv/getting-started/installation/), then install Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
. "$HOME/.local/bin/env"
uv python install 3.11
```

The system needs `git`, a working NVIDIA driver (`nvidia-smi`), and EGL/OpenGL libraries for headless MuJoCo video rendering. Keep the existing driver; the locked Python dependencies supply the tested Torch/CUDA runtime and video encoder.

```bash
# From this repository, with uv available:
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-mjlab.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_g1_data.py
.venv/bin/python -m pytest -q
.venv/bin/python -m examples.mjlab_g1.validate
```

The lock file records the tested Linux environment. For a different platform, resolve the extras instead: `uv pip install --python .venv/bin/python -e '.[mjlab,wandb,dev]'`. The learning core alone can be installed with `pip install .`; it does not depend on MJLab or Isaac Lab. The examples run from this source checkout.

The data script fetches 40 G1 LAFAN1 clips plus G1 assets at a pinned Hugging Face revision, preserves source notices, records SHA256 hashes, and compiles the original URDF into a free-base MJCF. It does not generate or replace the existing R1/BONES library.

## Native MJLab training

```bash
.venv/bin/wandb login
.venv/bin/python -m examples.mjlab_g1.train \
  --num-envs 1024 --iterations 2000 --log-dir logs/g1_mbdac_1024
```

The first 500 iterations fit the world model. Later iterations continue fitting it while updating actor and critic through 32-step imagined rollouts. `2000` is an initial run length, not a convergence claim. W&B, TensorBoard, JSONL metrics, runtime versions and checkpoints are saved under the log directory. `wandb.json` contains the actual run URL. No simulator gradient is used.

```bash
# Fast interface check; small networks, not a locomotion-quality test.
.venv/bin/python -m examples.mjlab_g1.train --num-envs 1024 \
  --iterations 3 --smoke --logger tensorboard --log-dir logs/smoke

# Continue a checkpoint; replay is refilled before updates when omitted.
.venv/bin/python -m examples.mjlab_g1.train --num-envs 1024 \
  --resume logs/g1_mbdac_1024/model_2000.pt \
  --iterations 2000 --compile --log-dir logs/g1_mbdac_1024_continued

# Actual physics evaluation, without automatic reset; reference left, policy right.
.venv/bin/python -m examples.mjlab_g1.evaluate \
  logs/g1_mbdac_1024/model_2000.pt --seconds 20 --output logs/evaluation_2000
```

Evaluation writes an MP4 and per-clip survival, return, joint RMSE and body-position RMSE, plus one-step/eight-step model errors in physical units on newly collected evaluation transitions. These diagnostics do not update model parameters or normalization. A 20-second prefix is identified as a prefix; it is not reported as full-clip success. Domain randomization and sensor noise are disabled for this evaluation. Training uses the supplied reference implementation's reset noise, friction/COM/calibration randomization, reward formulas, joint order, PD gains and per-joint action scales.

`--algorithm MBDACAMP` enables the additional adversarial prior. That extension has gradient-isolation tests and actual native 1024-world learning/evaluations; measured velocity quality remains limited. Its discriminator uses real policy transitions and motion reference pairs, while its style reward contributes gradients through imagined actor rollouts. This is an extension to SuperMimic, not a claim that its reference code already contains AMP imitation learning.

## Holosoma FastSAC with a differentiable world model

The learning core vendors the unchanged SDK-free `Actor`, `Critic` and categorical projection from Holosoma revision `d18d6cc50f872c15e904a22ceac22313cec955c8`. Apache-2.0 license/NOTICE and a source hash manifest are included in the wheel. Installing Holosoma, Isaac Sim, or another Torch/CUDA environment is unnecessary for this extension.

| Module | Responsibility and integration |
| --- | --- |
| `vendor/holosoma/fast_sac.py` | Original tanh Gaussian/`rsample` actor, LayerNorm networks, Q ensemble and C51 projection, byte-identical to pinned source |
| `vendor/holosoma/normalization.py` | Verbatim source empirical-normalization class, extracted without Holosoma runtime imports |
| `modules/holosoma.py` | History/goal layout, functional normalization with input gradients and fixed real-data statistics, Q expectation; preserves source normalization epsilon/statistics |
| `algorithms/mbdac_fast_sac.py` | WM fit, real-only Q/temperature updates, short model actor objective, target EMA, full save/load |
| Existing WM/replay/task modules | Physical-state codec, real autoregressive fit, analytic-first physical heads, reset-safe windows and observations/reward/termination |
| `examples/mjlab_g1/fast_sac.py` | Named G1 joint-bound/PD action contract and phase clamp matching native pre-reset reference |
| `examples/mjlab_g1/train_fast_sac.py` | Single-clip data provenance and native manager-based launch; no new reward or physics implementation |

The actor maximizes a discounted imagined sum of `reward - alpha * log_prob`, with soft survival and terminal `mean_Q - alpha * log_prob`. At horizon 0 its loss and every actor gradient match the original Holosoma actor objective. Horizons greater than 0 explicitly change that objective. Parameters of WM and Q are frozen during actor updates while state/action input derivatives remain live. C51 targets and temperature updates are detached. The source `explore()` and normalizer `forward()` disable gradients and therefore cannot be used unchanged inside BPTT; the numerical actor's differentiable sampling method and a functional normalization adapter are used instead.

**Q state is the controlled-system state.** The learner appends previous action and the complete policy observation history to task-provided physical/goal/context features. This is necessary for action-rate rewards and a history-conditioned policy; reusing the original scalar-V features alone aliases different immediate rewards and future actions. The G1 task also includes the exact reference clock in privileged Q features. Real replay stores actual current/physical-next observations, preserves their sensor noise and reset padding, and excludes starts whose preceding policy memory has already been evicted. Model actor rollouts update the same memory recursively. Tasks must still expose any additional task/actuator/reward memory in their physical state or critic features; the package cannot infer arbitrary hidden state.

FastSAC checkpoint state version 2 includes this Q memory contract. Full resumes of the earlier incomplete-Q prototypes are explicitly rejected; compatible actor/WM-only warm starts remain available. Original MBDAC/MBDACAMP checkpoint formats and behavior are unchanged.

`fast_sac.model_actor_weight` optionally blends this objective with the original FastSAC actor objective on real replay starts: `loss = weight * model_loss + (1-weight) * real_SAC_loss`. Default 1 preserves pure model actor updates and existing checkpoints; 0 skips model rollout for the actor. Intermediate values retain actual WM input gradients while anchoring the actor on real-state Q gradients. This is a configurable surrogate mixture, not a guarantee of monotonic improvement. CLI `--model-actor-weight`, `--log-std-min`, and `--log-std-max` make these choices explicit. Any narrower exploration bounds differ from the original WBT recipe and must be reported.

Alternatively `--action-std-min 0.01 --action-std-max 0.2` declares exploration in **raw control-action units stored by the WM/replay**, before native PD scaling. The adapter converts these to per-joint pre-tanh log-std bounds using the actor's action bounds. Because tanh is 1-Lipschitz, `action_scale * pre_tanh_std` upper-bounds the resulting action std. The full mean-action range, physical PD scales, densities/Jacobian correction and rewards remain unchanged. This is an explicit model-support/exploration choice, not part of the original Holosoma preset. Checkpoints include the converted per-joint buffers; metrics report the actual std upper bounds.

`--real-actor-batch` can give the real-state SAC component a large batch independently of the memory-limited `pi_batch` used for model BPTT. It applies to the mixed objective; the pure model/H=0 parity path keeps its documented `pi_batch`. Real and imagined objectives draw from the same real replay-start distribution but have independent Monte Carlo batch sizes.

For real transitions with stochastic/external command updates, adapters provide pre-reset `condition1`; real Q uses it directly. Deterministic motion clocks may use `task.advance` as the fallback. Imagined goals follow the task's explicit advance rule, independently of learned physical dynamics.

Optional transition `bootstrap` masks distinguish semantic task completion from ordinary time limits. They can suppress bootstrapping but never restore it after a true physical termination. If the adapter does not provide one, an optional pure `task.bootstrap_mask(physical_next_state, next_condition)` supplies it during collection; otherwise the original time-limit behavior remains. G1's finite-reference task returns zero at the final frame and its imagined termination head uses the same endpoint. Old G1 checkpoints retain their previous endpoint semantics during evaluation; full CLI resumes across this task-contract change are rejected. A compatible physical WM can still be reused explicitly.

`fast_sac.critic_warmup_updates` (default 0) holds the actor fixed for that many real Q/temperature/EMA optimizer steps after the WM-only phase. The Q update counter is checkpointed, so resume preserves the gate. `--actor-lr`, `--policy-frequency`, and `--compile-task` expose experiment settings; the last one compiles batched pure task heads as well as the WM. Long model rollouts now evaluate reward/termination once over `H*B`, following the existing MBDAC path; sequential and batched losses/all actor gradients are regression-checked.

Default schedule: initial 64 real steps; 500 WM-only outer iterations; then each outer iteration collects alternating 2/3 vector steps and performs **10 WM → 4 real Q/temperature/target updates with 2 actor updates interleaved**. This is a new hybrid schedule, not SuperMimic's 10:1:20 schedule or an unchanged Holosoma `learn()` loop. Q uses the original ensemble mean, distributional cross-entropy, tau 0.05, gamma 0.99, 501 atoms on [-20,20], source learning rates/AdamW decay and entropy target. The default Q batch is 256; the measured pilot below explicitly uses the source's 8192, independently of the model actor's batch 256. Model actor gradients are clipped at norm 1 for BPTT. Symmetry/CNN/distributed training and n-step Q returns are not implemented by this adapter; this WBT example uses one-step Q returns, as does the pinned WBT preset.

```bash
# Uses the same .venv and robot/motion assets as the existing G1 example.
.venv/bin/python -m examples.mjlab_g1.train_fast_sac \
  --smoke --num-envs 1024 --iterations 3 --logger tensorboard \
  --log-dir logs/g1_wm_fastsac_smoke

# Small pilot: original reference seconds 4--10, all networks initialized fresh.
# Model BPTT batch stays 256; the real-state SAC branch uses a larger batch.
.venv/bin/python -m examples.mjlab_g1.train_fast_sac \
  --num-envs 1024 --iterations 2000 --horizon 3 --compile \
  --critic-batch 8192 --real-actor-batch 8192 \
  --start-seconds 4 --clip-seconds 6 --model-actor-weight 0.1 \
  --action-std-min 0.01 --action-std-max 0.2 \
  --log-dir logs/g1_wm_fastsac_v5

.venv/bin/python -m examples.mjlab_g1.evaluate \
  logs/g1_wm_fastsac_v5/model_2000.pt \
  --motion-dir logs/g1_wm_fastsac_v5/motion --seconds 6 --video-seconds 6 \
  --continue-after-failure \
  --output logs/g1_wm_fastsac_v5_eval_2000
```

The subset records original/selected-file SHA256 and frame range; original data is never overwritten. `--start-seconds` selects another explicit window, `--critic-batch` changes only Q sampling volume, and `--wm-checkpoint` loads only a compatible native G1 physical model while leaving actor/Q/temperature fresh. `--wm-only-iters` sets its adaptation period. Report any previous WM training separately; a reused model is not a from-scratch speed result. Full resume checks the identical motion subset and algorithm recipe. Actor/Q math runs in FP32; the large WM uses the configured precision. Run length is an initial probe, not a convergence guarantee. Evaluate the actor in physical simulation and report WM fitting/collection/actor/Q costs together. New configs select `class_name="MBDACFastSAC"` through the same public runner. Full resume includes target Q, temperature/optimizer and update counters; actor-only inference/export needs no WM or simulator SDK. `load_cfg={"temperature": True}` selects a temperature warm start. Existing MBDAC checkpoints cannot directly initialize this different actor architecture.

### Measured small pilot, 2026-10-24

Run [uya7n6k7 on W&B](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-fastsac-tracking/runs/uya7n6k7) used native MJLab 1.2.0 `ManagerBasedRlEnv`, 1024 worlds, one unchanged G1 LAFAN1 reference window (original seconds 4–10), seed 42, and the command above. Actor, Q and WM all started fresh. Training source was `c104eefc16e938517cd8abb5253244e0e16ea534`. The actor objective was explicitly **0.1 model / 0.9 real-state SAC**, with model horizon 3; this is not evidence for the default pure-model objective or an unchanged upstream recipe.

Training took **1718.42 seconds (28 min 38 s)** including WM-only warmup and real collection; the whole process took 1740.37 seconds including initialization/log finalization. It collected 5,185,536 real transitions and checkpoint auditing confirmed 20,000 WM, 6,000 Q/temperature and 3,000 actor optimizer steps. A CPU evaluation of checkpoint 1000 ran during training, so this is a laptop trial timing rather than an isolated throughput benchmark.

Both checkpoints below were evaluated on CUDA in native physics, from the same initial reference state with seed 0, no observation noise/domain randomization, and no automatic resets. Metrics stop at the first failure; the 6-second videos continue physical simulation and show a red failure banner. The left panel is kinematic reference playback; the right panel is the actual policy.

| Checkpoint | Time until failure / 6 s | Joint RMSE before failure | Full tracking |
| --- | --- | --- | --- |
| 1500 | 1.02 s | 1.002 rad | Failed |
| 2000 | 2.12 s | 0.769 rad | Failed |

The final policy remains unusable for complete tracking. Reward/episode-length curves recover late in training but do not establish convergence; no equal-quality speed advantage over MBDAC or PPO is established. Earlier incomplete-Q debugging runs consumed a further 3024.01 process seconds (50 min 24 s); those are not valid method comparisons. Debug runs v2/v3 borrowed an existing WM, whose original training cost is recorded separately; this final run did not reuse it.

Validation is separate from this unsuccessful policy result: 41 native-environment tests pass (one optional skip), and the installed wheel passes 40 tests without simulator SDKs (two optional skips). Checks include upstream H=0 actor loss/all-gradient equivalence, a complete C51 optimizer step, reset-safe noisy policy memory, previous-action Q state, frozen WM/Q parameters with live actor input gradients, update/resume semantics and legacy-checkpoint rejection. Native contract validation found zero policy-history reconstruction error and verified applied PD action/pre-reset physical transitions. Export of the actual final actor through `ModelBasedRunner.export_jit()` runs without MJLab/MuJoCo/Holosoma/RSL-RL; eager versus exported outputs match exactly for batches 1 and 16. These checks do not validate the learned policy's behavior.

Local generated evidence (also uploaded to the W&B run): `logs/g1_wm_fastsac_v5_report/{summary.json,learning_curves.png}`, `logs/g1_wm_fastsac_v5_eval_{1500,2000}/{metrics.json,reference_left_policy_right.mp4}`, `logs/g1_wm_fastsac_v5_export/{policy.pt,audit.json}`, and `logs/fast_sac_native_contract.json`. These files are not required to install the wheel; checkpoints, data and generated logs remain outside source control.

## RSL-style use

```python
from mbdac.runners import ModelBasedRunner

cfg = {"algorithm": {"class_name": "mbdac.algorithms:MBDAC"}, "logger": "wandb"}
runner = ModelBasedRunner(env, cfg, log_dir="logs/run", device="cuda:0")
runner.learn(num_learning_iterations=2000)
runner.save("logs/run/checkpoint.pt")
runner.load("logs/run/checkpoint.pt")
policy = runner.get_inference_policy(device="cuda:0")
runner.export_jit("logs/run/policy.pt")
```

The vector environment returns a TensorDict with single-frame `actor` observations and `goal` features, and the RSL-style `(observations, rewards, dones, extras)` tuple. This is API similarity, not a drop-in replacement for an arbitrary PPO task: model-based learning also needs the physical task functions and pre-reset transition capture described in [the task contract](doc/task_contract.md).

Inference maintains observation history; call `policy.reset(dones)` after environment resets. The exported policy is stateless and takes `(observation_history, goal)` explicitly. It contains only the actor and normalization; the caller applies action clipping and its environment's action scale/PD interface. Export does not change live training parameters, device, or mode. Resume restores networks, optimizers, normalization, RNG, iteration and cumulative training time; physics state is reset and replay is refilled if not included. It is not a bitwise continuation of the simulator.

Like RSL-RL, `load_cfg` selects a warm start: `runner.load(path, load_cfg={"actor": True})` loads only the actor without advancing counters or restoring RNG. Available keys are `actor`, `critic`, `world_model`, `optimizer`, `replay`, `iteration`, `rng`, and `prior` for MBDACAMP or `temperature` for MBDACFastSAC. Omitting `load_cfg` requests full resume. Actor and model normalizers use real data. The original MBDAC/MBDACAMP critic normalizer uses detached imagined features; MBDACFastSAC's Q normalizer uses real physical features only.

## Differences from the reference training scale

The example preserves the reference's full network widths, 10-frame observation history, 8-step world-model fitting, 32-step actor horizon, learning rate, lambda-return critic and analytic task formulas. Laptop defaults use 1024 real environments, world-model batches of 128, actor batches of 256, and 256 replay time steps, rather than the reference's 4096 / 2048 / 4096 / 1000. These affect performance and must be reported in comparisons. Policy actions are clipped consistently in real and imagined rollouts. `--compile` optionally compiles model/analytic heads; the initial numerical validation used eager execution.

See the [update schedule audit](doc/update_schedule_audit.md) for the exact 10 WM / 1 actor / 20 critic optimizer steps per iteration, UTD definition, and the reference repository's additional 500-iteration warmup.

The [MBDAC review](doc/mbdac_audit_2026-10-24.md) records numerical equivalence, replay-resume fixes, trained-policy export and measured optimization. For new task rewards, [PhysicalSignals](doc/general_task_interface.md) selects analytic quantities first and automatically declares/captures missing physical heads.

The observed velocity baseline used a different task, AMP+PPO and hardware load. Timing a few tracking updates does not establish an advantage over that baseline. The [matched velocity benchmark](doc/velocity_benchmark.md) reuses the retained baseline's native MJLab task and expert loader; actual training duration and physical quality are reported separately. See [current measured progress](doc/progress_2026-10-24.md).

The first [matched AMP results](doc/velocity_results_2026-10-24.md) include actual
1000–4000 checkpoint evaluations of the independent prior. Survival improves,
but no checkpoint matches PPO's forward tracking quality; 4000 shuffles too
slowly. The shorter training time is therefore not an advantage at matched
policy quality. New native velocity runs directly reuse the upstream fork's
discriminator/normalizer/gradient penalty through a task-owned prior factory;
numerical module parity, native smoke tests and actual training through 5000
are evaluated in the final report. Existing checkpoints retain their legacy
prior automatically. See the benchmark guide before changing prior backends.

The [source-prior 3000 evaluation](doc/velocity_upstream_results_2026-10-24.md)
shows improving forward motion, with body XY RMSE 0.307 m/s versus PPO's 0.158.
The completed 5000 run improves gait and backward tracking, with mixed forward
quality; lateral/turning remain weak. Body and native torso-reward-frame errors
are separate. See [final delivery and comparison](doc/final_report_2026-10-24.md).

## Design contract

- Core learning code imports neither MJLab nor Isaac Lab.
- Dynamics learn from real simulator transitions; actor/critic learn from short imagined trajectories.
- Reward and observation formulas are reused. Learned heads predict missing **physical primitives** (velocity, contact probability, etc.), never weighted reward totals. Reward weights, targets, tolerances, and scales remain configurable without refitting the primitives.
- A physical quantity already present in the predicted state or analytic FK does not need a second learned head.
- Termination thresholds are applied to physical primitives; imagination uses smooth termination probabilities.
- Real transitions retain pre-reset next states, applied actions, context, and episode boundaries.

## Sources

Algorithm: [SuperMimic](https://arxiv.org/abs/2610.28480), sections 3 and appendices A-G.
Initial implementation reused under MIT from [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed), pinned at `a1508d42c9d1699a52e8233ab16d08c488441ab9`; see `licenses/` and file headers. This package is an independent implementation and is not the official SuperMimic release.

Interface reference: [RSL-RL](https://github.com/leggedrobotics/rsl_rl), public 5.x runner and TensorDict interfaces.
