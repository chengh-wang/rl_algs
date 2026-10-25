# rl_algs

Simulator-independent reinforcement learning for robotics, built with PyTorch
and installed as the **`mbdac`** Python package. Train with learned world models
or model-free FlashSAC through RSL-RL-style runners. Environment adapters own
physics, observations, rewards and action mapping.

[Installation](#installation) · [Quick start](#quick-start) ·
[Algorithm comparison](#algorithm-comparison) · [Examples](#examples) ·
[Documentation](#documentation)

## Supported algorithms

| Method | Configuration | Runner | Approach |
| --- | --- | --- | --- |
| MBDAC | `MBDAC` | `ModelBasedRunner` | Policy gradients through short learned-dynamics rollouts |
| MBDAC + AMP | `MBDACAMP` | `ModelBasedRunner` | MBDAC with an adversarial motion prior |
| WM + FastSAC | `MBDACFastSAC` | `ModelBasedRunner` | Real-data SAC updates with optional model-based actor gradients |
| FlashSAC | `FlashSAC` | `FlashSACRunner` | Model-free distributional SAC with real replay |
| SMP + FlashSAC | `SMPFlashSAC` | `SMPFlashSACRunner` | FlashSAC with a frozen diffusion motion prior |

Holosoma **FastSAC** is also available as an external comparison baseline in the
native velocity example. It is a separate implementation from **FlashSAC**.

## Installation

Requires Python **3.10+** and PyTorch **2.6+**. Install from this checkout;
the package is not published to PyPI.

```bash
git clone https://github.com/chengh-wang/rl_algs.git
cd rl_algs
python -m pip install -e .
```

The core uses PyTorch, NumPy, TensorDict and TensorBoard. Simulator SDKs are
optional. To add W&B logging and development tools:

```bash
python -m pip install -e '.[wandb,dev]'
```

## Quick start

The snippets below assume you have constructed an environment adapter named
`env`. Choose the adapter contract for your learning method.

### Model-free: FlashSAC

Provide `num_envs`, `num_actions`, `device`, `get_observations()` and an RSL-style
four-return `step()`. Observations are a dict or TensorDict with separate
`actor` and `critic` groups, each shaped `[num_envs, features]`.

```python
from mbdac.runners import FlashSACRunner

cfg = {"algorithm": {"class_name": "FlashSAC", "compile_mode": "default"}}
runner = FlashSACRunner(env, cfg, log_dir="logs/my_task", device="cuda:0")
try:
    runner.learn(num_learning_iterations=1000)
    runner.save("logs/my_task/model.pt", include_replay=True)
    policy = runner.get_inference_policy()
finally:
    runner.close()
```

Each FlashSAC iteration collects one vector step. Updates start after the default
warmup of **100,000 real transitions**; a short run with few environments may
only populate replay.

Actions use normalized `[-1, 1]` learner coordinates; the adapter maps them to
physical controls. Every done row must include pre-reset actor and critic
observations in `extras["terminal_observation"]`, with termination and timeout
flags kept distinct. See the [model-free contract](doc/smp_flashsac.md#输入与输出).

For SMP, use `SMPFlashSACRunner` and `class_name="SMPFlashSAC"` with an explicit
motion-prior reward/state contract. The [SMP guide](doc/smp_flashsac.md) covers
prior training, reward composition, configuration and simulator integration.

### Model-based: MBDAC

Use an adapter that additionally exposes `model_task`, goals, physical state
and pre-reset transitions, as described in the [model-based contract](doc/task_contract.md).

```python
from mbdac.runners import ModelBasedRunner

cfg = {"algorithm": {"class_name": "MBDAC"}}
runner = ModelBasedRunner(env, cfg, log_dir="logs/my_task", device="cuda:0")
try:
    runner.learn(num_learning_iterations=2000)
    runner.save("logs/my_task/model.pt")
finally:
    runner.close()
```

Choose `MBDACAMP` for an adversarial motion prior or `MBDACFastSAC` for the
WM + FastSAC learner. AMP also requires expert motion features.

Both runner families provide `load()`, `get_inference_policy()` and
`export_jit()`. Checkpoints store learner state, not simulator state. Replay
warmup precedes updates; iteration counts have different meanings across the
model-based and model-free runners and should not be compared as sample counts.

## Algorithm comparison

**Go1 flat velocity tracking · 1024 training environments · RTX 4070 Laptop
8 GB · MJLab 1.2.0 · seed 42.** These plots compare MBDAC, WM + FastSAC,
its matched real-only control and the external Holosoma FastSAC baseline.
FlashSAC and SMP + FlashSAC are not included in this benchmark.

### Performance versus time

![Go1 evaluation reward, velocity RMSE and survival versus logged checkpoint time](doc/assets/native_velocity_checkpoint_time.png)

### Performance versus real transitions

![Go1 evaluation reward, velocity RMSE and survival versus real transitions](doc/assets/native_velocity_real_transitions.png)

| Method | Logged time at first passing saved checkpoint | Real transitions |
| --- | ---: | ---: |
| FastSAC baseline | 3.679 min | 2,049,024 |
| MBDAC | 41.993 min | 9,025,536 |
| WM + FastSAC | 14.573 min | 2,059,264 |
| Matched real-only control | 7.603 min | 2,059,264 |

Passing requires XY RMSE ≤0.20 m/s, yaw-rate RMSE ≤0.30 rad/s and survival 1.0
for every evaluation command. Saved checkpoints were evaluated retrospectively;
their logged process times indicate checkpoint availability, not when the
evaluation result became available. Imagined samples are excluded from real
transition counts.

In this single-seed pilot, WM + FastSAC reaches acceptance earlier than MBDAC,
but shows no demonstrated wall-time or sample-efficiency advantage over the
FastSAC baseline or matched real-only control. G1 benchmark results and recipe
differences are in the full report.

[Protocol](doc/native_velocity_benchmark.md) ·
[Full results](doc/native_velocity_results.md) ·
[Plot data](doc/assets/native_velocity_evaluations.csv) ·
[Plot provenance](doc/assets/native_velocity_plot_provenance.json) ·
[W&B run](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/8490af7a)

## Examples

Examples and asset-preparation scripts run from the source checkout; the wheel
contains the learning core. Native velocity and G1 examples use MJLab managers
and require their own simulator dependencies.

<details>
<summary>Set up the tested MJLab 1.2 environment</summary>

The Linux lockfile targets Python 3.11, PyTorch 2.10, CUDA 12.8 runtime wheels
and MJLab 1.2.0. An NVIDIA driver is required; headless recording also needs
EGL/OpenGL and `ffmpeg`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-native-velocity.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

</details>

Train native Go1 velocity tracking:

```bash
.venv/bin/python -m examples.mjlab_velocity.train \
  --robot go1 --method wm_fastsac --recipe hybrid \
  --num-envs 1024 --iterations 100000 --max-train-seconds 3600 \
  --compile --logger tensorboard --log-dir logs/go1_wm_fastsac
```

Use `--robot g1` for G1, `--method mbdac` for MBDAC, or
`--recipe real_only` for the matched control. The external `--method fastsac`
baseline first requires `bash scripts/setup_native_velocity_baseline.sh`.

| Example | Setup and usage |
| --- | --- |
| Native Go1/G1 velocity | [Benchmark guide](doc/native_velocity_benchmark.md) |
| G1 motion tracking and AMP | [Data preparation and recipes](doc/implementation_recipes_2026-10-24.md) |
| Matched upstream AMP velocity | [AMP setup](doc/velocity_benchmark.md) |
| R1 SMP + FlashSAC | [Integration guide](doc/smp_flashsac.md) |

The R1 SMP integration was tested separately with MJLab **1.6** in its task
checkout. Its setup does not use this repository's MJLab 1.2 lockfile or extra.

## Documentation

- [Model-based environment contract](doc/task_contract.md)
- [Physical primitive heads and task formulas](doc/general_task_interface.md)
- [SMP / FlashSAC API, training and integration](doc/smp_flashsac.md)
- [Precision tasks: reward objectives, shaping and learner diagnostics](doc/precision_rewards.md)
- [World-model, actor and critic update schedules](doc/update_schedule_audit.md)
- [MBDAC implementation audit](doc/mbdac_audit_2026-10-24.md)
- [Method references, source attribution and notices](licenses/PROVENANCE.md)

Integration validation: **73 CPU tests passed**; the installed wheel passed
**70 tests with simulator imports disabled**. Optional tests were skipped in
each suite. Details and the separate R1 GPU update/restore checks are recorded
in the [integration guide](doc/smp_flashsac.md#验收范围).

> [!NOTE]
> The retained upstream SMP snapshot has no LICENSE file and is not covered by
> this project's MIT grant. See [SMP provenance](src/mbdac/smp/PROVENANCE.md) and
> [source notices](licenses/SMP-NO-LICENSE.txt).
