# MBDAC

Independent PyTorch package for **MBDAC** (model-based differentiable actor-critic) and **MBDACAMP** (the same learner with an AMP motion prior). The public runner follows RSL-RL conventions. Simulator gradients are not required.

Implementation is in progress. The current training and acceptance work uses **MJLab**, with native `ManagerBasedRlEnv` physics and managers. G1 tracking runs in 1024 parallel environments; full-network updates, physical evaluation, checkpoint resume and export have passed. Policy quality is still being evaluated. Other simulator backends are outside the current training scope.

## Fresh installation (Linux, NVIDIA GPU)

Use Python 3.11 and a working NVIDIA driver. The tested environment uses PyTorch 2.10.0 with its CUDA 12.8 runtime wheels; a separate CUDA toolkit installation is not required for this example. This creates an isolated environment and does not change existing Conda environments.

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

`--algorithm MBDACAMP` enables the additional adversarial prior. That extension has CPU gradient-isolation tests and a native 1024-world interface check; native learning quality remains unverified. Its discriminator uses real policy transitions and motion reference pairs, while its style reward contributes gradients through imagined actor rollouts. This is an extension to SuperMimic, not a claim that its reference code already contains AMP imitation learning.

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

Like RSL-RL, `load_cfg` selects a warm start: `runner.load(path, load_cfg={"actor": True})` loads only the actor without advancing counters or restoring RNG. Available keys are `actor`, `critic`, `world_model`, `optimizer`, `replay`, `iteration`, `rng`, and `prior` for MBDACAMP. Omitting `load_cfg` requests full resume. Actor and model normalizers use real data; the critic normalizer uses detached imagined features, following the reference method.

## Differences from the reference training scale

The example preserves the reference's full network widths, 10-frame observation history, 8-step world-model fitting, 32-step actor horizon, learning rate, lambda-return critic and analytic task formulas. Laptop defaults use 1024 real environments, world-model batches of 128, actor batches of 256, and 256 replay time steps, rather than the reference's 4096 / 2048 / 4096 / 1000. These affect performance and must be reported in comparisons. Policy actions are clipped consistently in real and imagined rollouts. `--compile` optionally compiles model/analytic heads; the initial numerical validation used eager execution.

See the [update schedule audit](doc/update_schedule_audit.md) for the exact 10 WM / 1 actor / 20 critic optimizer steps per iteration, UTD definition, and the reference repository's additional 500-iteration warmup.

The observed velocity baseline used a different task, AMP+PPO and hardware load. Timing a few tracking updates does not establish an advantage over that baseline. The [matched velocity benchmark](doc/velocity_benchmark.md) reuses the retained baseline's native MJLab task and expert loader; native learning quality and speed comparisons remain acceptance items. See [current measured progress](doc/progress_2026-10-24.md).

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
