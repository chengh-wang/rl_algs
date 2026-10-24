# Matched AMP velocity benchmark

This example imports the retained [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) environment at commit `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`. It is the same source as the earlier local 1.5k AMP+PPO experiment. The original checkout and experiment are preserved.

## Installation and run

```bash
# Pass an existing clean checkout at the pinned revision, or omit the argument
# to download a separate copy into ignored external/AMP_mjlab.
bash scripts/setup_amp_benchmark.sh /path/to/AMP_mjlab
.venv-amp/bin/python -m examples.mjlab_amp.train --algorithm MBDAC \
  --num-envs 16 --iterations 2 --smoke --logger tensorboard \
  --log-dir logs/velocity_interface
.venv-amp/bin/wandb login
.venv-amp/bin/python -m examples.mjlab_amp.train --algorithm MBDACAMP \
  --num-envs 1024 --iterations 5000 --compile --log-dir logs/velocity_mbdacamp_1024
.venv-amp/bin/python -m examples.mjlab_amp.validate
# Both the new actor and retained PPO checkpoint use this same evaluator.
.venv-amp/bin/python -m examples.mjlab_amp.evaluate \
  logs/velocity_mbdacamp_1024/model_5000.pt --seconds 20 --video \
  --model-diagnostics --output logs/velocity_evaluation
.venv-amp/bin/python -m examples.mjlab_amp.evaluate /path/to/ppo_model_1500.pt \
  --seconds 20 --video --output logs/ppo_evaluation
.venv-amp/bin/python -m examples.mjlab_amp.compare \
  logs/ppo_evaluation/evaluation.json logs/velocity_evaluation/evaluation.json \
  --output logs/velocity_comparison
```

The example's `.venv-amp` contains the upstream RSL fork and its supplied MJLab observation-manager patch, as the baseline requires. The main `.venv`, the old experiment's environment, host CUDA driver, and upstream tracked source files are unchanged. Installing a shared package through its normal editable installer does not require `sys.path` injection.

There is one deliberate dependency-metadata exception in this isolated example:
MJLab 1.2.0 declares `rsl-rl-lib==5.0.1`, whereas the upstream AMP fork reports
`rsl-rl-lib==2.3.1` and supplies its own configuration migration. Consequently
`uv pip check --python .venv-amp/bin/python` reports this one incompatibility.
The setup script installs the pinned fork with `--no-deps`; the actual task,
gradient updates and checkpoint evaluations have been tested with that combination.
This exception is confined to `.venv-amp`. The standalone core does not require
RSL-RL, and the main MJLab lock retains its official 5.0.1 dependency. Rerun the
example setup script after changing its dependencies; a general dependency
resolver can replace the AMP fork with the official package, which lacks this
fork's expert loader and runner. No package version is relabeled to hide the
exception.

For this velocity task, `--compile` compiles WM dynamics while leaving analytic task heads in ordinary Torch autograd (`compile_task=false`). Compiling the full contact/FK reward backward triggered a Triton 3.6 `PassManager::run failed` on the tested PyTorch 2.10 laptop setup. This changes execution strategy, not the reward or model. `--compile-task` opts into compiling those heads on a compatible stack. The failed attempt and its saved warmup checkpoint are preserved.

For timing a resumed experiment, record `started_unix`, `source_commit`, and
`resume_checkpoint` in its launch JSON. Set `prior_process_seconds` to the total
measured execution time of all earlier processes in that checkpoint lineage,
including failed compilation after the saved checkpoint. Pass the JSON through
`compare --candidate-launch`. The report distinguishes cumulative trainer time,
the current launch-to-checkpoint interval, and the total process cost. Idle time
spent fixing code is not training time; separate earlier preflight experiments
are not part of the final checkpoint lineage.

## Preserved real task

The native G1 flat environment factory owns the robot XML, torque-speed actuators, action scale, 5 ms physics / 20 ms controls, 17 walk/run reset clips plus one recovery clip, observations, rewards and weights, heading commands, command curriculum, pushes, friction/encoder/COM randomization, and delayed recovery termination. A zero-valued reward hook captures physical transitions before reset; it does not replace the native step loop or reward terms.

The privileged critic additionally receives the native delayed-recovery flag, counter and readiness state. Those determine reward and termination even for identical physical states; the physical world model does not need to predict them. The actor keeps the original proprioceptive/command interface.

The learner reads the last frame of the native observation manager's output, including its exact sensor values, order and noise sample. It maintains four-frame history and also has a three-dimensional goal encoder. Actor/critic hidden widths are 512/256/128 for this velocity example. These architecture differences and the 1024-world/batch adaptations must be reported against the retained 4096-world PPO run.

## Model task and basic physical heads

Native reward functions are reused against a differentiable Torch data view of predicted state and analytic FK. Missing physical measurements are predicted separately: joint acceleration in rad/s², foot-contact probability, and self-contact force vectors over physics substeps in N. They receive the real/predicted transition and physical context. No reward sum is a head target. Foot contact, force-threshold events, and failure conditions use smooth gates only during imagination.

Three approximations are explicit:

- Native reward managers observe derived kinematics lagging raw qpos/qvel by one physics substep. Analytic imagined FK uses the predicted state at the end of the control interval. The reward audit measures this timing difference separately from formula equality.
- External velocity commands are held over a short imagined horizon. Real collection retains the original heading controller and random command resampling.
- Real interval pushes remain enabled. Replay windows stop at a detected exogenous state boundary, so the deterministic world model does not fit an unobserved velocity jump. Collection and policy histories continue from the actual pushed state.

The AMP extension uses the baseline fork's expert loader, body order, feature definitions and reference files. Its discriminator learns from real policy/expert pairs; its differentiable style reward contributes to imagined actor learning. Quality and speed comparisons require actual training and matched physical evaluation; an interface smoke test is insufficient.

The evaluator uses nominal standing initialization, five fixed commands (forward, backward, lateral, turning and standing), no observation noise or perturbations, and no automatic reset. Native failure conditions are measured externally. Steady velocity and gait metrics exclude the first two seconds and all post-failure samples. `--model-diagnostics` additionally evaluates dynamics and each physical primitive head on newly collected real transitions; head errors are reported both with true states and with predicted rollout states.

The original `velocity`/RMSE fields measure pelvis velocity in its body frame.
Additional `anchor_velocity`/`anchor_*` fields measure torso XY velocity in its
yaw-only frame and torso angular velocity along world Z, matching the native
reward's commanded quantities. Both sets are reported separately. Earlier
prototype root/body metrics are retained with their original meaning. Reevaluate
the retained PPO checkpoint with the current evaluator when comparing the added
anchor fields; changing frames silently would invalidate the comparison.

Expert loading keeps the original generator alive across discriminator updates. The upstream loader chooses clips from its internal batch index; repeatedly creating a generator and taking its first batch would train on only the first clip. The adapter preserves the baseline's 20-batch cycle, covering all 18 clips and then repeating clips 0 and 1, before starting the next cycle.

MBDACAMP keeps the first 500 iterations strictly WM-only. Afterwards each outer iteration fits WM ten times, updates the discriminator once on real/expert data, updates the actor once through the frozen model/prior, and fits the critic twenty times. This avoids pretraining a discriminator against an actor that cannot yet change. The AMP extension is our design; SuperMimic does not prescribe an AMP schedule. Discriminator learning rate follows the learner schedule unless explicitly overridden in `amp.lr`.

## Direct upstream prior reuse

New velocity runs now default to `--prior upstream`. This uses the installed
fork's `Discriminator`, `Normalizer`, and `compute_grad_pen` directly, including
ReLU, ±10 normalization clipping, raw-expert gradient penalty, optimizer weight
decays, and its post-step normalization-update ordering. The fork updates those
statistics from normalized current-state samples; this behavior is retained for
the comparison rather than silently replaced. The task/style blend is unchanged.
The only reward wrapper change is retaining input gradients for BPTT; the source
PPO prediction helper has a `no_grad` context.

The example reads the upstream configuration's initial 1e-3 discriminator learning
rate and holds it fixed (`amp.lr`). WM/actor/critic retain the reference model-based
2e-4 schedule. PPO's KL-dependent joint learning-rate changes are not transferred
to the deterministic model-based actor. Update counts and batch sizes also remain
the documented laptop adaptations, so this is still not a pure actor-only ablation.

`--prior independent` retains the earlier generic prior. Existing checkpoints
without a backend field automatically resume with that legacy prior. To reuse a
WM-only legacy checkpoint and initialize a new upstream discriminator:

```bash
.venv-amp/bin/python -m examples.mjlab_amp.train --algorithm MBDACAMP \
  --num-envs 1024 --compile --resume /path/to/model_500.pt \
  --prior upstream --reset-prior --iterations 2500 \
  --log-dir logs/velocity_upstream_prior
```

This ends at outer iteration 3000. It restores actor/critic/WM, their optimizers,
normalization and counters, while intentionally initializing a new discriminator.
Use an actually WM-only checkpoint for this controlled restart; changing prior
architecture without `--reset-prior` is rejected by strict checkpoint loading.
`tests/test_upstream_prior.py` compares reward values, input gradients, a full
discriminator optimizer step and normalization statistics against the actual
fork, then checks checkpoint restoration. The native 1024-world smoke run is an
interface check, not evidence of improved policy quality.


## Completed local experiment

The native example now defaults to5000 iterations, reflecting the observed time
needed for a visible gait in this configuration; it is not a convergence or
quality guarantee. Source-prior4000 has forward body errors near PPO in one
probe, but worse native torso errors/slip. Source-prior5000 improves gait and
backward tracking, overshoots forward speed and remains weak laterally/turning.
See [the final report](final_report_2026-10-24.md). Actual policy outcomes are
measured separately from numerical integration checks.

Launch metadata may include `other_experiment_process_seconds` (or the older
`other_independent_prior_prototype_process_seconds`). `compare` reports this
separately from checkpoint lineage and also reports their sum. Those recorded
training costs exclude engineering/evaluation/smoke checks unless explicitly
included. This prevents hidden prototype cost from being mistaken for a
cold-start runtime improvement.
