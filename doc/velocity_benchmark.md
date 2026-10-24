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
.venv-amp/bin/python -m examples.mjlab_amp.train --algorithm MBDACAMP \
  --num-envs 1024 --iterations 2000 --compile --log-dir logs/velocity_mbdacamp_1024
.venv-amp/bin/python -m examples.mjlab_amp.validate
# Both the new actor and retained PPO checkpoint use this same evaluator.
.venv-amp/bin/python -m examples.mjlab_amp.evaluate \
  logs/velocity_mbdacamp_1024/model_2000.pt --seconds 20 --video \
  --model-diagnostics --output logs/velocity_evaluation
.venv-amp/bin/python -m examples.mjlab_amp.evaluate /path/to/ppo_model_1500.pt \
  --seconds 20 --video --output logs/ppo_evaluation
```

The example's `.venv-amp` contains the upstream RSL fork and its supplied MJLab observation-manager patch, as the baseline requires. The main `.venv`, the old experiment's environment, host CUDA driver, and upstream tracked source files are unchanged. Installing a shared package through its normal editable installer does not require `sys.path` injection.

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
