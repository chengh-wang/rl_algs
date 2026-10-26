# MBDAC project instructions

- Keep the learning core independent of simulator SDKs and robot-specific dimensions.
- Preserve the RSL-RL public runner calling convention. Framework imports belong in adapters or examples.
- MJLab examples must use native ManagerBasedRlEnv physics and managers.
- Store the applied action and physical next state before auto-reset; never fit dynamics across resets.
- Predict physical primitives, not weighted reward totals. Compose rewards and observations from the primitives with explicit units and frames. Do not train duplicate heads for quantities already represented in the model state or available from analytic kinematics.
- Keep real, imagined, and reference datasets distinct. Fit the world model to real transitions.
- The accepted laptop training scale is 1024 real worlds, WM batch 128 and actor batch 256. Preserve the reference update schedule and evaluate policy quality at this documented memory-limited scale.
- During actor updates freeze world model, critic, and discriminator parameters while preserving input gradients.
- Keep source attribution and licenses for reused SuperMimic / RSL-RL code.
- Do not delegate to subagents unless the human explicitly authorizes delegation.

## Git publication

- Publish approved, tested changes to `main` unless the user requests another branch.
- Use author and committer `chengh-wang <wang.chengh@northeastern.edu>`.
- Do not create `codex/*` branches for this repository.
- Preserve existing local work before switching branches. Keep superseded research
  snapshots separate from the current implementation; do not restore withdrawn
  learner changes while consolidating branches.
