# MBDAC review and AMP readiness — 2026-10-24

The shown G1 checkpoint 4000 is retained unchanged. It took 3435.28 s (57 min 15 s) of cumulative active training in 1024 native MJLab worlds, including WM warmup and compilation. No additional long MBDAC training is planned for this review. Only the AMP version proceeds to new long training.

## Correctness

- Rechecked the reference implementation's local SO(3) increments, history construction, pre-reset physical targets, action scales, differentiable survival, actor bootstrap and detached lambda-return critic. The default optimizer schedule remains ten WM, one actor and twenty critic steps per outer iteration after 500 WM-only iterations.
- WM/actor normalization uses real data. Critic normalization uses detached imagined features, matching the reference. Inference now explicitly disables autograd, without changing the live learner's mode or statistics.
- Actor updates freeze WM, critic and AMP prior parameters while preserving gradients through their inputs. A style-only regression confirms that contact/style gradients can reach the actor through frozen model/prior modules.
- Fixed an optional replay-resume edge case: retained replay must end at a collector boundary because simulator state is not in the checkpoint. Restored latest transitions are marked truncated; history reconstruction also honors actual boundaries instead of relying solely on episode counters. The shown G1 continuation refilled an empty replay and was not affected by this issue.
- The trained G1 actor at checkpoint 4000 was loaded/exported in an environment with no simulator SDK or RSL-RL installed. CPU FP32 outputs matched the exported policy exactly across streaming history and partial resets. Inference outputs require no autograd graph.

## Equivalent computation removed

Tracking has no learned physical heads, yet empty-head paths constructed state features and normalization inputs during collection, WM fitting and every imagined step. Those paths now return immediately. Checkpoint keys and the learning objective are unchanged.

A matched GPU check used 2048 actual native G1 transitions, the same trained checkpoint, the same sampled batches and random seed, full networks, actor batch 256 and horizon 32:

| Check | Maximum absolute difference |
| --- | ---: |
| WM loss | 0 |
| WM parameters after an optimizer step | 0 |
| Actor gradients | 0 |
| Actor parameters after an optimizer step | 0 |
| Critic parameters after its fitting phase | 0 |

Isolated eager actor/critic update median: 0.43072 s before and 0.39411 s after, an 8.5% time reduction over three measured repetitions per version. This excludes collection, WM fitting and compilation and is not an end-to-end training speedup claim. Evidence: `logs/mbdac_audit/comparison.json` and `trained_export.json`.

## General task interface

The new `PhysicalSignals` registry selects analytic providers first and creates heads only for declared missing physical quantities. It automatically captures immutable native measurement labels and checks shapes. The MJLab AMP example now uses it for acceleration, contact and force history. A simulator-free contact/relative-velocity reward example confirms that weights and tolerances can change without replacing physical heads. See [the task interface](general_task_interface.md); arbitrary black-box Python rewards are not automatically rewritten.

## AMP preparation

The expert iterator preserves the external baseline's 20-batch cycle over all 18 reference clips; recreating it for each update would silently use only clip zero. The privileged critic includes delayed-recovery bookkeeping, because those flags change reward/termination at otherwise identical physical states.

Discriminator updates now start together with actor updates after WM warmup, rather than saturating against an unchanged actor during warmup. Their style gradients are tested separately from analytic task reward. The short AMP pre-review run was manually interrupted during WM warmup and is excluded from quality/time comparisons. New AMP training starts fresh with the reviewed implementation.

26 tests passed in the MJLab environment. The independent SDK-free wheel checks, native reward/FK/observation audit and final AMP smoke are recorded under `logs/`. No Isaac Lab training is part of this acceptance work.

## Physical tracking result and limits

Checkpoint 5000 completed 26 of 40 entire reference clips in native physics without auto-reset, with mean normalized return 0.58493 across the full nominal clip durations. This includes walk, run, sprint, dance, jump and fighting clips; failures remain, particularly fall/get-up motions. Full-clip results must not be mixed with the earlier 20-second prefix return of 0.76334 at checkpoint 4000. The existing AMP+PPO comparison remains a separate matched velocity experiment; no performance advantage has been established yet.

## FastSAC + WM follow-up: small tracking goal passed

The source-pinned Holosoma numerical actor/C51 modules remain unchanged. The
new learner and its failure history are documented in README. Corrected Q state
includes previous action, measured policy memory and reference clock; finite
reference completion suppresses Q bootstrapping, while ordinary time limits
retain physical-next-state bootstrapping. Terminal-only reset bins are removed.
Optional Q-only warmup is resumed through saved optimizer counters. Batched and
sequential imagined reward losses/all actor gradients are regression-equivalent.

Local32-phase physical perturbations show the old Q gradient improves the
32-step real return in4/32cases, while model reward directions improve32/32.
The bounded repeat applies0clamps. These returns include action-rate cost and
are not a test isolating every dynamics derivative. Likewise, several recipe
changes were made together; no single-fix causal or speedup claim is justified.
The small-std probe's entropy contribution exceeded its positive tracking
reward, exposing an early-failure incentive. The successful recipe explicitly
uses rawstd0.05–0.8, actorLR5e-5, H32 and100%model actor objective, with100WM
adaptation iterations then800Q-only updates before10WM/4Q/1actor per iteration.

Only the physical WM from v5 is reused. Fresh actor/Q/temperature are audited.
The selected v6/model1500 completes6s in native CUDA physics without reset,
jointRMSE0.121609rad, globalbodyRMSE0.124699m; all6initial-phase cases also pass.
Original99D observations, goal/action order, reward and physical PD interface
are retained. No optional observation extension was used. Its21minnew training
and28min38s source-WM run total49min39s; former57min15s MBDAC used more motions.
This is a successful small demonstration, not equal-task benchmarking or broad
robustness/convergence certification.

Native tests45passed/1skip; installed core43passed/3skips. Actual trained actor
export matches eagerly at batch1/16 with0error without SDKs. A1024-world trained
full-resume update also passes. Raw physical trajectory/video, timing, source
manifest and checkpoint hashes are under `logs/g1_wm_fastsac_v6_report/` and
[W&B](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-fastsac-tracking/runs/vn81zz42).
