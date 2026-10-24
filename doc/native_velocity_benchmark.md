# Native velocity benchmark (2026-10-24)

Status: controlled v3 runs in progress; no WM speedup result yet.

Goal: improve simulator-independent WM+Holosoma FastSAC, test native MJLab G1
and Go1 velocity tasks, and compare MBDAC / upstream Holosoma FastSAC /
WM+FastSAC from scratch. This replaces neither earlier motion tracking results
nor the separate AMP experiment. No actor, critic, temperature or WM warm start.

MJLab 1.2.0 ships G1 and Go1 velocity factories, not Go2. The user accepted Go1
instead of Go2. Both use the native flat factory directly. Native physics,
managers, reward functions and weights remain shared across algorithms.

Predeclared primary benchmark: Go1 flat, seed 42, 1024 real environments. Each
method has fresh networks/replay/normalizers/optimizers; same task configuration,
5 ms physics / 20 ms control, command curriculum, noise, pushes and startup DR.
Algorithms retain their documented architectures/update schedules. These
differences are reported, so this is an end-to-end method benchmark, not an
isolated actor-gradient ablation. G1 is a second transfer test.

Evaluation: 20 s from standing, no resets, no noise/push/startup DR; commands
`[0,0,0]`, `[0.8,0,0]`, `[-0.5,0,0]`, `[0,0.5,0]`, `[0,0,0.5]`, eight replicas
each. Exclude only first 2 s, never post-failure samples. Root-body-frame velocity
matching native tracking rewards; XY RMSE = sqrt(mean(||v_xy-command_xy||²)),
yaw RMSE = sqrt(mean((omega_body_z-command_yaw)²)). Also report every command
separately and survival. Acceptance requires all 40 worlds complete and aggregate
XY <= 0.20 m/s and yaw <= 0.30 rad/s; report per-command errors to expose standing
or poor lateral/backward/turn behavior. Do not pool metres and radians.

Compare first evaluated checkpoint meeting both gates, with sampling intervals
and actual wall-clock times. Record process startup, compile, data collection,
WM fitting, actor/Q/V fitting, warmup, evaluation and total elapsed separately.
If a method never reaches the gate, report not reached, not a speedup. Initial
v3 per-method primary training budget: one active-process hour; pause earlier for non-finite
updates or clear physical/learning collapse, preserve and diagnose failures.
One seed is a laptop pilot, not a statistically established superiority claim.
Stop early after two consecutive 300 s evaluations meet the stricter per-command
gate and their evaluation returns differ by less than 5%; preserve the first
passing timestamp separately. The stopping decision does not alter rewards or
learning updates. Keep a complete source snapshot with every launched run.

Requested plots: reward versus wall-clock time and each velocity RMSE versus
wall-clock time, all three methods on common axes. Plot identical deterministic
evaluation returns as the primary reward comparison, with native training
episodic reward separately (its distribution changes with policy and curriculum).
Include total process elapsed and elapsed excluding evaluation versions; the
latter still includes imports, setup, compilation, diagnostics and checkpoint
I/O, so it is not pure learner time. Never relabel the
number of updates or simulated seconds as real elapsed time. Keep raw JSON/CSV,
checkpoint timestamps, native task/config hashes and method settings alongside
the figures. Evaluation points are discrete; report the first observed crossing
rather than inventing an exact time by interpolation.

Also report the stricter `all_command_pass` (both RMSE thresholds in every
individual command). Keep the originally declared aggregate gate as `pass`.
In particular, aggregate yaw alone can pass for a standing robot; the XY gate,
the per-command table and the stricter gate must accompany it. Replicas in this
nominal deterministic evaluation are repeatability copies, not 40 statistically
independent starts or a domain-randomization robustness claim.
All deterministic evaluation actors run in fp32, including learners trained in
BF16. Model-based runs log sampled real-buffer model rollout errors and physical
primitive-head errors separately; these are diagnostics, not held-out test errors.

Primary x-axis is real elapsed minutes (user's speed question). Secondary x-axis
is cumulative real simulator transitions (sample efficiency). An iteration
contains different data/update work in these algorithms, so iteration is shown
only in each method's diagnostics, never used as a common comparison scale.

No simulator derivatives. WM learns only real applied-action/pre-reset state
transitions. Physical contacts/forces get physical-unit heads where unavailable
analytically; reward expressions/weights/tolerances remain native and reusable.

## Implementation checks and approximations

The native zero-valued measurement term is appended after existing rewards. It
captures the applied raw action, physical state, physical contact labels and
swing-height memory before automatic reset. Timeouts retain physical terminal
observations. Pushes mark exogenous model boundaries, not fictitious continuous
dynamics samples. No custom physics loop, tracking reset or AMP dataset is used.

G1 and Go1 reward formulas agree with native per-term values to below 1e-8 when
given the same measured data. Noise-free native observations and reconstructed
observations agree to about 1e-6. Native reward kinematic caches lag raw qpos/qvel
by one 5 ms physics substep; end-state analytic FK therefore approximates that
timing in imagination. Both errors are logged separately in the contract JSON;
formula parity must not be described as exact physical reward prediction. Real
collection and evaluation always use actual unchanged native reward managers.
Commands are held over short imagined horizons; real heading updates, resampling
and pushes remain native. Contact predicates are softened only in imagination.

### Historical model-only v2 recipe (not the primary v3 hybrid)

The next three paragraphs document the earlier recipe. The frozen v3 settings
below supersede its WM update count, actor learning rate, warmup and replay size.

All actors receive the native single-frame actor observation; commands are also
passed as the existing model-based goal input (same information). MBDAC uses its
original deterministic BPTT/lambda-return V algorithm and native task adapter;
its history is set to one to match policy information. Holosoma FastSAC directly
inherits the pinned source setup, replay sampling and entire learn/update loop.
Symmetry is disabled for all methods. WM+FastSAC uses the same locomotion C51
support/discount/target EMA/Q widths as upstream (101 atoms, gamma .97, tau .125,
eight Q updates, actor every four Q updates); its model actor batch is 256 and WM
batch 128. WM actor LR 5e-5 and raw exploration std bounds .05–.8 retain the
documented prior model-based recipe; the upstream baseline retains its original
3e-4 actor LR and source log-std bounds. MBDAC retains its own original networks,
V optimizer and gamma .99. These are algorithm recipe differences, not controlled
ablations; a positive result alone cannot isolate the causal effect of WM.

Both model-based learners fit ten 8-step WM batches per update after 64 vector
warmup steps and collect two/three vector steps per outer update (UTD .4).
MBDAC holds its actor/critic for 500 updates. WM+FastSAC holds its actor for 500
updates and pretrains Q for 800 real-only updates during the last 100 WM-only
iterations. These operations execute sequentially on the GPU; the phases share
existing real collection/WM fitting iterations. This optional generic schedule
preserves Q warmup counts and removes 100 extra collection/WM iterations before
the first model actor update.
Upstream FastSAC collects one vector step per loop and starts fitting after step
10. Replay capacities remain source recipe values (256 model-based, 1024
upstream). Every schedule and its cost is saved; update counts are not equated.

Optional general precision optimization: WM+FastSAC actor/Q support CUDA BF16 and
fused AdamW. Default remains fp32/non-fused for checkpoint compatibility. The
CUDA BF16 actor objective and all actor gradients match unmodified Holosoma
network math under the same autocast context exactly in the contract test.
Analytic sensor FK prunes unobserved tree branches for either robot; G1 batch256
forward/backward measured 36.3 ms -> 3.92 ms, zero output error and max input
gradient error 4.77e-7. This is not an end-to-end speedup claim.

## Trial audit

`native_go1_fastsac_v1` is the initial successful source pilot: 6000 source steps,
first sampled aggregate/individual gates pass at 303.18 active seconds (2780),
final XY 0.055 m/s and yaw 0.052 rad/s. It is retained, not silently replaced.
`native_go1_wm_fastsac_v1` is an excluded prototype, stopped at 747 iterations:
its real Q features used predicted contact primitives rather than the native
critic sensor values. It used 526.12 process seconds (including evaluation).
This is a framework/adapter bug, not a reward tuning result.

Formal `*_v2` runs start all networks and replay from scratch with paired native
current/physical-next critic features for real Q fitting and normalizer updates;
imagined Q features still use analytic/learned physical heads. MBDAC's V and
imagined Q also receive the known previous action in the native critic action
slot. Unit tests prove real Q/H=0 updates bypass the learned feature function.
No old prototype checkpoints are reused. Source snapshots preserve launched code.

Generic scalar logging now packs detached native measurements before host
conversion, preserving mutable simulator-buffer snapshots and Python-float log
interfaces. Tests show collection replay and normalizer buffers are unchanged.
No robot name, reward target or task threshold appears in these core changes.

## Frozen v3 comparison

The source baseline remains `native_go1_fastsac_v2`; its original algorithm and
native observations/rewards are unaffected by the model feature changes below.
The model-based primary runs are `native_go1_mbdac_v3` and
`native_go1_wm_fastsac_v3`. Earlier model-only v2 runs remain development trials,
not silently omitted successes or definitive convergence failures.

The native velocity codec retains world yaw for physical world-force heads. The
previous yaw-invariant feature map could not identify rotated world forces from
otherwise identical body states. Local dynamics targets remain equivariant;
tests verify both properties. The same correction is applied to both model-based
methods and both robots. Real physics, native force sensors, rewards and command
distributions are unchanged.

The v3 hybrid retains large-batch real actor updates during model warmup. It uses
the source 3e-4 actor LR, source exploration log-std bounds and 8192 real actor/Q
batches. It collects one vector step per iteration, fits one WM batch of 128
8-step windows, fits eight real Q batches, and updates the actor every four Q
steps. Replay capacity is 1024 time rows and initial collection is 11 rows.
Before iteration500 the actor uses real SAC alone. Thereafter its objective is
0.9 real SAC + 0.1 differentiable 8-step model return (model actor batch256).
There is no teacher/checkpoint warm start and no task-specific recipe branch.

MBDAC retains its reference 10 WM batches, 32-step actor, V/lambda-return update,
500-iteration warmup and other original settings listed above. Source FastSAC
retains its pinned original loop. Therefore this is a comparison of complete
method recipes, not a claim that all neural architectures/optimizers are equal.

An additional `native_go1_real_only_v3` control uses the same new runner, policy,
Q, normalization, replay and native adapter as the hybrid, with WM fitting and
model actor weight disabled. It isolates the model contribution from integration
differences; its separate budget is 20 minutes. G1 uses exactly the same hybrid
recipe for a 30-minute/3000-iteration transfer trial. These limits were fixed
before the corresponding v3 policies trained.

## Artifact generation

After training has ended, `scripts/finalize_native_velocity.py` loads the first
checkpoint passing every command, or the final evaluated checkpoint when none
passes. It records this selection explicitly, evaluates in native physical
simulation again, saves raw trajectories, renders method-labelled videos and
produces the primary comparison plots. Its optional `--wait-pid` waits for the
confirmed matrix process before using the GPU. G1 is a fresh same-config task
test, not zero-shot weight transfer. No video joint smoothing is applied.

## Integration differences checked against the pinned learner

The upstream baseline inherits its complete training loop. The new runner's
`real_only` mode is an integration-matched ablation, not a step-for-step upstream
reimplementation. In particular:

| Item | Pinned upstream FastSAC | New runner, real-only and hybrid |
| --- | --- | --- |
| Actor/critic normalization | Sampled replay current/next batches; 65,536 rows each per loop | Current real collection for actor; current/next real collection for Q |
| Compilation | Q update, actor update, collection policy and normalization | WM forward compiled; real Q/actor update path remains eager |
| Actor phase within eight Q steps | After Q2 and Q6 | After Q4 and Q8 |
| Replay draws | World-stratified large batch split into eight; actor reuses Q batch | Independent uniform Q batches and separate actor draws |
| Actor gradient clipping | Disabled | Norm1 |
| Go1 actor input | Native48 | Native48 plus duplicate command3 |
| Go1 Q input including candidate action | Native critic72 plus action12 | Native critic72, previous action12, noisy actor history48, action12 |

Actor/critic normalization counts in actual checkpoints confirm this distinction:
source update3010 has393,216,000 samples in each normalizer; new update1000 has
1,035,264 actor/goal samples and2,070,528 Q samples. These are counts of updates
to statistics, not independent simulator samples. The new Q history and action
memory preserve the generic policy memory contract; they duplicate some native
features in this single-frame task. Thus the original-vs-new wall-time comparison
also includes compiler, memory-feature and normalization differences. The internal
real-only control can isolate the model contribution within this integration;
it does not isolate all integration costs relative to upstream.

The audit found matching C51 loss/reduction, atom support, twin-Q mean, gamma,
target EMA, action scale/tanh/density correction and temperature objective.
Timeouts bootstrap from physical pre-reset successors. For true terminations,
upstream uses reset successor observations while the adapter retains physical
terminal observations. Both zero Q bootstrap, but the successor distribution in
the alpha update can differ. No native reward was changed for these experiments.

`process_elapsed_s` in timing manifests is recorded before W&B shutdown and
environment cleanup. Matrix `process_s` includes the full child process where
available. Neither should be silently substituted for the other in a cost
ledger. Online curve points measure actual elapsed time at evaluation result
availability; `active_process_s` subtracts accumulated online evaluation work.
