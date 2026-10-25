# SMP extraction provenance and licensing limits

Algorithm source: https://github.com/SUZ-tsinghua/smp at commit
28d1dfd1b7b4447977fbdb2062167d9f87a2faaa. `vendor/model.py` and
`vendor/scheduler.py` retain that source unchanged. Feature transforms,
count-based normalization, score arithmetic, and DDPM generation derive from
that source. The source checkout contains no LICENSE file. No license grant is
inferred, and these sources are **not relicensed as MIT or Apache-2.0** by this
package's general project metadata. Permission for redistribution/use must be
resolved with the upstream owners as appropriate.

Extraction source: mjlab_unitree_r1, src/rl/smp/{prior,features,generation,pretrain}.py,
working checkout on 2026-10-24. Checkpoint model keys, EMA preference, float32
quantile scaling without clipping, L1 epsilon pretraining loss, diffusion defaults,
and reward/normalizer formulas are retained. The simulator-bound reader is
excluded. Pretraining now requires an explicit named schema instead of BODY24,
50-Hz, hand-filter, or fixed end-effector assumptions. It reads already prepared
kinematic NPZ arrays and never imports a simulator. Native simulation/FK data
preparation belongs outside this package.

`math.py` selects the wxyz quaternion formulas from installed MJLab 1.6.0
`mjlab/utils/lab_api/math.py`, whose header credits the Isaac Lab Project
Developers (2022-2025), BSD-3-Clause. The selected formulas are unchanged;
TorchScript decorators and unrelated SDK imports are omitted. The quaternion
matrix conversion helpers cite PyTorch3D in their original docstrings. Retained
BSD-3-Clause terms are in `licenses/smp-math-BSD-3-Clause.txt`; MJLab distribution
Apache-2.0 terms are retained in `licenses/smp-mjlab-Apache-2.0.txt`.

No upstream environment, event/reset writer, robot constants, simulation SDK, or
MotionFeatureReader implementation is included. Decoding returns arrays only;
it makes no claim of physical feasibility or stable initialization.
