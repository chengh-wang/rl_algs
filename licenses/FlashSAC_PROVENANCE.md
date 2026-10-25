# FlashSAC numerical components

Source project: https://holiday-robot.github.io/FlashSAC/ (identified in upstream README), commit `87edc9061150ae9e962dd84e6544e27a1554b3ab`.

The files under `src/mbdac/vendor/flashsac` are adapted from the pinned FlashSAC checkout. Its complete MIT license, including the original copyright, is preserved at `src/mbdac/vendor/flashsac/LICENSE`.

Vendored subset: FlashSACConfig and network initialization/action sampling/update orchestration from agents/flashSAC/agent.py; layer.py, network.py, update.py; agents/utils/{network,reward_normalization,distribution,scheduler}.py; buffers/{base_buffer,torch_buffer}.py.

Changes: imports relocated into the package; removed unused environment agent/rollout orchestration and external BaseAgent dependencies; replaced Gym shape/dtype-only descriptors with ShapeSpace; replaced unused JAX union typing with PyTorch/NumPy types. Network, loss, optimizer, reward-normalization and replay numerical equations are retained.

`src/mbdac/flashsac/{config,runner,learner}.py` adapt the existing local MJLab project learner integration into a standalone numerical core: external git/sys.path checks removed, CPU-safe CUDA RNG handling, caller-owned opaque compatibility contracts, and configurable initialization config factories. No simulator or task source is included.

## Original license

```text
MIT License

Copyright (c) 2026 Holiday Robotics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## Compatibility notes

The local learner adapter was extracted from `src/rl/flashsac/{config,runner}.py` and `src/rl/smp/learner.py` in the MJLab project. Those adapter contributions retain the source project's Apache-2.0 terms (see `smp-mjlab-Apache-2.0.txt`); the upstream numerical files retain the MIT terms above.

Intentional numerical compatibility includes the upstream reward-normalizer terminal carry behavior: `new_G_r = gamma * (1 - done) * G_r + reward`. The previous return is cleared on a done transition before including its reward; this terminal reward consequently remains in `G_r` on the following fresh-episode update. This extraction does not silently correct that behavior. Simulator resets do not imply learner-state equivalence.

Opt-in precision extensions retain that default path: the adapter provides a
corrected `episodic` normalizer and a constant-divisor `fixed` normalizer outside
the vendored files, configurable discount/value support, and a separately
versioned potential-based reward provider. The vendored `agent.py` and
`update.py` additionally forward a disabled-by-default `critic_diagnostics`
flag and report probability-weighted support diagnostics from existing critic
tensors. These diagnostics add no sampling, network forwards or loss terms.
They are local extensions, not features attributed to the pinned upstream
commit. Default fixed-RNG update/state parity and diagnostic on/off parity are
covered by tests. See [precision controls](../doc/precision_rewards.md) for the
intentional opt-in numerical changes and their limitations.

SHA-256 of the two diagnostic-extended vendored files:

| File | SHA-256 |
| --- | --- |
| `agent.py` | `fbbf0d1d117f46a848756c4a47c56e74df644f40e8db4cb7e518fe999dd47745` |
| `update.py` | `48a40323139b9896893e6129c65fda6bb9631244f7c007c33286968384c84f68` |

CPU checkpoints do not initialize or save CUDA RNG state. CUDA RNG is preserved for CUDA learners. Optional calibration fields default to None; legacy checkpoints missing those fields are accepted when current settings are None. Opaque caller contracts still require exact equality for learner resume. Initialization may compare an explicit caller-supplied subset of contract keys, and is declared as a new learner initialization rather than simulator resume.

The standalone regression suite performs exact fixed-RNG comparisons against the original local learner plus pinned upstream numerical implementation when `FLASHSAC_PARITY_CHECKOUT` and `FLASHSAC_ROOT` are supplied. It compares initial network parameters, sampled replay batches including normalized rewards, three action/update iterations (every metric and every model/BN tensor), and an original-format replay checkpoint's following update. Without that optional external fixture, all independent CPU tests run without the original checkout or Gymnasium.
