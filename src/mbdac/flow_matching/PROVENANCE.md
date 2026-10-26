# Flow Matching motion prior

The linear conditional path, velocity-MSE objective, residual scoring and
standalone tensor interfaces are local additions. They reuse the architecture,
feature preprocessing and count-based normalization from this package's SMP
extraction. This is not an official SUZ-tsinghua SMP implementation or release.

The underlying DiT and cosine mapping retain the source attribution and
licensing limitations in `mbdac/smp/PROVENANCE.md` and
`licenses/SMP-NO-LICENSE.txt`. The upstream snapshot has no LICENSE file; this
addition does not relicense it. No simulator, FlashSAC, robot asset or motion
dataset is copied into this module.
