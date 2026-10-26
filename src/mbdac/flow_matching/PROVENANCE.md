# Flow Matching Scored Motion Prior

The linear conditional path, velocity-MSE objective, continuous-time residual
scoring, count-weighted scoring statistics, strict checkpoint format and generic
tensor training interfaces are local additions.

The PyTorch DiT architecture in model.py is adapted from SUZ-tsinghua/smp at
commit28d1dfd1b7b4447977fbdb2062167d9f87a2faaa. Retained attribution and source
licensing limits are documented in src/mbdac/smp/PROVENANCE.md and
licenses/SMP-NO-LICENSE.txt. The upstream snapshot contains no LICENSE file;
this addition does not relicense that architecture. Runtime code is contained
in this module and does not import the source package.

No simulator, robot asset, reference dataset or RL learner is included in this
module. This is a local Flow Matching SMP variant, not an official release from
the upstream authors.
