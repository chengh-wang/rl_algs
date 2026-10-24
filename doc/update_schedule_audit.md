# Update schedule audit — 2026-10-23

Sources: [paper v1, Algorithm A.1 and Table A.5](https://arxiv.org/pdf/2610.28480v1), [reference train.py](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/supermimic/train.py), and [reference choices not specified by the paper](https://github.com/zita-ch/supermimic_vibed/blob/a1508d42c9d1699a52e8233ab16d08c488441ab9/README.md#deviations--choices-not-specified-by-the-paper).

An **outer iteration** is: collect fresh real transitions → fit WM → update actor → fit critic. The modules have separate optimizers. During actor backpropagation, WM and critic weights are frozen; gradients still pass through their state/action inputs. Afterwards only the critic optimizer fits detached lambda-return targets.

| Item | Paper | Reference repository | Active laptop run |
| --- | --- | --- | --- |
| Initial real collection | Not specified | 64 vector steps | 64 vector steps |
| WM-only warmup | Not specified | 500 outer iterations | 500 outer iterations |
| Real steps / outer iteration | UTD 0.4 | Alternate 2 and 3 | Alternate 2 and 3 |
| WM optimizer steps / iteration | 10 batches | 10 | 10 |
| Windows / WM batch | 2048 | 2048 | 128 |
| Autoregressive window | 8 steps | 8 | 8 |
| Actor optimizer steps / iteration | 1 | 1 | 1 |
| Imagined trajectories / actor step | 4096 | 4096 | 256 |
| Imagination horizon | 32 steps | 32 | 32 |
| Critic optimizer steps / iteration | 5 epochs × 4 minibatches | 20 | 20 |
| Parallel simulator worlds | 4096 / GPU | 4096 | 1024 |
| Replay capacity | 1000 steps / world | 1000 | 256 |

The paper's “policy updates per world-model update = 1” refers to one **WM fitting phase** containing ten minibatches. Counting individual `optimizer.step()` calls gives **WM : actor : critic = 10 : 1 : 20** after warmup, not 1:1:1. UTD 0.4 counts outer learning iterations per vector simulator step; it does not count individual WM minibatches.

With zero-based loop indexing, iterations 0…499 make 5000 WM optimizer steps and zero actor/critic steps. The next iteration (displayed iteration 501) performs another 10 WM steps before the first actor step, so 5010 WM optimizer steps have completed when the actor first changes. The critic then makes its first 20 optimizer steps. The 64-step initial collection performs no gradient updates.

The laptop batch sizes are 16 times smaller than the paper/reference, while real world count is 4 times smaller. Thus the update **frequency** matches but each iteration processes 16 times fewer sampled WM windows and imagined trajectories. Relative to incoming real transitions, sampled training volume is four times smaller. This is a reported hardware adaptation, not a full-scale reproduction.

The CPU schedule regression test counts actual optimizer calls and verifies both the WM-only boundary and the exact WM→actor→critic order. New run configs save all resolved defaults, not only CLI overrides. The running v1 experiment was launched before that logging improvement; `config_resolved_audit.json` records its audited effective settings without changing its training schedule.
