# Documentation index

Install and choose a public interface from the [repository README](../README.md).
Algorithm modules live in `src/mbdac`; task examples and simulator adapters live
in `examples`. Formal changes are published on `main` using the repository owner's
Git identity. Benchmark figures and their source data remain under `doc/assets`.

## Interfaces and training

| Guide | Scope |
| --- | --- |
| [Flow Matching SMP](flow_matching.md) | Standalone continuous-time velocity prior, generic feature-window training, scoring and checkpoint state |
| [FlashSAC and SMP runner](smp_flashsac.md) | Model-free runner contract, task-owned reward providers and the existing motion-prior integration |
| [Model-based task contract](task_contract.md) | Observations, physical transitions, goals and model-based adapters |
| [General physical task interface](general_task_interface.md) | Physical primitives, analytic kinematics and reward composition |
| [Precision rewards](precision_rewards.md) | Objectives, shaping and learner diagnostics |
| [Update schedule](update_schedule_audit.md) | World-model, actor and critic update schedules |
| [Implementation recipes](implementation_recipes_2026-10-24.md) | G1 preparation, task examples and training recipes |

`mbdac.flow_matching` supplies the prior algorithm independently of a policy
learner. The runner's explicit reward-provider contract controls how a caller
uses a score; selecting a runner does not select or load a Flow model.

## Benchmarks and reproducible evidence

| Report | Supporting material |
| --- | --- |
| [Native velocity protocol](native_velocity_benchmark.md) | [Results](native_velocity_results.md), [time comparison](assets/native_velocity_checkpoint_time.png), [transition comparison](assets/native_velocity_real_transitions.png), [plot provenance](assets/native_velocity_plot_provenance.json) |
| [Precision control results](precision_control_results.md) | [Figure](assets/precision_control/errors.png), [summary](assets/precision_control/summary.json), [manifest](assets/precision_control/manifest.json) |
| [Matched AMP velocity protocol](velocity_benchmark.md) | [Local results](velocity_results_2026-10-24.md), [upstream comparison](velocity_upstream_results_2026-10-24.md) |

## Historical audits and validation

These documents preserve the results and validation scope of the named snapshot:

- [MBDAC implementation audit](mbdac_audit_2026-10-24.md)
- [Acceptance record](acceptance_2026-10-24.md)
- [Validation record](validation_2026-10-23.md)
- [Final report](final_report_2026-10-24.md)
- [Progress record](progress_2026-10-24.md)
- [Benchmark progress log](../Progress/2026-10-24_native_velocity_benchmark.md)

[Central provenance index](../licenses/PROVENANCE.md) records source lineage and
licensing limits for all reused implementations.
