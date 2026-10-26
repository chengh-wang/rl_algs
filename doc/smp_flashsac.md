# SMP + FlashSAC：独立算法包与任务边界

本文记录既有的 model-free runner 与 motion-prior 集成。通用的 Flow Matching
SMP 算法使用独立的 [`mbdac.flow_matching` 接口](flow_matching.md)；调用方通过
明确的奖励组合接入评分，runner 不自动选择或加载 Flow 模型。

## 包及调用约定

此仓库名为 `rl_algs`，Python distribution/import 名为 `mbdac`。新增模块与已有
MBDAC、MBDACAMP、MBDACFastSAC 并存，不改变原有 `ModelBasedRunner` 的 WM 接口。
这里的 FlashSAC 来自 Holiday Robotics 固定版本，和 Holosoma FastSAC 分开命名。
没有使用 PPO 替换 FlashSAC，也没有把 world-model 数据伪造成 model-free 输入。

| 层 | 位置与职责 |
| --- | --- |
| 公开 runner | `mbdac.runners.FlashSACRunner` / `SMPFlashSACRunner` |
| 独立 learner | `mbdac.algorithms.FlashSAC` / `SMPFlashSAC`，或 `mbdac.flashsac` 的原低层 API |
| SMP 数学 | `mbdac.smp`：schema、history buffer、冻结 prior、误差归一化、DDPM 采样和解码 |
| prior 预训练 | `mbdac.smp.pretrain`：显式 schema、clip 内窗口、quantiles、epsilon L1 |
| simulator | 不在上述核心内：读取状态、FK、控制映射、奖励组合、reset/termination、机器人配置 |

公开签名沿用仓库 RSL-style 形状：

```python
runner = SMPFlashSACRunner(env, train_cfg, log_dir=None, device="cuda:0")
runner.learn(num_learning_iterations=1000, init_at_random_ep_len=False)
runner.save(path, infos=None, include_replay=True)
runner.load(path, load_cfg=None, strict=True, map_location=None)
policy = runner.get_inference_policy(device=None)
runner.export_jit(path)
runner.close()
```

`ModelBasedRunner` 仍只接受它原先支持的 model-based class；model-free 使用新增 runner。
`learn(n)` 的 n 表示额外 vector steps，不是 SAC updates。分别报告 vector steps、
real transitions 和 gradient updates。未达到 replay warmup 时只收集数据。

## 输入与输出

环境提供 `num_envs`、`num_actions`、`device`，以及：

```python
get_observations() -> TensorDict | dict[str, Tensor]
step(normalized_action) -> (observations, rewards, dones, extras)
reset()  # 从不保存 simulator state 的 checkpoint 恢复时需要
```

通过 `obs_groups={"actor":["actor"], "critic":["critic"]}` 指定有序拼接。
每个 group 为 `[N,D]`；actor/critic 可以不同维度。历史若已经由环境管理，则传其
固定拼接布局；此 runner 不隐式添加 history，不要求 `goal`。

输出 action 为 `[N,A]`，是 `[-1,1]` policy/replay 坐标。关节名、PD 参数、reference
target affine 映射及硬件限制由任务 adapter 定义。若 adapter 对 policy 坐标做 clipping，
返回 `extras["applied_action"]`；replay 保存实际应用的 policy action。

奖励为有限 `[N]` tensor；done 为 `[N]` 二值 tensor。`extras` 保留：

* `terminated`、`truncated`：成对显式给出；或 `time_outs` 区分普通 timeout。
  没有 timeout 信息时，done 按真正 termination 处理。
* `terminal_observation`：每一个 done 必须有 reset 前所有配置 observation groups；
  可传 N 行或按 done 顺序传 done-count 行。普通 timeout 从此状态 bootstrap，真正
  termination 截断 bootstrap。返回给下一次 action 的 observation 可以已经 reset。
* 可选 `log`/SMP diagnostics。无需 `model_transition`、codec 或 physical primitive heads。

`task_contract` 是调用方定义的可序列化观测/动作/任务版本字典；恢复时严格匹配。
核心不硬编码 R1、24 joints、MJLab 或任意机器人 frame 名。

## SMP 的明确组合方式

`smp.reward_source` 必须明确选择：

1. `environment`：任务已经计算组合奖励，提供 `env.smp_contract`，并可通过成对的
   `smp_state_dict` / `load_smp_state_dict` 保存在线 normalizer、GSI pool/counters 等持久状态。
   runner 不再次添加 SMP 分数。
2. `reward_provider`：`env.smp_reward_provider(rewards, extras)` 返回组合奖励，并提供
   `contract`、`state_dict`、`load_state_dict` 和 `reset()`。任务在 extras 中提供 pre-reset motion
   features/validity；provider 负责打分和显式组合。冷恢复必须调用 `reset()` 清除 episode history。

`FrozenSMPPrior.score([B,W,F], ...)` 只接收 tensor；特征归一化使用训练 q01/q99，不裁剪。
固定 t={8,15,22} 的 epsilon MSE 经每个 t 的在线计数均值归一化，
`r_SMP=exp(-w_s * mean_t(error_t / running_mean_t))`。prior 网络冻结；normalizer 是
独立持久状态。padding 窗口不能计分或更新 normalizer。任务 reset 清相应 history，
terminal history 必须先评分再清空。`MotionFeatureBuffer` 只处理传入的运动 tensor，
不访问 simulator。`MotionFeatureReader` 和生成状态写回不属于算法包。

GSI 为可选任务 reset capability。DDPM 生成/解码在独立包，FK 一致性、关节范围、
地面约束、拒绝采样及写回由任务实现。generated/reference/real 数据分别保存；
生成 reference 不会伪装成 real SAC replay transitions。

prior 预训练使用显式 schema，例如：

```bash
python -m mbdac.smp.pretrain --references /path/to/manifest.json \
  --feature-schema /path/to/feature_schema.json --output logs/my_prior \
  --epochs 2000 --device cuda:0
```

schema 包含有序 joint/end-effector 名称、root、sample_dt 和 wxyz quaternion 约定。
不跨 clip 取窗口；quantiles 仅来自训练集；可指定不相交的 validation clips。
R1 的 BODY24、五个 end effectors、50 Hz 默认值留在 R1 任务配置，未放进核心。

## 配置、保存和恢复

learner 参数可放 `algorithm` 字典（去除 `class_name`）或 `learner` 字典；冲突/未知键
报错。`RunnerConfig` 明确 replay capacity/warmup/batch、updates_per_step、网络更新
schedule、AMP/compile、seed/device，以及可选 entropy/temperature 校准。
SMP 可选 `actor_delay_updates`；等待期间 critic/target 继续更新，actor/temperature 不更新。

checkpoint 保存 actor/critic/target/temperature、BN、optimizer、scheduler、scaler、
reward normalizer/RMS、随机数、计数和可选 replay。公开 runner 额外保存 SMP 持久状态。
旧低层 checkpoint 格式保留，新增默认配置键兼容旧 checkpoint。全环境 public-runner
checkpoint 与旧 learner+sidecar 两种入口不同，不把它们静默混用。

checkpoint **不保存 simulator state**。冷恢复先导入持久状态，再 reset 收集器，清除
episode return/noise/history；replay 中已经完成的物理 transition 可以恢复。
源 actor/value 初始化是新的实验，不等同于恢复；可分别迁移 actor、匹配 critic/target
及 reward units，同时保持新 optimizer/replay/counters/temperature。

本次保持固定 FlashSAC 数学，包括上游 reward-return normalizer 的 terminal carry
公式。没有把修正算法细节混入结构迁移；来源及差异在 provenance 中列出。

## MJLab 1.6 / R1 的本地连接

MJLab 工作区通过正常安装本包连接，不安装本仓库 `.[mjlab]` 的旧可选版本来覆盖
现有 MJLab 1.6。算法核心不需要 Gymnasium、MJLab、MuJoCo 或外部 FlashSAC checkout。

任务侧采用项目原布局：

* `src/tasks/velocity/mdp/smp_observations.py`：按名称读取机器人状态。
* `mdp/smp_actions.py`：reference/home 映射及其校准；原生 ActionManager/PD 执行不变。
* `mdp/smp_rewards.py`：显式奖励公式、native cached rates 和 dt 校验。
* `mdp/smp_events.py`：GSI FK/拒绝检查、生成状态写入。
* `config/r1/smp_experiment_cfg.py`、`smp_parameters.py`、`smp_pretrain_cfg.py`：
  native manager CFG、实验奖励/动作/GSI 参数和 R1 prior schema。
* `rl/smp_env_adapter.py`：terminal-safe transition 和 RSL 四返回 bridge；
  `rl/smp_train.py` / `smp_evaluate.py`：任务运行集成。

原实现复用了 native manager，但 SMP 专属任务逻辑混在通用 `src/rl/smp`。
现已移动到上述任务路径；旧导入仅作兼容转发。基础环境的 observation/action/reward
manager、物理、reset、termination 以及 native CFG hash 保持不变。

**仍有明确的时序扩展**：native RewardManager 计算基础奖励，SMP bridge 在 native step
最终 forward 后计算 prior reward，再按实验 CFG 组合。SMP 不是一个普通
`RewardTermCfg`。直接注册普通 term 会提前读取一物理子步的 kinematics，从而改变
已训练 checkpoint 的评分时序。本次保留并公开此扩展，不声称所有附加奖励都经
RewardManager；迁成普通 term 属于另一个行为变更实验。

旧训练 CLI 继续使用原低层 API以保留成对 checkpoint 语义。新的
`RslSMPEnvAdapter` 使同一任务可以直接交给公开 SMP runner；不导入另一工作区的任务源。

## 验收范围

测试区分算法数值等价、无 SDK wheel 更新、真实 MJLab terminal/update/restore。
短 smoke 验证接口和有限梯度更新，不是重新评估策略成功率，也不构成新的训练成绩。
本次结果和具体命令记录在 MJLab 工作区的 `doc/smp_rl_algs_decoupling_20261024.md`。

本次 CPU suite 为 73 passed / 6 optional skipped；安装 wheel 并禁止 SDK 导入为
70 passed / 8 optional skipped（另排除一项需 source-only examples 的测试）。
原实现/新核心的固定 RNG 对照 14 项通过。实际 native R1 GPU smoke 使用 AMP 和
compile=default，8 个环境累计 20 vector steps、19 次 SAC updates，涵盖 GSI、
8 行 timeout、terminal snapshots、cold restore 和 CPU actor/JIT 导出。
原任务训练 CLI 也完成 12+4 vector steps、11+4 updates 的初始化/恢复测试。

可选 SMP 原实现 parity 需区分数据源与代码夹具：`MBDAC_SMP_SOURCE` 指原 MJLab
checkout（读取冻结 prior/546 窗口），`MBDAC_SMP_REFERENCE` 指迁移前的
`src/rl/smp` 实现目录。代码夹具若只是迁移后的兼容导出，测试明确跳过，避免
把新实现与自身比较。实际 MJLab 数学对照用 `MBDAC_SMP_SDK_PARITY=1` 显式启用。
