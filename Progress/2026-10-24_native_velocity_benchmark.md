# 2026-10-24：原生速度任务与三方法实验

目标保持 active。用户已同意用 MJLab 原生 Go1 替代 Go2；另一项任务是原生 G1。
本轮只做速度任务与通用 MBDAC/WM+FastSAC 框架，不改纸飞机、R1、AMP 或 motion tracking reward。

## 验收与比较

- 1024 个原生 `ManagerBasedRlEnv`，三种方法共用任务配置、物理、奖励、随机化与评估。
- 比较原 MBDAC、直接继承 Holosoma 源码训练循环的 FastSAC，以及 WM+FastSAC。
- 全部从随机网络与空 replay 开始；没有复用之前 G1 tracking 的 WM 或 policy。
- 主横轴为真实时间；另给真实 environment transitions 横轴。iteration 不跨方法等价。
- 图包含相同固定评估的 return、XY 速度 RMSE、yaw rate RMSE 与 survival。
- 原始阈值为 XY ≤ 0.20 m/s、yaw ≤ 0.30 rad/s；同时给更严格的逐指令阈值。
- 五类固定指令，每次 20 秒，禁止物理 reset，单独显示前进、后退、侧移、转向和站立。
- 两个连续评估逐指令达标且 return 变化 <5% 后停止；v3 正式比较每种方法最多一小时（不含评估）。
- 记录实际未达标、崩溃和开发预实验；不把不同任务、预训练或更新次数当作公平加速。

完整协议：[native_velocity_benchmark.md](../doc/native_velocity_benchmark.md)。

## 已实现的通用改动

1. 实际 Q 更新可直接接收成对的真实 critic features；想象 Q 才调用预测的物理 head。
2. 可选 controlled critic 接口保留 previous action；旧任务接口继续兼容。
3. BF16 / fused AdamW 为显式选项，默认 fp32 / 非 fused 保持旧 checkpoint 行为。
4. 保留原 model-only 配方；新的 hybrid 配方在 WM warmup 期间先进行真实 SAC 更新，之后混入模型梯度。
5. 标量日志统一打包再读回 CPU，并保存 simulator 可变 tensor 的快照。
6. FK 根据传感器所需的树路径裁剪分支，支持两种机器人和固定 link / 非零关节 anchor。

这些改动没有根据机器人名或某个 reward threshold 在学习核心中分支。
任务适配与原生 reward 视图位于 `examples/mjlab_velocity/`。

## 已验证

- 原生 Go1、G1 的 reward 公式在相同测量下误差低于 1e-8。
- 观测 / FK 位置和速度重建误差约 1e-6；各强制测试包含 8 个 timeout 和 8 个真实失败。
- 保留 reset 前物理状态，timeout bootstrap 不读取 reset 后的机器人。
- G1 传感器 FK 局部 forward/backward：36.3 ms → 3.92 ms，输出误差 0，梯度最大误差 4.77e-7。
- BF16 actor loss / 梯度与 Holosoma 对应 autocast 代码一致；C51 CUDA scatter 原子求和差约 1.2e-7。
- 新接口测试确认真实 Q / H=0 actor 不调用预测 critic feature 函数。
- 标量日志优化不改变真实 replay 或 normalization buffer。

注意：native reward 读取的部分运动学缓存比 raw state 晚一个 5 ms 物理步。
公式一致不等于想象 reward 精确；该时间近似与物理 head 误差单独记录。

## 预实验与正式运行

- `native_go1_fastsac_v1`：成功的源代码预实验，6000 步结束，最终 XY 0.055 m/s、yaw 0.052 rad/s。
- `native_go1_wm_fastsac_v1`：发现真实 Q 曾使用预测接触量；747 iteration 主动停止，保留 checkpoint。
  此轮不进入公平比较。过程耗时 526.12 s，不能隐藏为成功或训练崩溃。
- `native_go1_fastsac_v2`：正式基线，原生配置 SHA256
  `3d05202c49c7da8f065daa42fcddbbe52720035ea4dd273bc4c880e5aa4991f6`。
  3010 步的第一次训练后评估已逐指令达标，active process 309.97 s；6060 步稳定达标后结束，
  过程总计 641.41 s（含评估）。[W&B](https://forge.coreweave.com/wandb/sslab_motion_mimic-teams/mbdac-native-velocity/runs/ujc6scu7)。
- `native_go1_wm_fastsac_v2`：旧 model-only 预实验约 16 分钟仍主要站立，停止并保留原始日志。
- 修正世界坐标系力 head 所需的 yaw 信息后，`native_go1_mbdac_v3` 已完成；
  3520 次更新时首次观测逐指令达标（2451.07 s），XY 0.0525 m/s、yaw 0.0461 rad/s；
  3940 次更新再次逐指令达标，XY 0.0431 m/s、yaw 0.0344 rad/s，按稳定验收条件停止。
  总过程 2862.40 s（不含退出时的 W&B 同步），含同步的子进程完整耗时 2900.46 s。
- 当前 `native_go1_wm_fastsac_v3` 已进入真实训练，1024 世界，启动阶段显存约 5.4 GB，无 OOM。
  后续顺序运行同接口的 `native_go1_real_only_v3` 对照，
  以及 `native_g1_wm_fastsac_v3`。所有网络、replay 与 normalization 从零初始化。
- v3 hybrid 使用与原版相同的 3e-4 actor LR；WM warmup 后使用 0.9 真实 SAC + 0.1 模型 H=8 梯度。
  主比较仍是完整算法配方的比较，不能把差异全部归因于 WM。

`scripts/native_velocity_matrix.py` 负责顺序执行与记录，防止同时训练争用显卡。
它不覆盖已有 trial。每轮保存实际 source snapshot，最终从评估 JSONL 生成图。
视频由实际 native GPU trajectory 渲染，不做 joint smoothing。

当前不能宣称 WM+FastSAC 比原版更快：第三条正式曲线、同接口对照及 G1 实验尚未完成。

0.3.3 独立 wheel 验证：50 passed、5 个可选依赖跳过，28 个子模块不需要 simulator SDK。
源码和 wheel 内全部 30 项源码/数据文件与 5 项许可证逐字节一致。
SHA256：`1cca78c45fa242e0f7963c85bdcf1c2ea080aea15ba85ae93ce1311eadaf0bb9`。
