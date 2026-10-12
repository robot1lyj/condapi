# PLD-SAC 首个真实交互闭环

这是 PLD 式冻结 Pi＋残差 SAC 的工程版本，目标是修正接近、对准、下降、闭爪、抬起这一整段抓取；完成后 Pi 负责运输与放置。左右臂都可进入，同一时刻只修正一只臂，避免首轮把双臂交互的责任混在一起。框架中的动作解码、执行回执、SAC、版本与奖励接口可复用；抓取的进入/结束与视觉判定属于任务适配器。

原论文：[PLD §3.1/附录](https://arxiv.org/html/2511.00091v1)。算法参考：[SAC 原理与目标](https://spinningup.openai.com/en/latest/algorithms/sac.html)。当前是 **PLD-style SAC online**，不是完整 PLD 复现：采用冻结 ResNet18、YAM 关节短块、分批在线更新；本版没有移植 CalQL 初始化、论文的 probing 调度或完成最终 VLA 蒸馏。

## 已实现的链路

```mermaid
flowchart LR
    O[当前三相机/状态/已承诺队列] --> P[冻结 Pi / trained RTC]
    O --> E[冻结 ResNet18]
    P --> A[零均值初始化的高斯残差 Actor]
    E --> A
    A --> D[物理边界内的短块修正]
    D --> C[客户端实际执行与回执]
    C --> V[录像自动判定＋首批抽检]
    V --> R[末个动作块奖励 1 或 0]
    C --> B[不可变 Replay]
    R --> B
    B --> S[SAC 双 Q / Actor / 温度更新]
    S --> M[有版本的 Actor 包]
    M --> A
```

代码：`packages/pld-rl/src/yam_pld_rl/{block_contract,block_replay,sac,runtime}.py`；数据入口 `scripts/pld_rl/prepare_blocks.py`，学习入口 `scripts/pld_rl/train_sac.py`。Thor 原 RTC 启动器新增可选 `--pld-bundle` 与 `--pld-encoder`，不传则保持原 Pi。缺少 bundle 时拒绝 collect/eval；不悄悄当作 Pi 执行。只支持 trained RTC 的新修正协议，其他旧推理路径仍保留。

## 决策与动作

每次根据当前图像、状态、上个实际命令、当前 Pi 50×14 计划、活动臂、已承诺长度、已过去时间与可行幅度生成 7D 修正。神经网络不读取未来实际观测、奖励或未来阶段标签。

- 8 个可编辑 tick，约 0.267 秒，同块用一个 7D 随机样本；时间权重 `[0,1/3,2/3,1,1,2/3,1/3,0]`。前后回到 Pi，块内没有逐帧独立噪声；这也是本版可能限制修正能力的明确设计取舍。
- 实际一个 decision 包括当前观测之后的 `d` 个已承诺 tick 和后面 8 个修正 tick。`end_tick = observation_tick + d + 8`，下一决策观测必须正好从 end_tick 开始，不能重叠或缺失。异步推理时，客户端用已有 Pi 后缀形成下一次已承诺前缀。
- 物理边界、命令连续性按整个块及交还 Pi 的第一个 tick 计算可行幅度 `scale(s)`；动作 `a = scale(s) × tanh(z)`，再乘物理半径加到 Pi。Q 接收 `a`，不是未经约束的 `tanh(z)`。熵明确在潜变量 `tanh(z)` 上计算；不能把它误称为投影后机器人命令的概率密度。
- 前缀和非活动臂原值保留，含 float64 前缀；若 Pi 自己已违反配置边界，拒绝该请求，不能用 PLD 静默改写。客户端已有机械限位、碰撞/力矩保护仍是最终执行防线，这些关节命令检查不保证无碰撞。
- 确定性零均值初始化等于 Pi；`collect` 从小方差有界分布采样，**第一包是探索器，还没有学到成功修正**。不使用离线 IQL/TD3BC/ReBRAC 漂移的 actor。

## 奖励与训练

按一次完整抓取尝试给一个结果；不是每帧给一分。前面块奖励为 0，最后一个完整执行块奖励为 1（抬起并持住）或 0（明确空手），并终止 bootstrap。标注绑定 `attempt_id + final_decision_id + outcome_tick`，不依赖人工何时按键。至少 5 cm 是证据资格，不能单独判成功；没分割出积木、相机遮挡、未抬升不能自动判空。

首批可以分批回收录像后判定，无需实时跑 Qwen/DINO，也不要求人为高速接管。若设备故障、改写动作、时间不连续或证据不足，整次尝试保留原件并在审计中排除，不能写成任务失败。目前 importer 要求完整尝试有合格抬升窗口，故学习范围是“有合格终态证据的抓取”；完全未抬升的真实失败需后续扩展明确的失败终态合同。

SAC 目标：`y = r + 0.99^真实秒数 × (min Q_target(s', a') − α log π(u'|s'))`；终点直接 `y=r`。双 Q，自动熵温度，target EMA，critic:actor=2:1；冻结 Pi 与视觉编码器。采样先均匀选尝试，再选块；有同合同 probe 数据时 probe/collect 各半，缺 probe 时只用 collect 并记录实际配比，不伪造离线经验。

首轮配方 `configs/pld_rl/sac-block-v1.json`：3×256 量级网络、batch 256、lr 3e-4、1000 次 critic 更新，前 100 步只更新 critic，之后最多 450 次 actor 更新；保存 250/500/750/1000，先看 Q、动作饱和和真机结果，不按训练 loss 最小自动选部署包。首轮建议至少 60 次可用带修正抓取、双臂各至少 20 次且同时含成功/失败再更新；这只是开始试验的预算，不保证收敛。

**旧 14 集的限制**：旧 replay 是实际控制单步，缺少新策略决策状态、完整块执行回执，不能直接重命名为本版 SAC replay，也不能把其 critic 无校验地移到新输入架构。保留全部奖励、原始视频、特征和既有实验作审核/对照；如要作为同合同离线 buffer，需另做原始 request 级重建并证明对应，当前不作这一声明。新首批中的 probe 记录提供同合同零修正经验。

## 存储与验证

服务器沿用 `/home/wuyan/lyj/pld-grasp/`：`datasets/raw/` 原始包；`datasets/rewards/` 独立奖励版本；`datasets/replays/` 合同一致回放；`runs/` 配方/优化器/target/RNG/指标；`bundles/` 或 run 内 bundle 是不可变导出；`reports/` 审核与验证。训练入口核对 source/data/encoder/normalization 哈希，resume 须完全同源；跨轮 `--initialize-learner` 可换 replay，仍要求同策略合同。原始数据永不覆盖。

训练面板继续用现有 `training_dashboard.py`、原 `metrics.jsonl` 格式，增加 SAC 的双 Q 误差/分歧、actor loss、温度、熵、物理归一化残差 RMS/饱和、真实在线块数/秒数；尚无真实新数据时不把合成测试曲线列成训练成绩。抓取率、空抓率、左右臂、整轮 C/N、掉落率和耗时由真机 A/B 统计，不能由 Q 或 loss 推算。

本地只执行 NumPy/协议校验；GPU 数值、optimizer 恢复、真实视觉编码前向与 CLI 端到端测试在 Slurm 分配 GPU 上运行。验收结果写本目录 `validation.json`；合成测试不代表 Thor 延迟、3588 互通或真机效果已经通过。

## 客户端审核入口

下一步看 [客户端改造与首批采集说明](client-and-first-rollout.md)。本任务不修改 3588、不发送客户端任务指令。物理限位、真实 Pi 权重身份和新协议队列联调尚需客户端提供/确认；模板有意不可直接用于真机。
