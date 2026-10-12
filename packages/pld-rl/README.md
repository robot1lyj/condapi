# PLD 抓取专家：离线价值初始化与多算法对照

独立包，不读取或复用已退役 RL learner。VLA 继续使用原生 OpenPI；本包训练独立小 actor / critic。

此前两轮八组 critic-only 诊断使用 `configs/pld_rl/offline14-{a..h}.json`。用户继续授权四卡多论文实验，`configs/pld_rl/methods14-*.json` 比较 PLD-IQL、PLD-TD3BC、PLD-ReBRAC 主比较八组，另补两组低critic惩罚对照；每组 20,000 更新、每卡一组、分两波。本次更新小 actor / critic，Pi 与视觉冻结，不做生产部署。具体归[对照报告](../../docs/reports/rl/pld-methods14-20261012/README.md)。

## 实现与论文的关系

- PLD：[论文 §3.1](https://arxiv.org/html/2511.00091v1#S3.SS1)，冻结 VLA、残差 SAC、critic 初始化、后续离线/在线回放和蒸馏。
- Cal-QL：[论文式 5.1](https://arxiv.org/html/2303.05479#S5)，`mean(max(Q_proposal, MC_return)) - Q_behavior`，本版用十个归一化局部提案估计期望。
- 历史 critic-only 阶段明确评估记录中的零残差行为，用 logged-next-action 做 TD 备份，无 actor 更新和 SAC 熵项。因此称“Cal-QL 风格行为价值初始化”，不冒充完整上游 Cal-QL/PLD 复现。
- 冻结官方 ImageNet ResNet18 三视角特征，替代计划中的 ResNet10；仅训练 256×256 的双 Q。该替代须作为实验条件报告。奖励判定的 SAM/DINO/Qwen 输出不进入策略观测。
- 单步物理执行间隔初始化，动作 horizon=1；后续在线短块、因果入口、RTC 前缀与物理修正上限需独立验收。当前权重不声明在线可用。

## 数据与指标

E01 留出，E02–E14 训练，状态/动作统计不读取 E01。按 attempt 均匀抽样，再在 attempt 内抽 transition，防止长片段主导。

输入仅为当前/过去 RGB、state、过去动作、当前记录的 Pi 目标、活动臂和队列标量。奖励、闭爪时刻、后验阶段、最终结果不作为输入。关节目标相对当时 state 编码，夹爪保持原 absolute 值；不换单位，不把统计归一化尺度当部署物理边界。Q 评分活动臂的实际 7D 动作，另一臂行为通过状态与名义动作条件提供。

逐 tick/epoch/time/action 证明物理连续。相机失效移除相应观测转移，不压缩时间；有真实物理断裂时，仅终点相连后缀允许 MC 校准。每次尝试只在最后一条合法转移给一次结果奖励。缺少精确末端观测转移的尝试仍保留标签，但不硬塞进本轮 replay。

主要离线指标：entry MC RMSE（校准诊断）、TD loss、MC bias、双 Q 分歧、两臂 held/empty Q 与样本数。entry AUC 只比较同一初始窗口的分值，单集/少量失败样本不足以代表泛化或准确率。整轮 C/N 和真机抓取率属于后续实机评价，离线 loss 不替代它们。

工作站仅运行数据准备、纯数组/协议测试和看板；`cache_features.py`/`train_offline.py` 强制要求 Slurm+CUDA。所有训练数据和日志在服务器独立新目录保留，见 docs/03、04、11。
