# PLD 多算法离线对照：14 集固定数据

2026-10-12 用户授权继续并行实验，以 `pld-<论文方法>-<设置>-s<种子>` 命名。PLD 表示冻结 Pi + 小残差专家框架，后缀表示学习目标。这里是核心算法的 YAM 适配，不是原论文硬件、数据及完整线上实验的逐项复现。

## 配方

| 方案 | 数量 | 核心更新 | 设置 |
| --- | ---: | --- | --- |
| PLD-IQL | 4 | expectile V、数据内 Q、优势加权高斯策略似然 | expectile 0.7 / 0.9，各 seed 0/1；温度 3，权重上限 100 |
| PLD-TD3BC | 2 | 双 Q、目标动作平滑、Q 最大化 + 行为约束 | seed 0/1；α=2.5，BC 按动作维平均，策略每 2 步更新 |
| PLD-ReBRAC | 2 | next-action 行为惩罚进入 backup、actor 行为约束、critic LayerNorm | seed 0/1；actor β=1，critic β=1；策略每 2 步更新 |

共用 E02–E14 的 299 次抓取 / 29,842 条转移训练，E01 的 12 次抓取 / 1,235 条转移作开发验证。成功和空抓均保留，不加额外成功示范。数据版本 `pld-offline14-20261012-v2`、奖励、图像引用、冻结 ResNet18 缓存不变；先均匀抽尝试，再均匀抽尝试内转移。

每组 20,000 更新、batch 256、Adam 3e-4、FP32、TF32 关闭、τ=0.005、梯度裁剪 10，每 100 步记录、500 步验证、5,000 步保存。四卡分两波，每卡一组。IQL/TD3BC 为 2×256 ReLU，ReBRAC 为 3×256 ReLU + critic LayerNorm，因此不是只改变单一 loss 的消融。统一预算没有照抄 ReBRAC 的 1e-3 / batch 1024 / 百万步，也没有采用 IQL 的 cosine actor 衰减。

残差均值 `u=tanh(actor(x))`，Q 接收 `Pi_base + 0.25 × action_std × u`。std 只由 E02–E14 拟合，不转换原单位；这只是离线动作参数化，**不是已批准的物理部署限位**。活动臂输出 7D，左右共享网络。Pi / 视觉冻结；actor 零头初始化，不做单独 BC 预热。

IQL 使用状态无关对角高斯，均值 tanh，log std 裁在 [-5,2]，与作者公开配置的策略形式一致；诊断只用均值，不下发高斯样本。TD3 系方法在 u 空间加 σ=0.2、clip=0.5 的目标平滑噪声，再裁到 [-1,1]。实际时间折扣、按尝试抽样、残差坐标、零头初始化与冻结视觉是项目适配。

## 奖励与限制

抓取检查终点对应转移给持物 1 / 空手 0，中间 0，terminal 不 bootstrap，不复制奖励到每帧。整轮 C/N 只归档，不任意与局部奖励相加。数据全部是零修正行为；IQL 均值应保持零，不能称学会修正。TD3BC/ReBRAC 的非零偏移可能是 Q 外推误差，没有实际动作结果时，critic 自评增益不能证明成功率提升。

这些 run 是单 tick 离线学习，不代表整段动作/RTC 已完成。在线入口、物理投影、承诺前缀、连续性及取消语义尚未验收；本次不改 Thor 或客户端。

## 评价与归档

固定最后一步为主要对照，`best-diagnostic.pt` 仅便于分析。E01 已用于多轮调试，不是独立测试集，只有两次失败；左臂没有失败，AUC 无定义。

- 数值：Bellman 误差、Q 范围、双 Q 分歧、末步奖励 RMSE、梯度、有限值。
- 回报：逐尝试入口 MC RMSE / AUC、训练均值常数及常数 1。优化策略 Q 与实测行为回报不同，不能单凭 MC RMSE 判策略优劣。
- 动作：残差 RMS、边界饱和比例、左右臂预测；自评 gain 不是实测收益。
- 来源：实际更新次数、参数量、GPU 显存、更新/加载时间、配置/代码/数据 SHA、环境及 Slurm 号。

服务器新目录 `runs/pld-methods14-20261012-v1/` 保留配置、原始 metrics、每次验证的 311 次尝试预测、优化器/RNG/目标网络检查点及失败日志。原八组 critic-only 原件和 ID 不变，仅补统一展示名；复用 8766 看板。作业完成后补结果。

## 来源

- [PLD](https://arxiv.org/html/2511.00091v1)：冻结 VLA、残差专家、后续在线 RL / 蒸馏的主框架，本轮没有完成其线上 SAC。
- [IQL 论文](https://arxiv.org/abs/2110.06169)、[作者代码](https://github.com/ikostrikov/implicit_q_learning)：expectile、数据内 Q、优势加权策略。
- [TD3+BC 论文](https://arxiv.org/abs/2106.06860)、[作者代码](https://github.com/sfujim/TD3_BC)：目标平滑、延迟更新、归一化 Q / BC。
- [ReBRAC 论文](https://arxiv.org/abs/2305.09836)、[作者 CORL 实现](https://github.com/tinkoff-ai/CORL/blob/main/algorithms/offline/rebrac.py)：next-action 惩罚在 backup 内，actor 使用 min 双 Q 和动作维求和 BC。

按来源公式独立实现；审读代码 URL、访问日期、内容 SHA 归 [paper-sources.json](paper-sources.json)。原 ReBRAC raw 路径不可取，使用作者 CORL 版本并记录替代来源。
