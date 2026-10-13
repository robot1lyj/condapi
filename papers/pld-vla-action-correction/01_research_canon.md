# 事实与术语底稿

证据类型分为：文献已报告、项目历史记录、用户要求、研究假设。以下为 2026-10-12 研究初稿的事实快照；后续离线实验与 RTC 实现变化见 [2026-10-13 实验记录](experiments/20261013-rtc-rl/experiment-log.md)，历史结果没有由论文线程重新运行。

## 文献事实

| ID | 已核对内容 | 来源与定位 |
|---|---|---|
| L01 | PLD 为 Probe, Learn, Distill：冻结 VLA、残差 RL、分布对齐的混合采集、SFT 蒸馏。成功 base rollout、Cal-QL 初始化、离线在线混合回放属于原配方。 | [PLD v1](https://arxiv.org/html/2511.00091v1)，§3.1–3.2、§4.1、Algorithm 1 |
| L02 | PARTS 已有局部瓶颈 selector、verifier、受限 residual chunk、TD3+BC、成功重加权重训和交还 base。LEGO 示例为夹爪开度；通用框架可选其他动作维度。 | [PARTS v2](https://arxiv.org/html/2609.21788v2)，§III-B、附录 D |
| L03 | RLT 压缩 VLA 内部表示，actor 接收参考动作 chunk，critic 评估动作 chunk，actor 用参考动作正则。不是必须依赖实时人工纠正才能运行的定义。 | [RLT v2](https://arxiv.org/html/2604.23073v2)，§III，Eq.3–5、Algorithm 1 |
| L04 | ResiP 已在冻结 chunked BC 上使用闭环残差修正；快速小策略修正慢动作生成器不是新概念。 | [ResiP](https://arxiv.org/abs/2407.16677)，摘要与方法 |
| L05 | Real-Time EXPO-FT 已结合最新观测动作编辑、延迟观测、RTC 前缀条件及过滤备份。 | [Real-Time EXPO-FT](https://arxiv.org/html/2609.18207v1)，方法、附录 VII |
| L06 | Foresight Residual RL 已用后续技能成功概率塑造局部终止状态；“交接质量”本身不能作为原创主张。 | [Foresight Residual RL](https://arxiv.org/html/2607.16506v1)，§IV–V |
| L07 | ReCoVLA 已用 VLM 识别失败/恢复阶段、编译 reward mask，并用阶段前提 gate 防止奖励过早激活。其学习主要依赖仿真状态奖励。 | [ReCoVLA](https://arxiv.org/html/2606.09630v1)，§3.1–3.3 |
| L08 | RECAP 原文 §V-C 从 episode 成败标签定义稀疏奖励及剩余步数价值；不能把 RECAP 定义成必须逐阶段人工打分。 | [RECAP](https://arxiv.org/html/2511.14759v1)，§V-C，Eq.5 |
| L09 | Q-chunking 与 Adaptive Q-Chunking 已研究 chunk critic、多步备份与自适应动作时长。 | [Q-chunking](https://arxiv.org/html/2507.07969v2)、[AQC](https://arxiv.org/html/2605.05544v1) |
| L10 | RoboReward、Robometer、TOPReward 分别提供通用视觉语言奖励、轨迹比较监督与基于 token 概率的零样本进度估计。通用视频奖励不是新空白。 | [RoboReward](https://arxiv.org/html/2601.00675v2)、[Robometer](https://arxiv.org/html/2603.02115v1)、[TOPReward](https://arxiv.org/abs/2602.19313) |

## 项目事实

| ID | 事实 | 证据及边界 |
|---|---|---|
| D01 | 14 集自主 Pi rollout，用户确认均使用 20 万步 checkpoint，无 HIL。 | 会话确认；原审核还记录资产来源冲突，论文材料必须保留 provenance 复核项。 |
| D02 | 历史 342 闭爪提案中，327 具备本批 ≥5 cm 同尝试抬升及新鲜同臂图像资格；标签提案为 136 持物、191 空手，15 排除。 | `sources/user_materials.md` 中的审计文件与哈希。是资格/标注统计，不是实测 reward accuracy。 |
| D03 | 用户审核 13 条，覆盖 11 条中置信及 2 条额外样本；E01 已用于感知调试。 | 历史审核记录；不是独立随机测试。 |
| D04 | 该批存在助手视觉复核；尚无经过独立验收的全自动奖励器，也无残差专家真机改进结果。 | 历史 README；`approved_reward=null`。 |
| D05 | Pi-only 数据的 residual 都为零；不能识别未尝试修正动作的实际收益。 | 数据采集机制。可用于有限支持下的价值初始化，必须另做在线有界探索。 |

## 用户约束

- 框架应适用于多任务，LEGO 是起点；不把颜色框规则、固定裁剪或 5 cm 当通用方法定义。
- 整段抓取动作修正，不限夹爪开度或单一高度维度。
- 左右臂都覆盖；不把“按臂轮流介入”写成已验证的同时双臂联合控制。
- 自动跟踪关闭；不增加固定 1 秒停留；本轮不重新处理数据或开始训练。
- 人工可以抽检结果，不能假设操作者能在快速失败瞬间接管，也不能把慢键盘纠正当必需探索来源。

## 术语表

| 标准名称 | 本文含义 | 避免混用 |
|---|---|---|
| PLD | Probe, Learn, Distill，残差专家与分布对齐采集及蒸馏的框架 | 不是 PARTS 的简称 |
| PARTS | Policy Adaptation with RL on Targeted Subtasks | 不写 PART |
| 残差专家 | 在底座建议上增加有界修正的小策略 | 不是未训练即可改善的规则控制器 |
| 奖励器 | 对已发生的行为结果提供反馈 | 不等于预测未来回报的 critic |
| 证据资格 | 这一窗口是否足以判断目标结果 | 不等于成功，不等于置信分数 |
| 局部结果 | 某次动作是否实现局部目标 | 不等于完整任务成功 |
| 执行动作 | 投影、限幅、仲裁后真正下发并对应观测转移的命令 | 不等于 actor 原始输出；也不等于机器人实测运动 |
| RTC 已提交前缀 | 已进入执行承诺、当下不可改写的动作部分 | 不等于所有未来动作都不可修正 |
| Cal-QL | Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning | 价值尺度校准不等于奖励概率校准 |

## 禁止主张

1. 当前方案已证明提升成功率、样本效率、跨任务泛化或安全性。
2. 342/342 分类完成等于 100% 检测准确率；136/327 等于机器人整轮成功率。
3. 不可判片段就是失败；模型一致就是真值；5 cm 抬升就一定持物。
4. 更换 SAC/TD3、引入 DINO/SAM/VLM 或外挂 residual 本身就是算法创新。
5. 强制手工逐阶段标注是 RECAP 的必需条件。
6. 用同一奖励器给训练打分和给最终成功率裁判即可形成独立验证。

## 未解决事项

- “证据有效性与执行对齐”能否形成超出标准工程正确性的可辨认方法贡献？尚不能确认。
- 启发式切段是否漏掉未闭爪的对齐失败、未抬升失败？现有计数不能回答。
- 双臂干预重叠、原始录制动作语义、排队延迟、下游交接质量，须在相应实验前核实。
- 专家是否需要本体/任务适配？共享架构不等于共享权重泛化。
