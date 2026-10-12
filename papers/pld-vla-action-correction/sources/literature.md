# 公开文献登记与读取边界

检索日：2026-10-12。采用 nature-academic-search 的多来源查找与原始来源核查流程；当前未挂载其 academic-search MCP，使用可用网页检索及 arXiv/PMLR 原站 metadata 回退，不安装新运行环境。发现渠道含作者主页/官方代码，方法结论回到论文；没有引用二手解读作为原方法证据。

查询覆盖：PLD/PARTS/RLT、residual VLA、latent-noise steering、real-time action edit、reward model、handoff/foresight、action chunking、delayed rewards。按 arXiv ID 去重。未做引用次数排名或系统综述完整性声明。

“方法选读”指本轮核对相关方法和附录，不等于全文逐公式精读；“摘要核查”不支持超出摘要的方法细节。

| 编号 | 文献与读取版本 | 本轮读取范围 | 研究用途 |
|---|---|---|---|
| R01 | [PLD: Self-Improving VLA Models with Data Generation via Residual RL，2511.00091v1](https://arxiv.org/html/2511.00091v1) | §3.1–3.2、§4.1、Algorithm 1 与相关附录选读 | 主框架、成功先验与蒸馏 |
| R02 | [PARTS，2609.21788v2](https://arxiv.org/html/2609.21788v2) | §III、实验与附录 D 选读 | 局部实践、LEGO 合同 |
| R03 | [RL Token，2604.23073v2](https://arxiv.org/html/2604.23073v2) | 表示/actor/critic、Eq.3–5、Algorithm 1 | 小专家表示与动作 chunk |
| R04 | [EXPO-FT，2605.25477v2](https://arxiv.org/html/2605.25477v2) | §3–4、实验/冻结基线选读 | 联合 edit 与 VLA 更新 |
| R05 | [Real-Time EXPO-FT，2609.18207v1](https://arxiv.org/html/2609.18207v1) | §III–IV、prefix/filter 与附录 VII 选读 | 实时执行强近邻 |
| R06 | [Foresight Residual RL，2607.16506v1](https://arxiv.org/html/2607.16506v1) | §III–V、结果/边界 | 交接质量先例 |
| R07 | [ReCoVLA，2606.09630v1](https://arxiv.org/html/2606.09630v1) | §3、evaluation variants | reward compiler/gate |
| R08 | [π*0.6 / RECAP，2511.14759v1](https://arxiv.org/html/2511.14759v1) | 方法、§V-C/D、Eq.5 | 纠正阶段奖励误解 |
| R09 | [Q-chunking，2507.07969v2](https://arxiv.org/html/2507.07969v2) | 摘要/方法选读；[NeurIPS 登记](https://proceedings.neurips.cc/paper_files/paper/2025/hash/50348e8f9aef984abe0ea1ec2a326f78-Abstract-Conference.html) | chunk RL 先例 |
| R10 | [Adaptive Q-Chunking，2605.05544v1](https://arxiv.org/html/2605.05544v1) | 摘要/方法选读 | 自适应时长先例 |
| R11 | [RoboReward，2601.00675v2](https://arxiv.org/html/2601.00675v2) | 摘要与数据/奖励机制选读 | 通用视频 reward 对照 |
| R12 | [Robometer，2603.02115v1](https://arxiv.org/html/2603.02115v1) | 摘要、偏好与 progress、评估选读 | 轨迹比较 reward |
| R13 | [TOPReward，2602.19313v1](https://arxiv.org/abs/2602.19313v1) | 摘要＋[作者方法页](https://topreward.github.io/webpage/) | token 概率 reward |
| R14 | [ResiP，2407.16677v4](https://arxiv.org/abs/2407.16677v4) | 摘要＋[原论文](https://openreview.net/pdf?id=yzAd3vcAZR)方法概念核查 | 闭环残差先例 |
| R15 | [Cal-QL，2303.05479v4](https://arxiv.org/abs/2303.05479v4) | metadata、摘要与校准定义 | 价值初始化背景；非完整实现审计 |
| R16 | [Jump-Start RL，2204.02372](https://arxiv.org/abs/2204.02372) | 摘要、[ICML 原文](https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf) §4 | guide policy/roll-in |
| R17 | [SAC，1801.01290v2](https://arxiv.org/abs/1801.01290v2) | metadata/摘要；基础算法引用 | off-policy 熵正则背景 |
| R18 | [TD3+BC，2106.06860v1](https://arxiv.org/abs/2106.06860v1) | metadata/摘要＋[官方实现说明](https://github.com/sfujim/TD3_BC) | 行为约束基线 |
| R19 | [Residual RL for Robot Control，1812.03201v2](https://arxiv.org/abs/1812.03201v2) | metadata/摘要 | 残差控制起源背景 |
| R20 | [VLA-Precision，2609.04355v1](https://arxiv.org/abs/2609.04355v1) | 摘要＋[作者项目页](https://vla-precision.github.io/) | 补充阅读；尚不展开完整 ACoB 比较 |
| R21 | [Delayed, Composite, Partially Anonymous Reward，2305.02527v2](https://arxiv.org/abs/2305.02527v2) | 摘要；协作审阅的方法核查 | 延迟/匿名反馈先例 |
| R22 | [DSRL，2506.15799v2](https://arxiv.org/html/2506.15799v2) | §4 与方法动机选读 | 噪声空间探索 |
| R23 | [VLA-RL，2505.18719v1](https://arxiv.org/html/2505.18719v1) | §3、Algorithm 1 与 PPO/LoRA 选读 | 直接 VLA RL |
| R24 | [ReinFlow，2505.22094v1](https://arxiv.org/html/2505.22094v1) | 摘要、噪声网络及训练/推理差异选读 | flow likelihood 接口 |
| R25 | [Off-Policy RL with Delayed Rewards，ICML 2022](https://proceedings.mlr.press/v162/han22e.html) | PMLR metadata/摘要；协作审阅 §2–3 | 延迟奖励价值建模 |

## 引用文件

- [references.bib](references.bib)：按读取版本的可导入记录；多数保留 arXiv 形式，不臆填会议卷页。
- [verified-metadata.json](verified-metadata.json)：原站 title/author/date、版本 URL、抓取页面 SHA；只证明 metadata 来源，不是对所有方法结论的自动验真。
- JSRL 曾尝试不存在的 `v3` 返回 404，已回退原站实际记录；最终引用未使用失败版本。

## 后续精读欠账

1. 对 PLD、PARTS、RLT 制作逐模块复现表，查清 backbone、action frame、归一化、reset 与样本预算。
2. 核对不同版本论文与发布代码是否一致；当前未审计全部开源实现。
3. 若选择事件归属机制，进一步精读 delayed reward 与部分匿名反馈的假设，不能把有限状态理论直接移植成深度真机保证。
4. VLA-Precision、ReinFlow 等目前用于分类和补充阅读；尚不宣称已完成逐公式深读。
