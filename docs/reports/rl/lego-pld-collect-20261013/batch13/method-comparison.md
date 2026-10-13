# 首轮 PLD-Grasp：论文与官方实现对照

2026-10-13。审计对象是R01固定collect Replay上的离线更新，不是已完成在线学习闭环。训练代码cf79893，当前工作树审计基线c5b277e；没有启动新训练配方、部署学习actor或修改3588。检索论文/作者项目页，下载6份作者官方代码作只读审查，版本归[源码回执](method-source-receipt.json)。PLD、RLT、PARTS本次未从所查作者入口找到可核验的官方训练仓库，三者按原文公式/伪代码比较，不能把第三方复现当作者代码。

结论：首轮是有真实非零修正数据的残差SAC基线，缺少PLD的关键稳定配方。末端评价改善不等于入口动作价值已学好；目前证据不足以判断RL路线无效。之前将“修正变大”直接作为训练质量差的依据不充分，扩大残差可能是有效改进，也可能是Q外推，必须区分。

## 一手机制与实现证据

| 方法 | 原文/官方实现中的关键做法 | 对当前首轮的启示 |
|---|---|---|
| PLD | 成功的基础策略轨迹作先验池；Cal-QL价值初始化、基础策略采集预热；成功先验与在线经验混合；持续在线残差学习。论文明确不在actor损失施加显式行为约束。 | 当前没有Cal-QL，只有100次TD critic预热；所有输入是同一冻结collect快照，没有成功先验池。应补齐原配方，不能将加BC称为PLD原文。 |
| Cal-QL | 官方实现包含OOD动作提案、行为回报校准、保守项及TD项；离线阶段后再采集新策略轨迹。 | 单加MC回报MSE不等于Cal-QL；校准值和动作空间/密度必须匹配本项目映射。 |
| RLPD | 先验/在线混合回放、critic LayerNorm及ensemble，实际环境步进入buffer后更新。 | LayerNorm和双Q已有；它们不能替代先验经验及后续交互。官方pixel配置关闭entropy backup，但通用SAC默认可以开启，不能混称所有RLPD配置相同。 |
| HIL-SERL | 示范与在线buffer各取半batch；critic与actor分频；异步接收经验并发布网络。常用pixel SAC launcher关闭entropy backup，actor仍有熵项。 | 人工干预不是我们必须复制的部分；混合经验、目标语义和实际反馈必须明确。 |
| WSRL | 已学习策略先收集新分布的warmup经验，之后才更新；可选择不保留旧数据。 | 收集预热与100次梯度更新是不同概念；它能不保留离线池依赖已预训练策略与新交互，不能据此支持随机Q上直接离线裸SAC。 |
| RLT | 训练RL token后冻结；TD3式chunk critic；actor以Q和到VLA参考动作的平方距离共同训练；reference dropout防止只复制参考。 | 动作约束有论文依据，参考来自VLA，人工纠正可选；不能说采用参考约束就必须人工接管，也不能把零残差惩罚称为完整RLT。 |
| PARTS | 在线chunk TD3+BC；成功尝试拟合实际成功残差，可选失败向零残差约束；成功重加权重训后再次部署收集。 | 很适合本批有局部1/0的修正经验，但不是单纯给整个SAC batch加无差别BC。原乐高仅改夹爪，我们的7维关节/夹爪修正是扩展。 |
| DSRL | 在冻结扩散策略的噪声空间学习。官方pi0实现明确逐条采集新轨迹，再按新增经验规模更新。 | 更强行为先验是另一条稳定路线；它需要初始噪声接口，不能直接当当前残差actor的超参数变更。 |

来源：[PLD §3.1/B.2/C.1](https://arxiv.org/html/2511.00091v1#S3.SS1)、[RLT §IV-B/算法1](https://arxiv.org/html/2604.23073v2#S4.SS2)、[PARTS §III-B/III-D](https://arxiv.org/html/2609.21788v2#S3.SS2)。PLD仿真比较采用每任务50条成功基础策略轨迹；它不是我们13集一定不能学习的样本下限，也不能直接当真机通用需求。

已逐项检查的作者代码（固定revision）：

- [Cal-QL conservative_sac.py](https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/conservative_sac.py#L227)：实际回报参与保守提案的校准，随后形成保守损失；[主循环](https://github.com/nakamotoo/Cal-QL/blob/ac6eafec22e8d60836573e1f488c7f626ce8a77e/JaxCQL/conservative_sac_main.py#L200)在online阶段采集新轨迹并混合回放。
- [HIL-SERL learner](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/train_rlpd.py#L288)：两个buffer各半batch、网络发布与经验接收；[pixel launcher](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/serl_launcher/serl_launcher/utils/launcher.py#L85)关闭entropy backup。
- [RLPD训练循环](https://github.com/ikostrikov/rlpd/blob/c90fd4baf28c9c9ef40a81460a2e395092844f88/train_finetuning.py#L182)：按offline_ratio混合数据；pixel配置与通用SAC的entropy backup开关不同。
- [WSRL finetune.py](https://github.com/zhouzypaul/wsrl/blob/ad4dc1248a138bc15d6e053f2d1dba1b8cfbaca2/finetune.py#L338)：online warmup期间不更新，然后用新增经验更新。
- [DSRL pi0真机循环](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_real.py#L28)：采集轨迹与更新交替；[critic updater](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/critic_updater.py#L14)默认关闭entropy backup。
- [TD3+BC作者实现](https://github.com/sfujim/TD3_BC/blob/8791ad7d7656cb8396f1b3ac8c37f170b2a2dd5f/TD3_BC.py#L139)：Q项按数值尺度归一化，并与动作拟合项平衡；不等于PARTS新增的成功/失败区别处理。

## 当前代码究竟做了什么

当前`sac.py`使用标准TD双Q损失；actor只优化Q与熵，没有Cal-QL保守项、行为/参考约束。`train_sac.py`一次读取NPZ，1000步期间数据不增长；前100步只冻结actor更新，但TD目标仍查询未学习的零均值探索actor和随机初始化target Q，不是预训练价值初始化。采样是先等概率选尝试，再选尝试内转移；不是成功/失败重加权，也不是PLD成功基础策略池与在线池混合。R01全部collect，因此代码中probe/collect各半的分支并未启用；且probe/collect类别本身也不等于成功先验/在线数据类别。

已有正确组成包括：真实非零修正及来源链、完整尝试终点奖励、forced-noop排队尾部、终止mask、双Q独立初始化、LayerNorm、软目标更新、actor梯度路径和RTC合同。本次静态审查没有找到奖励全零、actor无更新、把终止后的虚构观测bootstrap等硬错误；这不等于实现完整或策略有效。

还存在具体设计差异：当前Q备份包含熵项，而Cal-QL默认配置、SERL pixel与DSRL pi0路径通常关闭；当前Q是soft value，不能直接当成功概率，也不能以Q超1或负数直接定错。当前gamma按真实秒折扣，不能直接和论文每决策gamma0.99比较。固定视觉特征没有训练时图像增强；RN18+7维残差包络也不同于PLD的RN10/末端动作和RLT/PARTS的VLA特征/完整chunk，这些是待消融的表示与控制差异，不是已经证实的失败根因。PLD附录还使用Tanh、AdamW和不同初始熵温度；不因名称相同就认为我们的ReLU/Adam/温度0.01逐项复现了原配置，温度数值更不能脱离动作与奖励单位照抄。

## 新检查：奖励信号是否传播到入口

在Thor空闲训练worker加载四个检查点，仅做前向。对每次尝试首个有可编辑动作余量的观测，给记录动作评分，并以整次尝试的实际标签计算排序AUC。这是对行为结果排序的诊断，不是新动作反事实成功率。脚本、逐尝试输出在本批`method-audit-v1/`及Thor run中保留。

| 检查点 | 入口训练AUC（425窗） | 入口验证AUC（42窗） | 验证终点AUC |
|---|---:|---:|---:|
| 250 | 0.597 | 0.601 | 0.938 |
| 500 | 0.482 | 0.628 | 0.974 |
| 750 | 0.414 | 0.616 | 0.974 |
| 1000 | 0.456 | 0.610 | 0.959 |

终点比入口容易：终点视觉已含抓取结果，且排队尾部往往是零修正。不能用终点AUC≈0.96推出抓取前动作选择有效。入口总体排序尚无稳定提升，是价值传播/表示/移动目标需优先核查的证据。按臂拆开，最后验证入口AUC左0.837（14窗/7正）、右0.563（28窗/4正）；训练入口左0.521、右0.426。两臂表现并不相同，验证正例少且仅一个episode，不能用总体AUC判定所有抓取阶段都未学习，也不能据此判死RL。

固定状态比较记录动作、零修正、actor均值和行为尺度随机提案：最后actor相对记录动作的验证平均Q自评增益仅0.0118，换行为尺度随机动作的平均绝对Q变化0.0054。没有这些新动作的真实结果，不能判断它在利用真实因果收益还是逼近Q误差。对照结果归[入口价值审计](sac-r01-entry-value-audit.json)。

## 建议下一轮比较的实现方向（本轮未启动）

建议先保留当前版本为`PLD-SAC-R01`基线，做两条明确区分的配方，而非把所有论文技巧混成一个算法：

1. **PLD-CalQL-SAC**：补校准保守价值初始化与成功先验/混合经验，先使价值在入口和中段有可信诊断，再放开actor并接入新rollout。原PLD不要求actor BC。若先用本13集成功尝试构建先验池，应明确它是有探索残差的成功经验，区别于原文零残差基础策略成功轨迹；不把执行动作改写为零残差。
2. **PARTS-TD3BC**：在相同YAM/RTC合同上实现成功残差拟合、失败可选零残差锚定、TD3目标平滑/延迟actor更新，以及成功经验重加权。动作仍覆盖整段抓取7维，不回退成只改夹爪；这也是相对原乐高实验的扩展。RLT参考动作正则可作为单独对照，不宣称用了该正则就是完整RLT。

必须落实的细节：

- 成功身份属于整次尝试。Replay中间块reward为0是正确的；不能拿`reward==1`直接筛成功BC样本，否则只拟合最后的零修正尾块。需要把终点标签传播成独立的`attempt_outcome`元数据，reward位置不改。
- 保守项的提案在已定义的latent/可执行动作映射下产生；不能把原Cal-QL的满维均匀分布/密度公式直接粘到非对称限幅、塌缩维度或RTC前缀上。强制零动作转移保留TD更新，不能要求actor修正已锁定前缀。
- 更充分地训练critic初始化可能有价值，不能将“不再无约束训练actor”误说成“不能再训练critic”。当前100步不能凭数值指定为充分；用固定评估协议核查价值尺度、入口/中段排序和保守提案差距。
- 静态阶段的`online_environment_steps`目前表示已录入collect数据量3126，训练期间新增真机转移为0。它不是在线闭环进度。下一轮应分别记录数据总量、新增交互、梯度步数、采样复用与策略版本。
- Bellman验证目标随actor/target Q变化且含随机采样，不是固定监督目标；不能要求单调下降。评估需独立随机状态，避免验证改变后续训练RNG。入口/中段已实现局部奖励的传播指标要与终点识别指标分开。
- 还需单独核查动作可纠正性：PARTS乐高原实验是一维夹爪且残差上限±0.50，我们是活动臂7维、夹爪±0.05/关节±0.02rad、固定8tick包络。更小幅的多维随机探索不一定覆盖真实偏差。先用录像/FK分析修正是否能触及失败所需位移与闭合量，不自动放大十倍、不因这个差异回退为夹爪专用框架。
- 完整策略验收仍需小批新策略自主rollout的抓起持住率、空抓率、左右臂表现和耗时；不要求人工实时键盘纠正。51个无建议窗口不硬标0，真实未抬起/掉落失败也不能永久从经验中缺席。

13集本身可继续用作首轮初始化/对照数据；先修训练配方与评价指标，再决定补多少新数据。没有证据支持先大量采集、把问题全部推给样本数，也没有证据支持当前actor已有效。
