# BEHAVIOR 1K 比赛实现与 π0.5 代码对照

这项工作复用 π0.5 预训练的 SigLIP、PaliGemma 和 Gemma 动作专家，在固定 50 个任务上增加相关噪声、跨层 KV 混合、阶段条件和多样本 Flow Matching，并在执行端做动作压缩与夹爪纠错。主要新增模型在比赛仓库的 `src/b1k/`，不是直接改写 OpenPI 的 `Pi0` 类。

本页作为源码学习入口，区分公开代码、作者描述和 YAM 适配判断。以下代码行为固定到 2026-10-08 下载的版本；它们不是本项目的新训练配方或部署合同。

## 固定版本与本地目录

所有目录相对实际工作树根目录。源码保留独立 Git 仓库和许可证，位于已被 `.gitignore` 忽略的 `downloads/behavior-1k-study/`。

| 对象 | 固定提交 | 本地位置 |
|---|---|---|
| [比赛实现](https://github.com/IliaLarchenko/behavior-1k-solution/tree/ca556f74a455cef7987a2be4537b5ac85cc56dd7) | `ca556f74a455cef7987a2be4537b5ac85cc56dd7` | `solution/` |
| [比赛指定 OpenPI fork](https://github.com/wensi-ai/openpi/tree/01177e0242a1c7e8fad2547caa0e987def614cda) | `01177e0242a1c7e8fad2547caa0e987def614cda` | `solution/openpi/` |
| [BEHAVIOR 仿真源码](https://github.com/StanfordVL/BEHAVIOR-1K/tree/684a83050ddd398de231e6aa7fc605bc34458d4b) | `684a83050ddd398de231e6aa7fc605bc34458d4b` | `solution/BEHAVIOR-1K/` |
| [官方 OpenPI 历史基线](https://github.com/Physical-Intelligence/openpi/tree/b29f81b6376f84606278a3350732a6cc806bb5e8) | `b29f81b6376f84606278a3350732a6cc806bb5e8` | `openpi-upstream/` |
| 本项目阅读基线 | `3025052d1c799374a9683e010fc06836afc6c625` | 当前工作树 `src/openpi/` |

官方基线取官方下载的 main `215abfb217dbac7d5f1273282331b9b1866c0479` 与固定 fork 的 `git merge-base`，不是任取最新 OpenPI 进行比较。历史基线的 `pi0.py` 与 fork 中的该文件逐字相同，因此可以清楚分开“fork 的机器人适配”与“比赛团队的新模型”。

[来源和 SHA256 清单](../../downloads/behavior-1k-study/source_manifest.json)记录 40 个阅读源文件、10 个差异产物以及仓库提交。仿真路径实际是 `BEHAVIOR-1K/OmniGibson/omnigibson/learning/`；比赛 README 示例省略了 `OmniGibson/` 这一层，学习时以固定树为准。

## 三层比较与阅读入口

第一层比较官方历史 OpenPI 与比赛 fork，阅读 [fork 完整差异](../../downloads/behavior-1k-study/diffs/01_upstream_to_behavior_fork.patch)。这里包含 B1K 的状态映射、数据加载、评估 wrapper、验证训练入口和额外 LoRA 变体；不是相关噪声等核心创新的所在。该差异排除了 `uv.lock` 和 notebook；[完整文件统计](../../downloads/behavior-1k-study/diffs/01_upstream_to_behavior_fork.stat)保留全部文件范围。

第二层比较官方 `Pi0` 与比赛 `PiBehavior`，阅读 [模型差异](../../downloads/behavior-1k-study/diffs/02_pi0_to_pi_behavior.patch)。另外保存了 model config、normalization、训练入口、policy 与 data loader 的逐文件对照。文件职责有拆分，统一 diff 是导航工具，不能仅凭新增行数判断贡献。

第三层比较官方历史 OpenPI 与本项目，阅读 [Pi0 差异](../../downloads/behavior-1k-study/diffs/07_upstream_to_condapi_pi0.patch)和 [Gemma 差异](../../downloads/behavior-1k-study/diffs/08_upstream_to_condapi_gemma.patch)。本项目已有训练时 RTC 和推理时 RTC，不能把比赛的 rolling inpainting 当成首次引入动作块连续性。

| 主题 | 比赛实现入口 | 官方或本项目对照 |
|---|---|---|
| 权重复用与新增参数 | `src/b1k/training/weight_loaders.py:30` | 官方 `src/openpi/training/weight_loaders.py` |
| 任务与阶段 token、注意力 | `src/b1k/models/pi_behavior.py:452,520` | 官方 `pi0.py:106`，本项目 `pi0.py:108` |
| 跨层 KV 混合 | `pi_behavior.py:35` 的 `KVCacheTransform` | 官方逐层 KV 对应 |
| 相关噪声与统计 | `pi_behavior.py:253,366`；`scripts/compute_norm_stats.py:262` | 官方 `pi0.py:189`；本项目 `pi0.py:194` |
| 多样本损失与 FAST | `pi_behavior.py:683`；`scripts/train.py:225` | 官方 `pi0.py:189`、`scripts/train.py` |
| 动作归一化 | `src/b1k/transforms_normalize.py:13,67` | 本项目 `src/openpi/transforms.py:115,149` |
| 推理约束 | `pi_behavior.py:403,906` | 本项目 `pi0.py:304,361` |
| 上一块动作重新编码 | `src/b1k/policies/pi_behavior_policy.py:26` | 本项目 policy 与 RTC 输入转换链 |
| 执行压缩与阶段投票 | `src/b1k/shared/eval_b1k_wrapper.py:126,176,190` | 本项目执行协议，不能直接替换 |
| 手写夹爪纠错 | `src/b1k/shared/correction_rules.py:151,255` | YAM 夹爪合同与实际反馈 |
| checkpoint 路由 | `src/b1k/policies/checkpoint_switcher.py:20` | 同架构不同任务组 checkpoint |

表中未带前缀的比赛文件路径位于 `solution/src/b1k/models/`；其余比赛路径相对 `solution/`。官方路径相对 `openpi-upstream/`。

## 数据到动作的实际路径

训练端先将仿真 proprioception 映射为 23D state。动作合同包含底盘、躯干、双臂和夹爪；delta mask 为 `(-3,3,-1,7,-1,7,-1)`。未来每个关节动作都减去动作块起点的当前状态，不是逐帧相邻差分。之后计算阶段标签、任务 ID、FAST token 和归一化动作，再 padding 到模型的 32D。

`ComputeSubtaskStateFromMeta` 按 `timestamp × 30 / episode_length` 将 episode 等分成该任务的阶段数，阶段数为 5–15。这是时间进度标签，不是人工标注的“已抓住”“已放下”等语义完成状态。

训练时真实阶段输入策略；推理时 wrapper 维护阶段、以预测历史投票更新，再用于下一次动作块预测。每个 episode reset 和任务切换都会重置历史、阶段和保留动作。它以很少的历史状态解决视觉歧义，但不能据此认为模型具备通用规划或可靠的语义成功检测。

模型输入结构是三相机图像、一个基础任务 token、四个融合阶段 token、离散 state token。图像与基础任务在同一注意力块，阶段与 state 在下一个块，FAST 为自回归后缀。阶段分类读取基础任务 token 输出，因此不会直接读取输入的真实阶段而发生标签泄漏。动作专家读取前缀，但其 KV 中的 FAST 部分先被裁掉，避免读取示范动作答案。

## 相关噪声改变了什么

官方 `Pi0.compute_loss` 对每条观测采样一次独立标准高斯噪声，构造 `x_t = t ε + (1-t) a`，目标为 `u_t = ε-a`。比赛保留这一 Flow Matching 目标，改变的是噪声分布和样本数。

统计脚本把已经按 delta 合同转换的动作块 padding 后展平，按每个时间位置和维度标准化，计算全矩阵的经验相关性；常量维度的行列置零、对角置一，并加入数值正则。Cholesky 因子保存到 `norm_stats['actions'].action_correlation_cholesky`。

模型加载时重构 `Σ = L Lᵀ`，应用 `Σ_reg = βΣ + (1-β)I`，再求 Cholesky。比赛配置 `β=0.5`。`generate_correlated_noise` 采样标准正态 `z`，计算 `ε = z L_regᵀ`。训练和推理都用该函数；相关性跨越时间与关节维度，不能用一次独立的逐关节低通滤波代替。

公开配置 `H=30,D=32`，因此实际加载矩阵是 **960×960**。论文以有效动作 `30×23` 说明的 690×690 矩阵，不等于公开实现的完整 padded 形状。迁移到 YAM 时必须按本项目真实动作合同、padding 和 horizon 重新估计；既有 norm 文件没有这个字段，矩阵也不能借用 R1Pro 数据。

相关因子存为 `nnx.Intermediate`，不能只拷参数 checkpoint 而丢弃对应 norm assets。训练初始化和推理 policy 创建都显式加载它；开启相关噪声但缺少矩阵会拒绝运行。

## 多样本训练与跨层 KV

`compute_detailed_loss` 先计算一次图像与任务前缀的 KV，然后使用 `jax.vmap` 对 15 组不同 `(noise,time)` 运行动作专家，平均动作损失。它复用昂贵的视觉前向和前缀计算，增加的是动作专家计算与激活需求；不是把一条动作轨迹展开成 15 个连续去噪步骤，也不能认为训练成本或效果提高 15 倍。

`KVCacheTransform` 使用两套互相独立的系数矩阵混合 K 和 V，再加 bias。系数从单位阵、bias 从零初始化，因此开始时保持 π0.5 的逐层对应关系。FAST token 在混合前已经移除。

公开比赛配置总损失为动作 MSE 加 `0.1 × stage CE` 和 `0.05 × FAST CE`。FAST 在训练端 teacher forcing，推理端禁止提供 FAST token；动作仍由连续 Flow Matching 生成。

需要读具体比赛配置而不是类默认值：`PiBehaviorConfig.use_knowledge_insulation` 默认 True，而 `pi_behavior_b1k_fast` 显式设为 False，动作损失可通过 KV 回传至主干。`num_flow_samples` 的函数与训练配置默认都是 1，比赛命名配置才设为 15。

`PiBehaviorWeightLoader` 从 π0.5 checkpoint 加载共有权重，对任务 embedding、阶段分类、融合网络、FAST 头和 KV transform 等新增参数保留各自初始化。新参数没有直接来自 π0.5 的已训练权重，不能把这个模型视作原 checkpoint 的无训练插件。

## 推理衔接与本项目 RTC 的区别

wrapper 保留上一块的一段**输出动作**。`PiBehaviorPolicy.infer` 将它与新观测一同送回输入 transform，重新做 delta 和归一化，然后传给模型的 `initial_actions`。对 delta 模型这是关键：保留的绝对动作必须相对新观测重新编码，不能复用上一块的归一化 delta 数值。

去噪每一步先做 Euler 更新，再在约束位置 O 上回填 `x_desired = (1-t) a_old + t z_fixed`；对自由位置 U，按 `δ_U = Σ_UO Σ_OO⁻¹ δ_O` 传播修正。代码使用正定线性求解和小正则，未显式计算矩阵逆。若传播修正的最大绝对值大于 1，会跳过 U 修正；O 约束仍保留。

代码仅在更新后的 `time_new > 0.3` 时执行回填，最后一段去噪释放约束。“soft”是时间范围上的释放，**最终前缀没有严格相等保证**。本项目 `sample_actions_trained_rtc` 对已承诺动作逐步硬保持，并在训练损失里排除前缀；比赛方案不等价于这一合同。若后续研究结合相关噪声和 RTC，必须先定义已承诺前缀与可调整未来动作，保留原路径及关闭/回退开关。

## 公开代码中需要单独核对的细节

以下是对固定源码的静态观察，不能当作比赛 checkpoint 实际训练历史或性能结论。

1. **关闭压缩时实际执行 20 步。** 默认压缩分支取 26 步，插值为 20 步，保留索引 `[26:30]`；夹爪变化使压缩关闭后，代码取前 20 步，保留 `[20:24]`。因此“预测 30、执行 26、保留 4”只覆盖默认压缩分支。
2. **冻结标志不等于冻结参数。** `freeze_vision_backbone=True` 控制 SigLIP 的 `train=False` 调用；训练参数由单独的 `freeze_filter` 决定，公开配置默认 `nnx.Nothing`，没有将图像参数加入排除或 stop-gradient。仅凭标志不能宣称视觉权重没有更新。
3. **最终阶段的回退分支有额外条件。** `update_current_stage` 将晋级、跳阶段和回退都包在 `next_stage <= max_stage` 内；达到最后阶段后，常规投票回退也不会进入。另有 radio 专用阶段重置规则；后续实验需单独检查阶段卡住和恢复路径。
4. **逐时间归一化的维度排除与论文描述需核对。** 公开 `NormalizeWithPerTimestamp` 在有统计时整张广播；统计脚本也为每个动作维度和时间位置生成 mean/std，本链路未见显式排除底盘速度和夹爪。不能把论文中的排除条件直接视为源码已落实。
5. **夹爪纠错依赖手写任务表。** 包括哪些任务某侧夹爪应始终张开、哪个阶段前不允许闭合，以及 R1Pro 的 `[-1,1]` 状态阈值。它不是通用的抓取成功感知器，不能直接套到 YAM 的夹爪单位和行为上。

## 对 YAM 的学习顺序

| 顺序 | 先弄清的问题 | 现有项目的区别 |
|---|---|---|
| 1 相关噪声 | 数据统计、Cholesky、训练与推理的分布是否一致 | YAM 原始 14D，模型 32D/H50；需要新的 train-only 统计与训练，而不是给既有 checkpoint 换噪声 |
| 2 多样本损失 | KV 复用、梯度路径、各样本平均和显存代价 | 保持原生 OpenPI trainer；服务器再验证数值和吞吐 |
| 3 KV 混合 | 单位阵初始化是否保持原行为，参数/导出如何保存 | 会新增可训练参数及推理算子，不能直接复用现有导出产物 |
| 4 阶段跟踪 | 时间等分是否足以表达抓取、放置、回撤，失败时怎样回退 | 可先阅读算法；语义阶段必须有数据证据，不能用进度标签冒充成功标签 |
| 5 动作块衔接 | 绝对动作重编码、硬前缀、可修改未来动作怎样划分 | 本项目已经有 RTC，优先比较约束语义和精度 |

任务 ID 替代语言是封闭任务比赛的选择，不应据此删除 YAM 的语言指令。动作执行压缩也依赖仿真控制周期与机器人合同，其 1.3 倍是轨迹执行速度，不是网络推理加速。

下一步逐函数讨论可从 `generate_correlated_noise → compute_detailed_loss → sample_actions` 开始，每项分别确认公式、张量形状、梯度和所需资产，再决定是否值得设计服务器消融。作者未提供充分消融，比赛排名不能证明每一项都必要。

## 下载恢复与静态核对

在一个尚不存在目标目录的检出中，可从工作树根目录恢复相同参考源码：

```bash
mkdir -p downloads/behavior-1k-study
git clone https://github.com/IliaLarchenko/behavior-1k-solution.git downloads/behavior-1k-study/solution
git -C downloads/behavior-1k-study/solution checkout --detach ca556f74a455cef7987a2be4537b5ac85cc56dd7
git -C downloads/behavior-1k-study/solution submodule update --init --depth 1 openpi BEHAVIOR-1K
git clone https://github.com/Physical-Intelligence/openpi.git downloads/behavior-1k-study/openpi-upstream
git -C downloads/behavior-1k-study/openpi-upstream checkout --detach b29f81b6376f84606278a3350732a6cc806bb5e8
```

本次核对包括四份下载检出的干净状态、子模块精确提交、阅读源文件 SHA256 和比赛 `src/` 与 `scripts/` 的 25 个 Python 文件 AST 解析。AST 通过只说明语法可解析；模型加载、梯度、显存、延迟和真实任务表现仍需在服务器上另行验证。未执行上游 setup、安装模型依赖、下载 checkpoint/数据、运行训练或推理。

该页和路由/历史随当前功能分支备份；忽略目录中的第三方代码与差异文件保留本地，可按上述固定提交重新取得。外部学习材料见[作者介绍](https://robot-learning-collective.github.io/winning-behavior-1k-challenge.html)与[技术报告](https://arxiv.org/html/2512.06951v2)，复现实现以固定源码为准。
