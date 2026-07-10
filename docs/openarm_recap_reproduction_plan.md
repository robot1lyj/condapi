# OpenArm KAI0 / Evo-RL / 组合复现计划

最后更新：2026-07-10

状态：**待用户确认；未启动新的 HQ Stage 全量评分、Site 标注服务或策略训练。**

本文档是 OpenArm 后续 KAI0、Evo-RL 和二者组合实验的唯一当前计划。它区分论文事实、官方代码事实和 OpenArm 适配，不能把工程建议写成论文结论。

## 1. 事实来源规则

计划中的结论按以下标签管理：

- `[论文]`：来自 KAI0 或 π*0.6/RECAP 论文。
- `[官方代码]`：来自 OpenDriveLab/kai0 或 MINT-SJTU/Evo-RL 发布代码。
- `[OpenArm适配]`：机器人维度、单位、相机键、数据路径等不得不做的适配。
- `[待确认实验]`：论文没有规定、由本项目提出的实验变量；用户确认前不得启动。

参考：

- KAI0 论文：`arXiv:2602.09021`
- KAI0 官方代码：`https://github.com/OpenDriveLab/kai0`
- π*0.6 / RECAP 论文：`arXiv:2511.14759`
- Evo-RL 官方代码：`https://github.com/MINT-SJTU/Evo-RL`

## 2. 三条并行路线

```text
路线 K：KAI0 复现
Stage Advantage + 小规模 TDA + 二值 AWBC

路线 E：Evo-RL 复现
HIL episode -> value -> n-step advantage -> ACP -> policy

路线 H：组合实验
KAI0 离线优势策略作为 Evo-RL 的初始化，再跑同一套 HIL/ACP
```

三条路线必须有独立数据集名、配置名和 checkpoint，不允许用同一个结果同时宣称三条路线成功。

## 3. OpenArm 硬合同

```text
task: Fold the T-shirt properly
state/action: 16D
layout: [右臂7关节, 右夹爪, 左臂7关节, 左夹爪]
arm joints: degrees
gripper: HQ motor degrees, 0=open, -66=closed
robot/ROS boundary: radians + normalized gripper only at client/runtime edge
```

- `[OpenArm适配]` OpenArm 只走 `LeRobotOpenArmDataConfig`、`OpenArmInputs`、`OpenArmOutputs`。
- `[OpenArm适配]` 现场/HIL 清洗只走 `scripts/convert_openarm_hq_dataset.py`。
- `[OpenArm适配]` 旧单位 `openarm_site_align_v1` 禁止训练；只用 `openarm_site_align_v1_deg`。
- `[官方代码]` HIL hold 等待帧不是人工动作；clean 导出必须丢弃。

## 4. Stage v1 与 KAI0 官方一致性审计

### 4.1 已对齐

| 项目 | KAI0 论文/官方代码 | OpenArm 当前实现 | 结论 |
|---|---|---|---|
| Task A 阶段 | flattening、folding 两阶段 | `flattening`、`folding` | 对齐 |
| 人工标注 | episode 起止 + 一个子任务分界 | 数据集已精修起止，只点 `flatten_done` | 等价适配 |
| 进度构造 | 每阶段线性插值，整体 0→1 | stage 0 为 0→0.5，stage 1 为 0.5→1 | 对齐 |
| 训练样本 | 同 episode 任意随机双帧 | 同 episode 随机双帧 | 对齐 |
| 监督目标 | 两帧 `stage_progress_gt` 之差 | `current - history` | 对齐 |
| 模型 | π0.5 backbone + 三层 MLP value head + tanh | 相同 | 对齐 |
| 损失 | action=0，value=1，MSE | 相同 | 对齐 |
| 输入 | 三路当前图 + 三路历史图 + state + prompt | OpenArm 三路相机键映射后相同 | 对齐 |

已有 200 条 HQ 标注审计结果：

- 200/200 都是两阶段 `flattening -> folding`。
- parquet 中 `stage_id` 只在 `flatten_done` 后从 0 切换到 1。
- 旧 sidecar 中个别 `fold_start` 与 `flatten_done` 有间隔，但 Stage parquet 的实际生成逻辑以 `flatten_done` 为准；后续 sidecar 统一只保留一个权威分界。
- Stage v1 `10000` 在 HQ val20 上已有 800 个随机帧对结果：MSE 0.0122、MAE 0.0875、方向准确率 91%、相关系数 0.951、R² 0.882。

### 4.2 尚未达到“完整官方复现”

- `[官方代码]` 官方 Stage config 是从 π0.5 checkpoint 初始化，示例训练到 100k；当前 Stage v1 只训练到 10k。
- `[OpenArm适配]` 当前 Stage v1 只用 HQ train180，尚未验证 Site 相机域。
- `[官方代码]` 正式 AWBC 是 `negative/positive` 二值；当前 OpenArm 旧构建脚本仍是三档。
- `[论文/官方代码差异]` 论文强调直接双帧 `relative_advantage`；官方发布离散脚本默认 `absolute_advantage`，但允许选择 `relative_advantage`。
- `[论文/官方代码差异]` 论文写显式 stage goal `g`；官方发布实现没有单独输入 `stage_id/g`，而是通过阶段进度监督和双帧视觉学习。OpenArm 当前实现与官方发布代码一致，不自行增加新 stage embedding。

因此 Stage v1 可以作为现有 HQ 自动评分器，但路线 K 的结果必须注明是 OpenArm 适配复现，不能宣称参数级完全复刻 KAI0。

## 5. 路线 K：KAI0 复现

### K0. 复现范围

第一版复现：

- Stage Advantage。
- 二值 AWBC。
- 小规模时间缩放和空间镜像 TDA。
- 相同数据上的普通 π0.5 BC 对照。

暂不混入：

- Evo-RL value/ACP。
- Model Arithmetic。
- 新的自定义 failure stage。

Model Arithmetic、DAgger 和 temporal smoothing 属于完整 KAI0 的其他模块，后续单列，不能把 K0 第一版称为“完整 χ0 复现”。

### K1. Site 阶段标注

- `[论文]` Task A 只有两个阶段：flattening、folding。
- `[OpenArm适配]` Site 151 条已经精修起止，每条只点一次 `flatten_done`。
- `[OpenArm适配]` 沿用现有 Site split：train `0:141`，val `141:151`；不再自行改成 131/20。
- 人工标注生成 `stage_progress_gt/stage_id`，用途是训练和验证 Stage scorer，不直接等同 positive/negative。

执行顺序：先用 Stage v1 在 Site val10 上评估；若跨域明显退化，再按官方 Stage 训练方式将 Site train141 加入 Stage 训练。论文没有给出 OpenArm 跨相机阈值，因此不写自定义 85% 等硬门槛，只完整报告同一组 MSE/MAE/方向准确率/相关系数/R²，并与 HQ val20 对照。

### K2. HQ / Site / TDA 原始优势

- HQ policy train 只用 `0:999`；`999:1199` 保持 policy holdout。
- 多 GPU 评分先只写 `relative_advantage/absolute_value/absolute_advantage`，不允许每个 shard 单独分桶。
- Site 使用通过 Site val10 审计的 Stage checkpoint 评分。
- TDA 不重新过 Stage 模型；按 `source_episode_index/source_frame_stride/mirror` 从原 episode 映射优势，保持官方 Train-Deploy Alignment 语义。

优势源处理：

- `[论文主分支]` 使用直接双帧 `relative_advantage`。
- `[官方代码核对]` 同时保留 `absolute_advantage` 统计，用于和官方发布脚本默认值核对，但第一版不据此再训练一套策略。
- 如果官方后续澄清论文实验实际使用 `absolute_advantage`，再修改主分支并留下决策记录。

### K3. 正式二值化

只允许：

```text
task_index=0 -> Fold the T-shirt properly\nAdvantage: negative
task_index=1 -> Fold the T-shirt properly\nAdvantage: positive
```

- `[论文]` 按 advantage 排序，最高 30% 为 positive，其余为 negative。
- `[论文]` Task A 是两个阶段。
- `[官方代码]` `stage_nums=2` 时，每个阶段分别计算 percentile。
- 全部 shard 合并后统一计算阈值；禁止三档和分片独立阈值。

官方 README 的 `--threshold 30`、脚本实现的 `100-threshold` 和帮助文字存在表述歧义。OpenArm 报告必须直接写最终 positive 实际比例，验收目标是论文定义的约 30%，不能只记录 CLI 参数。

### K4. 第一版 TDA 小预算

KAI0 官方 TDA 类型保持不变：

- `[官方代码]` time scaling：`extraction_factor=2`。
- `[官方代码]` time split 示例：30% episode 做抽帧、其余保持原始。
- `[官方代码]` space mirroring：视频水平翻转，同时左右臂 state/action 交换。

`[待确认实验]` 为满足“第一版加入 TDA、但不要加入过多”的要求：

- HQ 原始 999 条全部保留。
- 额外 TDA episode 总数上限暂定为原始 HQ 的 30%，约 300 条。
- 约 150 条 time scaling、150 条 mirroring；来源 episode 离散抽取。
- 不直接使用现有 2298 条 TDA 全量加入训练。
- Site 第一版不做 TDA，避免现场真实数据被增强样本淹没。

这 30% 总预算是 OpenArm 第一版实验预算，不是 KAI0 论文参数；用户确认后才执行。

### K5. AWBC 策略训练

- `[论文/官方代码]` 从原始 π0.5 base 开始全参数训练，不从 HQ99999 warm start。
- `[论文]` policy 训练表给出 80k steps、batch 128。
- `[官方代码]` 发布的 AWBC config 给出 100k steps、batch 256。
- 第一轮以论文 80k/batch128 为复现目标；如果硬件无法直接满足，只调整并行/梯度累积，不静默修改有效 global batch。
- 保存周期按官方 config 使用 5k/10k 级别 checkpoint，不预先指定哪个最好。

必须有普通 π0.5 BC 对照：相同 HQ + Site + 小预算 TDA、相同训练样本数，只去掉 Advantage prompt。该对照对应 KAI0 的 normal π0.5 baseline，不是额外自研路线。

## 6. 路线 E：Evo-RL 复现

数据仍由现有 Site HQ 5k / `4999` collector 持续采集：

```text
HIL raw HDF5/mp4
  -> openarm_hil_evo_v1 clean
  -> value train
  -> value infer
  -> n-step advantage / acp_indicator
  -> ACP policy train
```

固定对齐项：

- `[官方代码]` episode 必须有 `episode_success`。
- `[官方代码]` 保留真实 policy action、human action 和 `is_intervention`。
- `[官方代码]` `n_step=50`。
- `[官方代码]` `positive_ratio=0.3`。
- `[官方代码]` policy 训练 `indicator_dropout_prob=0.3`。
- `[官方代码/RECAP]` human correction 强制 positive；hold 不作为 human correction。

当前实现状态：

- HIL raw -> clean 转换已实现。
- JAX `ACPPromptTransform` 已实现。
- 当前 OpenArm ACP config 的 dropout 仍是 0.0，正式路线 E 前必须改成 0.3。
- OpenArm value-train/value-infer 固定入口仍未完成，不能绕过 value 模型直接把 intervention 当全部标签。

`[OpenArm适配/实验控制]` 路线 E 的策略初始化固定使用当前 Site HQ 5k collector 对应 checkpoint；这不是 Evo-RL 论文指定的 OpenArm 模型，而是为了只测 Evo-RL 带来的增量。

## 7. 路线 H：KAI0 + Evo-RL 组合

路线 H 只改变 Evo-RL 的策略初始化：

```text
路线 K 胜出的 KAI0 Stage-AWBC checkpoint
  -> 使用与路线 E 完全相同的 HIL 数据
  -> 使用同一个 value checkpoint
  -> 使用同样的 n_step / positive_ratio / dropout / steps
  -> Hybrid policy
```

- 不把 Stage label 和 Evo `acp_indicator` 合成第三种标签。
- 不重新解释 HIL success/failure。
- 不改变路线 E 的数据和超参数。
- 只有初始化 checkpoint 不同，才能回答“KAI0 离线底座是否帮助 Evo-RL”。

## 8. 数据集和模型命名

| 路线 | 数据集/模型建议名 | 含义 |
|---|---|---|
| K | `openarm_kai0_stage_scores_hq_v1` | HQ 原始 Stage 输出，未分桶 |
| K | `openarm_kai0_awbc_hq_site_tda_v1` | 二值 KAI0 训练集 |
| K | `pi05_openarm_kai0_awbc_v1` | π0.5 base -> KAI0 AWBC |
| K control | `pi05_openarm_kai0_bc_control_v1` | 同数据普通 BC |
| E | `openarm_hil_evo_v1` | Evo-RL clean + value/ACP 字段 |
| E | `pi05_openarm_evo_acp_v1` | Site HQ 5k -> Evo-RL |
| H | `pi05_openarm_kai0_evo_hybrid_v1` | KAI0 checkpoint -> 同一 Evo-RL 数据 |

## 9. 执行顺序和 GPU

计划确认后：

1. 修正旧三档脚本为“分片只评分 + 合并后官方二值化”，增加论文/官方参数报告。
2. HQ `0:999` 按空闲 GPU 分片评分；现有 HIL 现场采集不停止。
3. 同时启动 Site 151 条单边界标注服务。
4. Site val10 审计 Stage v1，必要时按官方方式训练 Site-aware Stage checkpoint。
5. 构建小预算 TDA、映射优势、统一二值化并完成数据审计。
6. 路线 K 跑 KAI0 AWBC 与普通 BC 对照。
7. HIL 数据达到可用规模后，路线 E 跑 Evo-RL。
8. 路线 K 和 E 都通过各自 smoke 后，路线 H 只替换初始化做组合实验。

六张 GPU 的用途是并行评分和独立路线实验，不在没有多机等价性验证时声称单个训练已经使用官方 8-GPU 配置。

## 10. 禁止项

- 不把三档 `bad/neutral/positive` 当 KAI0 正式复现。
- 不把当前 Stage 10k 宣称为训练参数级完整复现。
- 不隐藏论文与官方代码在 `relative/absolute advantage`、80k/100k、batch128/256 上的差异。
- 不把路线 K、E、H 合成一个无法归因的训练。
- 不把 TDA 2298 条全量直接灌入第一版。
- 不用自定义阈值替代论文/官方没有给出的指标。
- 不停止正在进行的 HIL 采集。

## 11. 关键入口

```text
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
scripts/evaluate_stage_advantage.py
scripts/openarm_stage_advantage_awbc.py
scripts/openarm_tda_awbc_from_source.py
scripts/augment_openarm_hq_tda.py
scripts/merge_openarm_lerobot_v21.py
scripts/convert_openarm_hq_dataset.py
src/openpi/training/advantage_dataset.py
src/openpi/models_pytorch/pi0_pytorch.py
src/openpi/training/config.py
src/openpi/transforms.py
```

当前旧 `pi05_openarms_dual_awbc_v1` 和三档 AWBC 输出只作历史追溯；用户确认本计划后再修改代码和启动远端任务。
