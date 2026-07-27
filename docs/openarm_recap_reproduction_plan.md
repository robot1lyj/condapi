# OpenArm KAI0 / Evo-RL / 组合复现计划

最后更新：2026-07-27

状态：**路线 K 的 80k、16-checkpoint 双域 sweep 和正式报告已完成；离线规则选中20k，但真机 A/B 未见改善，
当前 gpu25 人工恢复为79999。下一步使用79999采集30条定向 HIL，重点解决错误对角线、重复甩平和展开到折叠的
阶段切换；不得把继续甩平简单归因于 Stage 数据数量不足。**

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
HQ-Stage/HQ-Score + Site-Stage/Site-Score + TDA-S + 二值 K-Data

路线 E：Evo-RL 复现
HIL-Raw -> E-Value -> n-step advantage -> E-Data -> E-Policy

路线 H：组合实验
K-Policy 作为 Evo-RL 的初始化，再跑与路线 E 相同的 E-Data/ACP
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

## 4. HQ-Stage 与 KAI0 官方一致性审计

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
- `[OpenArm适配]` 当前 Stage v1 只用 HQ train180，因此领域定义固定为 HQ；按当前数据边界不用于 Site。
- `[官方代码]` 正式 AWBC 是 `negative/positive` 二值；当前 OpenArm 旧构建脚本仍是三档。
- `[论文/官方代码差异]` 论文强调直接双帧 `relative_advantage`；官方 README、AWBC README、脚本默认值和
  Task A 二阶段示例均用 `absolute_advantage` 离散，脚本仍允许选择 `relative_advantage`。正式路线 K
  跟随发布代码使用 absolute，relative 只保留为 scorer 诊断。
- `[论文/官方代码差异]` 论文写显式 stage goal `g`；官方发布实现没有单独输入 `stage_id/g`，而是通过阶段进度监督和双帧视觉学习。OpenArm 当前实现与官方发布代码一致，不自行增加新 stage embedding。

因此 Stage v1 的固定讨论名为 **HQ-Stage**：它负责 HQ 自动评分，也先作为 Site 的零样本迁移模型；若 Site
迁移不合格，它再作为 Site-Stage 的初始化。路线 K 的结果仍必须注明是 OpenArm 适配复现，不能宣称参数级完全复刻 KAI0。

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

### K1. 两个领域的 Stage 评分

HQ 与 Site 的来源必须始终分开说明：

```text
HQ -> HQ-Stage 模型推理 -> HQ-Score
Site-A150 -> HQ-Stage 直接推理 -> Site-DirectScore -> 迁移审计
                                              ├─ 合格 -> Site-Score
                                              └─ 不合格 -> Site-StageData -> Site-Stage -> Site-Score
```

**HQ-Score：**

- HQ policy train 只用 `0:999`；`999:1199` 保持 policy holdout。
- HQ-Stage 给每帧写入 `relative_advantage/absolute_value/absolute_advantage`。
- 多 GPU 分片只写原始分数，不允许各 shard 独立分桶。
- HQ 动态报告上图是从第 0 帧锚定到当前帧的直接预测
  `absolute_value[t] = Stage(frame_0, frame_t)`；下图不是上图差分，而是模型直接比较未来 50 帧得到的
  `relative_advantage[t] = Stage(frame_t, frame_{t+50})`。正值表示未来更接近完成，负值表示退步，接近 0
  表示停滞；尾部不足 50 帧的值会按实际间隔缩放，必须单独审计。
- 对 folding-only `536:999`，报告显示值会在 raw `absolute_value` 上加阶段起点0.5并裁剪到 `[0.5,1]`；
  raw 值仍保留为 `episode_relative_progress`，不会篡改 scorer 输出。
- HQ999 完成后必须通过 `audit_openarm_hq_stage_scores.py`：ID/有限值/范围完整，完整任务峰值 crossing
  比例至少95%、峰值P10至少0.60，folding-only episode-relative 峰值P10至少0.25，且 relative
  advantage 的P90/P10与近零比例不能塌缩。失败则停止 K-Data。

**Site-DirectScore / Site-StageData / Site-Stage / Site-Score：**

- `[论文]` Task A 只有 flattening、folding 两个阶段；人工边界用于构造 Stage 模型监督，不直接作为 AWBC advantage。
- Site-A151 中有 150 条 `success` 和 episode 95 一条 `failure`。成功子集固定简称 **Site-A150**；
  episode 95 固定简称 **Site-F1**，不进入当前 Stage 训练，留给后续恢复/HIL 路线。
- 第一步不训练新模型：HQ-Stage 直接推理 Site-A150，产物简称 **Site-DirectScore**。人工边界展开出的
  `stage_progress_gt/stage_id` 只用于核对预测，不写入最终 advantage。
- 迁移审计同时看：`absolute_value` 对人工进度的 MSE/MAE/相关系数/R²、随机帧对的方向准确率、预测
  0.5 crossing 与人工 `flatten_done` 的帧差，以及视频同步动态曲线。指标与 HQ20 已有基准并排报告，
  不能只凭一两集观感决定。
- 若 Site-DirectScore 的数值和动态曲线均保持 HQ-Stage 质量，直接提升为 **Site-Score**，150 条人工标注
  不再参与训练；这是改动最少的首选路径。
- 只有直接迁移明显退化时，才把 Site-A150 按原 split 分成 140 train + 10 val，构建
  **Site-StageData**。它只写 Stage 监督，不伪造最终 advantage。
- 备选模型简称 **Site-Stage**：从 HQ-Stage `10000` 初始化，在 Site140 上短程适配，并用 Site10 做主
  验证、HQ20 做遗忘保护；是否加入 HQ-A180 replay 及采样比例必须依据 Site-DirectScore 审计后单独立项，
  不预先写成 KAI0 论文结论。
- 通过验证的 Site-Stage 再对 Site-A150 全量推理生成 **Site-Score**。最终 Site-Score 必须保留模型预测的
  `relative_advantage/absolute_value/absolute_advantage`，而不是人工分界产生的分段直线。

**已否决实验：** 单边界线性插值得到的 Site-GT 累计进度是两段直线，50 帧增量近似两段常数，主要反映
阶段时长而不是视觉中的停滞、退步和恢复。远端派生数据 `openarm_site_gt_v1` 及 8766 报告服务已删除；
原始 Site、Site-A151 和 HQ-Stage 报告均保留。

**TDA-S：**

- TDA 不重新过 Stage 模型；只按 `source_episode_index/source_frame_stride/mirror` 从 HQ-Score 映射优势。
- 第一版只使用小预算 TDA，简称 **TDA-S**。

### K2. 合并前的尺度和来源审计

- HQ-Score 与 Site-Score 使用同一个进度范围 `[0,1]`、同一个 50 帧间隔和同一个裁剪规则。
- 合并前分别报告 HQ/Site 的 advantage 最小值、均值、最大值、分位数以及两个阶段的帧数。
- advantage 数值始终来自 Stage 模型；但 `[官方代码]` 两阶段分桶本来就读取 `stage_progress_gt`。
  因此 Site 使用已人工标注的 `flatten_done` 生成 `stage_id_awbc`，只决定“在哪个阶段计算30%阈值”，
  不把人工线性进度写成 advantage。
- HQ `meta/episodes.jsonl` 与首帧视觉审计确认 `0:360` 为完整任务、`360:536` 为明确
  layout+fold 完整任务、`536:999` 从首帧已完全平铺而只录 folding。HQ `0:536` 暂用 HQ-Stage
  `absolute_value>=0.5` 判阶段，`536:999` 固定为 folding；TDA-S 继承源 HQ 的阶段。
- 构建器会强制核对 `360:536` metadata 中连续的 layout prompt；范围与 metadata 不一致时停止，避免
  更换数据版本后静默沿用536边界。
- 若某一来源或阶段在二值化后 positive 比例异常，停止构建，不用静默重采样掩盖问题。

优势源处理：

- `[正式复现]` 使用官方发布流程默认的 `absolute_advantage = absolute_value[t+50]-absolute_value[t]`。
- `[诊断保留]` 直接双帧 `relative_advantage` 继续落盘并通过非塌缩闸门，用于解释论文方法和检查 scorer，
  但不决定第一版 K-Data 的 positive/negative。

### K3. 正式二值化

只允许：

```text
task_index=0 -> Fold the T-shirt properly, Advantage: negative
task_index=1 -> Fold the T-shirt properly, Advantage: positive
```

- `[论文]` 按 advantage 排序，最高 30% 为 positive，其余为 negative。
- `[论文]` Task A 是两个阶段。
- `[官方代码]` `stage_nums=2` 时，每个阶段分别计算 percentile。
- HQ-Score、Site-Score 和 TDA-S 合并后统一计算阈值；禁止三档和分片独立阈值。
- 报告必须按 `HQ/Site/TDA × flattening/folding` 分别列出 positive/negative 数量，确保 Site 没被 HQ 淹没。
- 正式构建器为 `scripts/build_openarm_kai0_awbc_dataset.py`：先审计完整的 HQ999、Site140 和 TDA-S，
  再按 HQ 任务范围/预测 crossing、Site 人工边界和 TDA 源映射得到 `stage_id_awbc`，对官方默认的
  `absolute_advantage` 分别取最高 30%；
  任一来源的正样本比例超出 `15%-45%` 即停止，不生成正式目录。

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
- 第一轮以论文 80k/batch128 为复现目标；当前正式任务使用 gpu12+gpu28 四卡，global batch128，可整除且与论文 batch 对齐。
- 保存周期按官方 config 使用 5k/10k 级别 checkpoint，不预先指定哪个最好。
- 正式配置固定为 `pi05_openarm_kai0_awbc_v1`，从原始 P05 初始化；本轮 gpu12+gpu28 组成 4 卡 JAX
  多节点作业，全局 batch128（每节点64、每卡32），workers8，80k steps，5k 保存一次。
- 正式训练前必须先以同一配置、同一全局 batch 和同一多节点启动器跑 20-step smoke；smoke 使用 workers0
  排除 DataLoader 子进程干扰，正式训练恢复基准选出的 workers8。

必须有普通 π0.5 BC 对照：相同 HQ + Site + 小预算 TDA、相同训练样本数，只去掉 Advantage prompt。该对照对应 KAI0 的 normal π0.5 baseline，不是额外自研路线。

### K6. 2026-07-27 真机与阶段切换结论

- `[运行事实]` 20k 与79999真机对比中，20k没有表现出更好的抓取、退出甩平循环或阶段切换能力；gpu25
  当前运行79999，正式离线20k选择记录继续保留，不改写历史结果。
- `[审计事实]` K-Data 并未在数量上过分强调展开阶段：materialized 展开阶段为1,687,264帧、
  538,760个positive；折叠阶段为2,614,138帧、804,858个positive。
- `[审计事实]` Site人工边界前最后300帧positive比例42.7%，边界后最初300帧为38.7%；Stage预测在干净
  示范边界前后的平均 `absolute_value` 从0.398升到0.542，因此不能仅凭真机重复甩平就断言 Stage
  模型没有识别示范中的阶段边界。
- `[实现事实]` `stage_id_awbc` 只用于离线分阶段计算 advantage 百分位，不进入策略提示词；展开与折叠
  的高优势样本都使用同一个 `Advantage: positive`。真机也不在线运行 Stage 模型，策略必须只凭当前
  图像和机器人 state 判断阶段。人工展开后的画面/手臂姿态若偏离示范分布，策略仍可能回到甩平模式。
- `[风险]` 当前 Stage/AWBC 数据主要来自成功示范，没有覆盖“错误对角线、无进展重复甩平、人工展开后仍
  重启甩平”等状态；局部50帧 advantage 可能把短暂抬升或画面变平误判为positive。
- `[下一批数据]` 首批30条必须是完整成功 HIL episode，而不是30条新的普通干净示范：10条错误对角线后
  松手重抓并完成；10条重复甩平后中止、恢复并完成；10条已经展开却不进入折叠时接管并完整示范折叠。
  全部保留策略失败前缀、真实 human intervention、`episode_success=true` 和 `recovery_success=true`。
- `[下一闸门]` 先用这30条失败/恢复轨迹审计 Stage/Value 对错误状态的判断，再进入 Evo-RL ACP；若完成
  HIL 后仍无法切换阶段，才立项 `stage × advantage` 四提示词和带迟滞的在线阶段条件实验。该实验是
  OpenArm 扩展，不得写成 KAI0 官方复现。

## 6. 路线 E：Evo-RL 复现

当前现实采集统一使用 gpu25 上的 K-Policy `79999`。它是真机 A/B 后的运行选择，不改写离线 sweep
选中 `20k` 的历史记录。采集器和路线 E 的训练初始化是两个概念：同一批 HIL 可以同时用于路线 E 的
受控对照和路线 H 的主力组合实验。

```text
K-Policy 79999 现实失败/接管/恢复
  -> HIL raw HDF5/mp4
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

`[OpenArm适配/实验控制]` 路线 E 的策略初始化固定使用 Site-5K；这不是当前采集器，也不是 Evo-RL
论文指定的 OpenArm checkpoint，而是为了只测 Evo-RL 相对历史 Site 微调模型的增量。路线 E 与路线 H
必须使用同一份 clean HIL、同一个 E-Value、同一组 ACP 标签和训练超参数。

## 7. 路线 H：KAI0 + Evo-RL 组合

路线 H 是当前主力下一步，只改变 Evo-RL 的策略初始化：

```text
当前真机运行的 K-Policy 79999
  -> 使用与路线 E 完全相同的 HIL 数据
  -> 使用同一个 value checkpoint
  -> 使用同样的 n_step / positive_ratio / dropout / steps
  -> Hybrid policy
```

- 不把 Stage label 和 Evo `acp_indicator` 合成第三种标签。
- 不重新解释 HIL success/failure。
- 不改变路线 E 的数据和超参数。
- 只有初始化 checkpoint 不同，才能回答“KAI0 离线底座是否帮助 Evo-RL”。
- 离线规则选出的 `20k` 继续作为历史 checkpoint；真机已经确认它没有优于 `79999`，因此本轮 H-Policy
  初始化使用 `79999`，不再重复以20k采集或训练。
- HQ-Stage 只用于失败前缀诊断，不替代 Evo 的 E-Value，也不直接生成 `acp_indicator`。

## 8. 统一命名

日常讨论一律优先使用下面的短名。短名是语义稳定的别名，不重命名已经存在或正在写入的磁盘目录。

### 8.1 数据集

| 短名 | 磁盘实体名/位置 | 内容 | 如何得到 | 是否经过 Stage 模型 |
|---|---|---|---|---|
| **HQ** | `high_quality_folding` | 原始 HQ 1199 集 | 已有数据 | 否 |
| **HQ-A200** | `high_quality_folding_v2p1_stage_train180` + `stage_val20` | HQ 人工 Stage 标注 180/20 | 历史人工标注 | 仅用于训练/验证 HQ-Stage |
| **HQ-Score** | `openarm_kai0_stage_scores_hq_v1_s*` | HQ `0:999` 原始 advantage 分片 | HQ-Stage 六卡推理 | **是，只用 HQ-Stage** |
| **Site** | `openarm_site_align_v1_deg` | 现场 151 集 HQ 单位合同数据 | HDF5 清洗转换 | 否 |
| **Site-A151** | `openarm_site_align_v1_deg/annotations/openarm_stage_v1.jsonl` | 151/151 单边界人工标注 | 人工点击 `flatten_done` | **否** |
| **Site-A150** | Site-A151 中 `quality=success` | 当前可用的 150 条现场人工边界 | 排除 Site-F1 | 迁移审计；必要时作监督 |
| **Site-F1** | Site episode 95 | 人工标记的 1 条 failure | Site-A151 | 不进入当前 Stage 训练 |
| **Site-DirectScore** | `openarm_kai0_site_scores_direct_v1` | HQ-Stage 对现场的零样本预测 | HQ-Stage 直接推理 | 审计后决定是否提升 |
| **Site-StageData** | `openarm_site_stage_v1` | 可选的 140 train + 10 val Stage 监督 | Site-A150 边界展开 | 仅在直接迁移失败时构建 |
| **Site-Score** | `openarm_kai0_site_scores_v1` | 通过迁移审计的现场非线性预测 | HQ-Stage 或 Site-Stage | **是，唯一正式现场分数** |
| **TDA-S** | `openarm_kai0_tda_small_v1` | 第一版小预算 HQ TDA，约 300 集上限 | HQ 变换 + HQ-Score 映射 | 不重新推理 |
| **K-Data** | `openarm_kai0_awbc_v1` | HQ-Score + Site-Score + TDA-S 的二值 AWBC 数据 | 合并后统一二值化 | 混合来源有明确 metadata |
| **K-Control** | `openarm_kai0_bc_control_v1` | 与 K-Data 同样本、普通 task prompt | 去掉 advantage prompt | 否 |
| **HIL-Raw** | 客户端 `openarm_hil_dagger` | policy/human/hold 原始 HIL | 现实采集 | 否 |
| **HIL-T30** | HIL-Raw 中首批30条定向完整成功 episode | 错误对角线、重复甩平、已展开不折叠各10条 | K-Policy79999失败前缀 + 人工接管恢复 | 只作诊断，不用 HQ-Stage 造标签 |
| **E-Data** | `openarm_hil_evo_v1` | clean HIL + value/ACP 字段 | HIL-Raw 清洗和 Evo value | Evo value，不用 HQ-Stage |

历史/禁用数据名：

- `openarm_site_align_v1`：旧弧度/归一化夹爪合同，禁用。
- `openarm_site_gt_v1`：已删除的线性 Site-GT 诊断产物，禁止重建为正式分数。
- `openarm_awbc_v1*`：旧三档 AWBC 实验，只作历史追溯。
- `openarm_hq_tda_aug_v1`：2298 集全量 TDA，不直接作为第一版 TDA-S。

### 8.2 模型

| 短名 | 配置/检查点 | 用途 | 当前状态 |
|---|---|---|---|
| **P05** | 官方 `pi05_base` | 所有正式 KAI0 policy 的初始化 | 已有 |
| **HQ-Policy** | `pi05_openarms_dual_hq` / `99999` | 1200 HQ 训练出的历史策略 | 已有 |
| **HQ-Stage** | `ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD` / `10000` | HQ 正式评分 + Site 直接迁移基线 | HQ999 评分与审计已通过 |
| **Site-Stage** | `ADVANTAGE_TORCH_OPENARM_SITE_FOLD` / `4000` | HQ-Stage -> Site-StageData 领域适配 | 直接迁移失败后已适配并通过 Site150 闸门 |
| **Site-5K** | `pi05_openarms_dual_site_align_v1_probe` / `4999` | 历史 Site 微调模型；路线 E 的受控初始化 | 已有，不是当前 collector |
| **Site-10K** | `pi05_openarms_dual_site_align_v1_base_10k` / `9999` | P05 直接微调 Site 的对照 | 已有 |
| **K-Policy** | `pi05_openarm_kai0_awbc_v1` | P05 -> K-Data AWBC | 80k与16-checkpoint sweep已完成；离线选20k，gpu25当前运行79999 |
| **K-BC** | `pi05_openarm_kai0_bc_control_v1` | P05 -> K-Control 普通 BC | 待实现/训练 |
| **E-Value** | `openarm_evo_value_v1` | HIL success/intervention -> value/advantage | 待实现/训练 |
| **E-Policy** | `pi05_openarm_evo_acp_v1` | Site-5K -> Evo ACP 受控对照 | HIL-T30清洗及E-Value完成后训练 |
| **H-Policy** | `pi05_openarm_kai0_evo_hybrid_v1` | K-Policy79999 -> 同一 Evo ACP | 当前主力下一模型 |

旧配置 `pi05_openarms_dual_awbc_v1` 使用旧数据名和 HQ-Policy warm start，不代表正式 K-Policy，禁止混用。

## 9. 执行顺序和 GPU

### 9.1 已完成基线

1. HQ999 已由 HQ-Stage 完成评分和质量审计；HQ policy holdout `999:1199` 未进入 K-Data。
2. Site-DirectScore 未通过迁移闸门；Site-Stage `4000` 通过 Site150 与 HQ 遗忘保护后生成 Site-Score。
3. TDA-S、1719集 K-Data、全量 norm、真实 OpenPI loader smoke 和四卡80k均已完成。
4. `5000/10000/.../75000/79999` 共16个 checkpoint 的双域 sweep、综合报告、完成审计和 gpu25
   WebSocket 合同验证均已完成。
5. 离线规则选择20k；真机20k未优于79999，因此运行选择恢复79999。两者必须继续分别称为“离线选择”和
   “当前运行选择”。

详细故障、修复和训练历史由 `docs/CHANGELOG.md` 与远端
`output/openpi/logs/openarm_kai0_pipeline_v1/` 保存，本计划不再重复运行日志。

### 9.2 当前主线（2026-07-27）

1. **固定采集器**：gpu25 保持 K-Policy79999、强制 positive prompt 和现有16D/角度制/HQ夹爪合同；
   采集过程中不切换20k或其他 checkpoint。
2. **采集 HIL-T30**：错误对角线后松手重抓10条、重复甩平后中止恢复10条、已经展开却不折叠时完整接管
   10条。每条都保留策略失败前缀、hold、人类动作和最终完整成功结尾。
3. **清洗与硬审计**：Raw 保留全部帧；Evo clean 丢弃 `session_state=intervention_hold`，只保留真实
   policy 与有限16D human VR 动作。逐集检查时间戳单调、视频同步、intervention 起止、success/recovery
   元数据、单位和夹爪范围；任一合同失败不得进入 E-Value。
4. **先诊断再训练**：对三类失败前缀分别绘制 HQ/Site-Stage 进度曲线和 E-Value 曲线。Stage 结果只回答
   “成功示范评分器如何看这些错误状态”，不能代替 Evo value 标签，也不能阻塞路线 E/H。
5. **补齐 Evo JAX 链路**：实现并验证 OpenArm E-Value train/infer，固定 `n_step=50`、
   `positive_ratio=0.3`、`indicator_dropout_prob=0.3`，再物化 E-Data。禁止把
   `is_intervention=1` 直接当作全部 ACP 标签。
6. **并行训练两个可归因结果**：E-Policy 从 Site-5K 初始化，H-Policy 从 K-Policy79999 初始化；
   两者使用同一 E-Data、E-Value、step 数、batch、学习率和随机种子。H-Policy 是主力候选，E-Policy 是
   Evo-RL 增量对照。
7. **真机固定协议比较**：至少分别统计正确对角线选择率、无进展甩平循环率、人工展开后进入折叠率、
   完整折叠成功率、每集接管次数和恢复成功率；不能只用训练 loss 或一次演示决定胜负。

### 9.3 并行但隔离的 OpenArm 实验

- `[待确认实验]` 可以另做 Site-heavy BC/KAI0 消融，使用加权采样而不是复制数据：第一版总采样权重
  Site/HQ/TDA 为 `50/40/10`；若做阶段感知采样，展开阶段暂用 `55/35/10`，折叠阶段暂用
  `40/50/10`。这些比例来自当前真机问题，不是 KAI0 官方参数。
- Site-heavy 实验不得与 E/H 共用实验名或被写成 Evo-RL 效果；先用短程 probe 判断减少 HQ 是否改善
  现场阶段切换，再决定是否长训。
- 当前不再追加150条同类黑衣普通成功示范。数据预算优先给 HIL-T30 的错误状态和恢复动作；后续是否补
  普通 Site 数据，由固定协议下的错误类型覆盖率决定。
- 如果完成 HIL-T30、E-Value 和 H-Policy 后仍然频繁“已经展开却继续甩平”，才启动
  `stage × advantage` 四提示词及在线阶段迟滞实验；它是 OpenArm 扩展，不属于 KAI0 或 Evo-RL 官方复现。

## 10. 禁止项

- 不把三档 `bad/neutral/positive` 当 KAI0 正式复现。
- 不把当前 Stage 10k 宣称为训练参数级完整复现。
- 不把 Site 人工线性插值直接当最终 advantage；Site-Score 必须来自通过迁移审计的 HQ-Stage 或 Site-Stage。
- 不隐藏论文与官方代码在 `relative/absolute advantage`、80k/100k、batch128/256 上的差异。
- 不把路线 K、E、H 合成一个无法归因的训练。
- 不把 TDA 2298 条全量直接灌入第一版。
- 不用自定义阈值替代论文/官方没有给出的指标。
- 不停止正在进行的 HIL 采集。

## 11. 关键入口

```text
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
scripts/build_openarm_hq_score_report.py
scripts/openarm_advantage_report.py
scripts/serve_openarm_advantage_report.py
scripts/monitor_openarm_hq_stage_scores.sh
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

当前旧 `pi05_openarms_dual_awbc_v1` 和三档 AWBC 输出只作历史追溯；当前运行只生成原始 Stage 分数，正式二值 AWBC 由合并后的独立步骤生成。
