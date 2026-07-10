# OpenArm RECAP 复现当前计划

最后更新：2026-07-10

状态：**待用户确认，尚未启动 HQ Stage 全量评分和新策略训练。**

本文档是 OpenArm 后续 Stage-AWBC、HIL 和 RECAP/Evo-RL 复现的唯一当前计划。旧训练流水和事故细节查 `docs/CHANGELOG.md`、git log 或远端日志。

## 1. 当前主线

两条任务并行，不互相等待：

```text
Track A：KAI0 离线优势底座
HQ 自动 Stage 评分 + Site 人工阶段标注/校准
  -> 二值 Advantage 数据集
  -> π0.5 base 全参数训练
  -> OpenArm Stage-AWBC 候选

Track B：Evo-RL / RECAP 现实闭环
现有 Site HQ 5k collector 持续采 HIL
  -> success/failure/intervention 数据
  -> value / advantage / ACP
  -> RECAP 第 1 轮策略
```

Track A 先利用空闲 GPU 做离线计算；Track B 继续现场采集，不因 Track A 暂停。

## 2. 硬合同

```text
task: Fold the T-shirt properly
state/action: 16D
layout: [右臂7关节, 右夹爪, 左臂7关节, 左夹爪]
arm joints: degrees
gripper: HQ motor degrees, 0=open, -66=closed
robot/ROS boundary: radians + normalized gripper only at client/runtime edge
```

- OpenArm 只走 `LeRobotOpenArmDataConfig`、`OpenArmInputs`、`OpenArmOutputs`。
- 现场/HIL 清洗只走 `scripts/convert_openarm_hq_dataset.py`。
- 旧单位数据集 `openarm_site_align_v1` 禁止训练；只用 `openarm_site_align_v1_deg`。
- HIL clean 必须丢弃 `session_state=intervention_hold` / `selected_source=hold`。

## 3. 数据和现有模型

| 资产 | 路径 / 名称 | 用途 |
|---|---|---|
| HQ | `/share/home/linyongjia/datasets/high_quality_folding` | Stage 自动评分、旧域动作先验 |
| HQ train | episode `0:999` | 新 Stage-AWBC 训练 |
| HQ holdout | episode `999:1199` | 不进入新策略训练 |
| Site deg | `/share/home/linyongjia/datasets/openarm_site_align_v1_deg` | 现场阶段标注、现场动作 |
| HIL clean | `openarm_hil_evo_v1` | Evo-RL value/ACP，等持续采集 |
| Stage v1 | `ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/.../10000` | HQ 自动评分起点 |
| HQ baseline | `pi05_openarms_dual_hq/openarms_hq_bs32/99999` | 历史真机基线，不删除 |
| HIL collector | `pi05_openarms_dual_site_align_v1_probe/.../4999` | 继续现场 HIL 采集 |

## 4. Track A：KAI0 Stage-AWBC

### 4.1 Site 人工标注的含义

Site 共 151 条，每条只人工点击一次：

```text
flatten_done = 展开结束 / 折叠开始
```

程序自动补齐：

```text
episode_start -> flatten_done -> episode_end
stage 0 = flatten / 展开
stage 1 = fold / 折叠
stage_progress_gt: 0 -> 1
stage_id: 0 / 1
```

人工阶段边界是 Stage 模型的监督和校准信号，**不是直接把每帧标成 positive/negative**。直接按时间差生成好坏会把动作速度误当成动作质量，禁止这样做。

Site 标注完成后离散抽 20 条作为 Stage 验证，其余 131 条用于现场 Stage 校准。验证集索引写入 sidecar，不能按连续尾部切分。

### 4.2 HQ 和 Site 评分

HQ：

1. Stage v1 `10000` 对 HQ train `0:999` 做全量推理。
2. 保存原始 `relative_advantage`、`absolute_value`、`absolute_advantage`，此时不分桶。
3. 评分任务按 episode 分片，多 GPU 只写各自 shard，最后统一合并。

Site：

1. 完成 151 条单边界人工标注。
2. 从 Stage v1 出发，用 Site train 131 做短校准，在离散 val20 上验收。
3. Site Stage 验收通过后，给 Site train 131 生成原始优势分。

HQ 和 Site 可以使用各自校准后的 Stage scorer，但必须分别按“域 × 阶段”计算阈值，避免两个相机域的数值尺度互相污染。

### 4.3 正式标签只有两类

正式策略训练只允许：

```text
task_index=0 -> Fold the T-shirt properly\nAdvantage: negative
task_index=1 -> Fold the T-shirt properly\nAdvantage: positive
```

规则：

- 使用 Stage 模型直接输出的 `relative_advantage` 作为主评分。
- 分别在 `HQ/Site × flatten/fold` 四个组内排序。
- 每组约最高 30% 标为 positive，其余标为 negative。
- 不使用旧的 `bad/neutral/positive` 三档方案。
- 不以 `absolute_value[t+50] - absolute_value[t]` 取代直接相对优势作为主标签。

### 4.4 合并数据集

目标数据集：

```text
/share/home/linyongjia/datasets/openarm_stage_awbc_hq_site_v1
```

组成：

- HQ train `0:999`：1 倍。
- Site train 131：按实际 frame 数重复，使 Site 占最终训练帧的约 35%～40%。
- v1 不加入 TDA，避免同时引入 Stage、现场域和增强三个变量。
- HQ/Site 各自完成二值化后再合并；合并后必须保持 source dataset / source episode 映射。

合并验收：

- 16D、degree、HQ gripper 合同不变。
- 三路视频、parquet、episode/frame index 对齐。
- 四个“域 × 阶段”组的 positive 比例约为 30%，不得塌缩。
- 抽查至少 20 条 HQ 和 20 条 Site 的优势曲线与视频。
- 重新生成该合并数据集自己的 `norm_stats.json`。

### 4.5 从 π0.5 base 训练

主模型必须从原始 π0.5 base 开始，不从 HQ `99999` warm start：

```text
pi05_base
  -> openarm_stage_awbc_hq_site_v1
  -> pi05_openarms_dual_stage_awbc_hq_site_v1
```

同数据并行跑三个实验，避免把“加入 Site”误判成“Stage 有效”：

| 节点 | 实验 | 作用 |
|---|---|---|
| gpu12 | Stage-AWBC，seed 1 | 主候选 |
| gpu14 | 普通 BC，相同 HQ/Site 组成 | Stage 因果对照 |
| gpu28 | Stage-AWBC，seed 2 | 稳定性复验 |

训练约束：

- 先做 100～1000 step smoke，再正式长训。
- 目标 global batch 64；先测显存，OOM 时回退 32，并按总训练样本数调整步数。
- 第一轮保存 `10k/30k/60k`，根据离线指标和真机 A/B 决定是否继续，不预设 `88k/100k` 最优。
- HQ `99999`、Site HQ 5k、Site base 10k 全部保留为对照，不提前宣布新模型替代旧模型。

## 5. Track B：HIL / Evo-RL / RECAP

该路线保持原计划并继续采集：

```text
Site HQ 5k / 4999 collector
  -> HIL raw HDF5 + mp4
  -> openarm_hil_evo_v1 clean
  -> value train / value infer
  -> complementary_info.value / advantage / acp_indicator
  -> JAX ACP policy train
```

- 当前 HIL collector 不等待 Stage-AWBC 新模型。
- HIL 记录完整自主成功、失败、人工接管和接管后结果。
- human VR 动作在 RECAP 中强制 positive；policy 动作由 value advantage 决定。
- KAI0 Stage 标签和 Evo-RL `acp_indicator` 是两套不同语义字段，不能相互覆盖。
- HIL 数据不加入本轮 Stage-AWBC v1；等 value/ACP 链路完成后再进入 RECAP 第 1 轮。
- RECAP 每轮使用累计数据，并从选定的固定预训练锚点重新微调，避免连续续训漂移。

## 6. 执行顺序

计划确认后按以下顺序执行：

| 顺序 | 任务 | GPU/人员 | 完成条件 |
|---|---|---|---|
| 1 | 修正 Stage 脚本为“只评分 + 统一二值分桶” | 本地 | 单测通过，不再出现三档正式标签 |
| 2 | HQ `0:999` 多卡分片评分 | gpu12/gpu14/gpu28 | 999 条都有原始优势列，分片可合并 |
| 3 | 启动 Site 标注服务 | 人工 + CPU | 151 条均有一个有效 `flatten_done` |
| 4 | Site Stage 校准与 val20 验证 | 空闲 2 GPU | 方向准确率 >=85%，相关系数 >=0.8 |
| 5 | Site 评分、二值化、HQ/Site 合并 | GPU + CPU | 数据合同和标签比例验收通过 |
| 6 | 三个策略实验并行训练 | 三节点各 2 GPU | smoke 正常并保存 10k/30k/60k |
| 7 | 离线检查 + 真机 A/B | gpu25 + 现场 | Stage-AWBC 胜过普通 BC 和历史基线 |

gpu12 若被他人占用，不抢占；先用 gpu14/gpu28，释放后再补分片。

## 7. Gate

| Gate | Go | No-Go |
|---|---|---|
| Site 标注 | 151/151 边界有效 | 修标注，不训练 Site Stage |
| Site Stage | val20 方向准确率 >=85%、相关系数 >=0.8 | 增加校准数据或停止 Site 自动评分 |
| AWBC 标签 | 每域每阶段约 30% positive，视频抽查合理 | 修 scorer/阈值，不训练 policy |
| 训练 smoke | loss 有限、显存稳定、checkpoint 可加载 | 修数据/config，不开长训 |
| 模型晋升 | 真机抓取、抬起、展开、成功率和重试综合胜出 | 保留为实验，不替代 HQ99999 |

## 8. 当前禁止项

- 不使用三档 `bad/neutral/positive` 作为正式 KAI0/RECAP 标签。
- 不在 HQ 评分完成前启动新策略训练。
- 不把 Site 人工阶段进度直接当动作好坏。
- 不让 TDA 进入第一版 HQ+Site Stage-AWBC 对照。
- 不把 train loss 当作最终模型选择标准。
- 不停止正在进行的 HIL 采集。

## 9. 关键入口

```text
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
scripts/openarm_stage_advantage_awbc.py
scripts/merge_openarm_lerobot_v21.py
scripts/compute_openarm_parquet_norm_stats.py
scripts/convert_openarm_hq_dataset.py
src/openpi/training/config.py
src/openpi/transforms.py
scripts/train.py
scripts/serve_policy.py
```

当前旧 `pi05_openarms_dual_awbc_v1` 和三档 AWBC 构建逻辑只视为历史实现；计划确认后再修改或新增正式 v1 配置，不直接复用旧输出。
