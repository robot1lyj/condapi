# OpenArm RECAP 复现当前计划

最后更新：2026-07-09

本文档是 OpenArm 后续训练、HIL、RECAP/Evo-RL 复现和 KAI0 辅助分支的唯一当前计划。它只保留现在要执行和不能忘的稳定事实；旧训练流水、事故细节和长解释查 `docs/CHANGELOG.md`、git log 或远端日志。

## 1. 当前判断

- HQ `99999` 没有在真机上稳定抓住衣服；当前第一故障点是抓取接触，其次才是完整折叠。
- 旧 `openarm_site_align_v1` 单位错误，禁止继续训练或真机结论；只保留追溯。
- 当前可用 collector policy 是 site_deg 候选，优先用于 HIL/rollout 采集和同场景 A/B。
- 主线并行推进两条复刻链：Evo-RL value/ACP 和 KAI0 Stage/AWBC；不排队，不引入自研主标签。
- OpenArm 与 Piper 数据链路已拆分；OpenArm 只走 `LeRobotOpenArmDataConfig`、`OpenArmInputs`、`OpenArmOutputs`。

## 2. 硬合同

OpenArm policy 数据合同：

```text
task: Fold the T-shirt properly
state/action shape: 16D
layout: [右臂7关节, 右夹爪, 左臂7关节, 左夹爪]
arm joints: degrees
gripper: HQ motor degrees, 0=open, -66=closed
robot/ROS boundary: radians + normalized gripper only at client/runtime edge
```

训练清洗：

- 现场/HIL 清洗只走 `scripts/convert_openarm_hq_dataset.py`。
- HIL raw 转 clean 只走 `scripts/convert_openarm_hq_dataset.py from-hil-hdf5`。
- HIL clean 导出必须丢弃 `session_state=intervention_hold` 或 `selected_source=hold` 的等待帧。
- `complementary_info.is_intervention=1` 只表示真实 human VR 动作已经控制机器人。
- KAI0 的 `stage_progress_gt/stage_id/task_index/tasks.jsonl` 全部后处理生成，不要求客户端实时写训练列。

## 3. 当前资产

| 资产 | 路径 / 名称 | 状态 | 用途 |
|---|---|---|---|
| HQ 原始 | `/share/home/linyongjia/datasets/high_quality_folding` | 已有 | 基础动作先验、HQ baseline、合并训练 |
| HQ split | train `0:999`, val `999:1199` | 已确认 | checkpoint sweep / holdout |
| Site old | `/share/home/linyongjia/datasets/openarm_site_align_v1` | 单位错误 | 只追溯，不训练 |
| Site deg | `/share/home/linyongjia/datasets/openarm_site_align_v1_deg` | 151 集，已转 HQ 合同 | 现场对齐、collector、短训 |
| HQ TDA | `/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1` | 2298 集，smoke 过 | 鲁棒性/合并候选 |
| HIL Evo | `openarm_hil_evo_v1` | 等真实 HIL 数据 | Evo-RL value/ACP |
| AWBC | `/share/home/linyongjia/datasets/openarm_awbc_v1` | 待复核 | KAI0 AWBC |
| Stage train/val | `high_quality_folding_v2p1_stage_train180/val20` | 已有 | Stage Advantage |

当前模型：

| 模型 | 路径 / 配置 | 状态 |
|---|---|---|
| HQ baseline | `pi05_openarms_dual_hq/openarms_hq_bs32/99999` | 可测，真机抓取失败 |
| Site HQ 5k | `pi05_openarms_dual_site_align_v1_probe/openarm_site_deg_151e_2gpu_5k_hq99999_20260707/4999` | 当前 collector 候选 |
| Site base 10k | `pi05_openarms_dual_site_align_v1_base_10k/openarm_site_deg_base_151e_2gpu_10k_pi05base_20260707/9999` | 第二 A/B 候选 |
| Stage v1 | `ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/.../10000` | 当前最优 Stage checkpoint |
| HIL ACP | `pi05_openarms_dual_evo_acp_hil_v1_probe` | 配置已落地，等 `acp_indicator` 数据 |

## 4. 双线方案

### Track A: Evo-RL / RECAP 风格

```text
HIL raw HDF5/mp4
  -> openarm_hil_evo_v1 LeRobot clean
  -> value train
  -> value infer
  -> complementary_info.value / advantage / acp_indicator
  -> pi05_openarms_dual_evo_acp_hil_v1_probe
  -> 真机 A/B
```

当前已落地：

- HIL raw -> LeRobot clean 转换入口。
- JAX `ACPPromptTransform`：按 `complementary_info.acp_indicator` 注入 `Advantage: positive/negative`。
- ACP 训练配置 `pi05_openarms_dual_evo_acp_hil_v1_probe`。

仍需落地：

- OpenArm value-train/value-infer 固定命令模板。
- HIL clean dataset smoke/report。
- 真机 A/B 汇总脚本。

### Track B: KAI0 Stage / AWBC

```text
同一批 LeRobot/HIL 数据
  -> stage boundary sidecar
  -> stage_progress_gt / stage_id
  -> Stage Advantage eval/train
  -> absolute/relative advantage
  -> task_index + meta/tasks.jsonl
  -> pi05_openarms_dual_awbc_v1
  -> 真机 A/B
```

当前阶段只对齐 KAI0 Task A 的两阶段：

```text
0 flatten / 展开拉平
1 fold / 折叠完成
```

不在 v1 引入复杂 `failure_stage`、三分类 prompt 或 Model Arithmetic。

## 5. 客户端 HIL 必录字段

默认目录：

```text
/tmp/openarm_hil/openarm_hil_dagger
episodes/episode_000000.hdf5
videos/observation.images.{base,left_wrist,right_wrist}/episode_000000.mp4
meta/info.json
meta/episodes.jsonl
```

逐帧必须有：

```text
timestamp / timestamp_ns
episode_index / frame_index
task / prompt
observation.state        # 16D HQ 合同
action / action.executed # 16D 最终实际执行动作
policy_action / policy_action_chunk
teleop_action / human_action
authority_source         # policy / human / scripted
selected_source          # policy / human / hold
session_state            # policy / human / intervention_hold
complementary_info.is_intervention
```

episode 级必须有：

```text
episode_success
episode_outcome          # success / failure / aborted
recovery_success
collector_policy_id
model_metadata
```

## 6. 当前下一步

| 优先级 | 任务 | 验收 |
|---|---|---|
| P0 | 真机 A/B：HQ `99999`、site HQ 5k、site base 10k | 同场景记录 grasp_contact、lift、flatten_entry、full_success |
| P0 | 采 HIL/rollout：自主失败、接管救回、接管后失败、自主成功 | raw schema 完整，hold 帧可过滤 |
| P1 | 转 `openarm_hil_evo_v1` 并生成 report/norm stats | 16D、单位、视频帧对齐、NaN、episode metadata 通过 |
| P1 | 跑 Evo-RL value/ACP smoke | 能回写 `value/advantage/acp_indicator` 并启动 100-1000 step |
| P1 | 复核 AWBC 数据 | `tasks.jsonl` 存在，positive/negative 不塌缩 |
| P2 | 短训 `hq_site` 与 `hq_tda_site` | 先 5k/10k，不直接 88k |

## 7. Gate

| Gate | Go | No-Go |
|---|---|---|
| G1 site A/B | site 候选抓取明显优于 HQ | 查单位、夹爪、推理执行，不开长训 |
| G2 HIL clean | 字段完整、hold 可删、视频对齐 | 修客户端或转换脚本 |
| G3 Evo-RL ACP | value/ACP smoke 可跑，真机不退化 | 回查 success/intervention 标注 |
| G4 KAI0 AWBC | label ratio 合理，AWBC smoke 可跑 | 回查 Stage 标注/打分 |
| G5 长训 | 短训真机 A/B 有正信号 | 不直接 88k |

## 8. 不做

- 不再训练或测试旧单位 `openarm_site_align_v1` 候选。
- 不做独立 HIL window BC；HIL 进入完整 episode 的 value/advantage/ACP 链。
- 不让客户端实时标 KAI0 stage；只保留可后标的帧号、时间和视频。
- 不把 KAI0 Model Arithmetic 当运行时模型路由。
- 不把 train loss 当最终选择标准；真机 A/B 优先。

## 9. 关键入口

```text
src/openpi/policies/openarm_policy.py
src/openpi/training/config.py
scripts/convert_openarm_hq_dataset.py
scripts/convert_openarm_hil_hdf5_to_lerobot_v21.py
scripts/compute_openarm_parquet_norm_stats.py
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
scripts/openarm_stage_advantage_awbc.py
scripts/openarm_tda_awbc_from_source.py
scripts/train.py
scripts/serve_policy.py
```

关键配置：

```text
pi05_openarms_dual_hq
pi05_openarms_dual_site_align_v1_probe
pi05_openarms_dual_site_align_v1_base_10k
pi05_openarms_dual_hq_tda_site_v1
pi05_openarms_dual_awbc_v1
pi05_openarms_dual_evo_acp_hil_v1_probe
ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD
```
