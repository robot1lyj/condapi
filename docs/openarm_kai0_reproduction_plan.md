# OpenArm RECAP / Evo-RL / KAI0 适配总计划

最后更新：2026-07-08 11:36 CST

本文档是 OpenArm + OpenPI 后续训练、部署、HIL、Stage/AWBC 的唯一计划文档。所有 Agent 只更新本文档，不新增分散计划文件。

当前主线不再是“只复现 KAI0”，而是先复现 **π*0.6 / RECAP + Evo-RL** 的闭环，再兼顾 KAI0 的衣物任务增强模块：

- **π0.5 / OpenPI**：作为 VLA 底座，保留三路图像 + 16 维 state/action + task prompt 的训练和推理路径。
- **π*0.6 / RECAP**：作为主线，复现“演示 + 自主试错 + 人工接管纠错 + value/advantage 条件训练”的闭环。
- **Evo-RL**：作为 RECAP 思路的开源工程参考，重点参考 value/advantage/indicator 回写和 ACP 训练链路。
- **KAI0**：作为衣物任务辅助模块，复用两阶段 Stage Advantage、TDA、AWBC；Model Arithmetic 暂缓。

## 0. 当前结论

### 0.1 不再重复的事

`site_grasp_probe` 不是一个新任务。它对应已经完成但已作废的旧单位 `site_align_v1_probe`：

```text
config: pi05_openarms_dual_site_align_v1_probe
dataset: /share/home/linyongjia/datasets/openarm_site_align_v1  # old radian-like site dataset
train split: episodes 0:141
holdout split: episodes 141:151
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_v1_probe_151e_4gpu_1k_tol005_20260706/999
```

后续不再重复测试这个旧 checkpoint。下一步是重新生成 `openarm_site_align_v1_deg`，再训练新的 site-only / HQ+site 短训候选。

### 0.2 关键事实

1. 用户现场反馈明确：HQ `99999` 只表现出“抬起来到桌面附近”的部分动作，没有稳定抓住衣服，因此不能把它判断成“已经学会折叠，只是现场轻微偏移”。
2. 当前第一故障点是 **抓取接触失败**，其次才是完整折叠、动作平滑、throughput。
3. 现场对齐数据必须以 `openarm_site_align_v1_deg` 为准；旧 `openarm_site_align_v1` 只保留作单位事故追溯。
4. 2026-07-07 复核发现旧 `openarm_site_align_v1` 与 HQ 单位错配：HQ 1200 集是 degree-like，site v1 是 radian-like + normalized gripper；旧 site 1k/5k/base10k 候选不再作为可用真机模型。
5. OpenArm policy 数据合同从现在起统一为 HQ contract：task prompt 固定 `Fold the T-shirt properly`，arm joints 用 degrees，gripper 用 HQ-style motor degrees（`0` open，`-66` closed）；ROS/机器人执行仍为 radians + normalized gripper，只在客户端或转换脚本边界转换。
6. 所有现场数据清洗统一入口为 `scripts/convert_openarm_hq_dataset.py`；默认将 raw gripper normalized `0.0` 视为全闭、`0.84` 视为全开，再映射到 HQ gripper motor degrees。
7. Stage Advantage v1 已可用，`10000` checkpoint 是当前最优离线版本：val20 / 800 paired-frame 上 MSE 0.00295、MAE 0.04340、方向准确率 96.75%、corr 0.9859、R2 0.9715。
8. TDA 增强数据 `openarm_hq_tda_aug_v1` 已生成并通过 smoke；但它不能替代现场数据，也不能单独证明抓取能变好。
9. HIL/recovery 真实接管数据还没有进入训练集；没有这类数据时，不要声称已经完成 RECAP/π0.6 风格闭环。
10. Model Arithmetic 不是模型路由。当前单任务、候选模型不足，暂缓。

### 0.3 当前总路线

```text
现有 HQ/OpenPI 底座
  -> openarm_site_align_v1_deg 生成 + norm stats
  -> site_deg_probe / HQ + site_deg / HQ + TDA + site_deg 短训候选
  -> HIL recovery 数据采集：success / failure / intervention / recovery
  -> Evo-RL 风格 value train -> value infer -> advantage / acp_indicator 回写
  -> ACP / AWBC policy 短训
  -> KAI0 两阶段 Stage Advantage 批量打分作为辅助信号
  -> 多候选 checkpoint sweep + 真机 A/B
  -> 可选 Model Arithmetic
```

不要跳过真机 A/B，也不要直接开一个 88k full train 后再猜原因。

## 1. 方法对齐

| 来源 | 它解决什么 | 对 OpenArm 的落地方式 | 不直接照搬的部分 |
|---|---|---|---|
| π0.5 | 用异构数据、高层语义子任务和低层动作训练通用 VLA | 保留 OpenPI/π0.5 作为底座；把 task/stage/advantage 放进 prompt 条件 | 我们没有 PI 的大规模 web/多机器人预训练数据，不能假设靠 prompt 就能泛化 |
| π0.6 / RECAP | 从部署经验、成功失败、人工纠错中继续变强 | 主线复现：HIL/recovery 数据 -> value/advantage -> `Advantage: positive` ACP/AWBC 训练 | 近期不做完整在线 PPO/SAC 训练大模型 |
| Evo-RL | RECAP 风格工程链路 | 主线工程参考：value train -> value infer -> `advantage/acp_indicator` writeback -> ACP policy train | Evo-RL 本地实现是训练侧参考，不直接替代 OpenPI 服务端 |
| KAI0 | 衣物任务里 `P_train / Q_model / P_test` 分布不一致 | 辅助模块：TDA、两阶段 Stage Advantage、AWBC prompt；后期 Model Arithmetic | 不把 Model Arithmetic 当运行时路由；不让客户端人工标复杂 failure taxonomy |
| ProcVLM / 过程奖励 | 用过程进展模型做密集 reward | 后续可作为 Stage/Value v2 参考，尤其抓取阶段进展判断 | 先不引入大外部 VLM 标注系统 |

参考链接：

- π0.5 paper: https://arxiv.org/abs/2504.16054
- π0.5 PMLR: https://proceedings.mlr.press/v305/black25a.html
- π*0.6 / RECAP paper: https://arxiv.org/abs/2511.14759
- π*0.6 blog: https://www.pi.website/blog/pistar06
- π0.6 model card: https://website.pi-asset.com/pi06star/PI06_model_card.pdf
- KAI0 paper: https://arxiv.org/abs/2602.09021
- KAI0 repo: https://github.com/OpenDriveLab/kai0
- Evo-RL repo: https://github.com/MINT-SJTU/Evo-RL
- 本地 Evo-RL: `/home/lyj/Evo-RL`

## 2. 资产表

### 2.1 数据集

| 逻辑名 | 路径 / 名称 | 状态 | 用途 | 注意 |
|---|---|---|---|---|
| HQ 原始 | `/share/home/linyongjia/datasets/high_quality_folding` | 已有，约 1200 集 | 基础动作先验、HQ baseline、合并训练主干 | 不代表当前现场相机/桌面分布 |
| HQ train split | episodes `0:999` | 已用于 HQ 训练 | HQ `99999` 训练 | val `999:1199` 未进入 HQ 训练梯度 |
| HQ val split | episodes `999:1199` | 可用于 sweep | checkpoint 离线评估 | 有损坏视频时要跳过或修复 |
| HQ + TDA | `/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1` | 已生成，2298 集 | 镜像/时间扰动、部署鲁棒性 | AWBC 时优先从源 HQ 打分后映射 |
| Stage train180 | `/share/home/linyongjia/data/high_quality_folding_v2p1_stage_train180` | 已完成 | 训练 Stage Advantage | 不作为 policy 主训练数据 |
| Stage val20 | `/share/home/linyongjia/data/high_quality_folding_v2p1_stage_val20` | 已完成 | Stage 验证 | 离散抽样 holdout |
| Site align v1 old | `/share/home/linyongjia/datasets/openarm_site_align_v1` | 历史错误单位，151 集 | 仅用于追溯 | radian-like + normalized gripper，禁止继续训练/真机测试 |
| Site align v1 deg | `/share/home/linyongjia/datasets/openarm_site_align_v1_deg` | 已重转并完成 norm stats | 现场分布对齐；site probe；合并训练高权重数据 | 默认 141 train + 10 holdout；HQ task；arm joints degrees；gripper HQ motor degrees |
| AWBC v1 | `/share/home/linyongjia/datasets/openarm_awbc_v1` | 生成/合并状态需复核 | Stage Advantage -> AWBC policy train | 先 smoke，再短训 |
| HIL / RECAP v1 | `openarm_hil_recap_v1` | 未有真实训练数据 | RECAP/Evo-RL value/advantage/ACP | 不作为独立 window BC 训练集 |
| 旧 OpenArms | `openarms_folding_v001/v002` | 已有 | 待审计辅助数据 | 不是 site align v1 |

### 2.2 模型与服务

| 名称 | 状态 | 路径 / 服务 | 下一步 |
|---|---|---|---|
| HQ baseline `99999` | 可推理，但真机抓取失败 | `/share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999` | 与 site probe 同场景 A/B |
| Site HQ 5k old | 已删除 | `openarm_site_v1_probe_151e_2gpu_5k_hq99999_20260706` | 旧单位合同错误；曾在 gpu25:6666 误测；2026-07-07 已清理 |
| Site align probe old | 已删除 | `openarm_site_v1_probe_151e_4gpu_1k_tol005_20260706` | 基于旧 radians site 数据；2026-07-07 已清理 |
| Site deg HQ 5k | 已完成 | `/share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_deg_151e_2gpu_5k_hq99999_20260707/4999` | 待开启 gpu25 推理服务和真机 A/B |
| Site deg base 10k | 已完成 | `/share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_base_10k/openarm_site_deg_base_151e_2gpu_10k_pi05base_20260707/9999` | 第二 A/B 候选；loss 更低但不单独判优 |
| Stage Advantage v1 | 可用 | `/share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000` | 批量预测 HQ/site/TDA 映射 |
| TDA smoke | 通过 | `openarm_hq_tda_aug_smoke_tiny_20260630/2` | 不单独作为主模型 |
| `hq_tda_site_v1` | 配置存在，待短训 | `pi05_openarms_dual_hq_tda_site_v1` | 先 5k/10k probe，不直接 88k |
| `awbc_v1` | 配置存在，待数据复核 | `pi05_openarms_dual_awbc_v1` | AWBC smoke 后短训 |

## 3. 核心问题与验证方式

当前不要靠主观判断解释失败，按下面假设逐个证伪。

| 假设 | 现象 | 验证 | 决策 |
|---|---|---|---|
| 现场视觉分布偏移 | 模型靠近桌面但抓点不准 | HQ vs site probe 同场景 A/B；三路相机对齐报告 | site probe 明显改善则进入合并短训 |
| state/action 或夹爪语义有错 | site probe 仍无法接触衣物或几乎不动 | 检查 action 16D、gripper dim 7/15、state fallback、动作范围、单位合同、真机回放 | 先修数据/控制，不开长训 |
| 推理/执行时延导致错位 | 离线动作合理，真机执行抖或慢 | FIFO vs TDA smooth；记录 infer_ms、publish hz、drop_count | 调部署，不靠数据训练硬补 |
| HQ checkpoint 过拟合或不是最佳 | 不同步数真机差异大 | HQ checkpoint sweep + 2-3 个真机 A/B | 选 warm start，不默认 `99999` 最佳 |
| 缺恢复经验 | 抓错后无法自救 | HIL/recovery 采集失败前后片段 | 进入 RECAP/DAgger 分支 |

## 4. OpenArm RECAP/Evo-RL 主线 + KAI0 辅助 v1 方案

### 4.1 训练数据流

```text
HQ 成功演示
  + HQ-TDA 源映射增强
  + site_align_v1 现场成功演示
  + HIL/recovery 自主成功、失败、人工接管纠错片段
    -> norm stats / schema 校验
    -> RECAP/Evo-RL: value train -> value infer -> advantage / acp_indicator
    -> KAI0: 两阶段 Stage progress / advantage 辅助打分
    -> prompt/task 写入：Advantage: bad / neutral / positive
    -> OpenPI ACP/AWBC policy 短训
    -> 真机 A/B
```

按 π*0.6 / RECAP 公开材料与本地 Evo-RL 代码重新归类：训练链路明确依赖的是 `observation/action/task`、episode 级 `episode_success`、frame 级 `complementary_info.is_intervention`，后处理再写回 `complementary_info.value/advantage/acp_indicator`。`policy_action`、`teleop_action/human_action`、`executed_action`、`collector_policy_id` 是 OpenArm HIL 为复现接管、追踪动作来源和排查部署问题保留的工程字段；`recovery_success` 可由 `episode_success + intervention spans` 派生，不作为 Evo-RL 当前代码硬依赖。KAI0 的两阶段 `task_stage/stage_progress` 后处理生成，不要求客户端人工标复杂失败阶段。

### 4.2 Prompt 条件策略

当前可落地的 v1：

```text
fold the cloth, Advantage: positive
fold the cloth, Advantage: neutral
fold the cloth, Advantage: bad
```

后续创新 v2，等 v1 跑通后再做：

```text
fold the cloth
Stage: grasp_cloth
Advantage: positive
```

理由：当前故障集中在抓取阶段，单纯全局 progress 可能无法区分“靠近衣服但未夹住”和“真正建立接触”。但 v2 需要改数据和 prompt hook，不能抢在 v1 前面。

### 4.3 不做的事

- 不重复训练 `site_align_v1_probe`。
- 不直接启动纯 TDA 88k full train。
- 不把 `openarms_folding_v001/v002` 当 site 数据。
- 不把 KAI0 Model Arithmetic 当作运行时路由。
- 不在没有 HIL/recovery 数据时宣称完成 RECAP。
- 不把复杂 `failure_stage` 作为 HIL 主标签；失败原因只做可选复盘字段。
- 不做独立的 HIL window BC / `recovery_v1_probe`；接管数据必须进入完整 episode 的 value/advantage/ACP 链路。

## 5. Agent 分工

| Agent | 负责人边界 | 现在要做 | 产物 | 验收 |
|---|---|---|---|---|
| Plan Owner | 统一计划、状态、门槛 | 维护本文档；合并多 Agent 结果；更新决策 gate | 本文档 | 每次更新写清“已完成/阻塞/下一步” |
| Eval/Deployment Agent | 真机推理和 A/B | HQ `99999` vs site probe `999` 同场景测试 | A/B 报告、视频、失败分类 | 至少记录抓取接触率、lift 成功率、延迟、动作范围 |
| Data Agent | 数据转换/合并/校验 | 生成/复核 `openarm_site_align_v1_deg`、构建 `openarm_hq_tda_site_v1` | merge report、validation report、norm stats | 16D、三路视频、episode split、NaN、task prompt、单位合同全通过 |
| Training Agent | GPU 训练 | 准备并运行短训候选，不直接开长训 | checkpoints、metrics、训练日志 | 5k/10k probe 完成，loss 无异常，checkpoint 可 serve |
| Stage/AWBC Agent | Stage 打分和 AWBC | 复核 AWBC shard 是否完成；合并 `openarm_awbc_v1`；跑 smoke | AWBC dataset、tasks.jsonl、label stats | bad/neutral/positive 分布合理，能启动训练 |
| HIL/Recovery Agent | 接管数据闭环 | 固定 HIL 字段；采首批真实接管/失败片段 | `openarm_hil_recovery_v1` | 可算 policy-human 差值，可分出 recovery segment |
| Infra Agent | SSH/GPU/服务 | 保证 gpu12/gpu14/gpu28 可用；服务端口可达 | tmux/log/GPU 状态 | 训练和 serve 不抢占关键 GPU |
| Research Agent | 论文和项目对照 | 跟踪 π0.6/RECAP、KAI0、Evo-RL 可落地差异 | 方案注释和引用 | 只给可执行建议，不堆概念 |

## 6. 接下来 48 小时任务

### P0. 真机 A/B：HQ baseline vs site probe

目的：回答 site 数据短训是否让抓取改善。

固定条件：

```text
same cloth
same table
same camera pose
same prompt
same reset protocol
same FIFO/TDA mode
```

最少记录 10 次尝试，指标：

```text
grasp_contact_rate: 夹爪是否接触/夹住衣物
lift_success_rate: 是否能带起衣物
flatten_entry_rate: 是否进入拉平/展开阶段
episode_success_rate: 是否完整折叠
episode_outcome: success / failure / aborted
failure_reason: grasp / manipulation / system / unknown  # optional debug only
latency: infer_ms / publish_hz / queue_drop
```

决策：

| 结果 | 下一步 |
|---|---|
| site probe 抓取明显改善 | 构建并短训 `hq_tda_site_v1` |
| site probe 无改善 | 暂停合并长训，检查数据转换、夹爪/action、推理执行 |
| site probe 改善但折叠不稳 | 进入 AWBC + HIL RECAP/ACP |

### P0. 复核 AWBC 生成状态

确认下面数据是否完整：

```text
/share/home/linyongjia/datasets/openarm_awbc_v1
/share/home/linyongjia/datasets/openarm_awbc_v1/meta/tasks.jsonl
/share/home/linyongjia/output/openpi/logs/openarm_awbc_v1_shards_balanced/
```

必须输出：

```text
episode_count
frame_count
bad / neutral / positive frame ratio
source: HQ original / TDA original / TDA time / TDA mirror
missing videos/parquets count
norm_stats status
```

### P1. 构建 `openarm_hq_tda_site_v1`

建议先做两个版本，不直接押一个比例：

```text
openarm_hq_site_v1_probe_data:
  HQ train 0:999 repeat 1
  site_deg train 0:141 repeat 6
  no TDA

openarm_hq_tda_site_v1_probe_data:
  HQ train 0:999 repeat 1
  TDA 0:2298 repeat 1
  site_deg train 0:141 repeat 6
```

原因：当前最大问题是抓取现场分布，不确定 TDA 是否帮助还是稀释现场梯度。两个短训候选能更快给答案。

参考命令：

```bash
python scripts/merge_openarm_lerobot_v21.py \
  --dst /share/home/linyongjia/datasets/openarm_hq_tda_site_v1 \
  --source hq,/share/home/linyongjia/datasets/high_quality_folding,0:999,1 \
  --source tda,/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1,0:2298,1 \
  --source site,/share/home/linyongjia/datasets/openarm_site_align_v1_deg,0:141,6 \
  --copy-mode hardlink \
  --overwrite
```

### P1. 准备短训，不开长训

候选：

| 实验名 | 数据 | warm start | 步数 | 目的 |
|---|---|---|---:|---|
| `site_align_v1_deg_probe` | site_deg only | HQ `99999` | 5k | 判断单位修正后现场抓取是否恢复 |
| `hq_site_v1_deg_probe_10k` | HQ + site_deg x6 | HQ `99999` 或 site_deg probe | 5k/10k | 判断 HQ 先验 + site 是否更稳 |
| `hq_tda_site_v1_deg_probe_10k` | HQ + TDA + site_deg x6 | HQ `99999` 或 site_deg probe | 5k/10k | 判断 TDA 是否帮助部署鲁棒性 |
| `awbc_v1_probe_10k` | AWBC bad/neutral/positive | HQ `99999` 或 best probe | 5k/10k | 判断 Stage/AWBC 是否提升抓取/进展 |
| `recap_acp_hil_v1_probe` | HIL full episodes + value/advantage/acp_indicator | best prior | 5k/10k | 复现 RECAP/Evo-RL 主线，不做窗口 BC |

## 7. 数据策略

### 7.1 现场数据 151/200 的定位

现场数据不是“再多一点演示”，而是对齐 `P_test` 的关键数据。当前 site 151 条已足够做第一轮 site 对齐验证；如果现场还继续录，新增数据优先补变化而不是重复：

- 衣服初始位置：左/右/前/后、偏斜、离机器人远近。
- 展开状态：皱褶、折叠程度、袖子/边缘摆放。
- 光照：亮/暗/反光。
- 轨迹波动：不同抓点、不同拉平路径。
- 接近失败边界：抓点不完美但最终修正成功。

单色黑衣服如果这些变化覆盖充分，80-100 条可作为 v1；151 条已经可以用于验证。继续采到 200 的价值主要在补失败边界和 HIL/recovery，而不是重复标准成功演示。

### 7.2 HIL/recovery 数据字段

HIL 数据必须先支持 RECAP/Evo-RL，不只是普通 BC。v1 不做复杂 `failure_stage` 主标签；字段按来源分级，避免把工程调试字段误说成论文或 Evo-RL 代码硬要求。

训练最小闭环必须能转成 LeRobot：

```text
timestamp / timestamp_ns
episode_index / episode_id
frame_index
task / prompt

observation.state                 # 16D，HQ contract: joints degrees, gripper HQ motor degrees
observation.images.base
observation.images.left_wrist
observation.images.right_wrist

action                            # 最终实际执行动作，16D；Evo-RL/LeRobot policy 训练读这个
complementary_info.is_intervention # 当前帧是否人工接管
```

每条 episode 结束时必须有：

```text
episode_success                   # success / failure；value train 和 value infer 的上游监督
episode_outcome                   # success / failure / aborted；转换时归一到 episode_success
```

OpenArm 强烈建议额外保留，方便复现接管和排查动作来源：

```text
complementary_info.policy_action  # 当前帧模型原本准备执行的动作，16D
complementary_info.teleop_action  # 接管时的人类动作，16D；未接管可为空
executed_action                   # 若 action 已是最终执行动作，可与 action 相同；用于显式审计
authority_source                  # policy / human / safety_stop / scripted
collector_policy_id               # policy checkpoint 或 human
policy_checkpoint

policy_action_chunk               # OpenPI 服务端返回的整段 chunk，shape (50, 16)；工程字段
action_chunk_id
step_in_chunk
model_metadata                    # action_dim=32, robot_action_dim=16, action_horizon=50, rtc_mode 等
safety_clipped
clip_reason
```

只作为部署 debug，可选保存：

```text
request_send_time_ns
response_recv_time_ns
server_infer_ms
round_trip_ms
publish_hz
drop_count
```

后处理生成，不要求客户端人工标：

```text
task_stage                        # 0 flatten / 1 fold
stage_progress
relative_advantage
absolute_value
absolute_advantage
acp_indicator
```

可选复盘字段，只用于 debug 和采集分布统计，不作为 v1 主训练标签：

```text
failure_reason                    # grasp / manipulation / system / unknown
operator_note
```

推荐首批规模：

```text
20-30 条自主失败 + 人工接管救回
10-20 条自主失败 + 接管后仍失败或中止
10 条自主失败且不接管
10 条自主成功或接近成功
```

用途：

- 人工接管动作：positive 或 high advantage 候选。
- 接管前 policy 动作：negative/neutral 候选。
- 自主成功：value/advantage 标定。
- 自主失败：防止 value model 把“看起来接近完成”的失败状态误判为好。
- 两阶段 task_stage：只服务 KAI0 Stage Advantage，不替代 RECAP/Evo-RL 的 success/intervention/recovery 主线。

## 8. Stage / AWBC 计划

### 8.1 当前 Stage v1

Stage v1 与 KAI0 Task A 对齐，只用两阶段：

```text
stage 0: flatten / 展开拉平
stage 1: fold / 折叠完成
```

已完成：

```text
annotated subset: 200 HQ episodes
train: 180 episodes
val: 20 episodes, discrete holdout
best checkpoint: step 10000
```

不要把之前诊断用的 5 个细粒度事件误认为 KAI0 原始 taxonomy。细粒度事件可以保留做分析，但 v1 不直接进入 `stage_progress_gt`。

### 8.2 AWBC 标签

当前脚本会写：

```text
relative_advantage
absolute_value
absolute_advantage
task_index: 0 bad / 1 neutral / 2 positive
meta/tasks.jsonl:
  fold the cloth, Advantage: bad
  fold the cloth, Advantage: neutral
  fold the cloth, Advantage: positive
```

验收门槛：

```text
bad/neutral/positive 都有样本
positive 不应全挤在 episode 末尾
bad 不应全是视频/读取异常
TDA mirror/time 的标签来自 source 映射，而不是盲目重打分增强视频
AWBC 数据可跑 100-1000 step smoke
```

### 8.3 Stage v2 创新方向

等 v1 真机 A/B 后再做：

```text
stage: approach_grasp / grasp_contact / flatten / fold / release
or
stage: grasp_cloth / flatten_align / fold_finish
```

v2 目标不是增加概念，而是解决当前抓取失败：让 value/advantage 明确知道“夹到衣服”是阶段进展，不是简单时间推进。

## 9. 训练与验证门槛

### 9.1 离线指标

| 指标 | 用途 | 趋势 |
|---|---|---|
| action MSE / MAE | 基础动作拟合 | 越低越好，但不能单独决定 |
| gripper timing error | 夹爪开合时机 | 越低越好 |
| stage progress MSE / MAE | Stage 拟合 | 越低越好 |
| pairwise direction accuracy | 判断哪帧更推进任务 | 越高越好 |
| corr / R2 | progress 曲线是否合理 | 越高越好 |
| label ratio | AWBC 数据质量 | 不应塌缩到单一类 |
| action smoothness | 推理动作抖动 | 越平滑越好，但不能牺牲接触 |

### 9.2 真机指标

真机指标优先级高于离线 loss。

| 指标 | 含义 | 第一阶段目标 |
|---|---|---|
| grasp_contact_rate | 是否真的夹到衣服 | 必须先提升 |
| lift_success_rate | 是否能把衣服带起来 | 抓取后验证 |
| flatten_entry_rate | 是否进入展开/拉平阶段 | 中间进展 |
| full_success_rate | 完整折叠成功 | 最终指标 |
| retry_cost | 平均重试次数 | 越低越好 |
| throughput | 单位时间完成数 | 成功稳定后再优化 |
| intervention_rate | HIL 中接管比例 | 迭代后应下降 |
| recovery_success_rate | 接管后是否救回 | RECAP/Evo-RL 主指标 |
| failure_reason_histogram | 可选失败原因分布 | 只指导下一轮采集，不作为 v1 主训练标签 |

### 9.3 checkpoint 选择

不要只看 train loss。标准流程：

1. HQ val split 离线 sweep：`50000 / 70000 / 85000 / 95000 / 99999`。
2. site holdout 离线评估：10 条现场 holdout。
3. 真机 A/B：同场景、同衣服、同 reset。
4. 只保留 2-3 个候选进入下一轮训练或 serve。

## 10. 决策 Gate

| Gate | 条件 | Go | No-Go |
|---|---|---|---|
| G1 site probe | site probe 真机抓取优于 HQ | 做 HQ/site/TDA 短训 | 查数据转换/夹爪/控制 |
| G2 combined short train | 5k/10k 候选离线和真机均不退化 | 扩到 20k/40k 或进入 AWBC | 调数据比例，不开 88k |
| G3 AWBC | AWBC smoke + 真机进展优于 SFT | 继续 AWBC/ACP | 回查 Stage 打分和 label ratio |
| G4 HIL RECAP/ACP | HIL 字段完整，能训练 value 并回写 `advantage/acp_indicator` | 跑 `recap_acp_hil_v1_probe` | 先修采集客户端或 value 标注链路 |
| G5 Model Arithmetic | 至少 3 个互补 checkpoint + OOD validation | 做权重合并 | 暂缓 |

## 11. 远端与命令索引

### 11.1 节点

```text
jump: ssh -p 12222 linyongjia@172.31.11.100
gpu12: 172.31.11.112
gpu14: 172.31.11.114
gpu28: 172.31.11.128
gpu25 serve: 172.31.11.125:6666
remote repo: /share/home/linyongjia/conda-pi/openpi
remote datasets: /share/home/linyongjia/datasets
remote outputs: /share/home/linyongjia/output/openpi
```

### 11.2 关键脚本

```text
scripts/convert_openarm_hq_dataset.py
scripts/merge_openarm_lerobot_v21.py
scripts/augment_openarm_hq_tda.py
scripts/annotate_openarm_tda_aug_metadata.py
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
scripts/openarm_stage_advantage_awbc.py
scripts/openarm_tda_awbc_from_source.py
scripts/compute_openarm_parquet_norm_stats.py
scripts/train.py
scripts/train_pytorch.py
scripts/serve_policy.py
```

### 11.3 关键配置

```text
pi05_openarms_dual_hq
pi05_openarms_dual_site_align_v1_probe
pi05_openarms_dual_site_align_v1_base_10k
pi05_openarms_dual_hq_tda_aug
pi05_openarms_dual_hq_tda_site_v1
pi05_openarms_dual_awbc_v1
ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD
```

## 12. 状态更新日志

### 2026-07-06 16:09 CST - Plan Owner - 大幅重排为 π0.6 / KAI0 / Evo-RL 组合路线

更新内容：

- 明确 `site_grasp_probe` 不再作为新任务；它就是已完成的 `site_align_v1_probe`。
- 把当前问题从“HQ 已学会折叠、只需现场微调”修正为“HQ 未稳定抓住衣服，第一故障点是抓取接触失败”。
- 将路线改为 OpenArm-RECAP-KAI0：site A/B -> HQ/site/TDA 短训 -> Stage/AWBC -> HIL recovery -> checkpoint sweep -> 可选 Model Arithmetic。
- 补充 Agent 分工、48 小时任务、数据策略、训练门槛和真机指标。
- 明确在没有 HIL/recovery 数据前，不宣称完成 RECAP/π0.6 风格闭环。

下一步：

1. Eval/Deployment Agent 先做 HQ `99999` vs site probe `999` 同场景 A/B。
2. Stage/AWBC Agent 复核 `openarm_awbc_v1` 是否完整、label ratio 是否合理。
3. Data/Training Agent 准备 `hq_site_v1_probe_10k` 和 `hq_tda_site_v1_probe_10k`，但等 G1 结果再开训。

### 2026-07-06 16:31 CST - Training Agent - 启动 site 5k 与 base 10k 对照训练

状态：训练中。

已完成：

- 新增 `pi05_openarms_dual_site_align_v1_base_10k` 配置；单位审计后改为等待 `openarm_site_align_v1_deg`，训练 episodes `0:141`，保留 `141:151` holdout。
- 在 gpu12 启动 HQ `99999` warm start 的 site 5k：

```text
tmux: openarm_site_hq5k_20260706
config: pi05_openarms_dual_site_align_v1_probe
exp: openarm_site_v1_probe_151e_2gpu_5k_hq99999_20260706
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_v1_probe_151e_2gpu_5k_hq99999_20260706
log: /share/home/linyongjia/output/openpi/logs/pi05_openarms_dual_site_align_v1_probe/openarm_site_v1_probe_151e_2gpu_5k_hq99999_20260706_gpu12.log
startup: step 69 reached at 16:31 CST, both gpu12 cards about 73.6GB and 100% util
```

- 在 gpu14 启动原始 π0.5 base warm start 的 site 10k：

```text
tmux: openarm_site_base10k_20260706
config: pi05_openarms_dual_site_align_v1_base_10k
exp: openarm_site_v1_base_151e_2gpu_10k_pi05base_20260706
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_base_10k/openarm_site_v1_base_151e_2gpu_10k_pi05base_20260706
log: /share/home/linyongjia/output/openpi/logs/pi05_openarms_dual_site_align_v1_base_10k/openarm_site_v1_base_151e_2gpu_10k_pi05base_20260706_gpu14.log
startup: step 16 reached at 16:31 CST, both gpu14 cards about 73.6GB and 100% util
```

实验目的：

- 比较 `site_align_v1_probe` 1k、HQ `99999` -> site 5k、原始 π0.5 -> site 10k 三个候选在真机抓取上的差异。
- 重点看 grasp_contact_rate / lift_success_rate，不以 train loss 单独决定。

### 2026-07-07 10:40 CST - Training/Serve Agent - 单位审计并暂停旧 site 训练/服务

状态：已暂停旧 site 训练/服务，等待 deg 数据重训。

已完成：

- HQ `99999` -> site 5k 已完成，最终 checkpoint `4999` 完整保存。
- site 5k 训练指标数值正常：loss 从 step 0 的 `0.2398` 降到 step 4980 的 `0.0215`，但该 loss 是在错误单位合同上收敛，不能证明真机可用。
- 原始 π0.5 base -> site 10k 在 step 1320 左右遇到 `DataLoader worker Segmentation fault`，可用 checkpoint 为 `1000`。
- 新增 `scripts/openarm_benchmark_loader.py`，用于压测 LeRobot loader 的视频后端和 worker 数。
- loader 压测结论：`torchcodec+2 workers` 平均约 6.9s/batch；`torchcodec+1 worker` 约 10.1s/batch；`torchcodec+4 workers` 约 7.6s/batch；`pyav+2 workers` 约 11.0s/batch。
- 将 `pi05_openarms_dual_site_align_v1_base_10k` 改为显式 `torchcodec+2 workers`，并将 `save_interval=200`、`keep_period=1000`，用更密的 checkpoint 降低 worker 偶发崩溃后的回退成本。
- gpu25:6666 已切换到 site 5k `4999` 推理服务；healthz 为 `OK`，WebSocket 假输入返回 `actions` shape `(50, 16)`。
- 单位审计结论：HQ `high_quality_folding` 全局 joint max_abs 约 `140.05`，为 degree-like；旧 site `openarm_site_align_v1` 全局 joint max_abs 约 `2.48`，为 radian-like，且 gripper 是 normalized，不是 HQ-style motor degrees。
- 已停止 gpu14 上旧 base10k watchdog 训练，释放两张 A800。
- 已停止 gpu25:6666 旧 site5k 推理服务，释放显存，避免现场继续误测。
- 新增统一入口 `scripts/convert_openarm_hq_dataset.py`：默认输出 `openarm_site_align_v1_deg`，task 对齐 HQ `Fold the T-shirt properly`，arm joints rad->deg，gripper normalized 经 `0.0 closed / 0.84 open` 标定后映射到 `[-66, 0]` HQ motor degrees。

### 2026-07-07 10:58 CST - Data/Training Agent - HQ contract 转换入口落地

状态：数据合同已修正，site deg 训练已完成，可进入 gpu25 推理 smoke。

已完成：

- 新增统一入口 `scripts/convert_openarm_hq_dataset.py`，支持 `from-hdf5` 和 `from-lerobot` 两种来源；后续现场清洗只走这个入口。
- HQ contract 固定为：task `Fold the T-shirt properly`，arm joints degrees，gripper HQ motor degrees（`0` open，`-66` closed）。
- 现场 raw gripper 标定为 `0.0` closed / `0.84` open，再映射到 HQ motor degrees；远端全量抽查 gripper 范围为 `-66..0`。
- 已重转 `/share/home/linyongjia/datasets/openarm_site_align_v1_deg`，151 episodes，joint max_abs 约 `142.06`，task 已对齐 HQ。
- 已计算 `/share/home/linyongjia/datasets/openarm_site_align_v1_deg/norm_stats.json`，训练 split `0:141`，共 374544 frames；OpenPI config 已确认能读取 `state/actions` norm stats。
- 已在 gpu12 启动 HQ `99999` -> site deg 5k：tmux `openarm_site_deg_5k_20260707`，log `/share/home/linyongjia/output/openpi/logs/pi05_openarms_dual_site_align_v1_probe/openarm_site_deg_151e_2gpu_5k_hq99999_20260707_gpu12.log`；step 0 loss `0.2434`，两张 A800 显存约 `73.6GB`。
- HQ `99999` -> site deg 5k 已完成 checkpoint `4999`，metrics `250` rows，step `0..4980`，loss `0.2434 -> 0.0218`，tail20 mean `0.0226`，grad_norm tail20 mean `0.1023`，最终 checkpoint 约 `42G`。
- π0.5 base -> site deg 10k 已完成 checkpoint `9999`，metrics `500` rows，step `0..9980`，loss `0.1602 -> 0.0156`，tail20 mean `0.0163`，grad_norm tail20 mean `0.0849`，checkpoint 约 `417G`。
- 已删除三个旧单位合同 checkpoint，释放约 `126G`：旧 site 5k、旧 site 1k、旧 base 1000。

下一步：

- 在 gpu25 先 serve `Site deg HQ 5k / 4999`，做 WebSocket health、假观测 action shape、动作范围和夹爪范围 smoke；不直接上真机执行。
- smoke 通过后做低速真机 A/B，第一候选是 `Site deg HQ 5k / 4999`，第二候选是 `Site deg base 10k / 9999`。
- 执行端夹爪建议先用防抖二值化：model gripper `<= -30` 发闭合，`>= -10` 发张开，中间保持上一次命令。
- 禁止继续使用旧 `openarm_site_align_v1` 单位合同下的 site5k/base10k 候选。

### 2026-07-08 10:43 CST - Training Agent - site_deg 两个候选完成并审计训练数据

状态：可以开始第一阶段推理测试，但顺序必须是 gpu25 smoke -> 动作范围检查 -> 低速真机 A/B。

训练完成：

- HQ `99999` warm start -> site_deg 5k：gpu12 无活跃训练进程，最终 checkpoint `/share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_deg_151e_2gpu_5k_hq99999_20260707/4999`。
- 原始 π0.5 base -> site_deg 10k：gpu14 无活跃训练进程，最终 checkpoint `/share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_base_10k/openarm_site_deg_base_151e_2gpu_10k_pi05base_20260707/9999`。

训练数据审计：

- `/share/home/linyongjia/datasets/openarm_site_align_v1_deg`：151 episodes，394900 frames，split `train=0:141`、`val=141:151`。
- task prompt 为 `Fold the T-shirt properly`。
- `observation.state` 和 `action` 都是 16 维。
- 关节范围：state arm abs max `142.06`，action arm abs max `140.0`，与 HQ degree-like 合同一致。
- 夹爪范围：state/action gripper `[-66, 0]`，与 HQ motor degrees 合同一致；`0` open，`-66` closed。
- 数据列名仍是 LeRobot 原始列 `observation.images.base/left_wrist/right_wrist`；训练配置用 `base_image_key="observation.images.base"`，`PiperInputs` 在训练/推理 transform 中重打包成模型需要的 `base_0_rgb/left_wrist_0_rgb/right_wrist_0_rgb`。

推理测试顺序：

1. gpu25 serve `Site deg HQ 5k / 4999`。
2. 本地或客户端假输入 smoke：`actions` shape 应为 `(50, 16)`，关节输出应是 degree-like，夹爪应落在约 `[-66, 0]` 合同内。
3. 低速真机执行，先看抓取接触率和 lift 成功率，不用 train loss 判优。
4. 若 `4999` 抓取仍差，再切 `Site deg base 10k / 9999` 做同场景 A/B。

### 2026-07-08 11:36 CST - Plan Owner - 主线收敛为 RECAP/Evo-RL 优先

状态：HIL 客户端合同已收敛，复杂失败阶段不进入 v1 主标签。

决策：

- 主线优先复现 π*0.6 / RECAP 与 Evo-RL：真实部署数据 -> value/advantage -> `acp_indicator` -> ACP/AWBC policy 训练。
- KAI0 作为衣物任务辅助模块：保留 TDA、两阶段 Stage Advantage、AWBC prompt，不把 Model Arithmetic 或复杂 failure taxonomy 作为当前主线。
- HIL v1 不要求客户端人工标复杂 `failure_stage`；每帧必须采 `policy_action/human_action/executed_action/is_intervention`，每条 episode 只标 `episode_success/recovery_success/episode_outcome`。
- `task_stage` 固定为两阶段：`0 flatten`、`1 fold`，由 Stage 模型或后处理生成；不作为客户端实时人工标注负担。
- `failure_reason` 只保留可选复盘字段：`grasp/manipulation/system/unknown`，用于分析采集分布，不作为 v1 主训练标签。

下一步：

- 客户端按第 7.2 节实现 HIL recorder；先采自主成功、自主失败、人工接管救回、接管后仍失败四类数据。
- Data Agent 将 HIL raw 转成 LeRobot v2.1，并保留 RECAP/Evo-RL 字段以便 value train / value infer / ACP writeback。

### 2026-07-08 11:45 CST - Plan Owner - 删除独立 HIL window BC 思路

状态：`recovery_v1_probe` 已从训练候选中删除。

决策：

- 不再设计“只截接管窗口做 BC”的独立实验。该做法会破坏长时序上下文，也不是 RECAP/π0.6、Evo-RL 或 KAI0 的主线。
- HIL 数据保留完整 episode：policy 控制、人类接管、恢复、失败/成功结果都不删不拼。
- 接管数据只通过 RECAP/Evo-RL 链路使用：训练 value，回写 `advantage/acp_indicator`，再做 ACP/AWBC policy 短训。
- KAI0 的两阶段 Stage Advantage 继续作为辅助 progress 信号，但不替代完整 HIL episode 的 value/advantage 训练。

下一步：

- HIL 到齐后先构建 `openarm_hil_recap_v1`，再跑 value/advantage/ACP smoke；不启动独立 window BC 微调。

### 2026-07-08 12:22 CST - Plan Owner - HIL 字段按 Evo-RL 代码重新归类

状态：已修正第 4.1 和 7.2 节，避免把工程扩展字段误标为 π*0.6/RECAP 或 Evo-RL 硬要求。

代码审计结论：

- Evo-RL 当前训练硬依赖：LeRobot 基础 `observation/action/task`，episode 级 `episode_success`，ACP/value 后处理字段 `complementary_info.value/advantage/acp_indicator`。
- Evo-RL 当前 HIL processor 明确使用：`teleop_action`、`is_intervention`、`success/terminate/rerecord`；接管时用 `teleop_action` 覆盖 policy action。
- `policy_action_chunk`、网络延迟、`action_chunk_id/step_in_chunk`、`model_metadata` 是 OpenArm/OpenPI 工程审计字段，不是论文或 Evo-RL 代码必需字段。
- `policy_action`、`executed_action`、`collector_policy_id` 对我们复现 RECAP 接管数据很有价值，但在当前 Evo-RL 代码里不是 value/ACP 训练硬依赖；客户端能采就采，最小闭环不能被这些字段阻塞。
