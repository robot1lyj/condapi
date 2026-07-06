# OpenArm π0.6 / KAI0 / Evo-RL 适配总计划

最后更新：2026-07-06 16:09 CST

本文档是 OpenArm + OpenPI 后续训练、部署、HIL、Stage/AWBC 的唯一计划文档。所有 Agent 只更新本文档，不新增分散计划文件。

当前路线不再是“只复现 KAI0”，而是学习三条证据链后做适合 OpenArm 的组合方案：

- **π0.5 / OpenPI**：作为 VLA 底座，保留三路图像 + 16 维 state/action + task prompt 的训练和推理路径。
- **π*0.6 / RECAP**：学习“演示 + 自主试错 + 人工接管纠错 + advantage 条件训练”的闭环。
- **KAI0**：学习衣物任务里的分布对齐、Stage Advantage、TDA、Model Arithmetic。
- **Evo-RL**：作为 RECAP 思路的开源工程参考，重点参考 value/advantage/indicator 回写和 ACP 训练链路。

## 0. 当前结论

### 0.1 不再重复的事

`site_grasp_probe` 不是一个新任务。它对应已经完成的 `site_align_v1_probe`：

```text
config: pi05_openarms_dual_site_align_v1_probe
dataset: /share/home/linyongjia/datasets/openarm_site_align_v1
train split: episodes 0:141
holdout split: episodes 141:151
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_v1_probe_151e_4gpu_1k_tol005_20260706/999
```

后续不再重复做同一个 site-only probe。下一步是把这个 checkpoint 和 HQ baseline 放到同一现场做 A/B，重点看抓取是否改善。

### 0.2 关键事实

1. 用户现场反馈明确：HQ `99999` 只表现出“抬起来到桌面附近”的部分动作，没有稳定抓住衣服，因此不能把它判断成“已经学会折叠，只是现场轻微偏移”。
2. 当前第一故障点是 **抓取接触失败**，其次才是完整折叠、动作平滑、throughput。
3. 现场对齐数据 `openarm_site_align_v1` 已完成转换、同步和短训；它是当前判断现场分布是否能拉动模型的核心证据。
4. Stage Advantage v1 已可用，`10000` checkpoint 是当前最优离线版本：val20 / 800 paired-frame 上 MSE 0.00295、MAE 0.04340、方向准确率 96.75%、corr 0.9859、R2 0.9715。
5. TDA 增强数据 `openarm_hq_tda_aug_v1` 已生成并通过 smoke；但它不能替代现场数据，也不能单独证明抓取能变好。
6. HIL/recovery 真实接管数据还没有进入训练集；没有这类数据时，不要声称已经完成 RECAP/π0.6 风格闭环。
7. Model Arithmetic 不是模型路由。当前单任务、候选模型不足，暂缓。

### 0.3 当前总路线

```text
现有 HQ/OpenPI 底座
  -> site_align_v1_probe 真机 A/B
  -> HQ + site / HQ + TDA + site 短训候选
  -> Stage Advantage 批量打分
  -> AWBC / ACP 训练
  -> HIL recovery 数据闭环
  -> 多候选 checkpoint sweep + 真机 A/B
  -> 可选 Model Arithmetic
```

不要跳过真机 A/B，也不要直接开一个 88k full train 后再猜原因。

## 1. 方法对齐

| 来源 | 它解决什么 | 对 OpenArm 的落地方式 | 不直接照搬的部分 |
|---|---|---|---|
| π0.5 | 用异构数据、高层语义子任务和低层动作训练通用 VLA | 保留 OpenPI/π0.5 作为底座；把 task/stage/advantage 放进 prompt 条件 | 我们没有 PI 的大规模 web/多机器人预训练数据，不能假设靠 prompt 就能泛化 |
| π0.6 / RECAP | 从部署经验、成功失败、人工纠错中继续变强 | 建 HIL/recovery 数据；训练 value/advantage；用 `Advantage: positive` 做 ACP/AWBC | 近期不做完整在线 PPO/SAC 训练大模型 |
| KAI0 | 衣物任务里 `P_train / Q_model / P_test` 分布不一致 | TDA、Stage Advantage、Heuristic DAgger、后期 Model Arithmetic | 不把 Model Arithmetic 当运行时路由；不在候选不足时先做 |
| Evo-RL | RECAP 风格工程链路 | 参考 value train -> value infer -> indicator writeback -> ACP policy train | Evo-RL 本地实现是训练侧参考，不直接替代 OpenPI 服务端 |
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
| Site align v1 | `/share/home/linyongjia/datasets/openarm_site_align_v1` | 已冻结，151 集 | 现场分布对齐；site probe；合并训练高权重数据 | 141 train + 10 holdout |
| AWBC v1 | `/share/home/linyongjia/datasets/openarm_awbc_v1` | 生成/合并状态需复核 | Stage Advantage -> AWBC policy train | 先 smoke，再短训 |
| HIL recovery v1 | `openarm_hil_recovery_v1` | 未有真实训练数据 | RECAP/DAgger/recovery | 不阻塞当前 site/TDA/AWBC 准备 |
| 旧 OpenArms | `openarms_folding_v001/v002` | 已有 | 待审计辅助数据 | 不是 site align v1 |

### 2.2 模型与服务

| 名称 | 状态 | 路径 / 服务 | 下一步 |
|---|---|---|---|
| HQ baseline `99999` | 可推理，但真机抓取失败 | `/share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999` | 与 site probe 同场景 A/B |
| HQ policy server | 可用 | `ws://172.31.11.125:6666` on gpu25 | 保留为 baseline |
| Site align probe | 已完成 1k | `/share/home/linyongjia/output/openpi/pi05_openarms_dual_site_align_v1_probe/openarm_site_v1_probe_151e_4gpu_1k_tol005_20260706/999` | 上真机看抓取改善 |
| Stage Advantage v1 | 可用 | `/share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000` | 批量预测 HQ/site/TDA 映射 |
| TDA smoke | 通过 | `openarm_hq_tda_aug_smoke_tiny_20260630/2` | 不单独作为主模型 |
| `hq_tda_site_v1` | 配置存在，待短训 | `pi05_openarms_dual_hq_tda_site_v1` | 先 5k/10k probe，不直接 88k |
| `awbc_v1` | 配置存在，待数据复核 | `pi05_openarms_dual_awbc_v1` | AWBC smoke 后短训 |

## 3. 核心问题与验证方式

当前不要靠主观判断解释失败，按下面假设逐个证伪。

| 假设 | 现象 | 验证 | 决策 |
|---|---|---|---|
| 现场视觉分布偏移 | 模型靠近桌面但抓点不准 | HQ vs site probe 同场景 A/B；三路相机对齐报告 | site probe 明显改善则进入合并短训 |
| state/action 或夹爪语义有错 | site probe 仍无法接触衣物 | 检查 action 16D、gripper dim 7/15、state fallback、动作范围、真机回放 | 先修数据/控制，不开长训 |
| 推理/执行时延导致错位 | 离线动作合理，真机执行抖或慢 | FIFO vs TDA smooth；记录 infer_ms、publish hz、drop_count | 调部署，不靠数据训练硬补 |
| HQ checkpoint 过拟合或不是最佳 | 不同步数真机差异大 | HQ checkpoint sweep + 2-3 个真机 A/B | 选 warm start，不默认 `99999` 最佳 |
| 缺恢复经验 | 抓错后无法自救 | HIL/recovery 采集失败前后片段 | 进入 RECAP/DAgger 分支 |

## 4. OpenArm-RECAP-KAI0 v1 方案

### 4.1 训练数据流

```text
HQ 成功演示
  + HQ-TDA 源映射增强
  + site_align_v1 现场成功演示
  + HIL/recovery 失败与纠错片段
    -> norm stats / schema 校验
    -> Stage/Value 预测 progress / advantage
    -> 离散为 bad / neutral / positive
    -> prompt/task 写入
    -> OpenPI policy 短训
    -> 真机 A/B
```

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

## 5. Agent 分工

| Agent | 负责人边界 | 现在要做 | 产物 | 验收 |
|---|---|---|---|---|
| Plan Owner | 统一计划、状态、门槛 | 维护本文档；合并多 Agent 结果；更新决策 gate | 本文档 | 每次更新写清“已完成/阻塞/下一步” |
| Eval/Deployment Agent | 真机推理和 A/B | HQ `99999` vs site probe `999` 同场景测试 | A/B 报告、视频、失败分类 | 至少记录抓取接触率、lift 成功率、延迟、动作范围 |
| Data Agent | 数据转换/合并/校验 | 复核 `openarm_site_align_v1`、构建 `openarm_hq_tda_site_v1` | merge report、validation report、norm stats | 16D、三路视频、episode split、NaN、task prompt 全通过 |
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
failure_stage: no_contact / wrong_contact / dropped / no_progress / unsafe / timeout
latency: infer_ms / publish_hz / queue_drop
```

决策：

| 结果 | 下一步 |
|---|---|
| site probe 抓取明显改善 | 构建并短训 `hq_tda_site_v1` |
| site probe 无改善 | 暂停合并长训，检查数据转换、夹爪/action、推理执行 |
| site probe 改善但折叠不稳 | 进入 AWBC + HIL recovery |

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
  site train 0:141 repeat 6
  no TDA

openarm_hq_tda_site_v1_probe_data:
  HQ train 0:999 repeat 1
  TDA 0:2298 repeat 1
  site train 0:141 repeat 6
```

原因：当前最大问题是抓取现场分布，不确定 TDA 是否帮助还是稀释现场梯度。两个短训候选能更快给答案。

参考命令：

```bash
python scripts/merge_openarm_lerobot_v21.py \
  --dst /share/home/linyongjia/datasets/openarm_hq_tda_site_v1 \
  --source hq,/share/home/linyongjia/datasets/high_quality_folding,0:999,1 \
  --source tda,/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1,0:2298,1 \
  --source site,/share/home/linyongjia/datasets/openarm_site_align_v1,0:141,6 \
  --copy-mode hardlink \
  --overwrite
```

### P1. 准备短训，不开长训

候选：

| 实验名 | 数据 | warm start | 步数 | 目的 |
|---|---|---|---:|---|
| `site_align_v1_probe` | site only | HQ `99999` | 已完成 1k | 已有，等真机 A/B |
| `hq_site_v1_probe_10k` | HQ + site x6 | HQ `99999` 或 site probe | 5k/10k | 判断 HQ 先验 + site 是否更稳 |
| `hq_tda_site_v1_probe_10k` | HQ + TDA + site x6 | HQ `99999` 或 site probe | 5k/10k | 判断 TDA 是否帮助部署鲁棒性 |
| `awbc_v1_probe_10k` | AWBC bad/neutral/positive | HQ `99999` 或 best probe | 5k/10k | 判断 Stage/AWBC 是否提升抓取/进展 |
| `recovery_v1_probe` | best + HIL recovery | best prior | 1k/5k | 等 HIL 数据后做恢复能力 |

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

HIL 数据必须能支持 RECAP，不只是普通 BC。每帧至少需要：

```text
policy_action
human_action 或 teleop_action
executed_action
is_intervention
authority_source
episode_success
failure_stage
policy_checkpoint
prompt
timestamp
camera frame id
latency / queue metadata
```

推荐首批规模：

```text
20-30 条 missed grasp recovery
10-20 条 dropped / wrong contact recovery
10 条 autonomous failure without intervention
10 条 autonomous success or near-success
```

用途：

- 人工接管动作：positive 或 high advantage 候选。
- 接管前 policy 动作：negative/neutral 候选。
- 自主成功：value/advantage 标定。
- 自主失败：防止 value model 把“看起来接近完成”的失败状态误判为好。

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
| failure_stage_histogram | 失败分布 | 指导下一轮采集 |

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
| G4 HIL recovery | 接管数据字段完整且能提取 recovery | 训练 recovery branch | 先修采集客户端 |
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
scripts/convert_openarm_site_hdf5_to_lerobot_v21.py
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
