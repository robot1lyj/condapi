# OpenArm KAI0 复现部署计划

本文档是 OpenArm + OpenPI 复现 KAI0 的统一协同计划。所有 Agent 的进展必须回写到本文档，不再新增分散的计划文件。

目标不是把 KAI0 所有模块一次性搬完，而是从我们已经有的 HQ 模型、真机推理链路、数据转换和 HIL 采集能力开始，按可验收的门槛推进。

## 0. 当前结论

最后更新：2026-07-02 11:05 CST

### 0.1 本轮决策

1. 当前第一优先级是修正 `P_test` 视觉分布，尤其是 `base` 主摄像头。第一轮真机显示机械臂起身到桌面阶段正常，失败集中在夹爪抓取定位，现场反馈指向主摄像头视角/安装位/画面分布与 HQ 数据集不一致。
2. 在主摄像头分布未对齐前，不继续把时间花在 TDA 参数微调上。当前 TDA smooth 已能真实运行，继续调 chunk 参数不能解决抓取目标看错的问题。
3. 现场对齐数据正在录制，第一版按 `openarm_site_align_v1` 冻结命名。若最终只有 150 条，先按 130 train + 20 holdout 启动 site probe；若扩到约 200 条，再按 180 train + 20 holdout 启动主线。
4. Stage Advantage v1 已完成，`10000` checkpoint 是当前最优：val20 / 800 paired-frame 上 MSE 0.00295、MAE 0.04340、方向准确率 96.75%、corr 0.9859、R2 0.9715。下一步不是继续训练 Stage，而是用它批量预测 advantage 并准备 AWBC。
5. TDA 增强数据已可作为合并数据源，但不要直接开纯增强 88k full train。当前训练优先级是 `site_v1_ft_probe` -> `hq_tda_site_v1` -> `AWBC_v1`。
6. Heuristic DAgger / Recovery 采集格式已经具备客户端基础，但当前还没有推理 HIL 接管数据。它不阻塞下一轮 site/TDA/AWBC 准备；有首批接管数据后再进入 recovery 分支。
7. Model Arithmetic 暂缓。它是多个 checkpoint 的权重空间合并，不是运行时模型路由；单任务、单模型阶段收益不高。
8. 2026-06-30 15:28 的执行决策：现在不三选一，而是两条主线并行。现场数据集 v1 立刻启动采集；TDA 增强完成后只做数据校验、norm stats、smoke/probe train，不直接开纯增强 88k full train。
9. `openarms_folding_v001` / `openarms_folding_v002` 不是现场数据集 v1；不要把这些既有 OpenArms 折叠数据目录当作 site 数据或 `site_v1_ft` 的输入。

### 0.2 活跃 HQ 推理服务

```text
node: gpu25
node_ip: 172.31.11.125
gpu: GPU0 / A800 80GB
port: 6666
server_uri: ws://172.31.11.125:6666
config: pi05_openarms_dual_hq
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999
pid: 887483
log: /share/home/linyongjia/output/openpi/logs/serve/openarm_hq_gpu25_6666.log
```

已验证：

```text
healthz: OK
listen: 0.0.0.0:6666
GPU memory: about 69GB / 80GB
metadata: {'action_horizon': 50, 'action_dim': 32, 'rtc_mode': 'off'}
dry-run actual output: actions shape = (50, 16)
local websocket connect: 0.0084s
local stable latency: mean 0.1333s, median 0.1331s, min 0.1299s, max 0.1387s
server infer_ms: about 86-100ms
```

注意：

- `metadata.action_dim=32` 是模型内部动作维度；经过 OpenArm output transform 后实际返回客户端的是 `actions (50, 16)`。
- 默认 `openpi_client.WebsocketClientPolicy` 在首次冷编译阶段可能触发 `websockets` ping timeout；本地稳定测试用 `ping_interval=None` 后正常。客户端应显式关闭 ping interval 或放大 timeout。

### 0.3 当前任务板

| 任务 | 状态 | 当前结论 | 下一步 | 证据 |
|---|---|---|---|---|
| A. HQ baseline 真机推理 | 阶段完成，视觉分布阻塞 | `ws://172.31.11.125:6666` 已跑通；机械臂起身正常；抓取失败指向主摄像头分布偏移 | 做主摄像头分布审计；对齐后复测 FIFO baseline | `/tmp/openarm_remote_policy_20260630_150301.log` |
| B. 客户端 TDA smooth | 阶段完成 | `tda_smooth` 真机可运行；急停链路可用 | 相机对齐后做 FIFO vs TDA A/B | OpenArm commit `cab9865`；IPC tmux `openpi_estop_test` |
| C. TDA 数据增强/重训 | 数据 ready，等 site 冻结后合并 | `openarm_hq_tda_aug_v1` 已生成并验收通过：parquet 2298、mp4 6894、约 62G；16D、time-scaling、mirror 互换和抽样视频帧数检查通过；`norm_stats.json` 与 tiny smoke checkpoint 已产出 | 准备 `hq_tda_site_v1` 合并/过采样脚本和配置；不单独开纯 TDA 88k full train | `/share/home/linyongjia/datasets/openarm_hq_tda_aug_v1`；`openarm_hq_tda_aug_smoke_tiny_20260630/2` |
| D. HIL / DAgger 采集格式 | 客户端补丁完成，暂无接管数据 | HIL mux/record/inspect 已在工控机 targeted build/test 通过；当前还没有推理 HIL 接管 episode | 先准备 inspect/转换和 recovery 数据命名；首批真实接管数据到位后再训练 recovery | OpenArm commit `232af15`；建议冻结名 `openarm_hil_recovery_v1` |
| E. Stage Advantage | 已完成，可用于 AWBC | 200 条标注子集训练完成；`10000` checkpoint 当前最优，val20 上 MSE 0.00295、MAE 0.04340、sign 96.75%、corr 0.9859、R2 0.9715 | 批量预测 HQ/TDA/site 的 advantage，离散化为 AWBC 标签；保留 `4000/6000` 作备份对照 | checkpoint `/share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000`；eval JSON `openarm_stage_v1_step10000_val20_gpu12_b8x100.json` |
| F. Model Arithmetic | 暂缓 | 需要多个互补 checkpoint 后再评估 | 等 HQ/TDA/Recovery/AWBC 至少两个模型可比较后再开 | KAI0 `model_arithmetic/README.md` |
| G. 现场数据集 v1 | P0，录制中 | 现场对齐数据按 150 条目标录制中，视为即将存在；冻结名建议 `openarm_site_align_v1`；`openarms_folding_v001/v002` 不属于 site 数据 | 录制期间先准备 schema 校验、split、norm stats、site-only probe 和混合数据构建 | 待冻结路径；建议 `/share/home/linyongjia/datasets/openarm_site_align_v1` |

### 0.4 数据资产与用途

| 逻辑名 | 建议/实际数据集名 | 状态 | 主要用途 | 不要误用 |
|---|---|---|---|---|
| HQ 原始数据 | `high_quality_folding` | 已有，约 1200 集 | 基础 BC 能力、HQ baseline、合并训练的主干数据 | 不能代表当前现场相机分布 |
| HQ + TDA 增强 | `openarm_hq_tda_aug_v1` | 已生成，2298 集 | 时间扰动、镜像、部署鲁棒性；用于 `hq_tda_site_v1` | 不能替代现场对齐数据 |
| Stage 标注训练集 | `high_quality_folding_v2p1_stage_train180` | 已完成 | 只用于训练 Stage Advantage | 不直接作为主 policy 数据 |
| Stage 标注验证集 | `high_quality_folding_v2p1_stage_val20` | 已完成 | Stage Advantage 离线评估 | 不参与 Stage 训练 |
| 现场对齐数据 | `openarm_site_align_v1` | 录制中，先按 150 条规划 | `site_v1_ft_probe` 与主力 `hq_tda_site_v1` 的关键数据 | 不要和 `openarms_folding_v001/v002` 混淆 |
| HIL 接管数据 | `openarm_hil_recovery_v1` | 暂无 | recovery / Heuristic DAgger 后续分支 | 不阻塞 site/TDA/AWBC 准备 |
| 旧 OpenArms 折叠数据 | `openarms_folding_v001/v002` | 已有 | 仅可作为待审计辅助数据 | 不是现场对齐 v1 |

### 0.5 录制期间可并行推进

| 并行任务 | 负责人类型 | 现在能做什么 | 等待什么 | 验收 |
|---|---|---|---|---|
| Site 数据冻结准备 | 数据 Agent | 写/跑 schema 检查、episode 统计、三路视频抽帧审计、holdout split 规则 | `openarm_site_align_v1` 实际路径冻结 | 150 条时 130/20，200 条时 180/20；16D、camera keys、prompt 全通过 |
| Site-only probe 配置 | 训练 Agent | 准备 `pi05_openarms_dual_site_align_v1` 配置，从 HQ `99999` warm start；准备 norm stats 命令 | site 数据落盘 | 500-1000 step smoke + 小步 probe 可启动 |
| HQ/TDA/site 合并 | 数据/训练 Agent | 准备物理复制或 manifest 合并策略；site 过采样 4-6x，TDA 不压过 site | site split 冻结 | 产出 `openarm_hq_tda_site_v1`，重新生成 norm stats |
| Stage Advantage -> AWBC | Stage Agent | 用 `10000` checkpoint 准备批量预测脚本、advantage 离散化规则、tasks.jsonl 生成 | site 数据可选；HQ/TDA 可先跑 dry-run | 出现 positive/neutral/bad 标签，AWBC smoke 可跑 |
| 真实部署评估基线 | Eval Agent | 固定 FIFO/TDA A/B 记录模板、成功阶段统计、失败分类、相机 metadata | site probe checkpoint | 同一现场布局下 HQ vs site probe 可复测 |
| HIL/Recovery 管线 | HIL Agent | 准备 `openarm_hil_recovery_v1` 命名、inspect、转换和字段验收 | 首批 policy-in-loop 接管 episode | policy/human/executed/intervention 字段齐全 |

## 1. KAI0 对齐原则

参考：

- KAI0 论文：`chi0: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies`，arXiv `2602.09021`，https://arxiv.org/abs/2602.09021
- KAI0 本地项目：`/home/lyj/kai0`
- KAI0 README：`/home/lyj/kai0/README.md`
- TDA：`/home/lyj/kai0/train_deploy_alignment/`
- Stage Advantage：`/home/lyj/kai0/stage_advantage/README.md`

版本核对：

```text
GitHub OpenDriveLab/kai0 main: 9d93078c757840f50e75248c5c5a94ab7b41e13a
local /home/lyj/kai0 HEAD:       9d93078c757840f50e75248c5c5a94ab7b41e13a
result: local kai0 is aligned with current upstream main as of 2026-06-30
```

KAI0 的核心是处理三种分布不一致：

| KAI0 概念 | 大白话 | OpenArm 当前对应 |
|---|---|---|
| `P_train` | 人类示教数据长什么样 | HQ folding 数据集，三路相机 + 16D state/action |
| `Q_model` | 模型学出来的偏好和错误 | HQ checkpoint 在真实运行中的动作习惯 |
| `P_test` | 真机部署时实际看到和执行到的情况 | 现在 IPC 相机、桌面、衣物、控制延迟、chunk 执行 |

第一轮真机反馈说明：当前最明显的问题是 `P_train` 和 `P_test` 的视觉分布不一致，不是单纯训练步数或 chunk 平滑问题。因此后续计划按这个优先级推进。

### 1.1 本项目采用的 KAI0 模块

| 模块 | 是否本阶段执行 | 本项目落地方式 |
|---|---:|---|
| Train-Deploy Alignment / TDA | 是 | 时空数据增强、真实部署 chunk 平滑、后续 recovery 数据 |
| Heuristic DAgger | 是 | 用 OpenArm HIL mux 记录 policy/human/executed/intervention 信号 |
| Stage Advantage | 是 | 建 OpenArm stage schema，标 `stage_progress_gt`，训练 advantage estimator，再做 AWBC |
| Model Arithmetic | 暂缓 | 多个 checkpoint 出来后做权重合并实验，不做 runtime route |

### 1.2 不做模型路由的原因

Model Arithmetic 不是模型路由。它把多个模型 checkpoint 在权重空间合并成一个模型，用于吸收不同数据子集的能力。我们现在是单任务折叠，且只有一个可用 HQ baseline；此时做路由会增加部署复杂度，但没有足够模型候选可路由。

等下面至少出现两个可比较 checkpoint 后，再评估 Model Arithmetic：

```text
HQ baseline
TDA augmented
Recovery finetuned
Stage Advantage / AWBC finetuned
```

## 2. 第一轮回看

### 2.1 已经证明可用的部分

| 链路 | 结论 | 证据 |
|---|---|---|
| gpu25 OpenPI policy server | 可用 | `healthz OK`，`actions (50,16)`，稳定延迟约 130ms |
| OpenArm OpenPI 客户端入口 | 可用 | IPC 成功连接 `ws://172.31.11.125:6666` |
| 16D action transform | 可用，但 metadata 待修 | 输出实际是 16D；metadata 仍写 32 |
| TDA smooth 客户端 | 可用 | `chunk_merge_mode=tda_smooth` 真机运行 |
| 急停链路 | 可用 | `soft_estop_latched`，effort/KP/KD 输出为 0 |
| HIL/DAgger 字段基础 | 可用 | OpenArm commit `232af15`，targeted tests 通过 |
| TDA 增强脚本 | 已完成 | `openarm_hq_tda_aug_v1` 已生成；norm stats 和 tiny smoke train 已通过 |

### 2.2 暴露的问题

| 问题 | 影响 | 当前判断 | 处理优先级 |
|---|---|---|---:|
| 主摄像头与 HQ 数据分布不一致 | 抓取定位失败，展开阶段无法进入 | 当前最大阻塞 | P0 |
| metadata `action_dim=32` | 严格客户端可能误判维度 | 不影响当前 transform 输出，但应修 | P1 |
| WebSocket 首次冷编译 ping timeout | 冷启动首个请求可能断 | 关闭 ping 或增大 timeout | P1 |
| 现场对齐数据仍在录制 | site/TDA 主力训练还不能最终冻结 | 录制期间先准备 schema、split、norm stats、训练配置和 AWBC dry-run | P0 |

### 2.3 第一轮结论

不要把第一轮失败解释成“HQ 模型完全不行”。它已经能在起身到桌面阶段给出合理行为，说明服务、客户端、单位转换、基础策略都不是第一故障点。下一轮应先把相机输入恢复到训练分布，或者明确转向“当前相机分布补采 + 微调/重训”。

## 3. 关键阻塞：主摄像头分布偏移

### 3.1 必须对比的项

相机审计 Agent 需要采当前 IPC 三路图片，并从 HQ 数据集中抽相同任务阶段样例，逐项对比：

```text
base camera:
  camera pose / height / angle
  FOV
  table occupancy
  cloth size in pixels
  gripper visible ratio
  crop / resize path
  exposure / white balance / lighting
  background clutter

left_wrist / right_wrist:
  wrist camera orientation
  image flip or rotation
  gripper visibility
  cloth center location
```

### 3.2 决策规则

| 观察结果 | 决策 |
|---|---|
| 当前相机能物理恢复到 HQ 视角 | 先恢复相机，再复测 HQ FIFO baseline |
| 当前相机不能恢复，但任务场景固定 | 补采当前现场分布数据；先按 150 条启动，扩到 200 条更稳 |
| 当前相机经常变化 | 必须把相机扰动纳入 TDA 增强和数据采集规范 |

### 3.3 验收产物

```text
docs/runs/openarm_camera_alignment_<date>.md
当前三路相机样例路径
HQ 数据集对照样例路径
对比图或拼图
结论：restore camera / collect current distribution / both
复测命令和复测结果
```

## 4. 下一轮任务板

### A. 相机分布审计与 baseline 复测

目标：先消除主摄像头分布偏移，再判断 HQ 模型真实能力。

步骤：

1. 在 IPC 采集当前 `base / left_wrist / right_wrist` 原始图和模型 resize 后图。
2. 从 HQ 数据集中抽取同阶段样例，做 side-by-side 对比。
3. 如果能恢复相机，先恢复硬件/裁剪/曝光。
4. 用 FIFO 模式复测 HQ baseline。
5. 再用当前 TDA smooth 参数复测。

验收：

```text
camera_alignment_report 存在
复测时 /openarm/joint_target 仍为单 writer
记录 FIFO 与 TDA 的成功/失败阶段
记录 action range、latency、publish hz
```

### G. 现场数据集 v1 采集

目标：如果现场相机型号、安装位、桌面布局和 HQ 数据集长期不同，就补一批当前现场分布数据，把模型训练分布拉到真实部署分布。

命名澄清：

```text
openarms_folding_v001 / openarms_folding_v002: 既有 OpenArms 折叠数据，不是现场数据集 v1
现场数据集 v1: openarm_site_align_v1；冻结后必须使用单独、明确的 dataset_id/path
```

建议规模：

```text
target_total_v0: about 150 episodes
train_v0: about 130 episodes
val/holdout_v0: about 20 episodes
target_total_v1_optional: about 200 episodes
train_v1_optional: about 180 episodes
val/holdout_v1_optional: about 20 episodes
```

建议组成：

```text
130 条左右: 标准现场布局下的成功完整折叠示教，作为 v0 train
20 条左右: 离散 holdout，只评估不训练
可选扩展 50 条左右: 受控变化或 policy-in-the-loop 接管恢复片段，进入 v1/recovery 分支
```

失败-only episode 可以采，但第一版不混入 BC 正样本；先作为诊断、Stage Advantage 负例候选或 evaluation set。

采集前必须冻结：

```text
camera model
camera mount / pose
resolution
crop / resize path
exposure / white balance
table layout
task prompt
state/action 16D 顺序
degree/rad 转换边界
```

推荐数据集名：

```text
/share/home/linyongjia/datasets/openarm_site_align_v1
```

验收：

```text
三路视频可读
state/action 均为 16D
150 条时 130/20 split 明确；扩到 200 条时 180/20 split 明确
记录现场 camera/layout metadata
至少 20 条 holdout 不参与训练
能与 HQ/TDA 数据 merge 或按权重采样训练
```

### B. 客户端部署卫生修复

目标：把第一轮发现的非核心但会反复踩坑的问题修掉。

待做：

1. 在 OpenArm WebSocket client 中设置 `ping_interval=None` 或足够大的 `ping_timeout`，避免 JAX 冷编译导致首次请求断开。
2. 修正或规避 metadata `action_dim=32` 与实际 `actions (50,16)` 的不一致。
3. 保持 `fifo` 为默认安全模式，`tda_smooth` 通过参数显式启用。
4. TDA trace 指标继续输出：

```text
infer_ms
queue_size
remote_fetch_count
drop_count
blend_length
max_abs_action_delta
```

验收：

```text
冷启动首次请求不因 ping timeout 断开
客户端严格校验时不会把 16D 输出误判为 32D
fifo 和 tda_smooth 可切换
targeted unit tests 通过
```

### C. TDA 增强数据完成与重训

目标：完成 `openarm_hq_tda_aug_v1`，重算 norm stats，先把 TDA 数据作为 site 合并训练输入准备好。

当前状态：

```text
node: gpu28
tmux: openarm_tda_aug_20260630 (completed)
dataset_src: /share/home/linyongjia/datasets/high_quality_folding
dataset_dst: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1
log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/augment_20260630_gpu28.log
parquet_done: 2298
video_target: 6894
encoder: libx264
decode: cuda decode where available
config: pi05_openarms_dual_hq_tda_aug
warm_start: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999/params
norm_log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/norm_20260630_gpu28.log
smoke_log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/smoke_tiny_20260630_gpu28.log
```

完成后必须检查：

```text
manifest.yaml
augment_report.json
meta/info.json
meta/episodes.jsonl
三路视频数量
state shape == 16
action shape == 16
right/left 8D mirror 顺序
gripper 只互换，不做 rad/degree 转换
```

训练顺序：

1. 增强完成后重新生成 OpenArm state/action norm stats。
2. 先跑 500-1000 step smoke train。
3. 纯 TDA full train 暂缓，不再作为当前主线。
4. 等 `openarm_site_align_v1` 冻结后，构建 `hq_tda_site_v1`，现场数据按 4-6x 采样权重进入主力训练。
5. 主力 checkpoint 输出后，与 HQ baseline 做同场景复测。

2026-06-30 smoke 结果：

```text
norm_stats: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1/norm_stats.json
norm_stats_check: state/actions mean/std/q01/q99 are all 16D
norm_script: scripts/compute_openarm_parquet_norm_stats.py
probe_exp: openarm_hq_tda_aug_smoke_20260630
probe_result: 500-step probe entered training and logged step 0, then stopped to avoid long decode/checkpoint cost
tiny_exp: openarm_hq_tda_aug_smoke_tiny_20260630
tiny_result: 3/3 steps completed; metrics logged for steps 0,1,2; checkpoint finalized at step 2
tiny_checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq_tda_aug/openarm_hq_tda_aug_smoke_tiny_20260630/2
tiny_metrics: loss step0=0.122777, step1=0.165189, step2=0.129028
decision: do not launch pure TDA 88000-step full train before site dataset / camera strategy is frozen
```

禁止：

- 不覆盖原 HQ 数据目录。
- 不把 degree 语义静默改成 radians。
- 不重新引入 `[-pi, pi]` 过滤删除 degree 样本。

### D. Real HIL / Heuristic DAgger 短 episode

目标：确认真实数据能支持 recovery 和 DAgger，而不是只支持普通 BC。

已经完成：

```text
OpenArm commit: 232af15
package: openarm_hil_rl
tools: openarm-hil-mux / openarm-hil-record / openarm-hil-inspect
format: openarm_hil_raw_hdf5_v3
entry: start_real_hil_dagger_openpi.sh --openpi-mode fifo|tda_smooth
targeted build/test: pass
fake HDF5 inspect: pass
```

真实 episode 需要包含：

```text
policy_action
human_action 或 teleop_action
executed_action
is_intervention
authority_source
intervention_start / intervention_end
failure_mode
success 或 outcome
policy_checkpoint
prompt
timestamp
camera frame ids
latency / queue metadata
```

验收：

```text
1 条真实短 episode 可读
openarm-hil-inspect 通过
能计算 human_action - policy_action
能筛选 recovery segment
```

### E. Stage Advantage v1

目标：建立 OpenArm 折叠任务的最小 stage schema，并产出可训练的 `stage_progress_gt`。

KAI0 原流程：

```text
Step 0: annotate stage_progress_gt
Step 1: train Advantage Estimator
Step 2: predict absolute_advantage / relative_advantage
Step 3: discretize advantage into task_index / tasks.jsonl
Step 4: AWBC training
```

OpenArm SA v1 stage taxonomy：

| stage_id | 名称 | 判定标准 |
|---:|---|---|
| 0 | flatten | 从 episode 起点到衣物被展开/拉平/对齐，已经进入可折叠状态 |
| 1 | fold | 从第一次明确折叠动作开始，到松爪和最终状态稳定 |

说明：

```text
此前文档中的 5 阶段 approach_grasp / spread_flatten / align / fold / release_finish
是 OpenArm 诊断拆分，不是 KAI0 论文的固定 Stage Advantage taxonomy。

SA v1 训练只使用 2 阶段，和 KAI0 Task A flatten-fold 思路对齐。
5 个细粒度事件可以作为可选诊断标签保留，但不要写入第一版 stage_progress_gt。
```

如果后续发现 2 阶段在 OpenArm 上不足，再升级为 3 阶段：

```text
grasp_flatten -> align -> fold_finish
```

但第一版不要直接上 5 阶段；阶段越多，人工边界越难一致，Advantage Estimator 的监督也更容易噪。

推荐先用 sidecar JSONL，不直接手改 parquet：

```json
{
  "dataset": "openarm_hq_v1",
  "episode_index": 12,
  "fps": 30,
  "task": "fold the cloth",
  "quality": "success",
  "stage_boundaries": [
    {"stage_id": 0, "name": "flatten", "start_frame": 0, "end_frame": 180},
    {"stage_id": 1, "name": "fold", "start_frame": 181, "end_frame": 380}
  ],
  "events": [
    {"name": "first_contact", "frame": 42},
    {"name": "flatten_done", "frame": 180},
    {"name": "fold_start", "frame": 181},
    {"name": "release", "frame": 350}
  ],
  "notes": ""
}
```

`stage_progress_gt` 生成规则：

```text
stage 0 = episode_start 到 fold_start - 1
stage 1 = fold_start 到 episode_end
Task A 两阶段最小人工标注只需要 flatten_done 一个分界点
fold_start = flatten_done + 1，由工具自动生成
stage_progress_gt = k / K + (1 / K) * frame_position_within_stage / segment_length
range: [0, 1]
monotonic: true within an episode
```

第一批建议：

```text
20-50 条 HQ 成功 episode: schema smoke
50-100 条 HQ 成功 episode: advantage estimator v1
10-20 条 recovery 成功 episode: 第二批加入
失败 episode: 先用于分析，不混入 AWBC 正样本
```

标注工具要求：

```text
三路视频同步显示：base / left_wrist / right_wrist
按键标注：flatten_done、quality、notes
输出 sidecar JSONL，不直接改 parquet
自动推断：start=0, fold_start=flatten_done+1, end=episode_length-1
自动生成 stage_progress_gt，并抽样回放检查
```

验收：

```text
stage_schema.md
annotation sidecar JSONL
轻量标注工具或脚本
stage_progress_gt 转换脚本或转换报告
至少 20 条 episode smoke 标注
stage_progress_gt 单调且范围 [0,1]
Advantage Estimator smoke train 可启动
AWBC 前 tasks.jsonl 中存在 "fold the cloth, Advantage: positive"
```

当前已落地工具：

```text
subset tool: scripts/subset_lerobot_v21.py
annotation server: scripts/openarm_stage_annotator.py
stage_progress writer: scripts/openarm_stage_progress.py
tests: scripts/openarm_stage_progress_test.py
tests: scripts/subset_lerobot_v21_test.py
```

当前本地 200 集标注子集：

```text
dataset: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_200
source: /share/home/linyongjia/datasets/high_quality_folding
format: LeRobot v2.1
episodes: 200
frames: 470491
parquet: 200
videos: 600
split: train180 / val20, val uses discrete holdout episodes 9,19,...,199 from original 200 episodes
camera keys: observation.images.base / observation.images.left_wrist / observation.images.right_wrist
state/action: 16D
```

当前训练/验证子集：

```text
train: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_stage_train180
val:   /home/lyj/storage1t/datasets/high_quality_folding_v2p1_stage_val20
remote train: /share/home/linyongjia/data/high_quality_folding_v2p1_stage_train180
remote val:   /share/home/linyongjia/data/high_quality_folding_v2p1_stage_val20
train episodes: 180, frames: 417933
val episodes: 20, frames: 52558
val selection: every 10th original episode's last item, [9, 19, ..., 199]
```

本地标注命令：

```bash
conda run -n lerobot-pi0 python scripts/openarm_stage_annotator.py \
  --dataset /home/lyj/storage1t/datasets/high_quality_folding_v2p1_200 \
  --port 8765
```

标注输出：

```text
/home/lyj/storage1t/datasets/high_quality_folding_v2p1_200/annotations/openarm_stage_v1.jsonl
```

生成 `stage_progress_gt`：

```bash
conda run -n lerobot-pi0 python scripts/openarm_stage_progress.py \
  --dataset /home/lyj/storage1t/datasets/high_quality_folding_v2p1_200 \
  --dry-run
```

验收后去掉 `--dry-run` 写回 parquet。写回后脚本会在 `meta/info.json` 增加：

```text
stage_progress_gt: float32[1]
stage_id: int64[1]
```

当前 Stage Advantage 训练配置：

```text
config: ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD
model: AdvantageEstimatorConfig(pi05=True, discrete_state_input=False)
loss: loss_action_weight=0.0, loss_value_weight=1.0
init: /share/home/linyongjia/data/pi05_base/model.safetensors
input images: base/left_wrist/right_wrist current frame + same-episode random history frame
label: progress = current stage_progress_gt - history stage_progress_gt
skip_norm_stats: true, matching KAI0 Stage Advantage recipe
```

当前远端正式训练结果：

```text
node: gpu28
tmux: openarm_stage_v1_bs32_no_ckpt_20260701
exp: openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701
log: /share/home/linyongjia/output/openpi/logs/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701.log
checkpoint dir: /share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701
launch: torchrun --nproc_per_node=2, batch_size=32, num_train_steps=10000, pytorch_gradient_checkpointing=false
observed memory: about 64.5GB nvidia-smi / 66.9GB torch reserved per A800
observed stable speed: about 4.34 sec/step after warmup, ETA about 12 hours from 2026-07-01 12:06 CST
best checkpoint: 10000
eval: val20 / 800 paired-frame, MSE 0.00295, MAE 0.04340, sign accuracy 96.75%, corr 0.9859, R2 0.9715
eval json: /share/home/linyongjia/output/openpi/eval/stage_advantage/openarm_stage_v1_step10000_val20_gpu12_b8x100.json
decision: Stage Advantage v1 is usable; next step is advantage prediction + AWBC, not longer Stage training
```

环境说明：

```text
本机当前没有 pi-conda 环境；本轮本地校验使用已有 conda env lerobot-pi0。
后续如需与远端训练完全一致，应在本机补建 pi-conda 后重跑同一组测试。
不要使用 uv。
```

### F. Model Arithmetic 后置实验

开启条件：

```text
至少两个 checkpoint 在同一相机分布下复测过
每个 checkpoint 有清楚的数据来源和训练配置
至少一个 checkpoint 与 HQ baseline 行为互补
```

候选：

```text
HQ baseline
TDA augmented
Recovery finetuned
Stage Advantage / AWBC finetuned
```

验收：

```text
权重合并脚本可复现
合并前后 checkpoint 都保留
同一真机场景 A/B/C 测试
不引入运行时多模型路由
```

## 5. 训练与数据路线

### 5.1 当前 HQ checkpoint 的定位

```text
/share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999
```

用途：

- 第一阶段 baseline 推理。
- TDA smooth 客户端测试。
- TDA augmented 训练 warm start。
- 后续推理平滑、延迟和安全链路回归测试。

禁止：

- 不覆盖。
- 不重命名为新实验结果。
- 不把它当作已经吸收当前相机分布的模型。

### 5.2 OpenArm 数据接口冻结

相机 key：

```text
observation.images.base        -> base_0_rgb
observation.images.left_wrist  -> left_wrist_0_rgb
observation.images.right_wrist -> right_wrist_0_rgb
```

State/action 统一 16D：

```text
right_joint_1
right_joint_2
right_joint_3
right_joint_4
right_joint_5
right_joint_6
right_joint_7
right_gripper
left_joint_1
left_joint_2
left_joint_3
left_joint_4
left_joint_5
left_joint_6
left_joint_7
left_gripper
```

单位：

```text
model side: joint degree 语义，沿用 HQ checkpoint
robot side: ROS/controller rad + gripper normalized
gripper: 单独处理，不参与 degree/rad 统一变换
```

Norm stats：

```text
每个新数据集必须重新生成 norm_stats.json
state 和 actions 都必须是 16D
不得用 [-pi, pi] 过滤误删 degree 样本
```

### 5.3 现场对齐数据训练策略

你的判断是合理的：如果现场相机型号、视角、桌面和 HQ 数据集不同，现场对齐数据不是“锦上添花”，而是把 `P_test` 拉回训练分布的关键步骤。

当前现场对齐数据还在录制中，按 150 条先规划；如果后续扩到约 200 条，只需要把 split 从 130/20 改成 180/20。推荐先做一个小而干净的数据集，而不是一上来追求数量。

```text
dataset: /share/home/linyongjia/datasets/openarm_site_align_v1
target_total_v0: about 150 episodes
train_v0: about 130 episodes
val/holdout_v0: about 20 episodes
target_total_v1_optional: about 200 episodes
train_v1_optional: about 180 episodes
val/holdout_v1_optional: about 20 episodes
prompt: fold the cloth
camera keys: base / left_wrist / right_wrist
state/action: 16D
unit: model side degree, robot side rad/gripper normalized
```

训练组合建议：

| 版本 | 数据 | 目的 |
|---|---|---|
| `hq_baseline` | 原 HQ | 保留对照，不覆盖 |
| `hq_tda_aug` | HQ + time scaling / mirror | 提升时空鲁棒性，但不一定覆盖现场相机型号差异 |
| `site_v1_ft_probe` | HQ checkpoint warm start + 现场约 130 train | 快速验证现场分布是否解决抓取问题 |
| `hq_tda_site_v1` | HQ + TDA augmented + site oversampling | 第一版主力候选，现场数据建议 4-6x 采样权重 |
| `awbc_v1` | HQ/TDA/site + Stage Advantage 标签 | 用 `10000` Stage 打分模型引入 advantage 条件，提升阶段内动作选择 |
| `recovery_v1` | policy-in-loop HIL 接管片段 | 后续补失败恢复能力，不阻塞当前 site/TDA/AWBC |

推荐顺序：

1. 现场录制期间先准备检查脚本和配置，不等数据全部完成。
2. 数据冻结后先固定 20 条 holdout，不参与任何训练。
3. 用 HQ checkpoint warm start 跑 `site_v1_ft_probe`，优先 500-1000 step smoke，再跑短 probe，验证现场抓取是否明显改善。
4. 再跑 `hq_tda_site_v1`，把 HQ、TDA 增强数据和现场数据合并，现场数据过采样 4-6x，避免被 1200/2298 集淹没。
5. 对合并数据重新生成 norm stats；不要复用 HQ、TDA-only 或 site-only 的 norm stats。
6. 用 Stage Advantage `10000` checkpoint 对 HQ/TDA/site 批量预测 advantage，先 dry-run 离散化，再跑 AWBC smoke。
7. HIL 接管数据到位后另开 `recovery_v1`，不要让尚未存在的 HIL 数据阻塞 site/TDA 主线。

验收：

```text
150 条左右现场 episode 可读；若扩到 200 条则更新 split
至少 20 条 holdout 固定
现场 camera/layout metadata 完整
norm_stats.json 重新生成且 state/action 为 16D
site_v1_ft_probe smoke train 通过
同一现场布局下 HQ baseline vs site_v1_ft_probe 有复测对比
Stage Advantage 10000 可批量预测 advantage 并生成 AWBC 标签
```

## 6. Gate 验收

| Gate | 名称 | 通过条件 |
|---:|---|---|
| 0 | Camera distribution decision | 有当前相机 vs HQ 对比报告，并明确 restore / collect / both |
| 1 | Site dataset decision | 现场对齐数据按 `openarm_site_align_v1` 冻结；150 条先跑，扩到 200 条更稳 |
| 2 | Client deployment hygiene | 冷启动不被 ping timeout 断开；metadata/shape 校验不误判 |
| 3 | FIFO baseline retest | 相机对齐或现场采集规范冻结后，HQ FIFO 低速真机日志完整 |
| 4 | Site dataset v1 freeze | 约 150 条现场 episode 可读，20 条 holdout 固定，metadata 完整；若扩到 200 条则同步更新 split |
| 5 | TDA augmented data freeze | 数据、视频、manifest、16D、norm stats 全部通过；tiny smoke checkpoint 已产出 |
| 6 | Site finetune smoke | `site_v1_ft_probe` smoke/probe train 通过，并与 HQ baseline 做现场复测 |
| 7 | TDA/site full train | `hq_tda_site_v1` 或同等主力候选输出 checkpoint |
| 8 | TDA A/B retest | 同场景下 FIFO vs TDA smooth 对比完成 |
| 9 | Real HIL episode | 真实短 episode 可 inspect，policy/human/executed/intervention 齐全 |
| 10 | Stage Advantage v1 | `10000` checkpoint 通过 val20 评估，并作为 AWBC 打分模型 |
| 11 | AWBC v1 | advantage 预测、discretize、AWBC smoke train 闭环 |

## 7. Agent 回写规范

每个 Agent 只允许更新两个区域：

1. `0.3 当前任务板` 中自己的任务行。
2. `8. 历史日志` 末尾追加一条记录。

不要在任务章节中间插入临时进度段落。需要新增事实时，先判断它属于：

```text
当前状态 -> 0.3 当前任务板
稳定接口 -> 5.2 OpenArm 数据接口冻结
阶段验收 -> 6. Gate 验收
过程记录 -> 8. 历史日志
```

日志模板：

```text
### YYYY-MM-DD HH:MM CST - Agent X - 标题

状态:
已完成:
证据:
阻塞:
下一步:
需要用户/其他 Agent:
```

证据必须是可复查对象，例如：

```text
commit hash
log path
tmux session
dataset path
checkpoint path
report path
test output summary
```

## 8. 历史日志

### 2026-07-02 11:05 CST - Plan Owner - KAI0 适配计划按 Stage/site 状态重排

状态：完成。

已完成：

- 将当前路线从“纯 TDA full train / 现场采集二选一”收敛为 `site_v1_ft_probe` -> `hq_tda_site_v1` -> `AWBC_v1`。
- 明确现场对齐数据冻结名为 `openarm_site_align_v1`；150 条时采用 130 train + 20 holdout，扩到 200 条时采用 180 train + 20 holdout。
- 将 Stage Advantage v1 标记为已完成，`10000` checkpoint 为当前最优，可进入批量 advantage 预测和 AWBC。
- 把 TDA 增强数据定位为合并训练输入，不再直接把纯增强 88000 step full train 作为当前主线。
- 增加数据资产表和“录制期间可并行推进”任务表，给数据、训练、Stage、评估和 HIL Agent 拆分边界。

证据：

```text
stage checkpoint: /share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701/10000
stage eval json: /share/home/linyongjia/output/openpi/eval/stage_advantage/openarm_stage_v1_step10000_val20_gpu12_b8x100.json
stage eval: MSE 0.00295, MAE 0.04340, sign accuracy 96.75%, corr 0.9859, R2 0.9715
tda dataset: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1
site dataset target: /share/home/linyongjia/datasets/openarm_site_align_v1
```

阻塞：

- 现场对齐数据仍在录制，主力 policy train 还不能最终冻结输入。
- 真实 HIL 接管数据还没有进入训练集，`recovery_v1` 暂不阻塞当前 site/TDA/AWBC。

下一步：

- 数据 Agent 先做 `openarm_site_align_v1` schema/视频/16D/split/norm stats 验收脚本。
- 训练 Agent 准备 `site_v1_ft_probe` 和 `hq_tda_site_v1` 配置，等 site 数据落盘即可启动 smoke。
- Stage Agent 用 `10000` checkpoint 先对 HQ/TDA dry-run 预测 advantage，并准备 AWBC 标签离散化。
- Eval Agent 固定 HQ baseline、site probe、TDA smooth 的同场景复测模板。

### 2026-07-01 12:06 CST - Agent E - Stage Advantage batch 调参与正式训练重启

状态：训练中。

结论：

- 原 `batch_size=16`、每卡 8、梯度检查点开启的 run 只占约 34GB/卡，稳定约 3.1 秒/步，20k steps ETA 约 17 小时，偏保守。
- 新增 `TrainConfig.pytorch_gradient_checkpointing` 开关，默认保持开启；本次正式 run 显式传 `--no-pytorch-gradient-checkpointing`。
- 短测结果：
  - `batch32 + checkpoint`: 约 41GB/卡，约 5.35 秒/步，吞吐约 6.0 samples/s。
  - `batch32 + no checkpoint`: 约 66.9GB/卡，约 4.6-5.0 秒/步，吞吐约 6.4-7.0 samples/s。
  - `batch64 + checkpoint`: 约 55GB/卡，但 step20 窗口出现约 137.6 秒/10步，吞吐不优。
- 正式训练改为 `batch_size=32`、`num_train_steps=10000`，样本数与旧 `batch16 x 20000` 同量级，优化后预计约 12 小时级别。

证据：

```text
tmux: gpu28 openarm_stage_v1_bs32_no_ckpt_20260701
log: /share/home/linyongjia/output/openpi/logs/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701.log
checkpoint dir: /share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_bs32_no_ckpt_10k_20260701
launch: torchrun --nproc_per_node=2, batch_size=32, num_train_steps=10000, --no-pytorch-gradient-checkpointing
step20: loss=0.1695, lr=2.87e-07, grad_norm=0.52, log window time=86.2s
observed GPU memory: about 64.5GB / 80GB per A800 by nvidia-smi, torch reserved peak about 66.9GB
```

下一步：

- 等 step1000 checkpoint 产出。
- 用 val20 做 Stage Advantage v1 评估：`stage_progress_gt` 拟合、pairwise progress sign accuracy、按 episode 的进度曲线平滑性。
- 评估通过后进入 KAI0 Step 2/3：预测 advantage，离散化到 AWBC 所需任务标签。

### 2026-07-01 11:19 CST - Agent E - Stage Advantage 200 条训练启动

状态：训练中。

已完成：

- 200 条 OpenArm HQ 子集标注已完成，并用 `--quality all` 写回 `stage_progress_gt` / `stage_id`。
- 离散抽出 20 条验证集，验证集为原始 200 条中的 `[9, 19, ..., 199]`；其余 180 条作为训练集。
- 训练/验证子集已同步到 gpu28。
- 将 KAI0 Stage Advantage 最小训练链路移植到当前 OpenPI：
  - `AdvantageLeRobotDataset`
  - `AdvantageEstimatorConfig`
  - `AdvantageEstimator(PI0Pytorch)`
  - `ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD`
- gpu28 远端 1 batch 数据链路通过，输入包含当前三相机 + 同 episode 随机历史三相机，`progress` 标签可进入模型。
- gpu28 1 step smoke 通过，`/share/home/linyongjia/data/pi05_base/model.safetensors` 可用作初始化。
- gpu28 两卡 DDP 正式训练已启动。

证据：

```text
train dataset local: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_stage_train180
val dataset local: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_stage_val20
train dataset remote: /share/home/linyongjia/data/high_quality_folding_v2p1_stage_train180
val dataset remote: /share/home/linyongjia/data/high_quality_folding_v2p1_stage_val20
config: ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD
tmux: gpu28 openarm_stage_v1_20260701
log: /share/home/linyongjia/output/openpi/logs/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_20260701.log
checkpoint dir: /share/home/linyongjia/output/openpi/ADVANTAGE_TORCH_OPENARM_FLATTEN_FOLD/openarm_stage_v1_train180_20260701
launch: torchrun --nproc_per_node=2, batch_size=16, num_train_steps=20000
observed speed: about 3.1 sec/step after warmup, ETA about 17 hours
observed memory: about 34GB / 80GB per A800 after warmup
```

下一步：

- 等 step1000 checkpoint 产出。
- 用 val20 做 Stage Advantage v1 评估：`stage_progress_gt` 拟合、pairwise progress sign accuracy、按 episode 的进度曲线平滑性。
- 评估通过后进入 KAI0 Step 2/3：预测 advantage，离散化到 AWBC 所需任务标签。

### 2026-06-30 20:54 CST - Agent E - Stage Advantage 改为单点标注

状态：完成工具修正，20 条已标注数据 dry-run 通过。

已完成：

- 确认当前 200 集 HQ 子集是精修数据：第一帧就是开始，最后一帧就是结束。
- 将人工标注从 4 个时间点收敛为 Task A 两阶段所需的单点标注：只点 `flatten_done`。
- `start=0`、`fold_start=flatten_done+1`、`end=episode_length-1` 由工具自动生成。
- 兼容已保存的 20 条旧格式标注，不需要重标。

证据：

```text
annotation file: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_200/annotations/openarm_stage_v1.jsonl
count: 20
episodes: 0-19
quality: success 20
dry-run: processed_count 20, skipped_count 0
stage_progress range: [0, 1]
```

下一步：

- 用单点 UI 继续扩标。
- 若先做 smoke，可直接对这 20 条去掉 `--dry-run` 写回 `stage_progress_gt`。

### 2026-06-30 16:54 CST - Agent E - Stage Advantage 标注工具和 200 集子集

状态：工具完成，进入人工标注前 smoke 阶段。

已完成：

- 核对 KAI0 upstream，`OpenDriveLab/kai0` main 与本地 `/home/lyj/kai0` 都是 `9d93078c757840f50e75248c5c5a94ab7b41e13a`，没有发现新的上游 Stage Advantage 标注工具需要同步。
- 从远端 HQ v2.1 数据集拉取前 200 集相关文件到本地，不做全量拉取。
- 生成规范本地子集 `/home/lyj/storage1t/datasets/high_quality_folding_v2p1_200`，保留三路视频和 16D state/action。
- 新增 LeRobot v2.1 子集工具、三路视频 Stage Advantage 标注服务、sidecar 到 `stage_progress_gt` 写回脚本和对应测试。

证据：

```text
source dataset: /share/home/linyongjia/datasets/high_quality_folding
local dataset: /home/lyj/storage1t/datasets/high_quality_folding_v2p1_200
total_episodes: 200
total_frames: 470491
parquet: 200
videos: 600
split: train 0:180, val 180:200
scripts/subset_lerobot_v21.py
scripts/openarm_stage_annotator.py
scripts/openarm_stage_progress.py
```

下一步：

- 用标注服务先标 20 条成功 episode，检查边界一致性。
- 用 `scripts/openarm_stage_progress.py --dry-run` 验证单调性和范围。
- 通过后去掉 `--dry-run` 写回 parquet，再启动 Advantage Estimator smoke train。

### 2026-06-30 15:42 CST - Plan Owner - 修正 Stage Advantage v1 阶段划分

状态：完成。

已完成：

- 明确此前 5 阶段划分是 OpenArm 诊断拆分，不是 KAI0 论文原始 Stage Advantage taxonomy。
- 将 SA v1 训练阶段改为论文 Task A 思路对齐的 2 阶段：`flatten` 和 `fold`。
- 保留 `first_contact`、`flatten_done`、`fold_start`、`release` 等细粒度事件作为诊断标签，但不进入第一版 `stage_progress_gt`。
- 增加轻量标注工具要求：三路视频同步、按键标边界、输出 sidecar JSONL、自动校验和自动生成 `stage_progress_gt`。

判断：

- 第一版 SA 不应直接使用 5 阶段。阶段越细，人工标注一致性越差，Advantage Estimator 的监督噪声越高。
- 对 OpenArm 折叠任务，先用 2 阶段足够跑通 KAI0 的 Stage Advantage 闭环；后续只有在 2 阶段无法区分关键失败模式时，再升级为 3 阶段。

下一步：

- 做一个轻量三路视频标注工具，先标 20-50 条成功 episode。
- 从 sidecar JSONL 生成 parquet 里的 `stage_progress_gt`，并用脚本验证单调性和范围。

### 2026-06-30 15:28 CST - Plan Owner - 下一步执行决策

状态：完成。

已完成：

- 确认下一步不是“增强训练”和“现场采集”二选一，而是并行推进。
- 现场数据集 v1 作为 P0 立刻启动：先采 20 条 smoke 验证格式，再扩到约 200 条。
- TDA 增强数据完成后只进入校验、norm stats、500-1000 step smoke 和短 probe train；暂不直接启动纯增强 88000 step full train。
- 纯增强模型只作为 ablation/probe，不作为下一阶段主力模型。主力候选应是 `site_v1_ft` 或 `hq_tda_site_v1`。

远端状态：

```text
node: gpu28
process: augment_openarm_hq_tda.py still running
dataset: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1
parquet: 2298
videos: 5011 / 6894
augment_report.json: missing
log file: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/augment_20260630_gpu28.log
```

判断：

- 现场 200 条数据耗时几天，必须现在启动，否则会拖住真正能解决相机/布局分布偏移的训练。
- 增强数据还没完成，即使完成也主要验证 TDA pipeline 和训练管线；它不能替代现场相机型号、安装位和桌面布局的数据。

下一步：

- 采集 Agent：冻结现场 camera/layout，先采 20 条 smoke episode。
- 数据 Agent：等 gpu28 增强完成后检查 report/video/16D，然后重算 norm stats。
- 训练 Agent：先准备 `site_v1_ft` 配置和数据合并策略；等现场数据 smoke 通过后再开始小微调。

### 2026-06-30 15:20 CST - Plan Owner - 纳入现场 200 条数据策略

状态：完成。

已完成：

- 将“现场布局和相机型号不同，需要补约 200 条现场数据”的判断写入计划。
- 新增 `G. 现场数据集 v1` 任务，明确 180 train + 20 holdout 的第一版目标。
- 增加 `5.3 现场数据集 v1 训练策略`，把训练路线拆成 `site_v1_ft` 和 `hq_tda_site_v1` 两步。
- 更新 Gate，把现场数据决策和现场数据冻结放到 TDA/site full train 之前。

判断：

- 如果现场相机/布局长期不同，现场 200 条数据是必要投入；只靠 TDA 平滑或已有 HQ 数据增强，很可能无法稳定解决抓取定位偏移。

下一步：

- 相机 Agent 先冻结现场相机、crop、曝光和桌面布局。
- 采集 Agent 先采 20 条现场 smoke 数据，确认格式正确后再扩到约 200 条。
- 训练 Agent 等现场数据集 v1 冻结后，优先从 HQ checkpoint warm start 做 `site_v1_ft`。

### 2026-06-30 15:17 CST - Plan Owner - 第一轮结果整理

状态：完成。

已完成：

- 把多 Agent 分散更新整理为统一计划结构。
- 明确当前主阻塞是 `base` 主摄像头分布偏移。
- 将后续任务重排为相机审计、客户端卫生、TDA 增强、真实 HIL、Stage Advantage、Model Arithmetic 后置。
- 增加统一回写规范，避免后续进度写法继续发散。

证据：

```text
docs/openarm_kai0_reproduction_plan.md
KAI0 paper: https://arxiv.org/abs/2602.09021
KAI0 local README: /home/lyj/kai0/README.md
KAI0 Stage Advantage README: /home/lyj/kai0/stage_advantage/README.md
```

下一步：

- 相机 Agent 先产出 `openarm_camera_alignment_<date>.md`。
- 客户端 Agent 修 ping timeout 和 metadata/action_dim 校验。
- 数据 Agent 等 gpu28 增强完成后回填 manifest、norm stats 和 smoke train。

### 2026-06-30 15:10 CST - Agent A/B - OpenPI HQ + TDA 真机首轮推理反馈

状态：阶段性完成，阻塞于主摄像头分布偏移。

已完成：

- IPC 端 OpenArm OpenPI 推理入口连接 `ws://172.31.11.125:6666` 成功。
- `remote_policy` 是 `/openarm/joint_target` 唯一 publisher。
- 实际运行参数：

```text
policy_type: openpi
action_unit: degrees
fps: 30
chunk_merge_mode: tda_smooth
prefetch_threshold: 12
tda_drop_max: 12
tda_min_overlap: 3
tda_blend_mode: linear
tda_blend_alpha: 0.5
```

- 急停路径已验证：`e+Enter` 触发 `/openarm/safety/set_soft_estop`，`/openarm/safety/state` 进入 `soft_estop_latched`，左右 `mode_state` 显示 `soft_estop_engaged=true`，输出 effort/KP/KD 为 0。
- 现场反馈：机械臂从启动起身到桌面阶段表现好。
- 当前失败点：主摄像头与 HQ 训练数据集分布差异较大，导致夹爪无法准确夹到目标，未完成第一阶段展开。

证据：

```text
IPC tmux: openpi_estop_test
policy log: /tmp/openarm_remote_policy_20260630_150301.log
server_uri: ws://172.31.11.125:6666
OpenArm TDA commit: cab9865
```

下一步：

- 采当前 IPC 三路相机样例图，与 HQ 数据集样例逐项比对。
- 主摄像头对齐后，复测 FIFO baseline 与 TDA smooth。

### 2026-06-30 14:54 CST - Agent D - HIL/DAgger 客户端远端验证

状态：客户端字段补丁完成。

已完成：

- OpenArm 客户端提交：`232af15 新增OpenPI HIL采集客户端`。
- 工控机 `/home/test/openarm_ros2_docker` 已快进到 `232af15f`。
- 工控机容器 targeted build 通过：

```text
openarm_hil_rl
openarm_remote_policy
openarm_bringup
```

- targeted tests 通过：

```text
openarm_hil_rl: 5 tests, 0 failures
openarm_remote_policy: 6 tests, 0 failures
scripts/test_runtime_guard.sh: pass
```

- 入口检查通过：

```text
openarm.bimanual.launch.py --show-args includes mink_joint_target_topic
openarm-hil-inspect --help
start_real_hil_dagger_openpi.sh --help
```

下一步：

- 现场停止当前推理后，运行 `start_real_hil_dagger_openpi.sh --openpi-mode fifo` 或 `--openpi-mode tda_smooth` 录真实短 episode。
- 用 `openarm-hil-inspect` 记录字段报告。

### 2026-06-30 14:47 CST - Agent D - HIL/DAgger 客户端字段补丁

状态：完成本地 fake episode 验证。

已完成：

- 不改 OpenPI 服务端，只在 OpenArm 客户端侧新增 optional HIL/DAgger 入口。
- `openarm_remote_policy` 新增 runtime trace topic：

```text
infer_ms
queue_size
remote_fetch_count
drop_count
blend_length
max_abs_action_delta
```

- 新增 `openarm_hil_rl`：

```text
openarm-hil-mux
openarm-hil-record
openarm-hil-inspect
```

- 新增 `openarm_hil_raw_hdf5_v3`。
- 新增 `start_real_hil_dagger_openpi.sh --openpi-mode fifo|tda_smooth`。
- fake HDF5 episode inspect 通过，确认 policy/human/executed、intervention、checkpoint/prompt 和 runtime 字段存在。

下一步：

- 已进入 14:54 远端验证记录。

### 2026-06-30 14:40 CST - Agent A - gpu25 推理服务重启与本地延迟测试

状态：完成。

已完成：

- 管理员放开 `gpu25:6666` 后，停止旧服务并重新启动同一 HQ checkpoint。
- 当前服务监听 `0.0.0.0:6666`，`http://172.31.11.125:6666/healthz` 返回 `OK`。
- 本机构造 OpenArm payload：

```text
observation.images.base: uint8[224,224,3]
observation.images.left_wrist: uint8[224,224,3]
observation.images.right_wrist: uint8[224,224,3]
observation.state: float32[16]
prompt: fold the cloth
```

- WebSocket 握手成功，metadata 为 `{'action_horizon': 50, 'action_dim': 32, 'rtc_mode': 'off'}`。
- 推理返回 `actions (50, 16)`。

延迟：

```text
connect: 0.0084s
warmup: 0.1401s
timed_1: 0.1299s
timed_2: 0.1387s
timed_3: 0.1331s
timed_4: 0.1331s
timed_5: 0.1316s
mean: 0.1333s
median: 0.1331s
min/max: 0.1299s / 0.1387s
server_infer_ms: about 86-100ms
```

观察：

- 默认 `openpi_client.WebsocketClientPolicy` 在首次冷编译阶段可能被 `websockets` ping timeout 断开。
- 自定义测试客户端关闭 `ping_interval/ping_timeout` 后通信稳定。

下一步：

- 已进入 15:10 真机推理记录。

### 2026-06-30 14:12 CST - Agent A - HQ policy server 启动

状态：完成。

已完成：

- 通过 `mu01` 二跳确认 `gpu25` 可访问。
- 确认 `gpu25` 有 1 张 A800 80GB，启动前无 compute app。
- 在 `gpu25` GPU0 启动 `pi05_openarms_dual_hq` policy server，端口 `6666`。
- 验证 `/healthz` 返回 `OK`。
- 验证 WebSocket 握手成功。
- dry-run 推理返回 `actions (50, 16)`。

服务信息：

```text
server_uri: ws://172.31.11.125:6666
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999
log: /share/home/linyongjia/output/openpi/logs/serve/openarm_hq_gpu25_6666.log
pid: 547844
```

观察：

- 首次 dry-run 推理耗时约 38.7s，包含 JAX 首次编译。
- metadata 显示 `action_dim=32`，但实际 output transform 后是 16D。

下一步：

- 已进入 14:40 重启与延迟测试记录。

### 2026-06-30 - Agent C - HQ TDA 增强进度

状态：增强数据集生成完成；norm stats 和 tiny smoke train 已完成。

已完成：

- 确认 gpu28 可经 mu01 进入，节点为 2x A800 80GB。
- 确认 HQ v2.1 数据集：

```text
/share/home/linyongjia/datasets/high_quality_folding
split: train=0:999, val=999:1199
```

- 新增 OpenArm 16D 专用增强脚本：

```text
scripts/augment_openarm_hq_tda.py
```

- 新增训练配置：

```text
pi05_openarms_dual_hq_tda_aug
target dataset: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1
warm start: HQ 99999/params
```

- gpu28 上 NVENC encoder 不可用，因此全量运行使用：

```text
--video-encoder libx264
--no-require-gpu-video
--use-gpu-decode
```

- 已在 gpu28 tmux 完成全量增强：

```text
tmux: openarm_tda_aug_20260630
log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/augment_20260630_gpu28.log
parquet_done: 2298
video_done: 6894
dataset_size: about 62G
```

- 验收已通过：

```text
meta/episodes.jsonl: 2298 lines
meta/episodes_stats.jsonl: 2298 lines
videos per camera: 2298
sample ffprobe: video frames match parquet rows for original/time-scaled/mirror/last episodes
sample parquet: state/action are 16D
sample mirror: right/left 8D swap is correct for state/action
```

- 已完成 parquet-only norm stats，避开官方视频解码慢路径：

```text
script: scripts/compute_openarm_parquet_norm_stats.py
output: /share/home/linyongjia/datasets/openarm_hq_tda_aug_v1/norm_stats.json
check: state/actions mean/std/q01/q99 are all 16D
log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/norm_20260630_gpu28.log
```

- 已完成训练链路 smoke：

```text
probe: openarm_hq_tda_aug_smoke_20260630 reached step 0 and logged loss, then stopped to avoid long decode/checkpoint cost
tiny_smoke: openarm_hq_tda_aug_smoke_tiny_20260630 completed 3/3 steps
metrics: loss step0=0.122777, step1=0.165189, step2=0.129028
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq_tda_aug/openarm_hq_tda_aug_smoke_tiny_20260630/2
log: /share/home/linyongjia/output/openpi/logs/openarm_tda_aug/smoke_tiny_20260630_gpu28.log
```

下一步：

- 暂不直接启动纯增强 88000 steps full train。
- 等现场数据集 v1 / 相机策略冻结后，再决定做 TDA-only probe 还是合并为 `hq_tda_site_v1`。
