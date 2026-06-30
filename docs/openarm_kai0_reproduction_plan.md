# OpenArm KAI0 复现部署计划

本文档用于把 KAI0 的有效模块复现到 OpenArm + OpenPI 链路中，并拆成可并行分配给多个 Agent 的任务。

当前策略：先不做模型路由，也不优先做 Model Arithmetic。先从已有 HQ 模型和已有客户端开始，按可验收的顺序推进：

```text
HQ baseline 真机推理
HQ + TDA 平滑真机推理
TDA 增强重训模型
Recovery / Heuristic DAgger 可用采集格式
Stage Advantage 标注和 AWBC 训练
```

## 0. 实时进度看板

最后更新：2026-06-30 14:12 CST

### 0.1 Agent 状态

| Agent | 当前状态 | 最近进展 | 下一步 | 证据/产物 |
|---|---|---|---|---|
| A - HQ Baseline 真机推理 | 进行中 | 已在 `gpu25` 启动 HQ policy server，health 和 dry-run 推理通过 | 用 OpenArm 客户端连 `ws://172.31.11.125:6666` 做真实 baseline | 日志 `/share/home/linyongjia/output/openpi/logs/serve/openarm_hq_gpu25_6666.log` |
| B - 客户端 TDA Chunk 平滑 | 等待回写 | 多 Agent 已开始工作，当前文档尚未收到实现状态 | 回填 `fifo/tda_smooth` 设计、测试状态、分支/提交 | 待填 |
| C - HQ 数据增强和重训 | 等待回写 | 多 Agent 已开始工作，当前文档尚未收到数据增强状态 | 回填数据集审计、增强脚本、norm stats、训练状态 | 待填 |
| D - Recovery / Heuristic DAgger 采集格式 | 等待回写 | 多 Agent 已开始工作，当前文档尚未收到字段审计结果 | 回填样例 episode、字段列表、缺失字段 | 待填 |
| E - Stage Advantage 标注和训练方案 | 等待回写 | 多 Agent 已开始工作，当前文档尚未收到 stage schema 版本 | 回填 stage schema、标注格式、首批标注计划 | 待填 |

### 0.2 当前 HQ 推理服务

```text
node: gpu25
node_ip: 172.31.11.125
gpu: GPU0 / A800 80GB
port: 6666
server_uri: ws://172.31.11.125:6666
config: pi05_openarms_dual_hq
checkpoint: /share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999
pid: 547844
log: /share/home/linyongjia/output/openpi/logs/serve/openarm_hq_gpu25_6666.log
```

验证记录：

```text
healthz: OK
listen: 0.0.0.0:6666
GPU memory: about 69GB / 80GB
metadata: {'action_horizon': 50, 'action_dim': 32, 'rtc_mode': 'off'}
dry-run actual output: actions shape = (50, 16)
first dry-run latency: 38.7s
```

注意：metadata 里的 `action_dim=32` 是模型内部动作维度；经过 OpenArm output transform 后，实际返回给客户端的 `actions` 是 `(50, 16)`。当前 baseline 可以先使用；后续做 RTC/TDA metadata 严格校验时需要把 metadata 修成 16，避免客户端误判。

### 0.3 进度更新规则

多 Agent 并行时，每个 Agent 回写本文档只更新两个位置：

1. `0.1 Agent 状态` 表中自己的行。
2. 文末 `9. 进度日志` 增加一条带时间戳的记录。

每条进度必须包含：

```text
时间
Agent
状态: 未开始 / 进行中 / 阻塞 / 已完成
证据: 日志、commit、数据集路径、checkpoint、样例 episode 或测试输出
下一步
```

## 1. 参考依据

### 1.1 主要参考项目

- KAI0 项目：`/home/lyj/kai0`
  - 主 README：`/home/lyj/kai0/README.md`
  - TDA：`/home/lyj/kai0/train_deploy_alignment/`
  - 数据增强：`/home/lyj/kai0/train_deploy_alignment/data_augment/`
  - DAgger：`/home/lyj/kai0/train_deploy_alignment/dagger/`
  - Stage Advantage：`/home/lyj/kai0/stage_advantage/`
- OpenPI 当前仓库：`/home/lyj/lyj/openpi`
  - 当前 OpenArm 训练配置：`pi05_openarms_dual_hq`
  - 当前模型输入内部相机 key：`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`
- OpenArm 客户端仓库：`/home/lyj/openarm_ros2_docker`
  - OpenPI 推理入口：`scripts/start_real_inference_openpi.sh`
  - 远端策略客户端：`src/openarm_remote_policy/openarm_remote_policy/`
  - 纯数据采集：`src/openarm_vr_recording/`

### 1.2 主要参考论文和思想

- KAI0 论文：
  - `χ0: Resource-Aware Robust Manipulation via Taming Distributional Inconsistencies`
  - arXiv: `2602.09021`
  - 作用：本计划的主参考，复现其中的 Train-Deploy Alignment、Heuristic DAgger、Stage Advantage。
- DAgger：
  - Ross, Gordon, Bagnell, `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning`, 2011
  - 作用：解释为什么要让策略进入自己的执行分布，再由人类接管补充修正数据。
- OpenPI / pi0.5：
  - 作用：作为当前策略模型、训练脚本、serve policy 和客户端协议的基础工程。

KAI0 的三个模块中，本阶段取舍如下：

| 模块 | 本阶段是否做 | 原因 |
|---|---:|---|
| Train-Deploy Alignment | 是 | 直接对应当前痛点：训练-部署分布差、推理 chunk 抖动、真实失败恢复 |
| Heuristic DAgger / Recovery | 是 | 能用现有客户端模型接管和人工接管采集 |
| Stage Advantage | 是，但第二批落地 | 需要额外标注 `stage_progress_gt`，不能空转 |
| Model Arithmetic | 暂缓 | 当前是单任务，且高质量 checkpoint 数量还不够；不是模型路由，也不是第一收益点 |
| 模型路由 | 不做 | 单任务折叠不需要 runtime route 多模型 |

## 2. 当前已有资产

### 2.1 已有 HQ checkpoint

```text
/share/home/linyongjia/output/openpi/pi05_openarms_dual_hq/openarms_hq_bs32/99999
```

该 checkpoint 是第一阶段 baseline，不覆盖、不改名、不作为实验产物重写。

用途：

- 真机 OpenPI baseline 推理。
- 客户端 TDA 平滑 A/B 测试。
- 后续 TDA 增强重训的 warm start 候选。

### 2.2 已有 OpenArm 客户端能力

`~/openarm_ros2_docker` 已有三条关键链路：

- VR 遥操作和纯采集：`scripts/start_pure_collection.sh`
- 普通远端推理：`scripts/start_real_inference_lerobot.sh`
- OpenPI HQ 推理：`scripts/start_real_inference_openpi.sh`

当前 OpenPI 入口仍标记为待现场验证，所以第一批 Agent A 的目标就是把它验证成可复现 baseline。

### 2.3 已有训练配置

当前 OpenPI 配置中已有：

```text
pi05_openarms_dual_hq
repo_id=/share/home/linyongjia/datasets/high_quality_folding
robot_action_dim=16
base_image_key=observation.images.base
action_style=relative
num_train_steps=100000
```

注意：

- OpenArm 当前 HQ 训练保持 degree 语义。
- 不要在复现实验中随意把历史 HQ 数据改成 radians。
- 夹爪维度要单独处理，不能和关节角统一做 rad/degree 变换。

## 3. 全局共享接口约定

所有 Agent 必须遵守同一份接口，不允许各自发明数据格式。

### 3.1 客户端 payload

客户端发给 OpenPI policy server 的输入：

```text
observation.images.base
observation.images.left_wrist
observation.images.right_wrist
observation.state
prompt
```

OpenPI transform 内部映射：

```text
observation.images.base        -> base_0_rgb
observation.images.left_wrist  -> left_wrist_0_rgb
observation.images.right_wrist -> right_wrist_0_rgb
```

### 3.2 State / action 顺序

统一 16D：

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

### 3.3 单位

模型侧：

```text
joint: degree 语义，沿用 HQ checkpoint
gripper: 单独映射，不能当普通角度统一处理
```

机器人侧：

```text
ROS / controller: rad + 夹爪归一化
客户端负责 degree <-> rad 和 gripper 映射
```

### 3.4 训练数据规则

- LeRobot 数据集版本不可原地修改。
- 增强数据必须新建版本目录。
- 每次重训必须重新生成 norm stats。
- 禁止使用会误删 OpenArm degree 样本的 `[-pi, pi]` 过滤。
- `norm_stats.json` 必须确认 `state` 和 `actions` 都是 16D。

## 4. 并行 Agent 拆分

### Agent A - HQ Baseline 真机推理

#### 目标

把已有 HQ checkpoint 稳定跑到 OpenArm 真机客户端，得到没有 TDA 平滑、没有 DAgger、没有 Stage Advantage 的 baseline。

#### 参考

- OpenPI serve：`scripts/serve_policy.py`
- OpenArm OpenPI 客户端：`~/openarm_ros2_docker/scripts/start_real_inference_openpi.sh`
- OpenArm 远端策略节点：`~/openarm_ros2_docker/src/openarm_remote_policy/openarm_remote_policy/remote_policy_node.py`
- OpenPI 远端推理说明：`docs/remote_inference.md`

#### 推荐做法

1. 在 GPU12 或 GPU14 启动 policy server，加载 HQ checkpoint。
2. 在 OpenArm 工控机/客户端运行 OpenPI 推理入口。
3. 先 dry run 检查 payload，不碰真机动作。
4. 再进入真机低速验证，确认 `/openarm/joint_target` 有且只有一个 writer。
5. 记录延迟、action shape、action range、publish rate、queue size。

#### 验收标准

必须全部满足：

- 客户端成功连接 policy server。
- server 返回 `actions`，shape 为 `(horizon, 16)` 或单步 `16D`。
- 客户端发送的三路图像 key 与训练 key 一致。
- `prompt` 与训练数据 task prompt 一致或有明确记录。
- action 能从模型 degree 语义转换成机器人 rad/夹爪命令。
- `/openarm/joint_target` 没有第二个抢控制 publisher。
- 失败时能明确区分是 server、payload、shape、单位、相机还是控制链问题。

#### 交付物

```text
docs/runs/openarm_hq_baseline_<date>.md
server 启动命令
client 启动命令
一次成功日志
metadata: action_horizon/action_dim/chunk_mode/server_queue
动作范围统计
问题清单
```

### Agent B - 客户端 TDA Chunk 平滑

#### 目标

在不改变 policy server、不改变 controller 的前提下，在客户端动作块队列中实现 KAI0 TDA 的 temporal chunk-wise smoothing。

#### 参考

- KAI0 TDA inference：`/home/lyj/kai0/train_deploy_alignment/inference/`
- OpenArm 当前 chunk FIFO：`~/openarm_ros2_docker/src/openarm_remote_policy/openarm_remote_policy/ws_policy_client.py`
- 当前客户端说明：OpenArm README 中明确当前只有普通 WebSocket + msgpack + 本地 chunk FIFO。

#### 推荐做法

只在 `ActionChunkBroker` 附近实现，保留现有 FIFO 行为。

新增参数建议：

```text
chunk_merge_mode=fifo|tda_smooth
tda_drop_max
tda_min_overlap
tda_blend_mode=linear|ema
tda_blend_alpha
```

行为建议：

1. `fifo`：完全保持当前逻辑，新 chunk 完整 append。
2. `tda_smooth`：
   - 新 chunk 到达时，根据已执行步数丢弃新 chunk 前部过期动作。
   - 保留旧队列尚未执行部分。
   - 在旧队列尾部和新 chunk 前部的 overlap 区间做线性融合。
   - 融合后替换队列尾部，再追加新 chunk 剩余部分。
3. 每次 reset、急停、人工接管、server reconnect 时清空平滑状态。

#### 必须记录的指标

```text
queue_size_before
queue_size_after
remote_fetch_count
drop_count
blend_length
max_abs_action_delta
infer_ms
publish_hz
```

#### 验收标准

必须全部满足：

- `fifo` 模式行为和原来完全一致。
- `tda_smooth` 可通过参数打开/关闭。
- action shape 永远保持 16D。
- 平滑逻辑不发布 ROS topic，只返回 action，不侵入 controller。
- 有离线单元测试覆盖两个 chunk 的融合结果。
- 真机前有 dry-run 日志证明 action delta 下降，且没有 action 维度错位。

#### 交付物

```text
TDA smooth 代码补丁
参数说明
单元测试
dry-run 日志
和 HQ baseline 的 A/B 对比表
```

### Agent C - HQ 数据增强和重训

#### 目标

从已有 HQ 数据出发，复现 KAI0 TDA 的时空数据增强，生成新数据集并训练 `openarm_hq_tda_aug_v1`。

#### 参考

- KAI0 data augment：`/home/lyj/kai0/train_deploy_alignment/data_augment/`
- KAI0 time scaling：`time_scaling.py`
- KAI0 space mirroring：`space_mirroring.py`
- OpenPI OpenArm config：`src/openpi/training/config.py`
- 数据集版本规范：`docs/dataset_versioning.md`
- norm stats 说明：`docs/norm_stats.md`

#### 推荐做法

第一版只做对 OpenArm 安全的增强：

```text
time scaling:
  extraction_factor=2
  可选 split_ratio=0.3，用 30% 数据做加速版本，再和原数据 merge

space mirroring:
  base image 可水平翻转
  left_wrist/right_wrist camera 互换并按实际视角决定是否翻转
  state/action 的 right 8D 和 left 8D 互换
  gripper 维度只互换，不做 rad/degree 变换
```

OpenArm 与 KAI0 Agilex/ARX 不同，不能直接假设 14D：

```text
KAI0 默认常见配置: 14D 双臂
OpenArm: 16D = right 7 + right gripper + left 7 + left gripper
```

因此空间镜像脚本需要显式适配 16D，不允许沿用默认 `left_dim=7/right_dim=7` 后把夹爪落在 rest 里混掉。

#### 训练建议

新增 config，不覆盖 `pi05_openarms_dual_hq`：

```text
pi05_openarms_dual_hq_tda_aug
repo_id=/share/home/linyongjia/data/openarm_hq_tda_aug_v1
robot_action_dim=16
base_image_key=observation.images.base
action_style=relative
weight_loader=HQ checkpoint 或 pi05_base
num_train_steps=88000 或 100000
```

推荐先做：

```text
smoke: 500-1000 steps
full: 88000 steps
```

如果从 HQ checkpoint warm start，必须确认：

- schema 相同；
- norm stats 是新数据重新生成；
- state/action 单位不变；
- action horizon 和 action dim 不变。

#### 验收标准

必须全部满足：

- 增强后数据集是新版本目录，不覆盖原 HQ。
- `meta/info.json`、`meta/episodes.jsonl`、parquet、videos 数量一致。
- 任取 episode 可读三路视频。
- state/action shape 全部为 16D。
- 镜像后 right/left 8D 顺序正确。
- norm stats 重新计算，`state` 和 `actions` 均为 16D。
- 训练 smoke 能启动并保存 checkpoint。
- full train 输出 checkpoint 和训练日志。

#### 交付物

```text
增强脚本或适配补丁
增强数据集路径
manifest.yaml
norm_stats.json
训练 config
smoke 训练日志
full checkpoint 路径
与原 HQ loss 曲线对比
```

### Agent D - Recovery / Heuristic DAgger 采集格式

#### 目标

确认现有 HIL/采集链路是否足够支持 Heuristic DAgger。如果不够，给出并实现最小字段补丁。

#### 参考

- KAI0 DAgger：`/home/lyj/kai0/train_deploy_alignment/dagger/`
- OpenArm 纯采集：`~/openarm_ros2_docker/docs/PURE_COLLECTION.md`
- OpenArm 采集机器人接口：`~/openarm_ros2_docker/src/openarm_vr_recording/openarm_vr_recording/openarm_robot.py`
- OpenArm 推理客户端：`~/openarm_ros2_docker/src/openarm_remote_policy/`

#### 当前判断

OpenArm 纯采集 v1 已经足够做普通 BC / SFT：

```text
observation.state
observation.images.*
action
task
timestamp
```

但完整 DAgger / Recovery 还需要知道每一帧到底是谁在控制，以及模型原始输出和人类修正之间的差异。

#### 必须审计的字段

录一条短 episode，然后 inspect HDF5 / raw dataset 是否有：

```text
policy_action
human_action 或 teleop_action
executed_action
is_intervention
authority_source
intervention_start
intervention_end
failure_mode
success 或 outcome
policy_checkpoint
prompt
timestamp
camera frame ids
latency / queue metadata
```

#### 推荐采集流程

第一版 Heuristic Recovery 不等策略自己完全失败，而是人为初始化到高风险状态：

1. 启动 HQ baseline 或 HQ + TDA smooth。
2. 让策略执行到明显不稳定或偏离阶段。
3. 人类进入接管模式。
4. 记录从失败附近到恢复成功的短 episode。
5. 保存 policy stream、人类 stream、最终执行 stream。
6. 标注 failure_mode 和 outcome。

#### 最小补丁原则

如果当前只能保存最终 `/openarm/joint_target`，不够。

最小补丁不是重写采集系统，而是在现有 raw episode 中增加字段或 sidecar：

```text
action.policy
action.human
action.executed
control.is_intervention
control.authority_source
control.policy_checkpoint
episode.outcome
episode.failure_mode
runtime.queue_size
runtime.infer_ms
```

#### 验收标准

必须全部满足：

- 能逐帧判断模型控制、人类控制、混合过渡。
- 能还原最终执行动作。
- 能计算人类修正量：`human_action - policy_action`。
- 能筛选 recovery segment。
- 能区分失败 episode、恢复成功 episode、普通示教 episode。

#### 交付物

```text
docs/runs/dagger_signal_audit_<date>.md
一条样例 episode 路径
字段列表
缺失字段清单
最小补丁方案
补丁后的样例 episode
```

### Agent E - Stage Advantage 标注和训练方案

#### 目标

为 OpenArm 折叠任务建立 Stage Advantage 的最小可训练版本：先标注，再训练 advantage estimator，再生成 AWBC 数据集。

#### 参考

- KAI0 Stage Advantage：`/home/lyj/kai0/stage_advantage/README.md`
- Advantage estimator 训练：`/home/lyj/kai0/scripts/train_pytorch.py`
- Advantage 数据集：`/home/lyj/kai0/src/openpi/training/advantage_dataset.py`
- 离散化：`/home/lyj/kai0/stage_advantage/annotation/discretize_advantage.py`

#### OpenArm 第一版 stage taxonomy

建议先用 5 阶段，不要一开始标太细：

| stage_id | 名称 | 判定标准 |
|---:|---|---|
| 0 | approach_grasp | 双臂接近衣物并建立有效抓取 |
| 1 | spread_flatten | 展开或拉平衣物，减少皱折 |
| 2 | align | 对齐关键边缘或折叠线 |
| 3 | fold | 执行主要折叠动作 |
| 4 | release_finish | 放置、松爪、结束姿态稳定 |

如果实际任务只需要 3 阶段，可降级为：

```text
grasp -> fold -> finish
```

但 stage 定义一旦进入训练数据，就不要频繁改。

#### 标注格式

推荐先用 sidecar JSONL，不直接手改 parquet：

```json
{
  "dataset": "openarm_hq_v1",
  "episode_index": 12,
  "fps": 30,
  "task": "fold the cloth",
  "quality": "success",
  "stage_boundaries": [
    {"stage_id": 0, "start_frame": 0, "end_frame": 85},
    {"stage_id": 1, "start_frame": 86, "end_frame": 180},
    {"stage_id": 2, "start_frame": 181, "end_frame": 240},
    {"stage_id": 3, "start_frame": 241, "end_frame": 330},
    {"stage_id": 4, "start_frame": 331, "end_frame": 380}
  ],
  "notes": ""
}
```

再由转换脚本生成每帧：

```text
stage_id
stage_progress_gt
```

KAI0 的 `stage_progress_gt` 公式是：每个 stage 内线性从 0 到 1，再映射到全局 0 到 1。

#### 第一批数据建议

优先标注：

```text
50-100 条 HQ 成功 episode
10-20 条 Recovery 成功 episode
少量失败 episode 只用于分析，不先混入 AWBC 正样本
```

#### 训练流程

1. 对 HQ 成功数据写入 `stage_progress_gt`。
2. 训练 Advantage Estimator。
3. 用 estimator 预测数据集，生成：
   - `absolute_advantage`
   - `relative_advantage`
   - `absolute_value`
4. 离散化 advantage，写回 `task_index` 和 `meta/tasks.jsonl`。
5. 训练 AWBC policy。
6. 推理时 prompt 必须使用训练时同样的 advantage prompt，例如：

```text
fold the cloth, Advantage: positive
```

#### OpenArm 适配点

KAI0 默认相机和任务名来自 Agilex/ARX，需要改成 OpenArm：

```text
top_head / hand_left / hand_right  -> base / left_wrist / right_wrist
14D action                         -> 16D action
KAI0 task prompt                   -> OpenArm task prompt
```

#### 验收标准

必须全部满足：

- stage schema 有明确文字定义。
- 至少 50 条 episode 有边界标注。
- 转换后 parquet 或 sidecar 能生成 per-frame `stage_progress_gt`。
- `stage_progress_gt` 单调递增，范围 `[0, 1]`。
- Advantage estimator 能完成 smoke train。
- eval 能写出 `absolute_advantage` / `relative_advantage`。
- discretize 后 `meta/tasks.jsonl` 中存在 positive/negative prompt。
- AWBC 训练能启动，并且推理 prompt 规则被记录。

#### 交付物

```text
stage_schema.md
annotation_format.jsonl
stage_progress_gt 转换脚本
50 条 episode 标注结果
advantage estimator smoke checkpoint
AWBC 数据集路径
```

## 5. 里程碑和汇合点

### Milestone 0 - 文档和接口冻结

目标：所有 Agent 都按本文档接口工作。

验收：

- 本文档合入仓库。
- 所有人确认 16D 顺序、相机 key、degree 语义。
- 确认 HQ checkpoint 路径。

### Milestone 1 - HQ baseline

目标：现有 HQ 模型可真实推理。

由 Agent A 交付。

通过条件：

```text
OpenArm 客户端收到 16D action
真机低速推理不出现 shape/key/unit 错误
记录 baseline latency/action/publish 指标
```

### Milestone 2 - HQ + TDA smooth A/B

目标：同一 HQ checkpoint 下比较 FIFO 和 TDA smooth。

由 Agent B 和 Agent A 联合交付。

通过条件：

```text
fifo baseline 可复现
tda_smooth 可开关
动作 delta 指标下降或至少没有变差
真机没有因平滑引入控制异常
```

### Milestone 3 - TDA 增强重训

目标：得到第一个增强重训模型。

由 Agent C 交付。

通过条件：

```text
openarm_hq_tda_aug_v1 数据集冻结
norm stats 重新生成
smoke train 通过
full train 输出 checkpoint
```

### Milestone 4 - Recovery / DAgger 数据可用

目标：采集格式足以支持 DAgger 和 recovery 筛选。

由 Agent D 交付。

通过条件：

```text
能区分 policy_action / human_action / executed_action
能标记 intervention
能筛选 recovery segment
至少有 10 条可读 recovery 样例
```

### Milestone 5 - Stage Advantage MVP

目标：跑通 OpenArm Stage Advantage 最小闭环。

由 Agent E 交付。

通过条件：

```text
50 条 HQ episode stage 标注
生成 stage_progress_gt
advantage estimator smoke train
生成 positive/negative advantage dataset
AWBC smoke train
```

## 6. 并行排期建议

第一天并行：

```text
Agent A: 启动 HQ baseline，定位 server/client/payload 问题
Agent B: 在本地实现 tda_smooth 和单元测试
Agent C: 审计 HQ 数据集和增强脚本适配点
Agent D: 录 1 条短 HIL/采集 episode 并 inspect 字段
Agent E: 定 stage schema 和标注格式
```

第一批汇合：

```text
A 给出 baseline 是否跑通
B 给出 fifo/tda_smooth dry-run
C 给出增强数据集方案和 norm stats 方案
D 给出 DAgger 字段够不够的结论
E 给出 stage schema v1
```

第二批并行：

```text
A+B: 真机 A/B
C: 增强数据 + 88k 重训
D: 补字段 + recovery 小批量采集
E: 50 条 HQ episode 标注 + stage_progress_gt 转换
```

## 7. 风险和禁止事项

### 7.1 禁止事项

- 禁止覆盖 HQ checkpoint。
- 禁止把 OpenArm HQ 数据静默从 degree 改成 rad。
- 禁止在客户端创建第二个 `/openarm/joint_target` writer。
- 禁止在没有验收的情况下把 TDA smooth 设为默认。
- 禁止把增强数据写回原 HQ 数据目录。
- 禁止 Stage Advantage 还没有标注时就空跑 AWBC。

### 7.2 主要风险

| 风险 | 影响 | 处理 |
|---|---|---|
| OpenPI 客户端入口未现场验证 | baseline 卡住 | Agent A 先 dry-run payload，再低速真机 |
| degree/rad 混淆 | 动作幅度错误，真机危险 | 所有日志打印 action range，客户端转换集中处理 |
| DAgger 只记录最终 action | 无法学习人类修正 | Agent D 必须补 policy/human/executed 三路信号 |
| 空间镜像维度错位 | 训练学坏左右手 | Agent C 必须做 16D 单元测试 |
| Stage 过细 | 标注慢且不一致 | 第一版 5 阶段，必要时降到 3 阶段 |

## 8. 最终完成定义

本计划不是以“代码写完”为完成，而是以下结果全部存在：

```text
1. HQ baseline 真机日志
2. HQ + TDA smooth A/B 日志
3. openarm_hq_tda_aug_v1 数据集和 checkpoint
4. DAgger/recovery 字段审计和样例 episode
5. Stage schema、50 条标注、stage_progress_gt 数据
6. AWBC smoke train 记录
```

达到这些结果后，再讨论是否补 Model Arithmetic。只有当我们已经有多个互补 checkpoint，例如：

```text
HQ baseline
TDA augmented
Recovery finetuned
AWBC finetuned
```

才有必要评估 KAI0 的 Model Arithmetic。单个任务、单个 checkpoint 阶段不需要模型路由。

## 9. 进度日志

### 2026-06-30 14:12 CST - Agent A - HQ policy server 启动

状态：进行中。

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
- metadata 当前显示 `action_dim=32`，但实际输出 transform 后是 16D。

下一步：

- OpenArm 客户端连接 `ws://172.31.11.125:6666`，做真实 baseline。
- 记录真实客户端 latency、action range、publish rate 和是否抖动。
- 后续修正 metadata action_dim，供 TDA/RTC 严格校验使用。
