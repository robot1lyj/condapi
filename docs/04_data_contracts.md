# 04 · 数据合同

本页是当前 YAM 训练数据的唯一语义 owner。服务器路径见 [02](02_installation_and_environment.md)，训练流程见 [03](03_training_and_evaluation.md)。OpenArm 16D 合同只在历史文档中保留，不能套用到 YAM。

## YAM 双臂合同

```text
state key:  observation.state
action key: action
state/action shape: (14,)
layout: [left 6 joints, left gripper, right 6 joints, right gripper]
images:
  observation.images.top_rgb
  observation.images.left_rgb
  observation.images.right_rgb
model action: internal 32D, horizon 50
policy output: real YAM 14D
```

YAM 与 YAM-ABC 使用同硬件配置；本仓库只实现训练数据和 policy transform，不实现机械臂控制。YAM state/action 的物理单位、正负方向、夹爪开闭范围必须从当前数据的 `meta/info.json`、feature metadata 和样本审计中确认；在证据完成前不写成 degree、弧度或归一化值。

## 图像和 prompt

LeRobot dataset 每行至少应能提供三路 RGB 图像和 state/action：

| 原始键 | OpenPI 槽位 | 要求 |
|---|---|---|
| `observation.images.top_rgb` | `base_0_rgb` | 必须存在 |
| `observation.images.left_rgb` | `left_wrist_0_rgb` | 必须存在 |
| `observation.images.right_rgb` | `right_wrist_0_rgb` | 必须存在 |
| `observation.state` | `state` | 14D |
| `action` | `actions` | `(horizon, 14)` 或单帧 14D |
| LeRobot task | `prompt` | `prompt_from_task=True` 时映射 |

视频读取后可以是 CHW 或 HWC，`YamInputs` 会统一到 HWC；正式数据应固定编码、fps、时间戳和 RGB 语义。缺少任一路图像不应静默补黑图，先修复数据或明确建立新版本。

## Action transform

当前 `LeRobotYamDataConfig` 默认假定 LeRobot action 是绝对目标：

```text
per arm: [6 joint delta, 1 gripper absolute]
two arms: (6, -1, 6, -1)
```

训练输入先由 `YamInputs` 将 `action` 改名为 OpenPI 的 `actions`，再由 `DeltaActions` 相对当前 `state` 处理 6 个关节；模型输入通过 `PadStatesAndActions` 补到 32D。输出路径先恢复 absolute，再由 `YamOutputs` 裁回 14D。若实际 action 已是 delta，必须在独立数据审计中确认后关闭 `use_delta_joint_actions`，不能重复转换。

## LeRobot 目录和版本

当前代码依赖 `lerobot==0.5.1`，优先使用 LeRobot v3 namespace。一个可训练数据版本应自洽地包含：

```text
<dataset>/
  meta/info.json
  meta/tasks.parquet 或等价 task metadata
  meta/episodes*.jsonl 或 v3 对应元数据
  data/...
  videos/...
```

实际目录结构必须以当前 LeRobot 版本的 metadata loader 为准。新增/删除 episode、重写 parquet、修改视频或改变 feature/unit 都创建新数据版本；不原地覆盖下载的 raw 目录。

服务器已经发现 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks`，但它在完成 metadata、feature、shape、视频和单位审计前只是原始/待审计数据，不能仅凭名称直接训练。

## Norm stats 和资产

YAM 资产 id 固定为 `yam`，训练和 checkpoint 约定为：

```text
assets/yam/norm_stats.json
```

norm 必须针对同一数据版本、同一 train episode split，并在 YAM action delta transform 后计算。禁止跨 OpenArm/Piper/YAM 合同或跨单位复制 stats；服务优先读取 checkpoint 内资产。

## 数据发布 gate

- `info` 中的 episode 数、task metadata、parquet 索引和视频清单一致。
- state/action 所有样本最后一维为 14，顺序是 `[左6+夹爪, 右6+夹爪]`，无 NaN/Inf。
- 三路图像首、中、尾样本可解码，时间戳不越界，颜色通道和 shape 一致。
- task/prompt 映射不为空，train/val split 明确且可复现。
- 物理单位和夹爪语义已写入数据版本的 audit/manifest；未确认前标记为待核验。
- norm stats、config、训练 split 和 checkpoint asset id 一一对应。

## 旧合同隔离

OpenArm 的 16D `[右臂7, 右夹爪, 左臂7, 左夹爪]`、HQ degree 语义和 Piper 14D transform 都是历史/legacy。它们不能与 YAM 的 14D `[左6, 左夹爪, 右6, 右夹爪]` 混合，也不能复用其 norm stats、动作顺序或训练结论。
