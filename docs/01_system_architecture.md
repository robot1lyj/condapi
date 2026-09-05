# 01 · 系统架构

## 当前训练数据流

```text
YAM LeRobot v3 数据
  -> LeRobotDataset / metadata task 映射
  -> YamInputs：三路 RGB + 14D state/action -> OpenPI 标准键
  -> DeltaActions：每臂 6 个关节转相对当前 state，夹爪保持 absolute
  -> norm stats -> Pi0/Pi0.5 模型输入
  -> 模型内部 32D action、50 步 horizon
  -> YamOutputs + AbsoluteActions -> YAM 14D action chunk
```

当前代码只负责训练数据、模型 transform 和训练后 policy 的通用输出；YAM 机械臂驱动、CAN、GUI、home pose 和控制频率不属于本仓库的训练适配范围。

## 代码模块

| 层 | 主要位置 | 责任 |
|---|---|---|
| 模型 | `src/openpi/models/`、`src/openpi/models_pytorch/` | Pi0、Pi0.5、视觉/语言 backbone 和 action sampling |
| 配置/训练 | `src/openpi/training/`、`scripts/train.py`、`scripts/train_pytorch.py` | `TrainConfig`、数据 loader、优化器、checkpoint |
| 数据变换 | `src/openpi/transforms.py`、`src/openpi/training/config.py` | YAM 输入、delta action、prompt、norm 和模型输入 |
| YAM policy | `src/openpi/policies/yam_policy.py` | 校验 14D 合同、映射三路图像、裁掉模型 32D padding |
| 服务 | `src/openpi/serving/`、`scripts/serve_policy.py` | 加载 checkpoint 和通用 WebSocket 推理 |
| 客户端 | `packages/openpi-client/` | 通用请求/响应协议；不实现 YAM 机械臂控制 |
| 数据工具 | `scripts/compute_norm_stats.py`、审计脚本 | norm、metadata、视频和 loader 预检 |

## YAM 配置路径

YAM 配置集中在 `src/openpi/training/config.py`：

- `LeRobotYamDataConfig`：默认双臂、14D、`assets/yam`，动作序列键为单数 `action`。
- `YamInputs`：接收 `observation.images.top_rgb`、`left_rgb`、`right_rgb`、`observation.state`、`action` 和 prompt。
- `YamOutputs`：把模型至少 32D 的 action chunk 裁回 14D；不足 14D 直接报错。
- `pi05_yam_lora`：当前首选低显存配置；对应 `gemma_2b_lora` 和 `gemma_300m_lora`，关闭 EMA。

YAM policy 不复用 OpenArm 或 Piper 的 transform。单臂 7D 只作为配置类的显式兼容选项，当前项目默认始终是双臂 14D。

## 数据加载兼容

当前仓库兼容 LeRobot 新旧 import 路径：优先使用 `lerobot.datasets.lerobot_dataset`，旧版本才回退到 `lerobot.common.datasets.lerobot_dataset`。LeRobot v3 的 `meta.tasks` 可能是 DataFrame，loader 会先归一化为 `task_index -> prompt` 映射。

LeRobot 原始键直接在 YAM policy boundary 处理，因此本仓库没有把独立 YAM-ABC 项目的数据转换器或机械臂控制层复制进来。若实际数据不是上述键，先写独立转换/审计产物，再接入训练配置。

## 模型与机器人边界

- OpenPI 模型统一需要标准 `image/state/actions` 结构和 32D padding；YAM 的真实合同只在 `YamInputs/YamOutputs` 边界出现。
- norm stats 在训练输入经 delta transform 后计算，checkpoint 内保存到 `assets/yam/norm_stats.json`；不能跨单位或跨数据版本复用。
- 推理服务返回 50 步、14D YAM 动作；任何单位转换、限幅、执行和安全检查必须由已核实的机器人侧系统负责。

## 历史边界

OpenArm 16D、Piper transform、KAI0/Evo-RL/HIL 方案只在变更记录和历史归档中保留背景；对应的当前专用代码已从默认训练树清理。它们不能改变当前 YAM 默认配置，也不能把 OpenArm 的单位或动作顺序套到 YAM。
