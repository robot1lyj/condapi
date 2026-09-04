# 01 · 系统架构

## 总体数据流

```text
LeRobot v2.1 数据
  -> dataset/config transforms
  -> Observation(state/images/prompt) + Actions
  -> pi0/pi0.5（JAX 主路径；pi0/pi0.5 亦支持 PyTorch）
  -> OpenArm policy wrapper
  -> 50 步、16D action chunk
  -> WebSocket server
  -> openpi-client / ROS runtime
```

训练与真机之间的边界是 policy server 和客户端：模型内部使用 OpenArm 数据合同；ROS 弧度、归一化夹爪和硬件安全限制只在客户端/运行时边界处理。

## 初始/复位位姿归属

policy server 只推理，不移动机器人，也不定义 OpenArm home pose。当前 OpenArm 配置没有 `reset_pose`；初始位姿、复位速度、夹爪复位和急停由机器人侧 ROS/driver/client `reset()` 实现。旧记录中的 `/home/lyj/openarm_ros2_docker` 在新平台尚未核实，不能作为新服务器默认路径；真机推理脚本是 `scripts/start_real_inference_openpi.sh`、`scripts/start_real_inference_lerobot.sh`，HIL 脚本是 `scripts/start_real_hil_dagger_openpi.sh`；`packages/openpi-client/runtime/runtime.py` 只负责调用环境的 `reset()`，不包含机械臂关节值。

`examples/aloha_real/constants.py:START_ARM_POSE` 和 `examples/aloha_real/real_env.py:DEFAULT_RESET_POSITION` 属于 ALOHA legacy，不得复制为 OpenArm 位姿。改变 OpenArm 位姿时必须同步检查碰撞/限位、相机标定和训练数据起始分布；不要在 policy config 或 `--rtc-metadata` 中伪造位姿。

## 代码模块

| 层 | 主要位置 | 责任 |
|---|---|---|
| 模型 | `src/openpi/models/`、`src/openpi/models_pytorch/` | pi0、pi0.5、视觉/语言 backbone 和 action sampling |
| 配置/训练 | `src/openpi/training/`、`scripts/train.py`、`scripts/train_pytorch.py` | `TrainConfig`、数据 loader、优化器、checkpoint |
| 数据变换 | `src/openpi/transforms.py`、`src/openpi/training/config.py` | repack、机器人 transform、prompt/模型输入 |
| 策略 | `src/openpi/policies/` | 将模型输出映射为机器人动作；OpenArm 入口见 `openarm_policy.py` |
| 服务 | `src/openpi/serving/`、`scripts/serve_policy.py` | 加载 checkpoint、WebSocket 推理、RTC 兼容开关 |
| 客户端 | `packages/openpi-client/` | 机器人侧请求、时间戳、运行时单位和 IO |
| 研究脚本 | `scripts/` | OpenArm 清洗、Stage/value、AWBC、HIL、审计和 smoke |

## OpenArm 配置路径

OpenArm 配置集中在 `src/openpi/training/config.py`，核心数据类是 `LeRobotOpenArmDataConfig`，输入/输出类是 `OpenArmInputs`/`OpenArmOutputs`。正式 KAI0 配置为 `pi05_openarm_kai0_awbc_v1`；Site probe、Evo ACP probe 和 advantage scorer 是独立配置，具体状态由 `docs/06_openarm_research_plan.md` 维护。

OpenArm 输入通常包括 base、left wrist、right wrist 三路图像、16D state、prompt 和可选的 ACP/metadata 字段。动作 horizon 当前为 50，policy wrapper 会校验 state/action 的最后一维为 16。

## 数据/模型边界

- 清洗后的 OpenArm 数据必须经过 `scripts/convert_openarm_hq_dataset.py` 或对应的 OpenArm 专用脚本；不要直接把 HIL raw 喂给训练 loader。
- `norm_stats.json` 必须与数据版本和配置绑定；不能跨单位合同复用。
- Stage/value 是离线评分器，不是默认在线控制器；KAI0 policy 只接收 policy prompt，不能假设线上有 `stage_id`。
- HIL 的 policy action、human action、hold 和 intervention 元数据必须在 clean 阶段保留/区分，Evo value 才能使用。

## Piper 边界

Piper 仍能通过 `pi*_piper_dual` 配置运行，但它是 legacy 支持路径，使用不同维度/transform/单位约定。新 OpenArm 代码、数据、norm stats、服务和研究结论不得依赖 Piper；需要追溯时只看 `docs/reference/legacy/piper.md`。

## 重要不变量

1. OpenArm 16D 顺序和 degree/HQ 夹爪语义不可改变。
2. 训练 config、数据 metadata、norm stats、checkpoint 和 serve config 必须成套核对。
3. RTC 新路径必须和旧推理路径共存，通过 `rtc_mode` 选择或回退。
4. WebSocket 层返回的动作必须有限且形状为 `(50,16)`；客户端再做机器人边界转换。
