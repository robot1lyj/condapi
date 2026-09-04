# 05 · 推理与 Rollout

本页是 OpenArm 服务、WebSocket 合同、RTC、HIL 和真机 rollout 的唯一操作 owner。服务器登录、GPU/tmux 审计和 checkpoint 目录见 [02 · 服务器与环境](02_installation_and_environment.md)，动作/单位/数据版本见 [04 · 数据合同](04_data_contracts.md)。

新平台的 OpenPI 代码、checkpoint、输出目录和服务节点尚未迁移核实；接管时只确认了 `wuyan@10.18.31.234` 登录入口、`yam` Conda 环境和一个独立的 Slurm 数据下载任务。按 [02](02_installation_and_environment.md) 完成计算节点 import/checkpoint gate 后，才能把本页模板落地；不能沿用旧服务器的 `pi-conda`、`gpu25` 或 `/share/home/...` 默认值。

## 1. Rollout 前的四个 gate

服务启动前必须同时满足：

1. **代码 gate**：服务端 commit、训练 config 和客户端代码已记录；服务运行在新平台经过验收的 Conda 环境，不是登录节点的系统 Python。
2. **checkpoint gate**：数字 step 目录包含 `_CHECKPOINT_METADATA`、`params/_METADATA` 和 `assets/<asset_id>/norm_stats.json`。缺任一项都不能部署。
3. **合同 gate**：OpenArm state/action 是 16D，关节为 degree，HQ 夹爪为 motor degree（`0=open`、`-66=closed`），动作 horizon 为 50；不能把 Piper 14D、弧度或 `[0,1]` 夹爪数据接进来。
4. **安全 gate**：急停、人工接管、工作空间、夹爪限位、相机/时间戳和动作频率均已现场确认；服务端只输出动作，不替代机器人侧限幅和安全控制。

在正式 K-Policy 上可先设置变量并做只读检查：

```bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/yam
export REPO_ROOT=/home/wuyan/lyj/YAM/YAM_code
export PYTHON="$CONDA_PREFIX/bin/python"
export OUTPUT_ROOT=replace_with_new_platform_output_root
export CONFIG=pi05_openarm_kai0_awbc_v1
export EXP_NAME=replace_with_exp_name
export STEP=replace_with_step
export CHECKPOINT_DIR="$OUTPUT_ROOT/$CONFIG/$EXP_NAME/$STEP"
export ASSET_ID=openarm_kai0_awbc_v1
export PROMPT='Fold the T-shirt properly, Advantage: positive'

cd "$REPO_ROOT"
test -f "$CHECKPOINT_DIR/_CHECKPOINT_METADATA"
test -f "$CHECKPOINT_DIR/params/_METADATA"
test -f "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json"
"$PYTHON" -m json.tool "$CHECKPOINT_DIR/assets/$ASSET_ID/norm_stats.json" >/dev/null
```

`replace_with_exp_name` 和 `replace_with_step` 是必须替换的占位符；不要把研究计划里的历史 step 或另一个数据版本的 norm stats 直接复制过来。服务代码会优先加载 checkpoint 内的 `assets/<asset_id>/norm_stats.json`，`--policy-repo-id` 只覆盖构造 transform 时的 `repo_id`，不能替代 checkpoint stats。

## 2. 参数与初始位姿

### 2.1 服务和 action-chunk 参数

| 参数 | 入口 | 当前建议 | 修改后检查 |
|---|---|---:|---|
| `port` | `scripts/serve_policy.py --port` | `6666` | 端口占用、真实 WebSocket smoke |
| prompt | `--force-prompt` / `--default-prompt` | K-Policy 强制 `Fold the T-shirt properly, Advantage: positive` | handshake 和 smoke 报告 |
| `rtc_mode` | `--rtc-mode {off,auto,only}` | 首轮 `off` | 按 §7 的三种协议分别 smoke |
| action horizon | `src/openpi/training/config.py` 的 model config | 模型 `50`，不能与数据合同脱节 | loader、server metadata、客户端形状 |
| 执行步长 | 机器人侧 `ActionChunkBroker(action_horizon)` 或等价 client 参数 | 不大于 `50`；真机先用短步长 | 记录每次重规划间隔和推理延迟 |
| 控制频率 | 机器人侧 ROS/client | 与 checkpoint metadata 的 `control_hz=30` 对齐 | 现场测频、时间戳和动作限幅 |
| 相机/队列/TDA | 新平台/机器人侧脚本（路径待核实） | 默认三路 `320x240@30 MJPG`、policy `30Hz`、prefetch `25`、`tda_smooth` | 先核实脚本归属，再用 `--help`/`--entrypoint-check` 和真实相机 smoke |

服务参数只影响推理协议，不会改变机械臂 home pose。模型 horizon 是一次返回的动作数；client 执行步长是多久重新请求一次，二者不要混写。

### 2.2 OpenArm 初始/复位位姿

本仓库没有 OpenArm 机器人驱动和初始关节常量。`serve_policy.py` 只输出动作；`packages/openpi-client/runtime/runtime.py` 只调用环境 `reset()`，不提供关节值。旧服务器记录中的 `/home/lyj/openarm_ros2_docker` 在新平台尚未核实，不能作为新服务器默认路径；真实机器人 ROS/driver/client 主机和 home pose 需要单独确认：

| 场景 | 修改位置 | 说明 |
|---|---|---|
| 真机推理/退出回零 | `scripts/start_real_inference_openpi.sh`、`scripts/start_real_inference_lerobot.sh` 的 `home_all()` | policy 启动前和退出时都会调用；两份脚本要保持一致 |
| 真机 HIL | `scripts/start_real_hil_dagger_openpi.sh` 的 `home_all()` | 同步检查 `--home-duration-sec` |
| VR/Teleop | `scripts/start_real_vr_teleop.sh` 的 `home_arms` | 只影响 teleop，不会自动改推理脚本 |
| 仿真初始值 | `src/openarm_bimanual_moveit_config/config/initial_positions.yaml` | 只影响 fake ros2_control，不会改变 DM 真机 |

一次性真机回零（先启动 bringup、确认只有一个 `/openarm/joint_target` 写入者，并在低速/急停可用条件下执行）：

```bash
# 在已核实的机器人 ROS/driver 主机执行；新平台暂未给出该路径
cd replace_with_verified_openarm_ros_root
ros2 run openarm_arm openarm-arm home both \
  --position 0 0 0 0 0 0 0 --gripper 0.9 \
  --duration-sec 5 --rate-hz 50 --wait-for-command-slot-sec 2
```

`--position` 是 7 个 ROS 弧度关节值；`--gripper 0.9` 是 ROS 开口量，不是训练合同里的 `-66` motor degree。没有夹爪控制时使用 `--no-gripper`。永久改 home 时只改上述 ROS 脚本的 `home_all/home_arms`，并执行 `bash -n scripts/start_real_*.sh`；推理/HIL 用 `--entrypoint-check`，VR teleop 另支持 `--self-check`。不要改本仓库 policy config 或 `--rtc-metadata`。

回零平滑参数由 `openarm-arm home` 控制：`--duration-sec 5`、`--rate-hz 50`、`--max-step-rad 0.016`、`--settle-sec 0.5`。出现抖动或冲击时优先增大 duration、减小 max-step，改完仍需低速空载复核；不要用调 policy 频率代替硬件回零限速。

改位姿后，先低速空载复位至少 5 次，检查碰撞、工作空间、急停、夹爪和三路相机；再记录 OpenArm 16D state，确认客户端转换到 degree/HQ 合同，并核对新 reset pose 与训练数据起始分布。分布变化明显时先采集/清洗新数据，不能只改 home 后直接沿用旧 checkpoint。

`examples/aloha_real/constants.py:START_ARM_POSE`、`examples/aloha_real/real_env.py:DEFAULT_RESET_POSITION` 和 ALOHA 的 `policy_metadata.reset_pose` 只对 ALOHA legacy 生效，不能复制到 OpenArm。`--rtc-metadata` 只用于握手 metadata。

## 3. 启动、健康检查和停止服务

### 3.1 用 tmux 启动

长时间服务不要依赖 SSH 前台会话。先在服务 GPU 节点完成 [02](02_installation_and_environment.md) 的 GPU 审计，再执行：

```bash
export SERVE_HOST=replace_with_allocated_service_host
export SERVE_PORT=replace_with_verified_service_port
export SERVE_SESSION=openarm_policy
mkdir -p "$OUTPUT_ROOT/logs/serve/$CONFIG"
tmux new-session -s "$SERVE_SESSION" -c "$REPO_ROOT"
```

进入 tmux 后逐行执行，完成后按 `Ctrl-b d` 脱离：

```bash
set -o pipefail
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/serve_policy.py \
  --port "$SERVE_PORT" \
  --force-prompt "$PROMPT" \
  --rtc-mode off \
  policy:checkpoint \
  --policy.config="$CONFIG" \
  --policy.dir="$CHECKPOINT_DIR" \
  2>&1 | tee "$OUTPUT_ROOT/logs/serve/$CONFIG/${EXP_NAME}_${STEP}.log"
```

`--force-prompt` 会替换客户端传来的 prompt；`--default-prompt` 只在客户端没有 prompt 时补全，不能作为 K-Policy 的强制条件。OpenArm 没有适合本项目的 `DEFAULT_CHECKPOINT`，必须显式使用 `policy:checkpoint`。

### 3.2 健康检查和日志

`/healthz` 是 HTTP 健康检查，不会触发模型推理：

```bash
curl --fail --silent "http://$SERVE_HOST:$SERVE_PORT/healthz"
tmux ls
tail -n 80 "$OUTPUT_ROOT/logs/serve/$CONFIG/${EXP_NAME}_${STEP}.log"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

`OK`、端口监听或进程存在只说明服务进程存活，不说明 checkpoint transform、动作维度或 prompt 合同正确。必须继续做第 4 节的真实 WebSocket smoke。停止服务前记录 smoke 报告和日志，然后在对应 tmux 中按 `Ctrl-c`；不要通过 kill/重启其他训练节点来“修复”服务。

## 4. 真实 WebSocket smoke

仓库脚本会连接真实服务，发送三路零图像、16D 零 state 和 prompt，检查返回动作及握手 metadata，并把结果原子写入 JSON。它验证的是**服务合同和可推理性**，不是衣服任务成功率：

```bash
cd "$REPO_ROOT"
"$PYTHON" scripts/smoke_test_openarm_policy_server.py \
  --host "$SERVE_HOST" --port "$SERVE_PORT" \
  --checkpoint "$CHECKPOINT_DIR" \
  --prompt "$PROMPT" \
  --timeout-seconds 600 \
  --output "$OUTPUT_ROOT/logs/serve/$CONFIG/${EXP_NAME}_${STEP}_smoke.json"
```

报告的 `passed` 必须为 `true`，并同时确认：

- `actions.shape == [50, 16]`，所有元素有限；
- metadata 为 `action_horizon=50`、`robot_action_dim=16`、`output_action_dim=16`；
- `action_unit=degrees`、`gripper_unit=hq_motor_degrees`、夹爪开/闭为 `0/-66`；
- 报告中的 host、port、checkpoint、prompt 与本次服务一致。

脚本使用 `openpi-client` 的 msgpack-numpy WebSocket 协议，不是普通 JSON HTTP API。机器人客户端至少要发送下列 observation 键：

```text
observation.state       float array, shape (16,)
observation.images.base uint8 image, HWC (推荐 224x224x3)
observation.images.left_wrist  uint8 image, HWC
observation.images.right_wrist uint8 image, HWC
prompt                  string
```

服务握手先返回 metadata；普通推理请求返回 `actions`，以及可选的 `policy_timing`/`server_timing`。服务端支持 CHW 图像和浮点图像的兼容解析，但真机应统一为 HWC `uint8`，避免客户端各自隐式缩放。一个 smoke 通过后，仍需用真实相机和真实 state 做低风险单步验证。

## 5. Rollout 分层和停止条件

按下面顺序逐级放量，每一级都绑定同一个 checkpoint、config、prompt 和数据版本：

```text
静态 checkpoint gate
  -> 零图/固定图 WebSocket smoke
  -> 仿真或离线回放
  -> 真机无物体/低风险单步
  -> 固定任务集小批量（人工接管）
  -> HIL raw
  -> clean/export
  -> 训练、离线评估和下一轮 rollout
```

出现以下任一情况立即停止：动作非有限或维度不是 50x16、单位/夹爪方向错误、prompt 未强制、相机/时间戳不同步、急停或人工接管失效、越过工作空间/夹爪限位、checkpoint 半写入。保留原始日志和数据，修复后从 checkpoint gate 重新开始；不要以“看起来能动”替代 smoke。

固定任务集至少记录：正确对角线选择率、无进展重复甩平率、已展开后继续甩平率、完整折叠成功率、每集接管次数、恢复成功率、推理延迟和动作越界次数。训练 loss 或单条成功演示不能替代真机结论。

## 6. HIL 采集合同

- **固定 collector**：一批数据从开始到结束只使用一个 checkpoint、一个 prompt 和一个服务配置；不要中途切换 20k/79999 或改变 `rtc_mode`。
- **raw 不覆盖**：保存 policy action、human/VR action、hold、intervention、视频、时间戳、`episode_success` 和 `recovery_success`；失败前缀必须保留，便于分析策略卡住的阶段。
- **clean 有规则**：清理可删除 `intervention_hold` 等等待帧，但不能删除真实 human VR correction，也不能把 hold 当接管；检查 16D、单位、时间单调、视频尾帧和成功结尾。
- **版本隔离**：raw 与 clean 使用不同目录/数据 id；导出只能写新目录，不能原地覆盖 raw 或冻结 K-Data。导出和 norm 规则见 [04 § 从 raw 转成新数据版本](04_data_contracts.md)。
- 当前研究计划中的 HIL-T30 是“错误对角线、重复甩平、已展开后未进入折叠”三类恢复各 10 条；它是计划数量，不代表已经采集完成，必须以远端 metadata 和审计报告为准。

## 7. RTC：协议、模式和回退

RTC 是可选的 action chunk 重规划路径。旧的普通 `obs -> policy.infer(obs)` 路径必须始终可用；OpenArm 第一次部署和每次 RTC 改动都先用 `--rtc-mode off` 做 baseline。

### 7.1 启动模式

```bash
# 默认/旧路径：忽略 RTC 字段
--rtc-mode off

# 收到 RTC envelope 时尝试 RTC；字段非法、模型不支持或推理失败则回退普通推理
--rtc-mode auto

# 必须收到并成功执行 RTC；非法或失败直接返回请求错误
--rtc-mode only
```

`rtc_mode` 会写入握手 metadata。正式 K-Policy 的 handshake 同时包含 `action_dim/model_action_dim=32`（模型内部维度）和 `robot_action_dim/output_action_dim=16`（机器人输出维度）。`auto` 的回退必须在 rollout 日志中记录 `server_timing.rtc_error` 或 `rtc_warnings`；不能把一次自动回退标成“RTC 成功”。

### 7.2 请求 envelope

普通客户端发送 observation 字典即可；RTC 客户端发送 msgpack-numpy 对象，结构如下（这是协议示意，不是 JSON HTTP 请求）：

```text
{
  "type": "infer",
  "obs": {"observation.state": ..., "observation.images.base": ..., "prompt": ...},
  "rtc": {
    "action_horizon": 50,
    "action_dim": 32,
    "prev_actions": [[... 32 values ...], ...],
    "d": <integer>,
    "s": <integer>,
    "reset": false
  }
}
```

`prev_actions` 必须是二维、**32D 模型空间**数值数组；正式 OpenArm 的前 16 维是 degree/HQ 动作，后 16 维按模型 padding 合同补零。服务端会检查 horizon/dim 与 metadata 是否一致，将 `d/s` 限制到合法范围，并把长度不足的历史动作补齐、过长的截断。`reset=true` 会忽略历史动作。客户端必须明确 `prev_actions` 的 degree/HQ 语义和 `d/s` 计数，不要把 16D 机器人输出直接当成 RTC `prev_actions`，也不要在服务端偷偷做弧度或夹爪转换。

当前 `WebsocketClientPolicy.infer()` 发送的是普通 observation；要使用 envelope，需要实现/审查能够发送上述 msgpack 对象的 RTC 客户端，并用 `server_timing.rtc_used` 验证实际走了 RTC。`only` 模式不要直接用于首轮真机。

### 7.3 RTC 验收矩阵

| 服务模式 | 普通 observation | 合法 RTC envelope | 非法/不支持 RTC |
|---|---|---|---|
| `off` | 普通推理 | 忽略 RTC，普通推理 | 普通推理 |
| `auto` | 普通推理 | RTC 推理 | 回退普通推理并记录 warning/error |
| `only` | 请求错误 | RTC 推理 | 请求错误 |

每次改动至少跑 `off` 普通 smoke，再跑 `auto` 合法/非法两组；只有旧路径和 RTC 路径都通过，才允许进入真机小批量。Piper 的 metadata 或 action transform 不属于 OpenArm RTC 合同。

## 8. Rollout 记录模板

每一集至少记录以下字段，存放在与 raw/clean 数据同名的 manifest 或实验报告中：

```text
date/time, operator, robot, checkpoint, config, git_commit
prompt, rtc_mode, server_host/port, client_commit
camera/state calibration, task_seed, intervention_count
success, recovery_success, failure_stage, failure_reason
action_shape/unit, infer_ms, safety_stop, raw_video/path
```

把跨实验的结论和事故摘要写入 [07 · 变更历史](07_change_log.md)；不要在本页复制某一次实验的成功率或当前 collector step。
