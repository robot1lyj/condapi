# 05 · 训练后 policy smoke

本页只保留 YAM 训练后 policy 的加载、协议和最小验收。YAM 机械臂控制、CAN、GUI、home pose、控制频率和真机安全不在本仓库适配范围；不要从独立 YAM-ABC-Reproduce 代码推断本项目训练合同。

## 1. 部署前 gate

必须同时满足：

1. 服务使用当前仓库 commit 和通过 import/loader gate 的 Python 3.12 环境。
2. checkpoint 参数元数据完整，且有 `assets/yam/norm_stats.json`。
3. 配置是 `pi05_yam_lora`（或明确记录的 YAM 配置），数据和 checkpoint 属于同一 14D 合同。
4. 真实 WebSocket smoke 确认输入键、输出 shape、有限值和握手 metadata。

## 2. 启动训练后 policy 服务

服务端没有 YAM 的默认 checkpoint，必须显式指定 checkpoint 和配置；在 Slurm GPU 节点用 tmux：

```bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/condapi-yam
export PYTHON="$CONDA_PREFIX/bin/python"
export REPO_ROOT=/home/wuyan/lyj/YAM/YAM_code
export CHECKPOINT_DIR=/path/to/complete/yam_pi05_lora_checkpoint
export PORT=8000

cd "$REPO_ROOT"
tmux new-session -s yam-policy -c "$REPO_ROOT"
```

在 tmux 中执行：

```bash
"$PYTHON" scripts/serve_policy.py \
  --port "$PORT" \
  --rtc-mode off \
  policy:checkpoint \
  --policy.config=pi05_yam_lora \
  --policy.dir="$CHECKPOINT_DIR"
```

`--policy.repo-id` 不是当前 CLI 的参数；若服务构造 transform 需要覆盖数据 repo，使用项目已有的 `--policy-repo-id`，但它不能替代 checkpoint 内 norm stats。首轮保持 `rtc_mode=off`，任何 RTC 改动必须保留普通推理路径并能自动回退。

## 3. YAM WebSocket 输入输出合同

YAM policy 输入使用和 LeRobot 导出一致的键：

```text
observation.state                  float array, shape (14,)
observation.images.top_rgb         RGB image
observation.images.left_rgb         RGB image
observation.images.right_rgb        RGB image
prompt                              scalar string（或由 task 注入）
```

图像可由 `YamInputs` 兼容 CHW/HWC，正式 smoke 建议使用 HWC `uint8`。握手 metadata 应包含 YAM、YAM-ABC-compatible、14D robot action、32D model action 和 horizon 50 等信息。推理返回：

```text
actions: float array, shape (50, 14), all finite
```

模型内部 padding 到 32D 只属于 OpenPI 模型边界，不应把 32D 直接当作 YAM 真实动作发送给机器人侧。

## 4. Smoke 验收

不要以端口监听、HTTP 进程存在或只返回 handshake 作为成功。用真实 `openpi-client` WebSocket 协议发送三路固定图像、14D 零 state 和 prompt，检查：

- 输入被接受且没有缺少 image/state/prompt 错误；
- 输出严格为 `(50,14)`，无 NaN/Inf；
- metadata 的 `robot_action_dim/output_action_dim=14`、`model_action_dim/action_dim=32`、`action_horizon=50` 与 checkpoint config 一致；
- 服务日志记录 commit、checkpoint、config、端口、prompt 和 smoke 结果。

当前仓库尚未提供独立的 YAM 真机 smoke 脚本；先使用 `yam_policy_test.py` 验证 14D transform，再用真实 YAM 客户端做 WebSocket smoke。不要把旧 OpenArm 16D 工具重新作为默认入口。

## 5. 停止和问题处理

服务停止前保存日志和 smoke 报告，在对应 tmux 中 `Ctrl-c`；不要 kill 其他用户进程或现有下载/训练作业。若出现以下任一情况，停止 rollout 并回到 checkpoint/data gate：

- shape 不是 14D 输入或 50x14 输出；
- 动作非有限、checkpoint/norm 不匹配或模型加载不完整；
- 三路图像缺失、通道/时间同步不明；
- 把 OpenArm/Piper 单位、顺序或 transform 混入 YAM。

真机执行动作前，机器人侧必须另行完成其硬件安全和限幅验收；本仓库不提供也不记录 YAM 的机械臂控制参数。
