# Piper 双臂远程推理（服务器端/客户端）

本说明针对双臂 Piper 数据集（`local/pen`）的远程推理流程。服务端跑模型，机器人端仅负责采集观测并发送到服务端。

## 0. 前提检查
- 已完成训练，且 checkpoint 目录存在（`params/` 或 `model.safetensors`，同时有 `assets/local/pen/norm_stats.json`）。
- 服务器与机器人网络互通，端口已放行。
- 远端服务使用 `python` 启动（离线环境不使用 `uv run`）。

## 1. 服务器端启动（policy server）
离线环境建议先设置：
```bash
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

启动服务（注意 `--port` 必须在 `policy:checkpoint` 之前）：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

切换为 `pi0`：
```bash
python scripts/serve_policy.py \
  --port 6666 \
  policy:checkpoint \
  --policy.config=pi0_piper_dual \
  --policy.dir /output/openpi/pi0_piper_dual/piper_ft_pi0/4999
```

可选：固定默认 prompt
```bash
python scripts/serve_policy.py \
  --port 6666 \
  --default-prompt "put the pen into the box" \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

## 2. 机器人端启动（remote client）
安装最小依赖包（机器人侧）：
```bash
pip install -e /home/lyj/lyj/openpi/packages/openpi-client
```

使用你现有的脚本（推荐）：
```bash
python /home/lyj/orin_VR/inference/run_piper_inference_remote.py \
  --teleop-config configs/piper_recording.json \
  --repo-id local/pen_eval \
  --single-task "put the pen into the box" \
  --policy-host 192.168.1.10 \
  --policy-port 6666 \
  --image-dtype uint8 \
  --resize-height 224 \
  --resize-width 224 \
  --action-horizon 50
```

要点：
- `--policy-host` 填服务器 IP（不要用 `0.0.0.0`）。
- `--resize-height/width 224` 建议开启（训练时会缩放到 224，客户端预缩放可减少带宽）。
- `--action-horizon` 要与训练配置一致（`pi05_piper_dual` 默认是 50，若你改过，以你的配置为准）。

## 3. 推理输入格式（与数据集对齐）
服务端期望的观测键（与 `meta/info.json` 一致）：
- `observation.state`：14 维（右臂 6 + 右夹爪 + 左臂 6 + 左夹爪）
- `observation.images.top_rgb`：HWC, uint8 或 float32
- `observation.images.left_wrist`：HWC, uint8 或 float32
- `observation.images.right_wrist`：HWC, uint8 或 float32
- `prompt`：任务描述字符串

相机命名必须匹配 `top_rgb/left_wrist/right_wrist`，否则服务端会报缺 key。

## 4. 图像尺寸与 dtype 建议
- 训练侧统一会 `resize_with_pad` 到 224×224。
- 推理时可直接发送 224×224（推荐）
- 建议用 `uint8` 传输（`openpi_client.image_tools` 会保持训练一致的处理逻辑）。

## 5. 常见问题
1) `--port` 报错  
必须放在 `policy:checkpoint` 之前。

2) 离线环境仍尝试下载  
确认已预下载权重到 `OPENPI_DATA_HOME`，并设置离线环境变量（见第 1 节）。

3) 提示 `Prompt is required`  
需要传入 `prompt` 或启动服务时指定 `--default-prompt`。
