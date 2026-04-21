# Piper 双臂远程推理（服务器端）

本说明针对双臂 Piper 数据集（`local/pen`）的服务端推理流程。

## 0. 前提检查
- 已完成训练，且 checkpoint 目录存在（`params/` 或 `model.safetensors`，同时有 `assets/local/pen/norm_stats.json`）。
- 服务器与机器人网络互通，端口已放行。
- 远端服务默认使用 `conda run -n pi-conda python` 启动（离线环境不使用 `uv run`）。

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
conda run -n pi-conda python scripts/serve_policy.py \
  --port 6666 \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /share/home/linyongjia/output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

切换为 `pi0`：
```bash
conda run -n pi-conda python scripts/serve_policy.py \
  --port 6666 \
  policy:checkpoint \
  --policy.config=pi0_piper_dual \
  --policy.dir /share/home/linyongjia/output/openpi/pi0_piper_dual/piper_ft_pi0/4999
```

可选：固定默认 prompt
```bash
conda run -n pi-conda python scripts/serve_policy.py \
  --port 6666 \
  --default-prompt "put the pen into the box" \
  policy:checkpoint \
  --policy.config=pi05_piper_dual \
  --policy.dir /share/home/linyongjia/output/openpi/pi05_piper_dual/piper_ft_pi05/4999
```

## 2. 常见问题
1) `--port` 报错  
必须放在 `policy:checkpoint` 之前。

2) 离线环境仍尝试下载  
确认已预下载权重到 `OPENPI_DATA_HOME`，并设置离线环境变量（见第 1 节）。
