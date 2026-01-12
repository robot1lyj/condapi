# Piper 双臂微调指南（pi0_base / pi0_droid）

本指南面向 `local/pen` 这类双臂 Piper 数据集（约 150 条演示）进行微调，关注小数据集稳定训练与现实评测记录。

## 1. action_dim=32 与基座权重的关系（必须理解）
- `action_dim` 是模型结构的一部分，**决定权重形状**。  
  官方基座（`pi0_base` / `pi0_droid`）均是按 `action_dim=32` 训练的。
- **如果你要加载这些权重，就必须保持 `action_dim=32`**；  
  如果把 `action_dim` 改为 14，形状不匹配，会导致权重无法加载，只能从头训练。
- 本仓库已自动做了 padding：  
  数据集动作是 14 维，经过 `ModelTransformFactory -> PadStatesAndActions` 会补零到 32 维，因此可以直接微调。

结论：  
**要微调 `pi0_base` / `pi0_droid`，就保留 `action_dim=32`。**

## 2. pi0_base vs pi0_droid 的影响
- `pi0_base`：更通用，跨任务/跨平台适配更稳，收敛慢一些但不容易过拟合。  
- `pi0_droid`：对 DROID 领域更偏向，可能更快收敛，但也可能更“偏分布”。  
  你可以把它视为一个更“强先验”的初始化，建议 **更小的学习率**。

推荐做法：两个都试，同一套超参只调整 `weight_loader`。

## 3. 小数据集（~150 条）推荐微调超参
下面是**起步值**，不一定最优，但适合先跑通并观察真实机器人表现：

- `batch_size`: 8 或 16（优先 8，避免过拟合/显存爆）
- `num_train_steps`: 3k ~ 8k
- `log_interval`: 20
- `save_interval`: 200 ~ 500（方便你经常落盘评测）
- `keep_period`: 1000 ~ 2000
- `lr_schedule.peak_lr`:  
  - `pi0_base`：2e-5 ~ 3e-5  
  - `pi0_droid`：1e-5 ~ 2e-5（更稳）
- `lr_schedule.warmup_steps`: 200 ~ 500
- `ema_decay`: 0.99 或 0.999（小数据集可略高）

> 如果你发现现实效果明显下降或过拟合迹象：降低 `peak_lr`、减少 `num_train_steps`、增大 `save_interval`（便于早停）。

## 4. 微调命令示例
以 `pi0_piper_dual` 为例（权重来自 `pi0_base`）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_piper_dual

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_piper_dual \
  --exp-name piper_ft_base \
  --checkpoint-base-dir /output/openpi \
  --batch-size 8 \
  --num-train-steps 5000 \
  --log-interval 20 \
  --save-interval 200 \
  --keep-period 1000 \
  --lr-schedule.peak-lr 2e-5 \
  --lr-schedule.warmup-steps 300
```

如果你要从 `pi0_droid` 微调：
1) 把 `src/openpi/training/config.py` 中 `pi0_piper_dual` 的 `weight_loader` 改成你的 `pi0_droid` 权重路径；  
   例如：`CheckpointWeightLoader("/path/to/pi0_droid/params")`
2) 用同样命令跑，但把 `peak_lr` 调小一点。

### 离线环境运行建议（无外网）
如果容器无法访问公网，建议先关闭所有可能联网的入口，并确保权重/资产都在本地缓存。

**离线强制关闭联网（先执行一次）**
```bash
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

**权重与资产本地化（避免访问 gs:// 或 HF）**
- 优先使用本地路径：`/openpi_cache/openpi-assets/checkpoints/.../params`
- 如果仍使用 `gs://`，请确保对应内容已缓存到 `OPENPI_DATA_HOME`（默认 `/openpi_cache/openpi-assets`）。

`uv run` 可能会尝试同步依赖并访问 PyPI。建议改用以下方式：

**方式 A：直接用已安装的 venv Python（推荐）**
```bash
python scripts/compute_norm_stats.py --config-name pi0_piper_dual

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi0_piper_dual \
  --exp-name piper_ft_base \
  --checkpoint-base-dir /output/openpi \
  --batch-size 8 \
  --num-train-steps 5000 \
  --log-interval 20 \
  --save-interval 200 \
  --keep-period 1000 \
  --lr-schedule.peak-lr 2e-5 \
  --lr-schedule.warmup-steps 300 \
  --wandb-enabled false
```

**方式 B：继续用 uv，但禁止同步**
```bash
UV_NO_SYNC=1 \
uv run --no-sync scripts/compute_norm_stats.py --config-name pi0_piper_dual

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
UV_NO_SYNC=1 \
uv run --no-sync scripts/train.py pi0_piper_dual \
  --exp-name piper_ft_base \
  --checkpoint-base-dir /output/openpi \
  --batch-size 8 \
  --num-train-steps 5000 \
  --log-interval 20 \
  --save-interval 200 \
  --keep-period 1000 \
  --lr-schedule.peak-lr 2e-5 \
  --lr-schedule.warmup-steps 300 \
  --wandb-enabled false
```

如果出现 `Permission denied` 的缓存问题，可临时指定可写缓存目录：
```bash
UV_CACHE_DIR=/openpi_cache/uv \
uv run --no-sync scripts/compute_norm_stats.py --config-name pi0_piper_dual
```

## 5. 训练曲线与日志在哪里？
Piper 配置已默认 `wandb_enabled=false`，避免任何联网行为。  
如需曲线，请显式开启 `--wandb-enabled true`（可能触网），并自行评估风险。

**输出目录 `/output/openpi` 里默认只有 checkpoint 与 assets**，不包含曲线。  
如果需要曲线，请自行启用 wandb 或外部日志方案。

## 6. 现实评测记录表（填写你自己的结果）
> 只填“真实机器人效果”，不要填 loss/curve。

| exp_name | init_weight | action_horizon | batch_size | peak_lr | steps | delta(12 joints) | real_world_result | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | pi0_base / pi0_droid | 50 |  |  |  | yes |  |  |
|  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |
