# Piper 双臂微调超参说明（pi0_base / pi0_droid）

本说明只讲“超参数含义 + 针对 150 条左右数据的建议”，并给出 pi0/pi05 的启动样例。

## 1. 先决条件（简要）
- 使用 `repo_id=local/pen`，本地路径需满足 `/data/local/pen`（详见训练说明文档）。
- 离线环境建议先设置：
```bash
export UV_CACHE_DIR=/openpi_cache/uv
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```
- 先计算归一化统计：
```bash
python scripts/compute_norm_stats.py --config-name pi0_piper_dual
```

## 2. 关键超参数含义与建议
下面是“含义 + 针对你数据的建议 + 为什么要这样”的核心表：

| 参数 | 含义 | 建议（~150 条数据） | 为什么 |
| --- | --- | --- | --- |
| `batch_size` | 每步训练用多少样本 | 8（显存紧张可 4） | 小数据更稳，过大易过拟合 |
| `num_train_steps` | 总训练步数 | 4000~8000 | 你的数据 `total_frames≈32k`，`batch=8` 时约 4000 steps/epoch，先跑 1~2 个 epoch 更稳 |
| `lr_schedule.peak_lr` | 峰值学习率 | pi0_base: 2e-5~3e-5<br>pi0_droid: 1e-5~2e-5 | droid 权重更“偏分布”，LR 要更小 |
| `lr_schedule.warmup_steps` | 预热步数 | 200~500 | 小数据初期易震荡，预热可稳住 |
| `save_interval` | 多久保存一次 | 200~500 | 方便频繁线下实测与早停 |
| `log_interval` | 多久打印一次 | 20~50 | 小数据易不稳定，便于观察 |
| `ema_decay` | EMA 平滑 | 0.99~0.999 | 输出更稳定，减小抖动 |
| `action_horizon` | 预测的动作长度 | 10~50 | 决定“看多远”，见下节 |

### 为什么步数不建议太大？
小数据很容易“学会训练集、失去泛化”。  
如果你用 `batch_size=8`、`total_frames≈32k`，每个 epoch 约 `4000` steps。  
先跑 1~2 个 epoch，现实评测效果不够再加步数，是最稳的方式。

### step 与 batch_size 的关系（怎么估算）
用 `meta/info.json` 的 `total_frames` 粗算：
```
steps_per_epoch ≈ total_frames / batch_size
epochs ≈ num_train_steps / steps_per_epoch
```
例子：`total_frames=32000`，`batch_size=8`  
`steps_per_epoch≈4000`，`num_train_steps=5000` 约等于 `1.25` 个 epoch。  
如果把 `batch_size` 改成 4，想保持 1~2 个 epoch，就需要把 `num_train_steps` 乘以 2。

## 3. action_horizon（action chunk）需要对齐帧数吗？
不需要对齐总帧数。  
`action_horizon` 是“模型一次预测多少个未来动作”，时间跨度为：

```
action_horizon / fps
```
你的数据 `fps=30`：  
- `horizon=10` ≈ 0.33s（更灵敏）  
- `horizon=20` ≈ 0.67s（中等）  
- `horizon=50` ≈ 1.67s（更长规划）  

建议：  
- 想更快响应：先用 10~20  
- 需要更长规划：保持 30~50  
训练和推理必须一致，否则动作时间尺度会错。

## 4. 训练命令样例（python）
下面示例以 `pi0_piper_dual` 为例，权重来自 `pi0_base`：
```bash
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

如果你要用 `pi0_droid` 权重：  
- 将 `weight_loader` 改成本地 `pi0_droid` 路径  
- 把 `peak_lr` 降到 `1e-5 ~ 2e-5`

再给一个 **pi05** 的微调样例（权重来自 `pi05_base`）：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_piper_dual \
  --exp-name piper_ft_pi05 \
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

## 5. pi0 vs pi05 的微调差异（官方定义 + 调参建议）
官方差异要点：
- **状态输入**：pi0 使用连续 state；pi05 会把 state 离散化并拼入语言 token。  
- **动作专家**：pi05 使用 adaRMSNorm 注入时间步；pi0 使用 MLP 融合时间/动作。  
- **默认 token 长度**：pi05 默认 `max_token_len=200`，pi0 默认更短。

对微调的影响与建议：
- **prompt 与 state**：pi05 依赖 prompt+离散 state，确保 `prompt_from_task=True`，且状态范围合理。  
- **学习率**：pi05 对分布更敏感，建议从 `2e-5` 起步，必要时再降。  
- **token 截断**：若出现 token 截断警告，适当提高 `max_token_len`。

## 6. 现实评测记录表（填写真实机器人结果）
| exp_name | init_weight | action_horizon | batch_size | peak_lr | steps | real_world_result | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | pi0_base / pi0_droid |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
