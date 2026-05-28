# Piper 双臂数据集训练规范（版本化目录方案）

当前推荐方案不再使用 `local/<alias>` 软链，也不再要求在服务器上修改 `config.py`。

训练时直接传入一份**不可变的版本化数据集目录绝对路径**。每个目录都应自包含：

- `meta/`
- `data/`
- `videos/`
- `norm_stats.json`
- 可选：`manifest.yaml`

推荐目录命名：

- `/share/home/linyongjia/datasets/piper_pen_v001`
- `/share/home/linyongjia/datasets/piper_pen_v002`
- `/share/home/linyongjia/datasets/piper_dish1_v001`
- `/share/home/linyongjia/datasets/openarms_folding_v001`

## 1. 核心规则

- 一个目录就是一个正式数据版本。
- 正式版本发布后不要原地修改；新增数据请新建 `v002`。
- `norm_stats.json` 默认与数据集目录绑定，直接写在数据集根目录。
- 训练命令永远显式传绝对路径，不再依赖 `local/pen` 这类别名。

## 2. Piper 默认配置

在 [`src/openpi/training/config.py`](../src/openpi/training/config.py) 中：

- `pi05_piper_dual`
- `pi0_piper_dual`
- `pi05_openarms_dual`
- `pi0_openarms_dual`

这些配置都应在命令行里显式覆盖为数据集绝对路径。

- `pi*_piper_dual`：14 维双臂 Piper 数据
- `pi*_openarms_dual`：16 维 `openarms_follower` 双臂数据

## 3. 训练前环境变量

```bash
unset WANDB_DISABLED
export WANDB_MODE=offline
export WANDB_SILENT=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export HF_HOME=/share/home/linyongjia/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_LEROBOT_HOME=/share/home/linyongjia/datasets
```

## 4. 计算归一化统计

默认会写到数据集根目录：

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id /share/home/linyongjia/datasets/piper_pen_v002
```

如果数据集是从更大的 LeRobot 数据集中抽出来的，建议先执行：

```bash
python scripts/repair_lerobot_subset.py \
  --dataset-dir /share/home/linyongjia/datasets/openarms_folding_v001
```

修复完 parquet/video/metadata 一致性后，再计算 `norm_stats.json`。

快速抽样检查：

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id /share/home/linyongjia/datasets/piper_pen_v002 \
  --max-frames 2000
```

如果你明确需要旧的 assets 目录布局，可以额外传：

```bash
  --output-dir /share/home/linyongjia/conda-pi/openpi/assets/pi05_piper_dual/piper_pen_v002
```

## 5. 启动训练

直接传数据集绝对路径：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_piper_dual \
  --exp-name piper_pen_v002_bs32_fsdp2 \
  --checkpoint-base-dir /share/home/linyongjia/output/openpi \
  --data.repo_id /share/home/linyongjia/datasets/piper_pen_v002 \
  --batch-size 32 \
  --log-interval 20 \
  --fsdp-devices 2 \
  --overwrite
```

训练 Pi0 时，将 `pi05_piper_dual` 替换为 `pi0_piper_dual`。

## 6. 一键入口脚本

仓库提供了一个薄封装：

```bash
conda run -n pi-conda bash scripts/piper_dataset_train.sh \
  --dataset-dir /share/home/linyongjia/datasets/piper_pen_v002 \
  --exp-name piper_pen_v002_bs32_fsdp2 \
  --batch-size 32 \
  --log-interval 20 \
  --fsdp-devices 2 \
  --overwrite
```

这个脚本会：

1. 校验 `meta/info.json` 是否存在
2. 默认先在数据集目录内生成 `norm_stats.json`
3. 再启动训练

训练曲线会写入实验目录：

- `wandb/`：离线 W&B run，不会联网同步。
- `metrics/metrics.jsonl` 和 `metrics/metrics.csv`：结构化指标。
- `metrics/plots/training_curves.png`：更密的聚合训练曲线。
- `metrics/plots/<metric>.png`：单指标曲线。

## 7. manifest.yaml 建议

建议每个版本目录附带一份人工可读的 `manifest.yaml`：

```yaml
dataset_id: piper_pen_v002
robot: piper
format: lerobot_v2.1
task: put the pen into the box
episodes: 149
source: piper_pen_v001 + appended_sessions_20260524
status: frozen
norm_stats: ./norm_stats.json
```

## 8. 排错原则

- 如果目录里的 episode 数变了，就新建版本目录，不要原地覆盖。
- 如果 `meta/info.json`、`episodes.jsonl`、`data/*.parquet` 数量不一致，这个版本就不应继续训练。
- 如果数据是从更大的 LeRobot 数据集抽出来的，必须额外检查 parquet 内部 `episode_index` 是否与文件名/metadata 一致。
- 不要在服务器上手改 `config.py` 去切数据集。
- 不要再新增 `local/pen`、`local/dish` 这类训练入口。
