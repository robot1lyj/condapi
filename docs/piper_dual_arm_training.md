# Piper 双臂数据集训练配置说明（openpi / conda 默认）

当前分支默认按训练服务器的非容器 `conda` 环境运行，环境名是 `pi-conda`。Docker 只保留为可选兼容方案，不再作为默认训练入口。

如果你要按步骤完成离线安装、补齐依赖并启动训练，先看：
[`docs/piper_conda_training.md`](./piper_conda_training.md)

## 1. 数据集与 repo_id（默认 `local/pen`）
Piper 双臂默认训练配置仍然使用 `repo_id=local/pen`。在当前服务器上，常见数据集目录位于：

- `/share/home/linyongjia/data/pen`
- `/share/home/linyongjia/data/towel_merged`
- `/share/home/linyongjia/data/dish1_codex_v1`

推荐做法：

- 固定 `HF_LEROBOT_HOME=/share/home/linyongjia/data`
- 默认配置保持 `local/pen`
- 如果切换数据集，优先在命令行里用 `--repo-id` 覆盖，例如 `local/towel_merged`

快速 smoke test：

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id local/towel_merged \
  --max-frames 2000
```

## 2. 数据集字段与映射（以 `pen` 为例）
`meta/info.json` 常见关键字段：

- `observation.state`: 14 维（右臂 6 关节 + 右夹爪 + 左臂 6 关节 + 左夹爪）
- `action`: 14 维（右臂在前、左臂在后，绝对关节角）
- `observation.images.top_rgb`
- `observation.images.left_wrist`
- `observation.images.right_wrist`

映射到模型输入：

- `top_rgb` -> `image.base_0_rgb`
- `left_wrist` -> `image.left_wrist_0_rgb`
- `right_wrist` -> `image.right_wrist_0_rgb`
- `observation.state` -> `state`
- `action` -> `actions`

## 2.1 orin_VR raw_hdf5 转 LeRobot v2.1
`openpi` 当前不是按最新 LeRobot 主线格式工作，而是依赖仓库里 pin 的旧版 `lerobot`，对应的本地数据布局仍是 LeRobot `v2.1`。

如果你的采集目录还是 `orin_VR` 的 raw 格式（`meta/info.json` 里有 `raw_format_version=orin_vr_raw_hdf5_v1`，例如 `dish1_new` / `dish2_new` 这种），先执行转换：

```bash
python ../data/convert_orin_vr_raw_to_lerobot.py \
  --source /path/to/raw_dataset \
  --target /path/to/output_dataset \
  --repo-id local/pen \
  --validate-openpi-piper-dual
```

常见例子：

```bash
python ../data/convert_orin_vr_raw_to_lerobot.py \
  --source /home/jetson/data/local/dish1_new \
  --target /home/jetson/data/local/dish1 \
  --repo-id local/dish1 \
  --validate-openpi-piper-dual \
  --overwrite
```

输入 raw 目录应包含：

- `meta/info.json`
- `meta/episodes.jsonl`
- `episodes/episode_*.hdf5`
- `videos/<camera_key>/episode_*.mp4`

输出会生成 openpi 可直接读取的 LeRobot v2.1 目录：

- `meta/info.json`
- `meta/episodes.jsonl`
- `meta/tasks.jsonl`
- `meta/episodes_stats.jsonl`
- `meta/stats.json`
- `data/chunk-000/episode_*.parquet`
- `videos/chunk-000/<camera_key>/episode_*.mp4`

## 3. Piper 双臂默认配置（已在配置里设置）
在 `src/openpi/training/config.py`：

- `repo_id="local/pen"`
- `action_sequence_keys=("action",)`
- `robot_action_dim=14`
- `use_delta_joint_actions=True`（只对 12 个关节做 delta，两个夹爪保持绝对）
- `swap_left_right=False`（右臂在前）
- `wandb_enabled=False`（离线默认禁用）

## 4. 默认 conda 运行环境
训练服务器默认使用 `pi-conda`：

```bash
conda activate pi-conda
```

如果不想激活环境，所有命令都可以写成：

```bash
conda run -n pi-conda python ...
```

## 5. 训练前环境变量（先执行一次）
```bash
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export OPENPI_DATA_HOME=/share/home/linyongjia/.cache/openpi
export HF_HOME=/share/home/linyongjia/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_LEROBOT_HOME=/share/home/linyongjia/data
```

## 6. 用 conda 环境启动（默认）
### 6.1 计算归一化统计
```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id local/pen
```

如需快速检查可加：

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id local/pen \
  --max-frames 2000
```

### 6.2 启动训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py pi05_piper_dual \
  --exp-name piper_dual_exp \
  --checkpoint-base-dir /share/home/linyongjia/output/openpi \
  --data.repo_id local/pen \
  --wandb-enabled false
```

训练 Pi0 时将 `pi05_piper_dual` 替换为 `pi0_piper_dual`。

## 7. 权重与资产本地化
训练配置默认使用 `gs://` 权重路径。离线环境必须确保已缓存到本地，例如：

```text
/share/home/linyongjia/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
/share/home/linyongjia/.cache/openpi/openpi-assets/checkpoints/pi0_base/params
```

如果不存在，需要在有网机器预下载后拷贝到 `OPENPI_DATA_HOME` 对应路径。

## 8. 数据与路径排错
- 不要再按 Docker 流程假设 `/data/local/...`、`/output/openpi`、`/.venv` 这些容器路径。
- 训练输出目录默认使用：

```bash
mkdir -p /share/home/linyongjia/output/openpi
```

- 新数据集先做一次 smoke test：

```bash
conda run -n pi-conda python scripts/compute_norm_stats.py \
  --config-name pi05_piper_dual \
  --repo-id <your_repo_id> \
  --max-frames 2000
```

- 如果 `compute_norm_stats` 在 `datasets/parquet` 阶段失败，先检查 `meta/info.json` 和 parquet 实际列名是否一致，尤其是图像 key 是否存在 `left_wrist` / `right_wrist` 与 `*_rgb` 的命名偏差。
- 如果 `transformers` 被重装过，需要重新覆盖 openpi patch：

```bash
conda run -n pi-conda python scripts/conda/patch_transformers.py \
  --openpi-dir /share/home/linyongjia/conda-pi/openpi
```
