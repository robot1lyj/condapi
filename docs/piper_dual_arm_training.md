# Piper 双臂数据集训练配置说明（openpi）

本说明以离线环境为前提，命令使用 `python` 直接启动（不走 `uv run` 同步）。

## 1. 数据集路径与 repo_id（local/pen）
`repo_id=local/pen` 在 LeRobot 中会解析为：
```
$HF_LEROBOT_HOME/local/pen
```
容器内默认 `HF_LEROBOT_HOME=/data`，所以应当存在：
```
/data/local/pen
```

你的数据在宿主机：
```
/share/home/linyongjia/data/pen
```
推荐用相对软链（宿主机和容器都能解析）：
```bash
mkdir -p /share/home/linyongjia/data/local
rm -f /share/home/linyongjia/data/local/pen
ln -s ../pen /share/home/linyongjia/data/local/pen
```
验证：
```bash
ls /share/home/linyongjia/data/local/pen/meta/info.json
# 容器内
ls /data/local/pen/meta/info.json
```

如果数据集名字变化：
- 改 `repo_id` 并创建同名软链：
```bash
ln -sfn /share/home/linyongjia/data/pen_v2 /share/home/linyongjia/data/local/pen_v2
```
然后把配置改成 `local/pen_v2`。
- 或保持 `repo_id=local/pen`，让旧名指向新数据：
```bash
ln -sfn /share/home/linyongjia/data/pen_v2 /share/home/linyongjia/data/local/pen
```

## 2. 数据集字段与映射（/share/home/linyongjia/data/pen）
`meta/info.json` 关键字段：
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

## 3. Piper 双臂默认配置（已在配置里设置）
在 `src/openpi/training/config.py`：
- `repo_id="local/pen"`
- `action_sequence_keys=("action",)`
- `robot_action_dim=14`
- `use_delta_joint_actions=True`（只对 12 个关节做 delta，两个夹爪保持绝对）
- `swap_left_right=False`（右臂在前）
- `wandb_enabled=False`（离线默认禁用）

## 4. 离线环境变量（先执行一次）
```bash
export UV_CACHE_DIR=/openpi_cache/uv
export WANDB_DISABLED=true
export HF_HUB_OFFLINE=1
export HUGGINGFACE_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## 5. 用 Python 启动（推荐）
### 5.1 计算归一化统计
```bash
python scripts/compute_norm_stats.py --config-name pi05_piper_dual
```
如需快速检查可加：
```bash
python scripts/compute_norm_stats.py --config-name pi05_piper_dual --max-frames 2000
```

### 5.2 启动训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_piper_dual \
  --exp-name piper_dual_exp \
  --checkpoint-base-dir /output/openpi \
  --wandb-enabled false
```
训练 Pi0 时将 `pi05_piper_dual` 替换为 `pi0_piper_dual`。

## 6. 权重与资产本地化
训练配置默认使用 `gs://` 权重路径。离线环境必须确保已缓存到本地，例如：
```
/openpi_cache/openpi-assets/checkpoints/pi05_base/params
/openpi_cache/openpi-assets/checkpoints/pi0_base/params
```
如果不存在，需要在有网机器预下载后拷贝到 `OPENPI_DATA_HOME` 对应路径。

## 7. 权限与缓存排错
- 不要用 `sudo uv run`，否则缓存会被 root 占用。
- 修复 uv 缓存权限：
```bash
sudo mkdir -p /home/linyongjia/.cache/uv
sudo chown -R linyongjia:linyongjia /home/linyongjia/.cache/uv
```
- `/app` 无写权限时修复：
```bash
sudo chown -R linyongjia:linyongjia /app
```
