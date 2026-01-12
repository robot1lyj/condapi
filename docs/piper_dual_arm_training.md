# Piper 双臂数据集训练配置说明（openpi）

本说明基于 `docs/offline_docker_deploy.md` 的离线 Docker 部署流程，并结合本地双臂数据集示例 `/share/home/linyongjia/data/pen`。

## 1. 需要修改的代码位置
- `src/openpi/policies/piper_policy.py`  
  新增/更新 Piper 双臂数据的输入输出转换：读取双腕/顶视图图像与 14 维状态/动作，并映射到模型需要的
  `base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb` 与 `state / actions`。
- `src/openpi/training/config.py`  
  新增 `LeRobotPiperDataConfig` 数据配置，并添加训练配置 `pi0_piper_dual`、`pi05_piper_dual`。

## 2. 数据集与键映射（/share/home/linyongjia/data/pen）
数据集 `meta/info.json` 中的关键字段（示例）：
- `observation.state`: 14 维（右臂 6 关节 + 右夹爪 + 左臂 6 关节 + 左夹爪）
- `action`: 14 维（**右臂在前、左臂在后**，顺序为 `right_*` 再 `left_*`）
- `observation.images.top_rgb`
- `observation.images.left_wrist`
- `observation.images.right_wrist`
- 动作为**绝对关节角**

映射到模型输入：
- `observation.images.top_rgb` -> `image.base_0_rgb`
- `observation.images.left_wrist` -> `image.left_wrist_0_rgb`
- `observation.images.right_wrist` -> `image.right_wrist_0_rgb`
- `observation.state` -> `state`
- `action` -> `actions`

## 3. 双臂 Piper 配置要点
针对 `/share/home/linyongjia/data/pen`，在 `LeRobotPiperDataConfig` 中建议设定：
- `action_sequence_keys = ("action",)`  
  数据集动作字段是 `action`（非 `actions`）。
- `robot_action_dim = 14`  
  双臂动作维度。
- `use_delta_joint_actions = True`  
  数据集动作是**绝对关节角**，建议开启增量转换，只对 12 个关节做 delta，两个夹爪保持绝对值。
- `swap_left_right = False`  
  数据集中 **右臂在前、左臂在后**，无需交换顺序。若你希望按“左在前”对齐训练习惯，再改为 `True`。
- `prompt_from_task = True`  
  使用 `tasks.jsonl` 中的任务描述作为 prompt。

## 4. Pi0 与 Pi0.5 的配置差异
### Pi0
- `model = Pi0Config()`  
- 状态输入是连续的，prompt 只包含任务文本。
- 推荐加载 `pi0_base` 作为初始化权重。
- 若希望加载官方权重，建议保持 `action_dim=32` 与默认 `max_token_len`，通过 padding 适配 14 维动作。

### Pi0.5
- `model = Pi0Config(pi05=True, discrete_state_input=True)`  
- 状态被离散化并拼入 prompt token。
- 使用 `pi05_base` 初始化权重。
- `max_token_len` 默认 200，一般足够双臂；若训练中出现 token 截断警告，可适当增大。

> 注意：如果修改 `action_dim` 或 `max_token_len`，将无法直接加载官方基座权重（形状不匹配），需要从头训练或换用匹配权重。

## 5. 训练前的容器挂载（离线 Docker）
默认数据目录为宿主机 `/share/home/linyongjia/data`，容器内映射为 `/data`，并设置 `HF_LEROBOT_HOME=/data`。
`repo_id=local/pen` 对应的实际路径是：
```
/data/local/pen
```
因此数据集在宿主机侧应位于：
```
/share/home/linyongjia/data/local/pen
```
容器内路径就是 `/data/local/pen`，对应的 `repo_id` 直接使用 `local/pen`（与 `meta/info.json` 一致），无需改动。

`local/pen` 里的 `local` 是 LeRobot 的本地数据前缀，表示从 `$HF_LEROBOT_HOME/local/pen` 读取数据，不是一个真实目录名。

如果你的数据现在在 `/share/home/linyongjia/data/pen`，可以二选一：
- 移动目录：`mv /share/home/linyongjia/data/pen /share/home/linyongjia/data/local/pen`
- 或创建软链：`ln -s /share/home/linyongjia/data/pen /share/home/linyongjia/data/local/pen`

如果你把数据放到了其它路径，二选一即可：
- 调整挂载：`scripts/docker/run_dev.sh --lerobot-data /你的/数据根目录`
- 或直接绑定：`scripts/docker/run_dev.sh --bind /实际数据集路径:/data/local/pen`

确保数据集在容器内可见，例如：
```bash
scripts/docker/run_dev.sh --gpus all --bind /share/home/linyongjia/data/local/pen:/data/local/pen -- /bin/bash
```
容器内默认设置 `HF_LEROBOT_HOME=/data`，因此 `repo_id` 应为 `local/pen`。

## 6. 计算归一化统计 + 启动训练
`scripts/compute_norm_stats.py` 无需改动，直接按配置运行即可。

```bash
# 1) 计算归一化统计
uv run scripts/compute_norm_stats.py --config-name pi05_piper_dual

# 2) 启动训练（输出到 /output/openpi）
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_piper_dual \
  --exp-name piper_dual_exp \
  --checkpoint-base-dir /output/openpi
```
如果训练 Pi0，替换为 `pi0_piper_dual`。
Piper 配置已默认 `wandb_enabled=false`，如需启用请显式传 `--wandb-enabled true`。

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
python scripts/compute_norm_stats.py --config-name pi05_piper_dual

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py pi05_piper_dual \
  --exp-name piper_dual_exp \
  --checkpoint-base-dir /output/openpi \
  --wandb-enabled false
```

**方式 B：继续用 uv，但禁止同步**
```bash
UV_NO_SYNC=1 \
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_piper_dual

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
UV_NO_SYNC=1 \
uv run --no-sync scripts/train.py pi05_piper_dual \
  --exp-name piper_dual_exp \
  --checkpoint-base-dir /output/openpi \
  --wandb-enabled false
```

如果出现 `Permission denied` 的缓存问题，可临时指定可写缓存目录：
```bash
UV_CACHE_DIR=/openpi_cache/uv \
uv run --no-sync scripts/compute_norm_stats.py --config-name pi05_piper_dual
```
