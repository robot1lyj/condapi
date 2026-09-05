# OpenPI + YAM 双臂训练适配

本仓库当前用于 YAM 双臂（与 YAM-ABC 同硬件配置）的 VLA 后训练。首选模型是 Pi0.5，当前服务器优先走 LoRA；OpenArm、Piper 和独立 YAM-ABC-Reproduce 代码只保留为 legacy/reference，不是本项目默认实现。

服务器入口、环境和迁移状态见 [服务器与环境](docs/02_installation_and_environment.md)。`/home/wuyan-lyj/YAM` 只作为 YAM 数据/训练合同的只读参考，本项目代码仍是本仓库。

## 当前训练合同

```text
state/action: 14D = [左臂6关节, 左夹爪, 右臂6关节, 右夹爪]
state key: observation.state
action key: action
image keys: observation.images.top_rgb / left_rgb / right_rgb
training action: 每臂6个关节维度相对当前state，夹爪维度保持absolute
model: Pi0.5, internal action_dim=32, action_horizon=50; YAM policy output=14D
norm asset id: yam
```

YAM 数据的物理单位以数据 metadata 和 audit 为准；不能套用 OpenArm 的 degree/HQ 夹爪或 ROS 弧度合同。

## 文档入口

1. [交接索引](docs/00_handoff_index.md)
2. [系统架构](docs/01_system_architecture.md)
3. [服务器与环境](docs/02_installation_and_environment.md)
4. [训练与评估](docs/03_training_and_evaluation.md)
5. [数据合同](docs/04_data_contracts.md)
6. [训练后 policy smoke](docs/05_inference_and_rollout.md)
7. [历史 OpenArm 研究归档](docs/06_openarm_research_plan.md)

## 最短训练路径

```bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/condapi-yam
export PYTHON="$CONDA_PREFIX/bin/python"
export REPO_ROOT=/home/wuyan/lyj/YAM/YAM_code
cd "$REPO_ROOT"
```

当前默认配置为 `pi05_yam_lora`。它的 `repo_id` 是占位值 `local/yam_bimanual`，正式训练前必须通过 CLI/config override 指向已审计的本地 LeRobot 数据集，并在该数据版本下重新计算 `assets/yam/norm_stats.json`。

```bash
"$PYTHON" scripts/compute_norm_stats.py pi05_yam_lora \
  --repo-id=/path/to/audited/yam_lerobot_dataset

"$PYTHON" scripts/train.py pi05_yam_lora \
  --data.repo-id=/path/to/audited/yam_lerobot_dataset \
  --exp-name=yam_pi05_lora_v001 \
  --num-train-steps=30000
```

首次运行先做 metadata、视频、loader 和 norm gate；不要把 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks` 仅凭目录名称直接喂给训练 loader，它需要先确认/转换为当前 LeRobot 合同。完整顺序见 [训练与评估](docs/03_training_and_evaluation.md)。

## 安全与同步

- 不在仓库写入服务器密码、token 或私钥；长训练使用 Slurm/tmux，不在登录节点训练。
- 每次中文 commit 后，必须把同一提交推送到 Gitea `origin` 和 GitHub `github`，作为双备份；不要 force push。
- 训练后服务只在 checkpoint gate 和真实 WebSocket smoke 通过后使用；YAM 输出应为有限的 `(50,14)` 动作。
