# 03 · 训练与评估

新平台命令先按 [02 · 服务器与环境](02_installation_and_environment.md) 加载 `miniconda3/26.1.1` 并激活 `/home/wuyan/.conda/envs/yam`，再使用其中导出的 `$PYTHON`。接管时该环境为 Python 3.13.12，且登录节点未通过 OpenPI import gate；`CODE_ROOT`、`OUTPUT_ROOT` 和 OpenArm checkpoint 尚未迁移完成，以下命令在 gate 通过前均只作模板。

## 训练前顺序

1. 固定数据集版本、episode split、config 和初始化 checkpoint；OpenPI 不会自动把 LeRobot split 意图当作训练 split。
2. 检查 `meta/info.json`、`meta/episodes.jsonl`、parquet/video 可读性和 OpenArm 16D/单位合同。
3. 按 [数据合同](04_data_contracts.md) 为该版本计算 norm stats；不要手工复制别的合同。
4. 做真实 loader smoke，再启动正式训练。

```bash
"$PYTHON" -m pytest scripts/train_test.py -q
```

## 通用单机入口

```bash
CONFIG=replace_with_config
EXP_NAME=replace_with_exp_name
STEPS=replace_with_steps
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" --num-train-steps "$STEPS"
```

PyTorch（仅 pi0/pi0.5）使用 `scripts/train_pytorch.py` 或 `torchrun`；JAX 是正式 OpenArm 主路径。配置注册表在 `src/openpi/training/config.py`，训练输出由 `checkpoint_base_dir/config/exp_name/step` 组织。

## 参数修改规则

| 参数 | 修改位置 | 说明 |
|---|---|---|
| `repo_id`、`train_episodes`、transform、初始化权重 | `src/openpi/training/config.py` 的新配置 | 数据版本或 split 改变就新建 config/实验名并重算 norm |
| `batch_size`、`num_workers`、`num_train_steps`、`log_interval` | config 或 `scripts/train.py` CLI | 多节点时由 [02](02_installation_and_environment.md#可调参数) 的 launcher 统一传入 |
| `save_interval`、`keep_period`、学习率/冻结规则 | config | 续训不能覆盖旧实验目录 |
| `model.action_horizon`、`action_dim` | model config + 数据/客户端合同 | OpenArm 正式值为 `50`、机器人输出 `16D`；改动后必须重跑 loader、server smoke 和客户端检查 |

不要直接修改 `pi05_openarm_kai0_awbc_v1` 作为试验；复制为新配置并记录初始化 checkpoint、数据、seed 和所有覆盖参数。

## 正式 K-Policy 配置

`pi05_openarm_kai0_awbc_v1` 是当前 KAI0/AWBC 正式配置：P05 base 初始化、OpenArm 16D、全局 batch 128、80k steps、每 5k 保存、TorchCodec 尾帧显式 PyAV fallback、训练 episode `0:1719`。它不能与历史 `pi05_openarms_dual_awbc_v1` 混称。

远端四卡 smoke、正式 80k、coordinator、tmux 和 resume 命令统一由 [02 · 服务器与环境](02_installation_and_environment.md) 的 5.4 节持有；不要在本页复制另一套 host/batch 参数。正式任务前先用相同 host/batch 做 20-step、`--num-workers 0` smoke；继续训练使用 `--mode resume`，不要只重启一个节点，也不要覆盖已有进度。

## KAI0 数据与 scorer gate

KAI0 的 HQ-Stage、Site-Score、TDA-S、K-Data 和 AWBC 二值化是独立阶段；具体数据命名、当前结果和下一批 HIL 只看 `docs/06_openarm_research_plan.md`。训练前至少通过：

- `scripts/audit_openarm_kai0_training_data.py`：来源、二值标签、真实 loader 样本和尾帧。
- Stage score 的 episode/有限值/范围审计；失败不得进入 K-Data。
- 混合 HQ/Site/TDA 的 norm 和 loader smoke；不能只抽 HQ 开头样本。

普通 BC 对照必须使用相同来源/样本预算，只去掉 Advantage prompt；不能把它和 KAI0 或 Evo-RL 结果合并归因。

## 离线 checkpoint 评估

这一步只作离线动作误差和 chunk 连续性参考，不等价于真机成功率。脚本中的 `rollout_drift` 是预留字段，当前不作为 gate；先通过 checkpoint gate，再在固定 holdout 上运行：

```bash
# HQ 原始数据的固定 holdout；K-Data 的 1719 集不是 999:1199 的 HQ holdout。
DATASET="$DATA_ROOT/high_quality_folding"
OUTPUT_ROOT=replace_with_new_platform_output_root
CHECKPOINT="$OUTPUT_ROOT/CONFIG/EXP_NAME/STEP"
"$PYTHON" scripts/evaluate_checkpoint.py \
  --config pi05_openarm_kai0_awbc_v1 \
  --checkpoint-dir "$CHECKPOINT" \
  --dataset "$DATASET" \
  --val-split "999:1199" \
  --output "$OUTPUT_ROOT/eval/EXP_NAME_STEP" \
  --verbose
```

报告至少保存 config、checkpoint、数据版本、val split 和 git commit；真机结论仍以 [05 · 推理与 rollout](05_inference_and_rollout.md) 的固定协议为准。

## checkpoint 验收

一个数字目录存在不等于 checkpoint 完整。部署前检查 Orbax `_CHECKPOINT_METADATA`、`params/_METADATA`、config、norm stats 和训练日志；再跑 policy server 的真实 WebSocket smoke，并把 checkpoint 路径绑定到输出报告。

## 结果记录

训练 loss 不是真机成功率。每个候选至少记录固定 prompt、checkpoint、数据版本、训练步数、rollout 次数、完整折叠成功率、正确对角线率、重复甩平率、已展开后进入折叠率、接管次数和恢复成功率。结果写入 `docs/07_change_log.md` 或研究计划的对应状态段，不在多个文档复制。
