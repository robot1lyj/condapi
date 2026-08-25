# 03 · 训练与评估

## 训练前顺序

1. 固定数据集版本、episode split、config 和初始化 checkpoint；OpenPI 不会自动把 LeRobot split 意图当作训练 split。
2. 检查 `meta/info.json`、`meta/episodes.jsonl`、parquet/video 可读性和 OpenArm 16D/单位合同。
3. 按 [数据合同](04_data_contracts.md) 为该版本计算 norm stats；不要手工复制别的合同。
4. 做真实 loader smoke，再启动正式训练。

```bash
conda run -n pi-conda python scripts/train_test.py
```

## 通用单机入口

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
conda run -n pi-conda python scripts/train.py <CONFIG> \
  --exp-name <EXP_NAME> --num-train-steps <STEPS>
```

PyTorch（仅 pi0/pi0.5）使用 `scripts/train_pytorch.py` 或 `torchrun`；JAX 是正式 OpenArm 主路径。配置注册表在 `src/openpi/training/config.py`，训练输出由 `checkpoint_base_dir/config/exp_name/step` 组织。

## 正式 K-Policy 配置

`pi05_openarm_kai0_awbc_v1` 是当前 KAI0/AWBC 正式配置：P05 base 初始化、OpenArm 16D、全局 batch 128、80k steps、每 5k 保存、TorchCodec 尾帧显式 PyAV fallback、训练 episode `0:1719`。它不能与历史 `pi05_openarms_dual_awbc_v1` 混称。

四卡多节点（gpu12 + gpu28）示例：

```bash
conda run -n pi-conda python scripts/launch_openarm_jax_multinode.py \
  --config pi05_openarm_kai0_awbc_v1 \
  --exp-name openarm_kai0_awbc_v1 \
  --num-train-steps 80000 --batch-size 128 --num-workers 8 \
  --mode overwrite --session-prefix kai0_awbc_v1 \
  --hosts gpu12 gpu28
```

正式任务前先用相同 host/batch 做 20-step、`--num-workers 0` smoke；继续训练使用 `--mode resume`，不要只重启一个节点，也不要覆盖已有进度。

## KAI0 数据与 scorer gate

KAI0 的 HQ-Stage、Site-Score、TDA-S、K-Data 和 AWBC 二值化是独立阶段；具体数据命名、当前结果和下一批 HIL 只看 `docs/06_openarm_research_plan.md`。训练前至少通过：

- `scripts/audit_openarm_kai0_training_data.py`：来源、二值标签、真实 loader 样本和尾帧。
- Stage score 的 episode/有限值/范围审计；失败不得进入 K-Data。
- 混合 HQ/Site/TDA 的 norm 和 loader smoke；不能只抽 HQ 开头样本。

普通 BC 对照必须使用相同来源/样本预算，只去掉 Advantage prompt；不能把它和 KAI0 或 Evo-RL 结果合并归因。

## checkpoint 验收

一个数字目录存在不等于 checkpoint 完整。部署前检查 Orbax `_CHECKPOINT_METADATA`、`params/_METADATA`、config、norm stats 和训练日志；再跑 policy server 的真实 WebSocket smoke，并把 checkpoint 路径绑定到输出报告。

## 结果记录

训练 loss 不是真机成功率。每个候选至少记录固定 prompt、checkpoint、数据版本、训练步数、rollout 次数、完整折叠成功率、正确对角线率、重复甩平率、已展开后进入折叠率、接管次数和恢复成功率。结果写入 `docs/07_change_log.md` 或研究计划的对应状态段，不在多个文档复制。
