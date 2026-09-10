# 2026-09-10 · Pi0.5 r4 诊断续训

## 运行身份

- Slurm allocation：`2064`，四张 RTX 4090，单节点 `gpu001`
- run：`lego_full_b64_r4_diag9361_20260910`
- control：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64_r4_diag9361_20260910`
- 远端 checkpoint：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64_r4_diag9361_20260910/5000`
- 来源 checkpoint：r3 `lego_full_b64_r3_20260909/5000`，保留原 r3，不覆盖
- 代码快照：`/home/wuyan/lyj/YAM/env-transfer/lego-full-a53bb00-diag9361`
- 原始训练快照基线：`a53bb001e4acb2cce486f2da83d6d8256439a188`
- 数据：`/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train`

## 诊断改动

诊断快照只增加可选观测，不改变模型、优化器、学习率、精度、batch、FSDP 或数据：

- `OPENPI_DIAGNOSTIC_GRADS=1`：每个训练步统计每个可训练参数叶子的有限性、非有限元素数量和有限梯度最大绝对值。
- 每 100 步写入 `diagnostics/gradient_diagnostics.jsonl`，非有限指标时额外写完整记录。
- `OPENPI_DIAGNOSTIC_BATCH=1`：保存当前 batch 的 LeRobot `index`、`episode_index`、`frame_index` 和 `task_index`。
- 诊断日志不写入普通 metrics，不改变 checkpoint 内容。

## 验收

- 2026-09-10 10:54 从 5000 checkpoint 恢复成功。
- 第一次启动因参数路径名称映射 API 写法错误，在 5001 步停止；该现场保留在 run 的 `metrics_instrumentation_v1/`、`diagnostics_instrumentation_v1/` 和 control 目录备份中。
- 修正映射后从同一 5000 checkpoint 重新启动。
- 10:58:46 数据加载器恢复到 step 5000；识别 51 个可训练参数叶子。
- 10:59 左右完成 step 5001/5011，loss 分别约 `0.01977/0.01896`，grad_norm 约 `0.0485/0.0505`，均有限；步速约 `3.38 s/step`，显存约 `9.4 GiB/GPU`。
- 首条诊断已包含 64 条 batch 的 episode/frame 来源，所有记录的梯度叶子均有限。
- 修正后的续训已推进到 step 5051；最近记录的 loss 约 `0.01781`、grad_norm 约 `0.04852`，仍为有限值，步速约 `3.38 s/step`。
- 本地只读看板已切换到该 r4 run，小时巡检也已更新为跟踪该 run。

## 当前边界

该 run 仍是诊断续训，不代表非有限梯度问题已经修复。若再次到达约 9361 步，优先读取 `diagnostics/gradient_diagnostics.jsonl`，按参数叶子和 batch 来源定位；不根据单次有限运行宣称数据或硬件根因已排除。
