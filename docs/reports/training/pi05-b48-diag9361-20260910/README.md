# 2026-09-10 · Pi0.5 batch48 诊断续训

## 状态

- 用户要求主动把稳定性实验从 global batch64 切换到 batch48。
- 原 r5 `lego_full_b64_r5_diag9361_20260910` 已保留，停止前运行到约 step5840；其完整 checkpoint 5000 未修改。
- 新 run：`lego_full_b48_diag9361_20260910`，Slurm allocation 2064、gpu001、FSDP4，从完整 checkpoint 5000 恢复。
- 新 run 启动后已完成恢复，并以 batch48 进入训练；启动日志显示输入 batch 形状为 `(48, ...)`，诊断识别 51 个可训练参数叶子。

## 配置

- global batch：48（每卡12）
- FSDP：4
- num_workers：8
- BF16 计算、FP32 参数/优化器状态、EMA关闭
- 学习率、数据、随机种子、诊断开关和保存间隔保持不变
- 代码快照：`/home/wuyan/lyj/YAM/env-transfer/lego-full-a53bb00-diag9361-b48`
- 训练日志：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b48_diag9361_20260910/train.log`

代码快照由诊断快照硬链接复制；仅在新快照中将入口的 batch 配置改为读取 `LEGO_BATCH_SIZE=48`，旧快照和 r5 不变。

## 监控

本地看板已切换到 batch48 的 metrics 路径。每小时巡检当前 b48；若再次出现 NCCL、SIGSEGV 或 GPU Xid 同类故障，保留现场并按授权创建独立的 batch32 续训，不覆盖历史 run。
