# openpi_docker 使用流程（统一版）

## 0) 运行前必填
- `DATASET_NAME`：数据集文件夹名（在 `/share/home/linyongjia/data/` 下）
- 任务来源（二选一）
  - `TASK_MODE=dataset`
  - 或 `TASK_PROMPT="..."`
- 训练配置：`CONFIG_NAME`（`pi0_piper_dual` / `pi05_piper_dual`）
- 实验名：`EXP_NAME`
- 训练命令：`TRAIN_CMD`

## 1) 启动训练（tmux，推荐）
```
DATASET_NAME=pen \
TASK_PROMPT="test" \
CONFIG_NAME=pi05_piper_dual \
EXP_NAME=test \
TRAIN_CMD="XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi05_piper_dual --exp-name test --checkpoint-base-dir /output/openpi --batch-size 8 --num-train-steps 5000 --log-interval 20 --save-interval 200 --keep-period 1000 --lr-schedule.peak-lr 2e-5 --lr-schedule.warmup-steps 300 --no-wandb-enabled" \
bash /home/lyj/.codex/skills/openpi_docker/scripts/train_entry.sh
```

## 2) 训练前自动动作
- 创建/检查数据集软链
- 检查归一化统计（无则自动计算）
- 离线环境变量注入（WANDB/HF_OFFLINE）

## 3) 训练过程
- 临时容器运行（便于复现/迁移）
- 日志、metrics、plot 自动输出

## 4) 训练结束
- 自动复盘生成：`$MONITOR_DIR/metrics_latest_review.txt`
- 成功结束后自动清理 `active_run.env`

## 5) 统一停止（唯一入口）
```
CLEAN_NORM_STATS=1 CLEAN_METRICS=1 \
  bash /home/lyj/.codex/skills/openpi_docker/scripts/tmux_stop_run.sh
```
