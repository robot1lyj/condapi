# 11 · 模型无关训练看板

## 定位与启动

看板只读本地指标，不依赖模型环境、不上传 W&B、不启动或停止训练。模型环境继续按系列使用 Conda；训练器继续复用 OpenPI 或 LeRobot，新增模型不需要复制看板或训练循环。

```bash
# 在仓库根目录；第一个例子回放已归档的真实 Pi 日志，不访问服务器
python3 scripts/training_dashboard.py --runs-config configs/dashboards/example.json
# 任意单次运行
python3 scripts/training_dashboard.py --metrics /absolute/run/metrics.jsonl --model evo1 --backend LeRobot
```

打开 `http://127.0.0.1:8765`。顶部切换运行；自动显示已有数值指标，默认至多添加八个辅助图，其余通过“添加指标”选择，× 可移除。训练损失图固定保留；没有 loss 的模型可通过自定义指标图观察。未知 batch、目标步数、精度、GPU 数留空，绝不套用 Pi 配置。

当前 Pi 服务模板 `scripts/training_dashboard.service` 跟踪 batch32 续训，并通过 `--history-remote-metrics` 继承历史指标。恢复运行时，父 run 的记录会保留到当前 run 的第一条 step 之前；重叠 step 由当前 run 覆盖，避免同一训练步重复绘制。当前服务显式指定首阶段目标 80k、总体目标 324,194、每 5k 保存；余弦学习率周期 162097 作为独立参数保留。

单次运行也可以用 `--history-metrics /path/to/parent.jsonl` 指定本地父日志；远端父日志使用同一 SSH 主机上的重复 `--history-remote-metrics /absolute/parent.jsonl`。当前 run 尚无记录时，看板先显示父 run；当前 run 写入后自动拼接为连续曲线。父日志只读，镜像失败时保留上次缓存。

## 任意框架的接入合同

每个文件只承载一次运行，rank 0 单写者，以追加并换行结束的 JSONL 为首选：

```json
{"schema_version":1,"step":10,"metrics":{"loss":0.123456789,"action/mse":0.002,"step_seconds":0.5,"samples_per_second":64}}
```

`step` 为非负整数，`metrics` 为命名的有限数值；键可以自定义或嵌套（嵌套按 `/` 展平）。不要把时间戳、epoch、子模型内部步数冒充统一训练步号。数值不转 BF16/FP16、不四舍五入、不自动换算单位。生产者不得在 metrics 内重用 step 等结构字段。

标准库写入器可供任何原生框架的日志 callback 调用，无需导入 Torch/JAX：

```python
from vla_platform.metrics import write_metrics

# 需安装 packages/vla-platform，或将其 src 加入 PYTHONPATH。
# 分布式时仅主进程调用；标量张量先由训练器转成 Python 数值。
write_metrics("/absolute/run/metrics.jsonl", 10, {"loss": 0.123456789})
```

亦支持 OpenPI 的平铺 JSONL、带表头 CSV，以及 Hugging Face `trainer_state.json` 的 `log_history`。`.csv`/`.json` 自动选择格式，其他扩展名默认 JSONL；可显式设置 format。CSV/JSONL 忽略尚未写完的末行；被改写中的不完整 Trainer JSON 暂时返回 503，页面保留该运行的上次数据。

常见 step/global_step/steps、loss/train/loss、lr、eval_loss 自动映射。非标准名称通过配置声明，不解析任意终端文字：

```json
{
  "schema_version": 1,
  "runs": [{
    "id": "experiment-a",
    "name": "自定义动作模型",
    "model": "custom-vla",
    "backend": "native",
    "metrics": "../../runs/a/metrics.csv",
    "format": "csv",
    "field_map": {"step": "iteration", "loss": "objective"},
    "metric_labels": {"action_mae": "动作 MAE / 已确认单位"},
    "loss_semantics": "由训练器声明：例如最近 20 步均值",
    "parameters": {"计算精度": "由启动配置明确填写"}
  }]
}
```

相对路径以配置文件所在目录为基准。可选 batch_size、stage_steps、total_steps、save_interval 必须是正整数。field_map 仅改名，不把毫秒当秒；单位换算应在生产者明确完成。自定义训练器只需提供上述任一种格式，不要求进入本项目模型注册表。

## LeRobot 原生接入

共享入口 `adapters/lerobot/train.py` 自动捕获原生 `MetricsTracker.to_dict()` 的数值，写到 `<output>.metrics/metrics.jsonl`（原生输出目录的同级目录，避免破坏 LeRobot 对新输出目录的检查）。Pi 保持其原生 `metrics/metrics.jsonl`。

实现是 root logger filter，不重写优化器/训练循环；LeRobot 重置 handlers 后仍有效。固定上游 `2774d9bddcbbda50e697e162e89e7eaada8d7105` 在主进程完成原生区间平均/跨卡归约后记录 tracker。显示值沿用这些统计语义，不等于瞬时单 batch 损失。

原生 step_s → step_seconds，samples_per_s → samples_per_second，lr → learning_rate；gpu_mem_gb → gpu_peak_memory_gib，表示峰值分配，不与 Pi 活跃分配混为一谈。其他数值指标原样保留。文本形式的 eval_loss 和没有结构化日志的环境评估不会伪装成完整精度采集；需用 callback 写标准事件。日志写入失败警告一次，不中断原生训练；页面日志变旧仍需检查源日志。换 LeRobot 版本后复核 tracker 日志接口。

“看板可接入”不等于 Evo-1/MolmoAct2/其他模型已通过真实 YAM GPU 训练验收，注册状态不因此自动升级。

## 如何读指标

- loss：训练器的优化目标，适合比较同一任务/配置的变化；不同模型的 loss 定义与尺度不同，不能直接排名模型效果。
- 验证损失、动作 MAE/MSE、成功率：仅当生产者提供才显示；成功率还须声明样本数与任务设置。训练损失下降不代表机械臂任务成功。
- 单步耗时：一次训练迭代的时间，不是 Thor 的推理时延。近期 ETA 只作进度估计，不包含未来停机与评估。
- 吞吐：生产者提供的 samples/s 优先；仅已知全局 batch 与每步秒数时推导。未知时留空。
- 平滑：只影响曲线展示，导出保留读到的原始数值。loss 摘要取最新有 loss 的记录，步号取最新事件。

同一步训练/评估事件合并；步号回退视为续训回滚，丢弃旧分支的未来记录。因此异步延迟事件应先按训练步序整理，不要把多次运行混写一文件。单文件最多 32 MiB、显示最近 20,000 行；超限应归档或分段，不能悄悄认定读取成功。

看板会按指标源的最后更新时间做新鲜度检查。源文件超过 600 秒（10 分钟）没有更新时，顶部状态变为“训练异常：指标超过10分钟未同步”，并显示检查训练进程、远端日志和 SSH 镜像的告警；同步请求本身失败时仍显示“远端同步异常”。尚未产生第一条指标时保持“等待日志”，不把启动空窗误报为训练异常。

HTTP 只提供固定页面、`/api/runs` 和已配置 ID 的 `/api/metrics?run=ID`，不能请求任意路径。默认仅监听 localhost。单源保留 SSH 原子镜像、失败保留缓存；多源使用独立本地缓存，不自动向多个服务器发起连接。
