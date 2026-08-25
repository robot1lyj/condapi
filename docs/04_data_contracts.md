# 04 · 数据合同

## OpenArm 机器人合同

```text
task prompt: Fold the T-shirt properly
state/action: 16D = [右臂7关节, 右夹爪, 左臂7关节, 左夹爪]
arm joints: degree
HQ gripper: motor degree, 0=open, -66=closed
action horizon: 50
```

训练和 server metadata 使用上述合同；ROS/runtime 才把关节转成弧度、把夹爪转成归一化值。任何 14D、弧度或 `[0,1]` 夹爪数据都不能直接混入 OpenArm 正式训练。

## LeRobot v2.1 版本

一个数据版本对应一个冻结目录，不在原目录覆盖。目录至少应包含：

```text
<dataset>/
  meta/info.json
  meta/episodes.jsonl
  data/chunk-*/episode_*.parquet
  videos/chunk-*/...
  norm_stats.json
  manifest.yaml                 # 推荐
```

新增/删除 episode、重写 metadata/parquet、修改视频编码/字段或重算 norm stats，都应创建新版本名（如 `openarm_<task>_vNNN`）。训练配置显式指向绝对数据目录；不依赖软链别名或隐式 split。

## 发布前审计

- `info.total_episodes`、`episodes.jsonl` 行数、parquet episode 编号和视频文件数一致。
- parquet 内 `episode_index` 与文件名/metadata 一致，行数不超过视频真实可读帧数。
- 时间戳映射到视频帧时不越界；抽子集后先 dry-run `scripts/repair_lerobot_subset.py`，再决定是否生成新版本。
- 首、中、尾样本都能用训练 loader 解码；TorchCodec 尾帧 fallback 只能处理明确的 end-of-stream 情况。
- 变更来源、episode 数、单位、清洗脚本、split 和 norm 路径写入 manifest/审计报告。

## Norm stats

OpenArm policy 的 state/action stats 应由训练版本的 parquet 重新计算：

```bash
conda run -n pi-conda python scripts/compute_openarm_parquet_norm_stats.py \
  --dataset /share/home/linyongjia/datasets/<DATASET> \
  --episodes <TRAIN_EPISODES>
```

这个脚本按 OpenArm relative action 规则计算并原子写入 `norm_stats.json`。只有机器人 action space、单位、夹爪语义和任务分布都明确一致时，才可考虑复用预训练 stats；上游 generic radians/[0,1] 说明已移到 legacy，不适用于 OpenArm。

## HIL 数据

Raw HIL 保留 policy action、human/VR action、hold、intervention、时间戳、视频和 episode 成功/恢复元数据。clean 导出使用 `scripts/convert_openarm_hq_dataset.py` 或对应 OpenArm HIL 流程：

- 丢弃 `session_state=intervention_hold` 等等待帧，但不删除原始 raw。
- 只有真实人类 VR 动作标为 intervention；不能把 hold 或自动动作当接管。
- 检查有限 16D human action、单位/夹爪范围、时间单调、视频同步和 episode 结尾。
- Evo ACP 的 `complementary_info.acp_indicator` 必须来自 value/advantage 流程，不可直接把 `is_intervention=1` 当作全部标签。

## 数据与实验隔离

HQ、Site、TDA、HIL-Evo、K-Data、E-Data 使用独立目录和 metadata。不要覆盖原始数据、跨版本复制 norm stats，或让 KAI0 scorer、Evo value、普通 SFT 共用无法归因的派生标签。
