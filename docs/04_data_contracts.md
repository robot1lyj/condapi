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

推荐的 `manifest.yaml` 至少记录：`dataset_id`/源目录、转换脚本与 git commit、episode 总数和各来源/ split、16D/单位合同、视频 codec/fps、norm stats 计算 episode、审计报告路径和生成时间。manifest 是交接索引，不替代 LeRobot 的 `meta/*.jsonl`。

## 服务器数据集矩阵

以下是当前代码/研究计划中实际出现的数据集；状态以目录中的 metadata 和审计报告为准，不以目录名猜测内容。

| 数据集 | 服务器位置 | 用途与 split | 状态 |
|---|---|---|---|
| `high_quality_folding` | `$DATA_ROOT/high_quality_folding` | HQ 原始 1199 集；policy train `0:999`，holdout `999:1199` | KAI0 来源 |
| `openarm_site_align_v1_deg` | `$DATA_ROOT/openarm_site_align_v1_deg` | OpenArm degree/HQ 合同 Site 151 集；Site probe train `0:141`，保留 10 集验证 | 当前有效 Site |
| `openarm_hq_tda_aug_v1` | `$DATA_ROOT/openarm_hq_tda_aug_v1` | HQ 全量 TDA 2298 集 | 归档增强集；第一版 K-Data 只取 TDA-S 300 集 |
| `openarm_kai0_awbc_v1` | `$DATA_ROOT/openarm_kai0_awbc_v1` | 正式二值 K-Data 1719 集：HQ 999、Site 420、TDA 300 | 正式 K-Policy 输入 |
| `openarm_hil_evo_v1` | `$DATA_ROOT/openarm_hil_evo_v1` | HIL clean 数据，供 E-Value/ACP | 计划/探针，需看 metadata |
| `openarm_site_stage_v1` | `$DATA_ROOT/openarm_site_stage_v1` | Site-Stage 领域适配数据 | scorer 适配产物 |
| Stage/reference 数据 | `$REFERENCE_ROOT/high_quality_folding_v2p1_stage_*`、`$REFERENCE_ROOT/openarm_stage_mix_site_v1` | Stage/value 训练与评分 | 不直接作为 policy repo |

K-Data 中的 Site 420 是通过 Site-Score 选择/重复后物化的训练来源（不是原始 Site-A151 的集数）；原始 Site、Site-StageData、Site-Score 和 K-Data 的 episode 数不能互相替代。实际来源比例必须以 `episodes.jsonl` 的 `source_kind` 和正式 audit 为准。

禁用 `openarm_site_align_v1`（旧弧度/归一化夹爪合同）、旧 `openarm_awbc_v1*` 三档标签和已删除的线性 `openarm_site_gt_v1`。正式 K-Data 的 1719/999/420/300 不是“目录里大概有多少”，而是 `audit_openarm_kai0_training_data.py` 的硬 gate。

## repo_id、assets 和 checkpoint 的关系

训练 config 中的 `repo_id` 决定读取哪套 episode；`AssetsConfig.assets_dir/asset_id` 决定从哪里读取 norm stats。正式 OpenArm config 让两者都指向 `$DATA_ROOT/openarm_kai0_awbc_v1`，但自定义 config 可能不同，必须分别核对。训练保存 checkpoint 时会把 stats 复制到：

```text
$CHECKPOINT/assets/<asset_id>/norm_stats.json
```

服务阶段优先读取 checkpoint 内的 stats，而不是现场重新计算或从另一数据集覆盖；因此 checkpoint、config、数据版本和资产目录必须成套晋级。

## 发布前审计

- `info.total_episodes`、`episodes.jsonl` 行数、parquet episode 编号和视频文件数一致。
- parquet 内 `episode_index` 与文件名/metadata 一致，行数不超过视频真实可读帧数。
- 时间戳映射到视频帧时不越界；抽子集后先 dry-run `scripts/repair_lerobot_subset.py`，再决定是否生成新版本。
- 首、中、尾样本都能用训练 loader 解码；TorchCodec 尾帧 fallback 只能处理明确的 end-of-stream 情况。
- 变更来源、episode 数、单位、清洗脚本、split 和 norm 路径写入 manifest/审计报告。

## Norm stats

OpenArm policy 的 state/action stats 应由训练版本的 parquet 重新计算：

```bash
DATASET=/share/home/linyongjia/datasets/replace_with_dataset
TRAIN_EPISODES=0:141
conda run -n pi-conda python scripts/compute_openarm_parquet_norm_stats.py \
  --dataset "$DATASET" \
  --episodes "$TRAIN_EPISODES"
```

这个脚本按 OpenArm relative action 规则计算并原子写入 `norm_stats.json`。只有机器人 action space、单位、夹爪语义和任务分布都明确一致时，才可考虑复用预训练 stats；上游 generic radians/[0,1] 说明不属于当前 OpenArm 合同，不能拿来替代本页规则。

## 从 raw 转成新数据版本

转换只写入新目标目录；`from-hdf5`/`from-hil-hdf5` 先用 `--dry-run` 检查输入，再把同一命令的 `--dry-run` 换成 `--overwrite`；`from-lerobot` 没有 dry-run，必须先确认 source 可读且 destination 是新目录。任何情况下都不要原地改正式数据：

```bash
# Site/HDF5 -> degree/HQ LeRobot v2.1
conda run -n pi-conda python scripts/convert_openarm_hq_dataset.py from-hdf5 \
  --src /storage1t/ipc \
  --dst /share/home/linyongjia/datasets/openarm_site_align_v1_deg \
  --dataset-id openarm_site_align_v1_deg --val-count 10 \
  --verify-video-frames --copy-mode hardlink --dry-run

# 旧 OpenArm LeRobot 单位转换 -> 新 degree/HQ 目录
conda run -n pi-conda python scripts/convert_openarm_hq_dataset.py from-lerobot \
  --src /share/home/linyongjia/datasets/openarm_site_align_v1 \
  --dst /share/home/linyongjia/datasets/openarm_site_align_v1_deg \
  --dataset-id openarm_site_align_v1_deg \
  --policy-joint-unit degrees --policy-gripper-unit dataset_degrees \
  --copy-mode hardlink --overwrite

# HIL raw -> Evo clean LeRobot v2.1
conda run -n pi-conda python scripts/convert_openarm_hq_dataset.py from-hil-hdf5 \
  --src /tmp/openarm_hil/openarm_hil_dagger \
  --dst /share/home/linyongjia/datasets/openarm_hil_evo_v1 \
  --dataset-id openarm_hil_evo_v1 --verify-video-frames --dry-run
```

上面的 raw 路径是输入示例，不代表每台服务器都存在；HDF5/HIL dry-run 通过后，重新执行对应命令并将 `--dry-run` 换成 `--overwrite` 才会落盘。转换后必须重新检查 metadata、视频、norm stats 和 config split。原始 HIL 不能被 clean 导出覆盖。

## HIL 数据

Raw HIL 保留 policy action、human/VR action、hold、intervention、时间戳、视频和 episode 成功/恢复元数据。clean 导出使用 `scripts/convert_openarm_hq_dataset.py` 或对应 OpenArm HIL 流程：

- 丢弃 `session_state=intervention_hold` 等等待帧，但不删除原始 raw。
- 只有真实人类 VR 动作标为 intervention；不能把 hold 或自动动作当接管。
- 检查有限 16D human action、单位/夹爪范围、时间单调、视频同步和 episode 结尾。
- Evo ACP 的 `complementary_info.acp_indicator` 必须来自 value/advantage 流程，不可直接把 `is_intervention=1` 当作全部标签。

## 数据与实验隔离

HQ、Site、TDA、HIL-Evo、K-Data、E-Data 使用独立目录和 metadata。不要覆盖原始数据、跨版本复制 norm stats，或让 KAI0 scorer、Evo value、普通 SFT 共用无法归因的派生标签。
