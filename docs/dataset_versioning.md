# 数据集版本化规范（简单目录方案）

本规范用于当前 `LeRobot v2.1` 训练链路，不依赖软链，不依赖 `LeRobot v3`。

## 1. 目标

- 一个数据集版本对应一个固定目录。
- 正式版本发布后不原地修改。
- 训练、归一化统计、排查都直接使用绝对路径。

## 2. 目录布局

推荐统一放在：

```text
/share/home/linyongjia/datasets/
```

单个版本目录示例：

```text
/share/home/linyongjia/datasets/piper_pen_v002/
  meta/
  data/
  videos/
  norm_stats.json
  manifest.yaml
```

## 3. 命名规则

推荐格式：

```text
<robot>_<task_or_dataset>_vNNN
```

例子：

- `piper_pen_v001`
- `piper_pen_v002`
- `piper_dish1_v001`

## 4. 什么时候必须新建版本

出现以下任一情况，都不要原地改旧目录，必须新建下一个版本：

- 新增或删除 episode
- 重写 `meta/info.json`
- 重写 `episodes.jsonl`
- 重写 `data/*.parquet`
- 修改视频编码或相机字段
- 重算并替换 `norm_stats.json`

## 5. manifest.yaml 最小字段

建议至少包含：

```yaml
dataset_id: piper_pen_v002
format: lerobot_v2.1
robot: piper
task: put the pen into the box
episodes: 149
created_at: 2026-05-25
source: piper_pen_v001 + appended_sessions_20260524
status: frozen
norm_stats: ./norm_stats.json
```

## 6. 训练规范

- 训练命令一律显式传 `--data.repo_id /abs/path/to/dataset`
- 不再使用 `local/<alias>`
- 不再通过修改服务器上的 `config.py` 切换数据集

## 7. 发布前检查

至少检查以下一致性：

- `meta/info.json` 中 `total_episodes`
- `meta/episodes.jsonl` 行数
- `data/chunk-*/episode_*.parquet` 数量
- 每路相机 `videos/chunk-*/.../episode_*.mp4` 数量

如果这些不一致，这个版本不应进入训练。
