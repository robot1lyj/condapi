# ABC-130k 折纸盒下载与整集隔离

用户要求：把折纸盒数据下载到服务器，并排除有问题的 episode。目标 prompt 精确匹配 `fold the paper box`，不混入纸飞机、其他折叠任务，也不改变原 train/val 分区。

## 当前状态

服务器 CPU-only Slurm 作业 **2638 / abc-paper-box** 已启动，2 CPU、12 GiB 内存、0 GPU，运行在 gpu001；耳机训练 2637 继续运行。下载根目录为：

```text
/home/wuyan/lyj/YAM/YAM_data/ABC-130k-paper-box-20261029
```

本回执的状态是 **DOWNLOADING_AND_SCREENING**，不是下载完成、全量清洗完成或训练 READY。现场以 `squeue -j 2638`、根目录 `download-2638.log`、随后生成的 `progress.json` 和最终 `completion.json` 为准。网络、权限、磁盘容量或 I/O 失败会停止重试后报错，不会被计为坏 episode。完成前不发布最终通过清单。

源元数据统计（排除前）：

| split | episode 数 | 30 fps 帧数 | 小时 |
|---|---:|---:|---:|
| train | 2,387 | 10,728,844 | 99.341148 |
| val | 29 | 128,807 | 1.192657 |
| 总计 | 2,416 | 10,857,651 | 100.533806 |

这些是来源数量，不是筛查后保留数量。实际保留/隔离结果以逐集审计为准。

## 来源身份与格式

复用现用 `lerobot/abc_130k_v3_{train,val}`，通过服务器可达的 ModelScope 国内镜像读取，固定镜像提交：

- train：`5b1e1b0e4384922c118b0fd4a68d803feea37d73`
- val：`f3f651fccf9a010ce18369a78e62bf17da435a61`

镜像 metadata 下载校验其 API SHA256；已在工作站核对 train 四个 episode metadata Parquet、val 一个 metadata Parquet 的 SHA256 与现用固定 HF revision 完全相同。HF train 为 `68651e4929d9fb00f798937b2d62617cab5c771d`，val 为 `46ca817c39d3df08115a75c31b3cd3867e3d875a`。未逐个证明镜像所有 data/video 整文件与 HF 相等：`plan.json` 保留镜像的文件路径、大小、整文件 SHA256 和提交，提取后的本地文件另记实际 SHA256；不能把范围读取写成完整源文件哈希验收。

原库把不同任务装在同一 Parquet/MP4 内。仅 train 折纸盒即引用 593 个数据分片，以及 top/left/right 分别 1,122 / 1,185 / 1,168 个视频分片。全部原样下载会占用数百 GB，因此采用有界 HTTP Range：

1. 对数据分片读取原表，只保留所选任务的完整 episode 行，保持原 14D state/action、原 episode/frame/index 和数值，不计算 delta、padding 或 norm。
2. 对视频从 episode 起点前最近的关键帧复制原始压缩包，到原 episode 终点为止；不重编码。必要的关键帧预滚仍保留，sidecar 和最终 manifest 的 `offset_s` 指明真实 episode 起点。使用时必须按此 offset 对齐，不能把整个 MP4 无偏移视为单集观测。
3. 每集逐帧解码目标时间区间进行审核；原来源不修改，不删坏原件或隔离文件。源目录名 `top/left_wrist/right_wrist` 的语义仍按 04 映射，不按字典顺序猜左右。

下载目录是带来源和偏移的审计导出，不是直接可交给现有训练入口的标准 LeRobot 发布目录；后续训练转换和配方仍需另行审核与确认。可用空间低于 150 GiB 时主动停止，保留已下载内容。

## 本轮整集筛查范围

- 全集 14D state/action shape、有限值、连续帧号、30 fps 时间戳与元数据长度。
- 三路视频完整目标区间的解码帧数、逐帧时间戳。
- 64×64 灰度下，连续相邻帧平均绝对差 `<0.25` 且连续持续 `>1 s`，整集隔离待复核。
- 单路超过 25% 帧平均亮度 `<5` 或 `>250`，或超过 25% 帧灰度标准差 `<3`，整集隔离待复核。
- 相邻重复比例仅记录。实际首集 top 样本相邻重复比例 0.4321，但最长连续重复仅 1/30 s，不能将 30 fps 重采样造成的规律重复直接断言为损坏。
- 原始夹爪略超标称范围不裁剪、不据此自动剔除；不套用 OpenArm/Piper 的单位或动作规则。

这些是本轮保守自动隔离规则，不是人工验证的上游故障归因。**未覆盖完整 Pantheon 原始 UUID 标注、原生采样时间缺口、动作/视觉语义一致性、全任务成功标签或真机标定。** 现用 LeRobot 不携带原始 UUID，旧 Pantheon 7 条身份抽查只覆盖乐高/耳机；本次不能将其未命中称为折纸盒无标注问题，也没有移植 Pantheon EE 动作或 frame mask。

产物：`plan.json` 冻结来源和本轮规则；`source-metadata/` 保留元数据；`{train,val}/` 保存数据和含预滚的原编码视频；`audit.jsonl` 记录逐集结果；全部结束后 `manifests/{train,val}.jsonl` 只列通过本轮检查的完整 episode，`completion.json` 绑定 plan/audit 哈希。隔离原因保留在审计日志和本地文件中。

## 代码与验证

入口：[download_abc_task.py](../../../scripts/download_abc_task.py)。代码提交 `634fc5b` 已推送两个配置远端；服务器仅从 Gitea 取得，并从该提交导出独立快照到 `/home/wuyan/lyj/YAM/code-snapshots/abc-paper-box-20261029`，未更新原训练快照或 Python 环境。执行使用现有独立数据 prefix `/home/wuyan/.conda/envs/yam-xr1-data`。

11 项轻量文件/元数据/编码检查通过，覆盖 HTTP 精确范围与错误响应、关键帧预滚、重封装前后目标帧像素相同、冻结检测、非目标 episode 排除、断点复跑，以及既有数据审计器检查。真实 train source 23 的 top 视频完成无重编码范围提取，读取 27,262,976 字节，4,442 个目标帧全部解码且时间戳通过。Ruff 与 `git diff --check` 通过；没有运行任何训练循环、模型 smoke 或 GPU 检查。

复跑使用同一冻结目录（仅用于原作业已退出后，不并行启动第二份）：

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
nice -n 15 /home/wuyan/.conda/envs/yam-xr1-data/bin/python -u \
  /home/wuyan/lyj/YAM/code-snapshots/abc-paper-box-20261029/scripts/download_abc_task.py \
  --root /home/wuyan/lyj/YAM/YAM_data/ABC-130k-paper-box-20261029
```

该入口强制要求 CPU-only Slurm 分配，并用文件锁拒绝重复下载。以上命令需放入 CPU-only Slurm 作业中执行。
