# 04 · 数据合同

## DAgger 纠正数据发布合同（2026-09-21方案）

状态：拟实施，尚未验收现场新数据或导出器。采集/训练操作见 [DAgger手册](reference/lego_dagger_playbook.md)。已有14D、三路RGB、absolute动作原值和Pi训练时delta规则继续适用。

原始完整rollout保留自主、人工、过渡、暂停与复位；训练版本只选择审核通过的连续专家片段，不覆盖原件。发布为当前LeRobot v3兼容数据＋来源/审核sidecar；sidecar不是模型必需输入，拟定字段不是现有实现声明。

| 内容 | 合同 |
|---|---|
| 观测 | 三路 `observation.images.top_rgb/left_rgb/right_rgb`；`observation.state[14]` 为观测对应的follower反馈状态 |
| 专家动作 | `action[14]` 为时间对齐、审核通过、实际提交follower的absolute目标；不是下一帧反馈，不是未经映射的主臂读数；训练时才做delta |
| 原始动作链 | 分别保留policy预测、人工目标、仲裁/过渡目标和实际提交目标，缺失值显式标记，不相互冒充 |
| 时间 | 帧索引、各相机时间、state时间、命令提交时间、policy tick及跨设备时钟映射/误差；不能以读日志时间冒充电机写入时间 |
| 控制权 | `policy/expert/transition/hold/reset`；单臂接管须逐臂标记。首版完整14D专家监督要求两臂标签语义均有效，否则排除或另案做维度mask |
| 来源 | round、原始rollout、纠正事件、连续片段、起止原帧/时间、布局/采集组、采集模型及控制版本、norm指纹 |
| RTC审计 | control epoch、request ID、观测tick、chunk起始tick、承诺前缀与实际采用动作索引；可用单独请求表关联，避免逐帧重复存整块 |
| 结果 | 纠正原因、恢复是否成功、整任务结果、无效/排除原因；整任务失败不自动否定其中已成功的局部纠正 |

split按原始rollout/场景采集组隔离，同源片段不可跨train/val；原验证集和各轮val不进入后续训练。片段内不得有观测缺口、控制权切换或人工改场景。H50首版仅取未来50步真实标签均有效的起点，长度L贡献max(0,L-49)，每个标签有源帧映射；不拼接过滤缺口、不用复制末帧凑专家标签。短片段留在原始库，待有效位mask实现并验收后再使用。

M0配套norm可作为适配坐标系继承，C1统计只用于范围审计；这不同于宣称norm由混合新数据计算。manifest分别记录norm来源与训练数据来源。任何单位/夹爪方向不一致、明显越界先阻断发布并复核；不静默改变单位或重算norm。完整新训练版本需审核清单、视频解码、时序检查和真实loader边界检查，当前状态保持planned。

## 2026-09-16 · 现场同款kit与数据范围复核

用户确认现场为ABC-130K同款官方YAM kit、三台D405、各640×480，优先复现原数据任务。这是用户确认的硬件条件，不等于安装视角、像素尺度、标定及时间合同均已独立验收。

随后用户明确现场普通2×2小颗粒底面约16 mm。14:51最新记录三路视频均640×480；追加抽取ABC六集共18帧显示物体形态/相对尺度、背景、指形和容器存在差异，源积木物理尺寸及全量比例未知，不能仅因同kit就认定同任务分布。抽样出处和224几何预览归 [最新证据](reports/training/redesign-20260916/README.md#乐高分拣3最新异步测试)；现场数据如何转换为专家标签、同步、14D绝对目标、30fps与H50边界需独立验收，不能把本次失败的policy_action直接当专家示范。

服务器metadata：train为4458集/10,374,181帧、val为69集/165,142帧，均30fps、1个任务，文本为 `sort the legos into containers by color`。train约96.06小时，含val约97.59小时。三路已发布视频均224×224；抽查source episode95（转换后episode0）frame100，均有上下近黑补边，与640×480等比缩放到224方形相符；一集不能代表全量或现场图像一致。快照、图像和局限见 [证据](reports/training/redesign-20260916/README.md)。未改数据、单位或norm；小集实验设计归 [03](03_training_and_evaluation.md#2026-09-16--训练重设计先复现乐高分拣再比较模型)。

## 2026-09-08 全量发布与归一化完成

实时核对 v3 续跑日志：`CONVERSION_COMPLETE` 为 2026-09-08 07:20:07 +08:00。
已发布版本 `/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/{train,val}`；
train 共4,458轨迹、10,374,181帧。本节替代后文9月7日“尚在转换”的历史状态。

`scripts/compute_yam_norm_stats.py` 通过只读数值列、逐集 H50 分块和同一 DeltaActions 计算全部训练帧，
不包含 val，不裁掉最后不足 batch 的帧，不修改原始数据。12个代表性首中尾/边界样本与真实
LeRobot→YamInputs→DeltaActions 逐值一致，三相机解码成功。统计为14D，模型变换随后补到32D。
所有关节 delta，夹爪 absolute；源 metadata 合同为关节 radian、夹爪0闭1开，raw→port 单位保持
仍是发布者文档与数值范围支撑的推断，不是独立机械臂标定结论。

持久化资产目录（训练应设置 `data.assets.assets_dir=.../pi05_h50`，`asset_id=yam`）：

```text
服务器 /home/wuyan/lyj/YAM/training-assets/lego_lerobot_v1_20260907/pi05_h50/yam/norm_stats.json
本地   assets/lego_lerobot_v1_20260907/pi05_h50/yam/norm_stats.json
SHA256 c496b738470e432b9da02b22a45c8c309b3db8412d73723944a7cbdc62305be9
```

`assets_dir` 指 `yam` 的父目录。本地资产不进入 Git；数值文件、逐文件来源哈希、
加载器对照与进度记录同目录保存。训练集 conversion manifest SHA256：
`9844c0b86b93fd40adf251275d770899ba413800d2a0cba11d4e9bdae40544dc`。
state计数10,374,181，action计数518,709,050；全量计算164.27秒。
分位数使用原 RunningStats 的5000-bin近似直方图，累计改用float64。
该数值路径只支持本仓库已发布的单episode/Parquet布局，遇到其他布局拒绝运行。
旧转换 manifest 内 `norm_stats=not_computed` 是发布时的历史快照，保持不改；新 norm 的来源由独立
`provenance.json` 记录。[详细证据](reports/training/pi05-full-20260908/README.md)。

本页是当前 YAM 训练数据的唯一语义 owner。服务器路径见 [02](02_installation_and_environment.md)，训练流程见 [03](03_training_and_evaluation.md)。OpenArm 16D 合同只在历史文档中保留，不能套用到 YAM。

## YAM 双臂合同

```text
state key:  observation.state
action key: action
state/action shape: (14,)
layout: [left 6 joints, left gripper, right 6 joints, right gripper]
images:
  observation.images.top_rgb
  observation.images.left_rgb
  observation.images.right_rgb
model action: internal 32D, horizon 50
policy output: real YAM 14D
```

RTC时间索引：当前 `create_torch_dataset` 按 `[0,1,…,49]/fps` 加载50步 `action`，YAM发布集为30fps，故样本中 `action[0]` 与 observation/state 对齐同一 LeRobot 行时间戳，步距约33.33ms。这是训练数据索引，不证明真实部署中相机曝光→控制命令的物理延迟为0。3588异步部署应把同步观测归属的30Hz policy tick 显式传给Thor，完整协议见 [RTC冷手册](reference/thor/13_trained_rtc_inference.md)。

YAM 与 YAM-ABC 使用同硬件配置；本仓库只实现训练数据和 policy transform，不实现机械臂控制。YAM state/action 的物理单位、正负方向、夹爪开闭范围必须从当前数据的 `meta/info.json`、feature metadata 和样本审计中确认；在证据完成前不写成 degree、弧度或归一化值。

## 图像和 prompt

LeRobot dataset 每行至少应能提供三路 RGB 图像和 state/action：

| 原始键 | OpenPI 槽位 | 要求 |
|---|---|---|
| `observation.images.top_rgb` | `base_0_rgb` | 必须存在 |
| `observation.images.left_rgb` | `left_wrist_0_rgb` | 必须存在 |
| `observation.images.right_rgb` | `right_wrist_0_rgb` | 必须存在 |
| `observation.state` | `state` | 14D |
| `action` | `actions` | `(horizon, 14)` 或单帧 14D |
| LeRobot task | `prompt` | `prompt_from_task=True` 时映射 |

视频读取后可以是 CHW 或 HWC，`YamInputs` 会统一到 HWC；正式数据应固定编码、fps、时间戳和 RGB 语义。缺少任一路图像不应静默补黑图，先修复数据或明确建立新版本。

YAM 配置默认视频后端为 `pyav`，便于 conda 打包后使用随 PyAV 提供的 FFmpeg 库；
若计算节点另行通过 TorchCodec/系统 FFmpeg 导入与首中尾解码验收，可通过 `--data.video-backend=torchcodec` 切换。

## Action transform

当前 `LeRobotYamDataConfig` 默认假定 LeRobot action 是绝对目标：

```text
per arm: [6 joint delta, 1 gripper absolute]
two arms: (6, -1, 6, -1)
```

训练输入先由 `YamInputs` 将 `action` 改名为 OpenPI 的 `actions`，再由 `DeltaActions` 相对当前 `state` 处理 6 个关节；模型输入通过 `PadStatesAndActions` 补到 32D。输出路径先恢复 absolute，再由 `YamOutputs` 裁回 14D。若实际 action 已是 delta，必须在独立数据审计中确认后关闭 `use_delta_joint_actions`，不能重复转换。

## LeRobot 目录和版本

当前代码依赖 `lerobot==0.5.1`，优先使用 LeRobot v3 namespace。一个可训练数据版本应自洽地包含：

```text
<dataset>/
  meta/info.json
  meta/tasks.parquet 或等价 task metadata
  meta/episodes*.jsonl 或 v3 对应元数据
  data/...
  videos/...
```

实际目录结构必须以当前 LeRobot 版本的 metadata loader 为准。新增/删除 episode、重写 parquet、修改视频或改变 feature/unit 都创建新数据版本；不原地覆盖下载的 raw 目录。

服务器已经发现 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks`，但它在完成 metadata、feature、shape、视频和单位审计前只是原始/待审计数据，不能仅凭名称直接训练。

## Norm stats 和资产

YAM 资产 id 固定为 `yam`，训练和 checkpoint 约定为：

```text
assets/yam/norm_stats.json
```

norm 必须针对同一数据版本、同一 train episode split，并在 YAM action delta transform 后计算。禁止跨 OpenArm/Piper/YAM 合同或跨单位复制 stats；服务优先读取 checkpoint 内资产。

## 数据发布 gate

### ABC 乐高子集：上传期间的清洗流程

2026-09-07 只读核验：`ABC-130k-two-tasks/lego_sorting` 是任务筛选后的原始导出，
不是可直接加载的 LeRobot v3 数据集。manifest 声明 train 4458、val 69 episodes，30 FPS；
布局是 `manifests/{train,val}.jsonl`、`{train,val}/data/source-file-NNN.parquet` 和
`{train,val}/videos/{top,left_wrist,right_wrist}/episode-NNNNNN.mp4`。
parquet 保留原始 episode_index、frame_index、index、task_index，以及 language_persistent/events。
抽查 train/source-file-000.parquet 有 11082 行，state/action 均 14D；语言列可为空。
当次清点 4527 条全部为 `pending_upload`，首条 train episode 95 缺 top/right_wrist 视频；
这是上传中的瞬时观察，不能作为后续完成状态。清点清单指纹：train manifest SHA-256
`1fabf4a2b83224f85a724146fb37ccf17928f91c00bd0a697e470980e54fb3a6`，val
`9927aa682e78662bec5584e83728c2e9eb5fd86323ab4bc46a90294f255bfe08`。

清洗按以下顺序推进，每一步保留原始 train/val 分离：

1. **上传清点（已实现）**：`scripts/audit_yam_subset.py --inventory-only` 对照 manifest 检查每条 episode
   的 parquet 和三路视频；缺失、空文件、近期修改或读取期间变化标为 `pending_upload`。
   报告包含 manifest SHA-256、源 repo/revision、episode ID、文件长度和 mtime；mtime 稳定仅是预检，
   不是上传完成证明。冻结时仍须上传方确认完成并复核报告。
2. **结构清洗（已实现审计，不修改原始文件）**：完整模式验证每个 episode 行数、14D 有限数值、
   连续 frame_index、30 FPS 时间戳，逐帧解码三路视频并核对时间戳和帧数。
   已稳定但失败的样本标为 `rejected`，人工核查/重传后重跑；不自动裁短视频、补零、插值或删帧。
3. **语义确认（待完成）**：确认左右顺序、关节单位、夹爪范围、action 是否绝对目标。
   manifest 的 `task` 用于统一 prompt：`sort the legos into containers by color`。
   数值大小只能提示单位，不能证明单位；`validated_structure` 不等于可训练。
4. **发布转换（已实现并通过合成数据回读；真实完整 episode 待验收）**：只将验收通过的完整 episode 写入新版本目录，
   train/val 各自生成独立 LeRobot v3 数据集。使用当前 LeRobot writer 生成 metadata/task/episode 表，
   重编号索引并保留 `(source_repo, source_revision, split, source_episode_index)` 映射。
   相机映射为 top→top_rgb、left_wrist→left_rgb、right_wrist→right_rgb。
   manifest 视频时间戳指向原始长视频；当前逐 episode 视频必须使用本地 PTS，不重复裁切原始区间。
5. **发布验收**：真实 loader 验证首中尾/跨 episode action chunk，核对源 split 无泄漏，
   norm 仅使用 train；再做短训练。正式版本记录纳入/拒绝清单、源指纹和转换参数。
   DAgger 后续追加独立版本与来源标签，不混入本轮 holdout。

可在上传期间运行清点（仅标准库；JSON 写到 stdout，由调用方留存到原始数据目录外）：

```bash
python3 scripts/audit_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting --inventory-only
```

上传完成后的结构审计应放到计算节点，用项目环境运行同一命令并去掉 `--inventory-only`。
可先加 `--limit 2`，表示每个 split 最多审计两条，不代表全量验收。
报告始终保留 `trainable=false`，直到独立的语义、转换和 loader 发布 gate 完成。

- `info` 中的 episode 数、task metadata、parquet 索引和视频清单一致。
- state/action 所有样本最后一维为 14，顺序是 `[左6+夹爪, 右6+夹爪]`，无 NaN/Inf。
- 三路图像首、中、尾样本可解码，时间戳不越界，颜色通道和 shape 一致。
- task/prompt 映射不为空，train/val split 明确且可复现。
- 物理单位和夹爪语义已写入数据版本的 audit/manifest；未确认前标记为待核验。
- norm stats、config、训练 split 和 checkpoint asset id 一一对应。

## 转换工具操作

`scripts/convert_yam_subset.py` 使用已安装的 LeRobot 0.5.1 writer 生成 v3 metadata、task 表、episode 表、
parquet 和视频；train/val 分别转换，默认要求该 split 的全部 episode 完整。
原始 parquet 的 state/action 以 float32 原值写入，不在转换时做 delta 或单位缩放；delta 仍由训练 transform 处理。
空语言列由 manifest 的非空 task 补入标准 LeRobot task。

转换必须提供经审计的 JSON 合同：`state_action_names` 为脚本 `JOINT_NAMES` 的完整左6+夹爪/右6+夹爪顺序，
`joint_unit`、`gripper_unit` 为已确认单位，`action_mode` 为 `absolute` 或 `delta`，`evidence` 标明确认依据。
工具只能检查声明是否完整，不能代替硬件/来源文档核验；不能用测试中的 synthetic 单位发布真实数据。
若声明 action 已是 delta，训练配置必须关闭 `use_delta_joint_actions`。

```bash
"$PYTHON" scripts/convert_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting \
  /home/wuyan/lyj/YAM/YAM_data/audited/lego_train_v001 \
  --split train --contract /path/to/reviewed_yam_contract.json
```

首轮可通过 `--episode-ids` 显式选择少量已上传完整的源 ID；该子集选择会写入 provenance，不会自动跳过缺失 episode。
不存在、近期修改、空文件、14D/时间戳异常、视频帧数不匹配均停止转换。
工具对源文件记录并复查 SHA-256，在 `<output>.incomplete` 完成写入、finalize 和 loader 首中尾回读后才改名为输出目录。
失败目录保留，不自动覆盖或续写；重跑使用新版本路径。原始数据只读。

当前版本视频通过临时 PNG 和 LeRobot 默认 H.264/yuv420p（CRF30）重新编码；这不是视频无损复制。
临时 PNG 会占用额外磁盘空间，适合先小样本验证；正式全量前需核验画质、吞吐和可用空间。
`conversion_manifest.json` 记录 split、原始 repo/revision/episode 映射、源文件散列、合同和编码方式。
转换成功不意味着训练就绪：norm 尚未计算，真实 GPU smoke 与最终训练验收另行完成。

## 旧合同隔离

OpenArm 的 16D `[右臂7, 右夹爪, 左臂7, 左夹爪]`、HQ degree 语义和 Piper 14D transform 都是历史/legacy。它们不能与 YAM 的 14D `[左6, 左夹爪, 右6, 右夹爪]` 混合，也不能复用其 norm stats、动作顺序或训练结论。

## 2026-09-07 Lego 实测清洗状态

用户后续明确收窄范围：本轮“清洗”仅指开源数据完整性验收，不做语义质量筛选、夹爪裁剪、
静止帧删除、重新划分 train/val 或全量视频重编码。范围警告不影响完整性通过判定。
保留现有原始数据，逐集检查 Parquet 数值/帧序列及三相机完整解码、帧数和时间戳；
发现损坏只报告具体文件，不自动删除或替换。没有上游逐文件校验值时，不宣称与上游字节级一致。

本节为本次上传后的最新实测，不把先前的 pending_upload 清点当作当前结论。
原始路径仍为 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting`，原始文件只读。

- 4458 train + 69 val，共 4527 集、10,539,323 帧：文件齐备；全量 14D、有限值、帧序号和时间戳检查通过。
- 287 集的 observation.state 夹爪值略超名义上限，最大约 1.003627；action 夹爪均在 [0,1]。
  此项标记为 warning，不裁剪读数、不删除整集；不能从数值范围独立证明物理单位。
- train/95（3428 帧）与 val/421（2567 帧）完成全部三路视频解码验证。
- train/95 的实际 LeRobot 转换与首/中/尾帧回读通过。小样本发布到
  `/home/wuyan/lyj/YAM/YAM_data/processed/lego_sorting/smoke-train-20260907`，不是全量训练集。
- 全量视频逐帧审计已在 tmux `lego-full-audit` 启动，限制单核、nice 19、最长 24 小时；尚未完成。
  报告目录 `/home/wuyan/lyj/YAM/env-transfer/lego-full-audit-20260907`，
  `full.json.incomplete` 不作为完成结果；`full.json` 发布后还须检查 counts，不能仅凭进程结束宣称全部合格。

全量数值报告为 `/home/wuyan/lyj/YAM/env-transfer/lego-clean-20260907/lowdim.json`。
新增 `audit_yam_subset.py --lowdim-only --progress-every 100`，仅验证数值，状态为 validated_lowdim，
绝不冒充 validated_structure；warning 不自动转成 rejected。默认 full 模式才逐帧解码全部视频。

本次转换合同保留在同目录 `contract.json`，副本写入小样本的 conversion_manifest.json。
[XDOF 原始格式说明](https://huggingface.co/datasets/XDOF/ABC-130k/blob/main/README.md) 明确弧度和夹爪 0=闭、1=开；
[固定版本 LeRobot 字段定义](https://huggingface.co/datasets/lerobot/abc_130k_v3_train/blob/68651e4929d9fb00f798937b2d62617cab5c771d/README.md)
确认左右臂 14D 顺序。将原始单位用于该 LeRobot port 仍是与样本范围一致的推断，尚未做 raw-to-port 逐值对照或实机标定。
当前转换保留数值、不做尺度转换，输出 training_verified=false；norm stats 与训练验收仍未完成。

## 2026-09-07 全量 LeRobot 转换启动

用户已授权全量转换，但禁止覆盖原始数据。当前任务在完整性验收之外增加格式转换，不增加语义筛选。
`scripts/convert_yam_subset.py --video-mode copy` 使用 LeRobot 0.5.1 的公开 metadata API 重建 v3 数据：
视频独立复制、目标 SHA-256 校验并完整解码，不重编码、不建软/硬链接；数值不缩放、不裁剪、不删帧。
逐集重建连续索引、任务标签和来源映射；video image stats 不伪造，OpenPI norm stats 仍需另算。
旧的 reencode 模式保留供小样本/显式使用，批量入口固定使用 copy。

批量入口 `scripts/run_yam_conversion.sh`，远端 tmux `lego-convert-v1`，单 CPU 核、nice 19、最长 48 小时。
按 val → train 顺序转换，发布目标：
`/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/{val,train}`。
运行期间父目录带 `.incomplete`；两个 split 都通过回读才发布父目录。既有目录拒绝覆盖，失败产物保留排查。
日志 `/home/wuyan/lyj/YAM/env-transfer/lego-convert-v1-20260907/conversion.log`；
只有 `CONVERSION_COMPLETE=...` 才代表整个版本发布完成。当前已启动，不能声明全部完成。

转换使用已验证的 `/tmp/condapi-yam-smoke.CzQI6oAj/env`，输入输出均在共享盘；正式共享环境安装独立继续。
约 92.6GB 输入（视频约 91.5GB），输出另占空间。每路视频复制时记录 SHA-256，复制后检查目标 SHA-256，
全部源文件在发布前重查 size/mtime，Parquet 另重查 SHA-256；不是对视频做发布时二次源文件全量哈希。
不变性检查不能防御外部刻意改内容并恢复同 size/mtime 的操作，转换期间原始数据应冻结。

验证：本地 28 项相关测试通过，服务器 7 项转换测试通过；真实 val/421 复制转换及回读成功。
全量作业确认逐集推进后，停止旧 tmux `lego-full-audit` 的重复检查，保留旧日志和未完成报告，
其 `full.json.incomplete` 不能当作全量通过证据；完整性验收由新转换流程执行。

## 断点续跑（2026-09-07 当前入口）

全量任务已切换为支持 `--video-mode copy --resume` 的版本。既有 val/69 集经源/目标哈希、数值和索引
复核后复用，训练集从带检查点的新流程开始；没有重做或覆盖已完成验证集，也没有改动原始数据。

- 检查点在共享盘的 `train.incomplete/resume_identity.json` 和 `resume_records/`（相对于版本暂存目录），
  不放在临时环境目录。记录源 manifest、文件 size/mtime、Parquet 哈希、合同、episode 选择和 LeRobot 版本。
- 每路视频复制、SHA-256 和全帧解码通过后，fsync 数据，再原子写入检查点；续跑校验目标哈希后复用，
  不重复复制/解码。没有有效检查点或校验失败的派生文件移到 `.interrupted-*` 备份后重做，不删除原始数据。
- 中断的 metadata 不直接追加：从校验后的文件重建，原有 metadata 代次保留；已有 Parquet 先逐值比较，
  一致则复用。未知旧 `.incomplete` 目录不自动接管，源数据或配置改变时拒绝混用。
- split 和整个版本各有文件锁，禁止并发写入；已发布 split 只读复核，损坏时拒绝覆盖。重启不保证自动启动，
  但检查点保留；可用下面的入口恢复，已有同名 tmux 会拒绝重复启动。

在本地执行：

```bash
ssh yam-server 'bash /home/wuyan/lyj/YAM/env-transfer/lego-resume-v2-20260907/scripts/resume_lego_server.sh'
```

入口优先使用已出现 INSTALL_COMPLETE 的正式环境，否则使用已验收的临时环境；二者均不可用时停止，
需恢复环境或显式指定 `YAM_ENV_PREFIX`，不会因此删除检查点。tmux 仍为 `lego-convert-v1`，当前日志改为
`/home/wuyan/lyj/YAM/env-transfer/lego-resume-v2-20260907/resume.log`。新版代码在独立执行快照，不改服务器 Git 仓库。
进度 `reused_videos` 表示本集复用视频数；最终仍以 `CONVERSION_COMPLETE` 为整个版本发布标志。

验证：本地 32 项相关测试、服务器 11 项转换/续跑测试通过，覆盖中途异常、派生视频损坏、配置变化、
未知目录、并发锁、完成版本复核和批处理二次运行。实际服务器已输出 REUSED_COMPLETED_SPLIT=val 并继续训练集转换。

## OpenWAM YAM 数据投影

2026-09-22 接入 `adapters/openwam/data.py`；当前输入限 LeRobot v2.0/v2.1，每 episode 独立 parquet/MP4。原始数据只读，不自动转换版本、不写回统计。14D 次序仍为 `[左6关节, 左夹爪, 右6关节, 右夹爪]`，状态读 `observation.state`，监督读同一行的单数 `action`；要求调用者确认数据为绝对目标，本次不推断物理单位、不做 Pi delta 或 EEF/FK 转换。两个夹爪保留连续值，不翻转、二值化或重标单位。

三相机按 `top_rgb / left_rgb / right_rgb`（完整键前缀 `observation.images.`）排列为上部全宽、左下/右下各半宽的原生 OpenWAM L 形 RGB 拼图，训练和离线推理复用同一函数。禁止缺失腕部相机时静默填黑。动作窗长为 `num_frames - 1`，视频每 `video_stride` 抽帧；例如 33/4 得 H32 与 9 帧视频，与 Pi H50 不同。窗口不跨 episode；末尾动作零填充并屏蔽 loss，视频复制最后一帧并用 `video_mask` 标明填充；proprio 仅为窗口起点状态。

动作/状态分别使用所选训练 episode 的 min-max 或 z-score 统计。统计 JSON 带源 metadata/parquet 哈希和清单，原生 checkpoint 保存 `normalization_stats.npy` 的 `joint`（动作）与 `joint_state`（状态）两个区块。推理使用原生定向 normalizer：状态归一化使用 state 统计，动作反归一化使用 action 统计，保留 fps 为返回动作周期的来源。

14D 模型不 padding；80D 预训练模型要求配方明确指定 14 个唯一槽位。先在原始 14D 空间归一化，再 scatter 到 80D 并构造维度 mask，推理先 gather 再反归一化。此映射不宣称与上游 EEF 槽位有相同物理语义；没有已核对的槽位合同不得声称 pretrained policy 可直接控制 YAM。动作头维度不一致的微调会拒绝，不能悄悄随机重建头。配置与验收状态归 [10](10_vla_platform.md#2026-09-22--openwam-微调接入)。
