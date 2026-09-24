# 10 · 多模型接入层操作

## 当前状态

状态汇总更新至 2026-09-24。2026-09-09 曾按用户授权将多模型工作树与当时 main 的 Pi 全量训练/续训修复整合；当前功能分支与 main 的关系以 Git 为准。XR-1 原生训练入口和 YAM HIL sidecar 转换入口已写入本地分支；真实 HIL 训练集尚未定位，未启动训练或部署 Thor。架构 owner 为 [01](01_system_architecture.md#多模型接入层)，设备信息归 [02](02_installation_and_environment.md)，模型精度与 Thor 验收归 [08](08_thor_edge_deployment.md)，模型无关训练看板归 [11](11_training_dashboard.md)。

| 部分 | 状态 |
|---|---|
| 控制层 | 标准库核心 v0.2；模型/数据/算法组合、源文件 inventory、分组 split、计划与运行哈希、独立 Conda 子进程和模型包完整性 |
| Pi | 现有 OpenPI 训练/参考推理/回放/ONNX 导出入口已接线；新环境 GPU 执行未验证 |
| LeRobot 后端 | 共用原生训练 launcher 已实现并用替身测试；无自建 trainer/processor；尚无通用离线推理入口 |
| Evo-1 | 本地/服务器专用环境已安装并通过CPU检查；YAM真实训练/推理仍待接入 |
| MolmoAct2 | 原生LeRobot共享后端已注册；两端独立环境CPU检查通过、108包版本一致；仅普通版，不含Think，真实YAM/GPU仍待验收 |
| Xiaomi-Robotics-1 / XR-1 | 官方 5B checkpoint 和独立服务器模型/清洗环境已准备；本地 `xr1` 后端、固定上游源码、训练配置预检、原生训练入口及 HIL sidecar→末端原生 JSON/视频转换入口已接入。50h 乐高 LeRobot→末端派生版正在服务器生成；FK 审计、训练统计、GPU 训练及部署 IK 尚未验收 |
| FastWAM、VLA-JEPA | 注册 planned；不得运行或报告已支持 |
| Conda | Evo-1已有独立环境规格和104个wheel的锁；Pi等bootstrap仍不代表模型环境已安装 |
| Thor | 原 Pi 容器、TensorRT 引擎、报告保持原状；没有部署此次改造 |

历史 2026-09-09 CPU 回归曾有平台测试52项、与 `scripts/thor`、`skills/mlops-memory/tests` 和下载完整性测试合跑170项通过；这些数值不代表 v0.2 新模块已验收。框架测试替身不冒充模型运行；独立Evo环境实际导入/processor检查见 [环境证据](reports/environments/evo1-20260908/README.md)，MolmoAct2的原生精度/数据差异见 [接入说明](reference/molmoact2_integration.md)。

XR-1 的本地原生训练入口已接入，尚不能把现有 YAM 14D joint action 数据直接拿来开训。所选 checkpoint、版本锁、`decord` wheel 元数据修复和原生 state/action 语义见 [XR-1 环境报告](reports/environments/xr1-20260924/README.md)；操作与待完成项见下文。

不复制 LeRobot 的 registry、trainer、processor 或 dataset 实现；`configs/models/*.toml` 只选后端、policy_type 和已接通能力，入口集中在 `adapters/<backend>/backend.toml`。具体模型在子进程中调用上游；Pi 调用既有 OpenPI。原作者代码用于对照，不强制每个模型维护双实现。RLinf 的 DAgger/RL 接入不是当前范围。

## 2026-09-24 · XR-1 原生训练入口

`configs/models/xr1-5b.toml` 选择独立 `xr1` 后端，`third_party/xr1/` 固定 Xiaomi 上游 `0dd7aef8dc87296246aae812a1f59ccb708e5546` 源码、许可和逐文件 SHA256。`adapters/xr1/` 只做配置合成、数据门槛和 Slurm 启动；调用上游 `tools/train.py`，不复制训练循环。服务器已安装的独立 prefix 和官方 5B 权重仍按 [02](02_installation_and_environment.md#xr-1-服务器环境-2026-09-24) 与 [环境报告](reports/environments/xr1-20260924/README.md) 核对。本地快照与服务器早先安装的源码 revision 一致，服务器尚未同步本仓库的新接入代码。

先由已标定的离线运动学在**新目录**生成 XR-1 原生 JSON 和三路同步视频绝对路径；每条训练 episode 需有当前/目标末端位置与旋转矩阵、6关节与夹爪状态/目标、显式固定底盘和腰部字段。关节状态能映射到 60D 的左/右各6关节和夹爪槽，缺少的第7关节槽为0；14D 关节目标不能填进末端 action。按 train episode 运行上游 `tools/compute_normalize.py` 得到 30×60 action mean/std 和 1×60 state q01/q99，在 `normalize.json` 加入 `train_json_sha256`，每个训练 JSON 的绝对路径映射到 SHA256。另存 `fk_audit.json`，记录 `source_contract`、`source_dataset`、`source_revision`、逐条 `train_episode_ids`、`fk_model` 绝对路径及 `fk_model_sha256`、相同的 `train_json_sha256`，并且只在逐值核实坐标系/物理单位和目标时序后把 `frames_and_units_verified`、`target_alignment_verified` 置为 true。split 按源 rollout 隔离；val 不进入训练统计。现阶段**没有**已审核的 FK、派生 JSON、统计和部署 IK，示例路径不可运行。

YAM 新增的 `hil.xr1_dataset` 是 FK 与连续人工专家段的 sidecar owner。本仓库 `adapters/xr1/prepare_hil.py` 将每个不少于 30 帧的 sidecar 段转成单独的原生 JSON 与重编码视频，并保存源时间/tick/相机索引和摘要；命令与数据门槛见 [04](04_data_contracts.md#yam-hil-人工专家帧到-xr-1-原生训练数据)。现阶段本地仅发现 mock 录制，服务器尚未定位真实 HIL 原始 episode，因此该入口没有生成可训练的真实版本；上段所述训练统计与 FK audit gate 仍有效。

用户随后指定服务器的 50h 乐高 LeRobot 选集。该资产不是 HIL episode，本仓库因此另提供 `adapters/xr1/prepare_lego.py`：按既有 2,337 条选集和 LeRobot v3 episode 边界使用同一 YAM 末端坐标合同派生 JSON；服务器 XR-1 的 decord 无法读取源 AV1，故按逐集 `start` 偏移将选中视频重编码到新的 H.264 目录。不生成虚构的 HIL tick 或专家标记。50h 派生资产的来源、状态和单位限制归 [04](04_data_contracts.md#50h-乐高-lerobot-数据的-xr-1-末端派生版)。

准备好这些资产后，将 [示例 recipe](../configs/native/xr1-yam.example.json) 另存为实验专用文件，填入真实绝对路径和批量/步数，然后只读生成平台计划：

```bash
python3 scripts/vla.py plan configs/experiments/xr1-yam.toml train --run-id xr1-yam-plan
```

平台计划仅核对声明和文件哈希，不表示配方已通过数据门槛。用服务器 XR-1 prefix 执行 `python adapters/xr1/train.py --recipe <绝对路径> --output <新目录> --check-only` 才执行原生 JSON、视频、统计、FK 审计和 checkpoint SHA 检查；此步骤不构造模型或进入训练循环。真正启动必须在获准的 Slurm GPU allocation 中执行 `scripts/vla.py run ...`；wrapper 会把解析后的上游配置与审计哈希留在新运行目录，W&B 仅本地离线记录。启动前仍需核对当前作业资源、CUDA/import、视频解码与首条数据。已实现训练接口不等于真实 YAM 训练或推理通过。XR-1 原生输出是末端目标；推理前还需独立验证 YAM 的 IK、限位、坐标变换及 Thor 服务时序，当前模型声明没有 `infer` 能力。

## 模块化配置后端 v0.2

首版按用户确定的范围提供**可复用标准库核心与 CLI**，暂不提供多用户 REST API。旧 `schema_version = 1` 实验继续按原有入口执行；新的 `schema_version = 2` 实验按组件组合，模型权重和数据仍在仓库外，各系列继续调用原生训练器。后端由 `packages/vla-platform/src/vla_platform/` 的 inventory、splits、composition、project、runtime 和 artifacts 模块组成；文件清单与命令均不执行 shell 插值。模型/算法/数据的名称不是兼容性的证据，必须通过具体 IO 与来源检查。

| 组件 | 配置/产物 | 编译时核对 |
|---|---|---|
| 模型 | `configs/models/*.toml` 的 `io.<operation>` | 后端、方法、状态/动作空间、相机顺序、合同及已开放操作 |
| 数据 | `configs/datasets/*.toml` + episode inventory JSONL | 源格式、版本、14D joint 或经 FK 派生 EE 标签、源文件哈希 |
| 分割 | `configs/splits/*.toml` → 新目录 `manifest.json` + `READY` | 固定 seed，按 `group_id` 整组分配，train/val/test 互斥，源清单或文件变化拒绝复用 |
| 算法 | `configs/algorithms/*.toml` | 后端/方法/输入语义；只允许类型相同的显式覆盖；底层 loss/processor/optimizer 仍由原生训练器实现 |
| 运行 | `configs/experiments/*.toml`、环境、机器人合同 | 生成完整 Conda 命令、输入哈希、run ID 和组件收据；运行前重查 split/模型包 |

### 生成 inventory 与 split

先准备审核后的 JSONL 文件列表，每行至少有 `episode_id`、`group_id`（原始 rollout 或同场景采集组）、`task_id`、`frames`、`source_files`。`source_files` 是相对数据根目录或绝对文件路径，必须完整列出会影响该 episode 解码/标签的 Parquet、三路视频和元数据文件；XR-1 还要用 `asset_uri` 标明原生 JSON 文件，并把该 JSON 引用的视频列入清单。平台仅核对清单中声明的文件，不会自动发现漏列资产。示例的一行格式：

```json
{"episode_id":"17","group_id":"session-2026-09-24-a","task_id":"sort lego","frames":780,"source_files":["meta/info.json","data/episode_000017.parquet","videos/top/episode_000017.mp4","videos/left/episode_000017.mp4","videos/right/episode_000017.mp4"]}
```

已有 YAM 原始 val 不应重新混入 train 池；只对允许重新划分的源 train 池建立一个数据组件。`inventory create` 在数据根目录之外的新路径写入源文件哈希与 episode 清单，不复制或修改视频/Parquet。它检查路径不逃出数据根目录、文件存在且无重复；来源、分组和帧数仍须先完成采集审计。

```bash
python3 scripts/vla.py inventory create --source-root /path/to/source-train --listing /path/to/reviewed-episodes.jsonl --output /path/to/new-audit/episode-inventory.jsonl
python3 scripts/vla.py split create configs/splits/yam-grouped.toml --output /path/to/new-split
python3 scripts/vla.py split inspect /path/to/new-split/manifest.json
```

将数据组件中的 `source_root`、`episode_manifest`、`source_revision` 填成真实值后再创建 split。分割结果只含成员 ID 和源/配方哈希；每个非零比例至少需要一个独立 group，组过少直接失败。输出目录必须不存在且位于原始数据目录之外；半途失败不自动覆盖重试目录。`split inspect` 和训练运行前重新计算源文件哈希并重放分配。比例按 group 完整性尽量逼近，不能承诺每个任务或每个比例精确分层；需要任务分层时先扩展且版本化分割策略，不能手工改 manifest。

### 组合训练与推理

第二版示例在 `configs/experiments/*-modular.example.toml`。将占位路径换成实际资产后运行 `vla plan`；它核对模型、数据、算法、机器人合同、split 和原生训练配置的 train 成员必须完全一致。Pi 使用薄启动器把选中 episode 交给现有 OpenPI `train.main`，还要求同一 train 集的 H50/delta norm provenance、显式基础 `/params` 及 `vla hash-tree /path/to/params` 得出的完整目录 SHA256；计划锁定 norm 文件哈希，运行前复核基础权重目录哈希。OpenWAM 核对其原生 JSON `dataloader.dataset_dir`、`episodes`、动作/相机语义和训练统计的成员清单，并锁定统计文件哈希；XR-1 核对原生 JSON 列表，并继续要求 FK/单位/时序审计。三者都不产生第二套训练循环。Pi 模块化首版只支持 `pi05_yam` 全量新 run，未提供 LoRA 或自动 resume；旧正式 Lego 路线不受影响。

```bash
python3 scripts/vla.py datasets
python3 scripts/vla.py algorithms
python3 scripts/vla.py plan configs/experiments/pi-train-modular.example.toml train --run-id pi-new-001
python3 scripts/vla.py plan configs/experiments/xr1-modular.example.toml train --run-id xr1-new-001
```

推理/导出组合要求已封装的模型包 `bundle_manifest`：逐文件哈希必须通过，模型、合同、model_version 与 checkpoint 路径必须一致。当前 v2 推理声明覆盖 Pi 和 OpenWAM 的原生离线接口；XR-1 没有 YAM IK/末端到关节转换验收，仍无 `infer` capability。训练后 policy 的 Thor 服务、数值精度和跨 IPC smoke 仍按 [05](05_inference_and_rollout.md) 与 [08](08_thor_edge_deployment.md) 单独验收。

`plan` 只读，`run` 在本机系列 Conda prefix 启动；不自动 SSH、提交 Slurm 或更新 Thor。训练 wrapper 另要求获准的 Slurm GPU allocation，本地工作站禁止训练。运行目录只创建一次，记录 `plan.json`、源 SHA、开始/结束状态和原生日志；命令退出成功不等于模型验收。当前示例是模板，路径未填、数据未审核、GPU 未验证时不可运行。DAgger 每轮训练配方仍须按本仓库既有规则逐轮展示与确认。

此核心为以后服务 API 留下清晰边界：HTTP 层只提交已授权的配置与操作，复用相同的 plan/run/split 逻辑；身份、租户隔离、任务队列、配额、持久事件和审计仍需独立实现，现阶段不能宣称已提供多人在线服务。

## 工作树与 Git

主目录 `/home/wuyan-lyj/condapi` 的 main 与当前工作树分支是独立检出。用 `pwd`、`git branch --show-current`、`git worktree list` 确认当前目录。代码和版本化记忆随分支变化；未合并前主目录不会看到这里的更新。

```bash
git status --short
git diff
python3 scripts/vla.py models
```

默认只提交/备份当前分支；明确授权合并时，先把最新 main 合入功能分支解决冲突并检查，再将干净的主检出 fast-forward 到验收提交，向两个已配置远端推送，不能 force push。不自动同步 Thor/服务器。工作树合并不是搬文件或删除目录，分支和工作树可保留。共享权重、Conda prefix、外部数据并不受工作树隔离保护。

当前 Pi 正式全量入口仍为 `scripts/launch_lego_full.sh` → `scripts/train_lego_full.py`，每 5k 保存与 committed checkpoint 续训保护保留。`configs/experiments/pi-train.toml` 是通用全量配置示例，不替代已有 Lego 正式启动参数。服务器故障未解决，本地禁止运行训练循环，包括 debug/CPU smoke；仅做静态、配置和看板验证。多模型指标接入不提升尚未 GPU 验收的模型状态。

文件记忆读取当前工作树版本，只按当前任务展开摘要与相关章节；检索与旧账本迁移规则归 [记忆系统](09_memory_system.md)。可选 `docs/cache/runtime/` 账本保留检索计量与去重历史，无默认累计额度，不把会话 ID 固化为长期操作步骤，也不将账本字节当作当前上下文占用。

## 命令与环境

所有命令在工作树根目录运行。控制层 Python 3.11+，模型环境独立。可选安装：`python -m pip install --no-deps -e packages/vla-platform`，随后使用 `vla`；不需要安装根目录 OpenPI 大依赖。

```bash
python3 scripts/vla.py models
python3 scripts/vla.py backends
python3 scripts/vla.py plan configs/experiments/pi-reference.toml infer --run-id pi-plan-001
python3 scripts/vla.py env plan configs/environments/evo1-workstation.toml
```

`plan` 只输出命令，不下载/安装/运行模型。先替换实验配置中的 `/path/to`、模型版本和 action_dt_s；0.1 只是示例，不是对 YAM 频率的确认。

`env create` 显式创建不存在的 prefix，不更新现有环境；`env audit` 只列包，不证明 CUDA 可用。现有 `condapi-yam` 服务器 prefix 仅引用，不自动迁移。真正的模型环境还需锁定源码、Torch/CUDA/Transformers 和处理器依赖，x86 服务器与 ARM Thor 分别核对。

```bash
python3 scripts/vla.py run configs/experiments/pi-reference.toml infer --run-id pi-infer-001
```

`run` 是本机执行，不会自动 SSH。服务器训练在已审计计算节点的 Slurm allocation/tmux 内启动，框架不是 Slurm 提交器。Thor profile 必须在实机运行；infer/benchmark 包装现有 MAXN session，退出恢复 120W；不设置开机 MAXN。Conda 缺失或 profile 未安装时直接失败。

输出位于 `runs/platform/<run-id>/`：plan、started、console.log、finished 以及模型产物。目录不可覆盖；成功退出标为 `command_succeeded_not_model_accepted`。运行元数据是命令与源码追溯，并非完整模型验收清单，数据/训练参数指纹仍由模型侧产物补齐。

Pi `infer` 是一次本地请求的参考路径，图像为本机 RGB 文件，示例见 `configs/requests/pi-example.json`。每次新进程，不实现常驻缓存或网络服务。时间包含首调用编译，不能拿它与旧 TensorRT 稳态 104ms 比较。未合并 LoRA 的 PyTorch 推理会拒绝；JAX 保持 checkpoint 原始加载精度。

## 模型包

模型包保留原格式，不强制转换成 JAX。recipe JSON 必须包含 schema_version=1、model、model_version、code_revision、contract_id、format、precision.storage/compute，以及 files 列表（role/path）。必须包含 weights、model_config、preprocessing、normalization、contract、reference 六类文件；contract 为本版本 TOML。目录权重要逐个列文件，不能只记录目录名。LoRA adapter 还需 base_model_sha256。旧原型 plugin 字段不再接受；不重写已经留存的历史记录。

```bash
python3 scripts/vla.py bundle seal /path/to/package/recipe.json --output /path/to/package/manifest.json
python3 scripts/vla.py bundle check /path/to/package/manifest.json
```

路径只能在包内；检查每个文件 SHA256。seal 不复制、裁剪、量化或合并权重，也不覆盖 manifest。完整性通过不等于格式能加载、参考输出一致或可部署。模型精度批准记录与模型包分离。

## Evo-1 接入判断

### 2026-09-16 · Evo-1对照实验的范围

用户提出Pi0.5成本高、100000抓取效果差，随后提出在采集现场数据期间训练Evo-1；建议提前做独立小预算试验。数据划分、两阶段预算、现场衔接和成功率准则归 [四卡试验](03_training_and_evaluation.md#采集期间的evo-1四卡试验)。模型仍为planned，未完成YAM真实训练/Thor推理，不因研究推荐开放capability。

本轮核对 [官方LeRobot文档](https://huggingface.co/docs/lerobot/evo1)：原生实现使用 `OpenGVLab/InternVL3-1B-hf`，阶段切换默认重应用冻结规则；stage2加载stage1策略后新建优化器/调度，不是训练状态原样resume。该 VLM 已按参考配方固定到 revision `014c0583a0d4bedf29fbe2dbff4f865eb998e171`，并完成本地下载、服务器交接和 RTX 4090 完整 VLM 权重加载；交接证据见 [模型报告](reports/environments/evo1-model-transfer-20260916/README.md)。其公开LIBERO参考使用2×H100，不能推导本项目4×4090速度。在线文档可能晚于已安装版本，运行前核对支持字段，不直接升级既有环境。

优先VLM起点的原生stage1→stage2并明确动作头是否新初始化；模拟器checkpoint先验身份/processor/动作语义。三图、真实14D、双夹爪连续语义、24D padding和Evo独立统计遵守下方规则，Pi32D/delta norm不迁入。先在获准服务器验短GPU容量/吞吐和保存重载，再在Thor原生路径计时；Pi TensorRT经验不能当Evo已可部署。

### 固定源码检查与YAM合同

已检查 LeRobot commit `2774d9bddcbbda50e697e162e89e7eaada8d7105` 的 `configuration_evo1.py`、`modeling_evo1.py`、`processor_evo1.py`；这个 commit 是 API 调研固定点，不是已经验收的依赖锁。

- 配置支持 max_views=3，默认图像 448×448，内部 state/action padding 为 24D，chunk_size 默认 50。模型预测 padded chunk，必须经过原生后处理器裁回真实动作维度；不能直接把内部24D给控制侧。
- 原生 processor 已做 padding、归一化、反归一化、动作裁剪；保存/重载原生 processors，不重新写一套。不要把 Pi norm 资产套给 Evo，也不要把内部24D改成32D来模仿 Pi。
- YAM 需显式配置真实 state/action feature 为14D和三路相机。必须关闭 LIBERO 单夹爪二值化；不能只设置 gripper_index=6 而遗漏右夹爪13。
- 动作表示属于训练产物合同。若用 YAM absolute 原始 action，则训练和部署都保持该语义；若实验选择 delta，必须明确 processor、转换顺序及对应统计，不能暗中沿用 Pi 的 delta norm。
- stage1 默认冻结 VLM、训练动作头；stage2 默认解冻相关 VLM 分支，不是 LoRA。stage2 会重新应用阶段默认值，不能只沿用 stage1 checkpoint 的冻结状态。
- 训练 FP32 主参数和 BF16 autocast 需检查真实配置；`vlm_dtype`、`use_amp` 是不同开关。当前未在本地跑 Evo backward 或推理。
- LIBERO checkpoint 的7D语义不是 YAM14D。修改配置/裁剪输出不能把7D模型变成已训练的YAM策略；必须进行正确机器人适配和微调。

接入顺序：固定 LeRobot 实现 → 专用 Conda 依赖审计 → 检查 checkpoint config/processors → YAM batch 经原生 processor → 一次真实前向/反向和保存重载 → 开放模型声明中的对应操作 → Thor 原生推理计时。后续 FastWAM/VLA-JEPA 复用同一个 LeRobot 后端，但视频帧采样、文本编码和动作头仍用各自原生实现。

### 四卡训练落地前的固定版本检查

以下为2026-09-16核查，固定源码仍为`2774d9bddcbbda50e697e162e89e7eaada8d7105`；本地源码/安装包与服务器配置、trainer两个文件哈希一致，服务器观察与范围见 [只读快照](reports/training/redesign-20260916/evo1-readiness-20260916.json)。FlashAttention依赖和完整 VLM 权重加载已完成 GPU smoke，但完整Evo policy、YAM batch和训练仍未验收；环境安装证据归 [02](02_installation_and_environment.md#evo-1--lerobot-独立环境)，详细实测归 [FlashAttention报告](reports/environments/evo1-flash-attn-20260916/README.md) 和 [模型交接报告](reports/environments/evo1-model-transfer-20260916/README.md)。

- **注意力实现：** `internvl3_embedder.py`仅在`use_flash_attn`且`is_flash_attn_2_available()`时选择`flash_attention_2`，否则为`eager`，不是自动选择SDPA。服务器环境已安装并在GPU节点确认检测为True、实际选择`flash_attention_2`；BF16 `(1,2048,16,128)` kernel前向P50约0.263ms，强制math-only SDPA约3.733ms，约14.18倍差异。这是kernel微基准，不是完整Evo训练吞吐。
- **图像/token：** 基础VLM的448图像与`image_seq_length`绑定，单改`image_resolution=224`会被原生校验拒绝。保持三图448，明确top/left/right输入映射；三图token和指令须完整容纳，不能随意把`max_text_length=1024`大幅缩短。ABC224视频放大到448不恢复细节，现场640×480的等比补边/缩放几何须与训练样本对齐，不能假设两种源图直接resize就等价。
- **精度/冻结：** stage1的`vlm_dtype=bfloat16`用于冻结VLM，stage2显式改为`float32`且`use_amp=True`；仅打开AMP不会将已有BF16主权重升为FP32。保留`apply_training_stage_defaults=True`，stage2加载stage1后应实查视觉、语言、动作参数的requires_grad/dtype。有效梯度检查点开关是`policy.enable_gradient_checkpointing`，stage1无VLM梯度时原生实现会禁用该分支的checkpointing。
- **DDP/步数：** 原生`torchrun`入口可用，4进程候选`parallelism.dp_replicate=4, dp_shard=1`；`batch_size`按进程，累积为`accelerator.gradient_accumulation.steps`。`lerobot_train.py`每microbatch递增step并调用scheduler，累积同步才更新优化器；`AcceleratorConfig.build`设`step_scheduler_with_optimizer=False`。因此训练上限/保存/warmup按microstep换算，两个计数都要记录。现有共享launcher未接通分布式，不能把其骨架当可运行命令。
- **14D/统计：** 第一条路线保留原始absolute action；显式14D输入输出、`postprocess_action_dim=14`、`binarize_gripper=False`，内部24D padding/MIN_MAX用Evo原生processor。双夹爪6/13维及单位逐值验收，不复用Pi delta/32D norm。native trainer在`pretrained_path`且`resume=False`时，会用当前dataset stats覆盖normalizer/unnormalizer；stage1→stage2用同一D1及stats较直接，改现场数据时必须审计这种变化，不能把加载旧processor等同于冻结旧统计。
- **部署：** 训练H50与一次实际执行多少步分开配置。原生默认flow采样32次，不能拿Pi的10次去噪/约100ms TensorRT成绩推导Evo在Thor更快；训练可先完成，但机器人成功率比较前仍须接通Evo原生推理和同一控制合同。

固定依据：[embedder](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/internvl3_embedder.py)、[训练器](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/scripts/lerobot_train.py)、[Accelerator配置](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/configs/accelerator.py)。这里记录候选配置及验收条件，没有修改依赖锁、实现新trainer或开放Evo能力声明。

## 接入一个 LeRobot 模型

1. 在 `configs/models/` 声明模型的 `backend = "lerobot"` 和上游 `policy_type`，不新增 `plugins/<model>/` 或复制训练脚本。
2. 选择该系列的 `configs/environments/`，固定其源码和依赖。共用后端不要求共用 Conda；不同系列可以固定不同的上游版本，接口变更时需重测共享入口。当前 backend 中的 revision 只是 API 参考，运行器未强制校验安装版本。
3. 实验用 `model`、`environment`、`contract` 加 `[parameters].native_config` 指向上游原生训练 JSON；数据、图像采样、stage、精度、优化器和模型参数都留在原生配置内，不再翻译为自定义统一模型配置。路径相对项目根目录解析，文件哈希写入运行计划。
4. `adapters/lerobot/train.py` 检查 policy.type，然后直接调用 `lerobot.scripts.lerobot_train`。只覆盖输出目录、禁用 W&B、禁用最终/中途 Hub 上传并限定本机执行，不修改 dtype、归一化或 loss。当前仅支持新运行；resume、分布式启动器和远端提交暂走独立原生工作流，不伪装为已接入功能。
5. YAM 样例与保存重载验证后再开放该模型的 `train`；`infer` 需另行接通共享原生 policy + processor 路径，不能因 train launcher 存在就标为可推理。原生权重和 processors 是部署交接物，seal 仅补充哈希，不创造另一套权重格式。

现有 `evo1-yam.toml` 是待替换路径的实验骨架，仍会明确拒绝执行；不是可直接训练的 YAM 配置。Evo环境安装与模型capability分开：安装位置、依赖和CPU检查见 [02](02_installation_and_environment.md#evo-1--lerobot-独立环境)；基座 VLM 已完成权重交接，但尚未发起训练，也未改变 Thor 服务或 3588 控制侧。

训练入口 API 依据固定源码：[原生训练入口](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/scripts/lerobot_train.py)、[训练配置](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/configs/train.py)。

源码来源：[Evo config](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/configuration_evo1.py)、[模型](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/modeling_evo1.py)、[处理器](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/processor_evo1.py)。

## 2026-09-22 · OpenWAM 微调接入

**已实现接口，未验收 GPU 训练或 Thor 推理。** `configs/models/openwam.toml` 选择独立 `openwam` 后端，支持 `train` 和离线 `infer`。本次仅接入模型侧，不启动训练、不更换 Thor 服务。初次训练前仍需数据单位/绝对目标语义审核、模型依赖锁定、计算节点容量与真实保存/恢复验收。

### 固定源码与环境

本地 OpenWAM 的 Git revision `7c5861e45cfe1339a0323f0e0b03a3316c37971c` 原样快照到 `third_party/openwam/`。`UPSTREAM.json` 保存源仓库、revision 与逐文件 SHA256；入口校验快照。只迁入模型/数据处理/原生 trainer、配置、训练入口及许可，不带权重、原仓库 Git 历史或未跟踪资料。快照中的 README 是上游说明，部分完整上游文档链接不随运行时快照迁入。本项目的变更在 `adapters/openwam/`，不另写训练循环。

`environments/openwam.yml` 仅创建 Python 3.12/pip。审计过的 GPU 计算节点上，在独立 `vla-openwam` Conda prefix 安装 `python -m pip install ./third_party/openwam`；它不与 Pi/Evo 的依赖混装。上游 `pyproject.toml` 使用版本范围，**不是已验证的 CUDA 锁文件**；`docker/constraints-cu128.txt` 仅为上游参考，不能据此声称服务器或 Thor 可用。安装后应保存完整包锁、CUDA/DeepSpeed 审计和对应骨干依赖。此轮只在临时测试目录增加 Hydra/OmegaConf/h5py 等轻量依赖，没有安装模型环境。

### 数据接口

当前 reader 读取 **LeRobot v2.0/v2.1 每 episode 文件与 v3.0 共享分片**；正式服务器发布目录实测为 v3.0、14D、30fps。复用原生 episode metadata 读取器，按 episode_index 选择共享 Parquet 内的记录，并按每相机 from_timestamp/fps 定位共享视频；不原位升级数据。相机、14D 次序和语义归 [04](04_data_contracts.md#openwam-yam-数据投影)。缺失/损坏相机、非有限值、fps/时间戳不一致会报错，不自动换 episode 或填黑腕部图像。

先准备明确的训练 episode ID JSON 列表，统计工具只读取这些 episode，输出路径必须位于源数据目录外且尚不存在：

```bash
python adapters/openwam/prepare.py --dataset /absolute/yam-v2 \
  --episodes /absolute/train-episodes.json --output /absolute/new-run/yam-stats.json
```

该命令只做数据审计/统计，不运行训练。产物包含 `joint` 和 `joint_state` 各自的 min/max/mean/std、episode 清单、fps、metadata/parquet 哈希。训练启动复查哈希，生成原生 `normalization_stats.npy`，交由上游 checkpoint 保存器复制。不要拿 Pi 的 delta/分位数 norm 来替代。当前只支持 min-max/z-score；统计不读取验证集。

### 微调与恢复

`configs/native/openwam-yam.example.json` 是**接口示例，不是已批准实验配方**；路径、episode 清单和预算均须替换。DAgger 各轮仍按 AGENTS 的逐轮配方确认要求执行。`configs/experiments/openwam-yam.toml` 将本轮 JSON 交给后端：

```bash
python scripts/vla.py models
python scripts/vla.py plan configs/experiments/openwam-yam.toml train --run-id openwam-yam-r1
python adapters/openwam/train.py --config /absolute/recipe.json --output /absolute/new-artifacts --check-only
# 仅在获准的 Slurm GPU allocation、独立 OpenWAM 环境中执行：
python scripts/vla.py run configs/experiments/openwam-yam.toml train --run-id openwam-yam-r1
```

`--check-only` 只组合配置/检查 metadata，不导入 trainer、不构建模型。正式入口要求 Slurm allocation，按 `nproc_per_node` 调用原生 torchrun + Hydra/Accelerate/DeepSpeed 训练。平台仅标记 target 不会自动 SSH，因此必须在计算节点执行；本地禁止训练。入口关闭 W&B 发送与 Hugging Face 在线下载；模型资产须预先准备。

- `training.finetune_ckpt_path`：完整原生 checkpoint **run 目录**，新输出、step 从零开始。继承 checkpoint 模型配置，再应用显式 `model` 修改，禁止静默改变动作头维度/framework/variant。不使用父目录替代实际含 `config.yaml` 的 run 目录。
- 常见 80D 预训练 checkpoint：模型保持 `action_dim=state_dim=80`；显式开启 `unify_action` 并提供 **14 个不同的整数槽位**，state/action 使用同一映射。槽位需结合该 checkpoint 的原训练语义确定，不能把关节角冒充 EEF xyz/rot6d。没有自动猜测或默认映射。
- 示例的 14D 模式要求已有兼容 14D checkpoint。若从预训练视频骨干建立新 14D 动作专家，将 `finetune_ckpt_path` 设为 null，配置骨干的本地 `model_path`；这属于新动作专家训练，不是完整 80D policy 权重微调。
- `training.resume_ckpt_path`：与 finetune 互斥；恢复原生 `accel_state_step_*`，复用原 run 的权重/优化器/scheduler/RNG。入口要求保存的 model、dataset、project、training 以及进程数一致；变更配方应另开 finetune。上游会进一步检查全状态完整性和 norm 一致性。**上游正常结束会移除续训全状态**，只有权重时应 warm-start，不能假称严格续训。

`training.max_steps`、`save_steps`、`global_step` 是上游 **micro-batch step**；`opt_step` 才是优化器更新次数。不要把梯度累积后的有效 batch 或更新预算算错。原生日志 hook 同时写 `metrics.jsonl`：事件 step 为 microstep，`optimizer_step` 独立保存；只有主进程写，非有限指标计数保留。它可供现有看板消费，不改变 loss/optimizer。模型产物位于平台 `artifacts/checkpoints/<上游时间目录>/`；平台层另存 resolved config、来源与数据审计。checkpoint 应完整交接 `config.yaml`、safetensors、normalization_stats.npy、tokenizer/component assets，不能只拷权重文件。

### 离线推理与验收

`configs/experiments/openwam-reference.toml` 复用平台请求合同；替换 checkpoint、请求和模型版本后使用 `vla plan/run infer`。图像为本地 RGB 文件，入口复用训练时三相机拼图及原生 checkpoint normalizer，返回 `(num_frames - 1) × 14` 绝对动作，动作周期从保存的 dataset fps 计算。每次请求独立进程；不是常驻 IPC 协议。默认不启用 compile/DiT cache，性能优化须单独验收。

本次轻量测试覆盖配置规划、源快照、训练集统计隔离、14D/80D 掩码、时间窗口/末尾填充、实际 MP4 解码、三相机顺序、原生 checkpoint norm 正逆变换、微调动作头拒绝和恢复合同。测试不运行模型前后向或训练循环。真实 checkpoint 加载、GPU loss/梯度、容量/吞吐、实际中断恢复、任务成功率与 Thor 精度/延迟仍待独立验收；现阶段不开放平台 Thor target，避免绕过容器和部署 gate。
