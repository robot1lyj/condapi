# Repository Guidelines

## 项目定位

本仓库是 YAM 双臂（与 YAM-ABC 同硬件配置）VLA 多模型训练与 Thor 推理平台。控制层复用各模型原生训练器、处理器和权重格式；Pi 使用 OpenPI，LeRobot 系列复用 LeRobot，不复制第二套训练循环。按模型系列隔离 Conda 环境，当前 Pi 默认路线为 `pi05_yam` 全量微调，训练在服务器 GPU 上执行。架构与模型状态分别归 `docs/01_system_architecture.md`、`docs/10_vla_platform.md`；Pi 特有规则不能套用其他模型。

Thor 只负责推理，3588 负责相机、机械臂和控制，两台 IPC 用网线直连。OpenArm、Piper、独立 YAM-ABC-Reproduce 仅作历史参考；不得把其控制代码移入本仓库。

`/home/wuyan-lyj/YAM` 是外部只读参考目录；以实际 cwd / `git rev-parse --show-toplevel` 为工作树根目录，不硬编码主检出路径写文件。

## Context OS 记忆规则

- 本文件 `AGENTS.md` 是唯一项目规则入口；`docs/cache/` 和编号化 `docs/` 是按需记忆与事实 owner。不要另建 `CLAUDE.md`、`.agents`、`.codex` 或平行缓存。路由归 `docs/cache/context_index.md`，详细事实归相关编号文档，历史原因归 `docs/07_change_log.md`。
- 启动只用宿主已注入的本文件；**不默认读取** kernel、index、mode、交接页或历史。先确定当前决策与缺失事实；已知 owner 时直接读相关章节，未知 owner 时查 index；跨主题恢复时才读 kernel，需要具体操作边界时才读一个 mode。已有上下文不重复读取。
- 完整证据和未完成事项留在现有 owner；更新当前状态时替换旧摘要，不把进度逐条追加到热记忆，不为普通进度新建记忆文件。候选经验、失败尝试、单位/版本、适用条件、反例与来源按需记录在 owner；历史通过不能当成当前运行就绪，临时状态使用前复核。
- 无默认累计读取额度。`memory_gate.py` 是可选检索/证据工具，其字节包和 `docs/cache/runtime/` 账本都不是模型上下文 token 计量；旧账本保留历史并按 [09](docs/09_memory_system.md) 原位迁移。搜索与日志先过滤；必要完整条件不截断，原始证据不因节省上下文而删除。
- 热文件预算、写回与复核流程归 [09](docs/09_memory_system.md)；通用方法归 `skills/mlops-memory/`，不在 skill 复制项目事实。

## 编号化文档所有权

架构归 `docs/01_system_architecture.md`，环境归 `02_installation_and_environment.md`，训练归 `03_training_and_evaluation.md`，数据合同归 `04_data_contracts.md`，推理协议归 `05_inference_and_rollout.md`，Thor 部署归 `08_thor_edge_deployment.md`，多模型操作归 `10_vla_platform.md`，看板归 `11_training_dashboard.md`，记忆规则归 `09_memory_system.md`。交接导航归 `00_handoff_index.md`；`06_openarm_research_plan.md`、`07_change_log.md`、`docs/reference/` 为按需读取的历史或参考。详细路由只在 index 维护。

## 环境与代码边界

`packages/vla-platform/` 控制层仅依赖 Python 标准库，不导入 Torch/JAX/LeRobot；各模型在独立 Conda prefix 调用原生入口，不使用 uv 或一个混装环境。`environments/bootstrap.yml` 不代表模型依赖/GPU 已就绪。环境、安装命令、模型目录及审计步骤按需查 `docs/01_system_architecture.md`、`docs/02_installation_and_environment.md` 和 `docs/10_vla_platform.md`；登录节点不运行长训练。本地架构改动不自动更新服务器环境或 Thor 容器。

## 当前 YAM 数据边界

- Pi 当前训练产物为 JAX/Flax checkpoint；新模型可以保留其 LeRobot/PyTorch 原生产物，不要求绕行 JAX。部署转换必须保留原始权重与 LoRA，精度验收依据见 `docs/08_thor_edge_deployment.md`，不得默认接受 BF16 转存或 FP8/NVFP4 量化。
- 双臂合同固定为 14D `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`；YAM 数据的具体物理单位必须由数据 metadata/audit 确认，不能擅自套用 OpenArm degree 或 ROS 弧度。
- 图像键固定为 `observation.images.top_rgb`、`observation.images.left_rgb`、`observation.images.right_rgb`；动作键为单数 `action`；状态键为 `observation.state`。
- 训练默认将每臂 6 个关节动作转为相对当前状态的 delta，夹爪维度保持 absolute；mask 为 `(6,-1,6,-1)`。
- 当前使用 `pi05_yam` 的全量微调配置，正式入口为 `scripts/train_lego_full.py`；具体运行参数和续训边界归 `docs/03_training_and_evaluation.md`，不自动回退为 LoRA。OpenPI 模型内部为 32D、action horizon 为 50，YAM policy 输出裁回 14D。
- Pi 的 YAM 训练使用 `LeRobotYamDataConfig`、`YamInputs`、`YamOutputs`；其他模型使用独立适配器并保持 YAM 原始数据语义，不能强制复用 Pi 的 padding、delta 或 norm。禁止把 OpenArm 16D 或 Piper 14D transform 当作 YAM 默认路径。

## 不可违反的边界

- 2026-09-22 用户明确要求：DAgger每轮生成或修改训练配置、训练命令、作业文件之前，先展示该轮具体训练配方并取得明确确认；A/B可整组确认，配方变更重新确认。数据审核、统计与预算草案可先完成；上轮/总体授权不替代本轮配方确认，同一已确认合同恢复不重复询问。确认记录与操作归 `dagger-flywheel` 技能和本轮实验产物。

- 2026-09-09 用户明确要求：本地工作站禁止运行训练循环或训练 smoke（包括 CPU/debug 小模型），会造成卡顿；仅做静态检查、轻量配置/协议测试和看板验证。训练执行验证留待服务器恢复后在获准的计算资源上进行。
- 2026-09-23 用户反馈当前 Thor Pi 136000 policy 左臂行为异常；在未复核现场输入与结果前不得继续驱动真机。来源与待核条件见 `docs/reports/thor/rtc-20h-136000-20260923/README.md`；当前服务/设备状态每次操作前重新观测。
- 不提交凭据、token、私钥、服务器密码；不删除远端数据/权重/缓存，除非用户明确授权。
- 长训练使用 Slurm 作业或 tmux；端侧推理默认在 Thor 本地容器/进程执行，Thor↔3588 的直连以太网协议是生产数据通道。端口监听不等于推理可用，必须做真实本地推理和跨 IPC 直连 smoke。
- 涉及端侧推理与部署时只操作 Thor 侧；不得读取、修改、同步或替代 3588 的机械臂控制、相机采集和系统部署。
- Thor 推理默认使用 Docker + NVIDIA Container Toolkit，按模型系列隔离容器，Pi 系列共用一个服务，通过配置/checkpoint 选择模型；模型依赖安装在系列镜像内，容器规划与精度验收由 `docs/08_thor_edge_deployment.md` 持有。
- 数据转换只写新目录；原始 YAM 数据和现有下载任务不可覆盖、停止或删除。
- 2026-09-07 用户追加授权：确实损坏的 Lego episode 可修复或整条隔离排除；优先从固定上游 revision 恢复并逐帧验证，保留坏原件、哈希和修复记录，使用双写锁更新仅被修复文件的断点签名。不可把网络/权限/容量问题视为数据损坏；不随意裁帧，不改变其他数据和下载任务。非必要不永久删除。
- 任何 RTC 改动都必须保留旧推理路径，并可通过 `rtc_mode` 关闭或自动回退。

## Git 自动化

默认只提交/备份当前功能分支；合并前按当前授权吸收最新 main、核对并快进，不自动同步 Thor/服务器代码或删除远端环境、权重和实验原件。历史合并授权与原因见 `docs/07_change_log.md`，具体双远端命令见 `docs/02_installation_and_environment.md`。

完成请求后执行 `git diff --check`；代码检查按任务与适用指令执行。只暂存本任务文件，中文提交；本地把同一分支提交推送到已配置的 Gitea `origin` 和 GitHub `github`，核对两端提交。服务器只从 Gitea 获取代码，不访问 GitHub。Git 身份为 `wuyan_lyj <linyongjia@wuyanai.cn>`；不 force push、不在 URL/历史中写凭据、不推送未知远端。
