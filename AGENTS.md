# Repository Guidelines

## 项目定位

2026-09-08 起在独立工作树分支推进多模型接入层改造：复用 LeRobot 的模型、数据处理器和训练器，本仓库只增加配置、Conda 运行、模型交接与 Thor 部署适配；不重写第二套 LeRobot。模型放 `configs/models/`，代码入口按 `adapters/openpi`、`adapters/lerobot` 后端组织，不再为每个模型复制插件/训练循环。下文 OpenPI 规则仍适用于 Pi 后端，不能套用到所有模型。当前阶段与操作见 `docs/10_vla_platform.md`，架构 owner 为 `docs/01_system_architecture.md`。

本仓库是 OpenPI 的 YAM 双臂训练适配分支，当前默认任务是使用 LeRobot 数据对 YAM（与 YAM-ABC 同硬件配置）进行 VLA 后训练，并把训练后 policy 部署到 NVIDIA Jetson AGX Thor 端侧推理。系统由两台 IPC 组成：Thor 只负责模型推理，3588 负责相机采集、机械臂控制和控制侧逻辑，两者通过网线直连交换数据。首选模型是 Pi0.5，首选低显存路线是 LoRA；训练仍在服务器 GPU 上，模型不再放在远程推理服务器。OpenArm、Piper 和独立 YAM-ABC-Reproduce 代码只作为 legacy/reference，不是本项目默认实现。

`/home/wuyan-lyj/YAM` 是外部 YAM 参考目录，只读查看训练数据合同和模型适配信息；主检出目录是 `/home/wuyan-lyj/condapi`，工作树以实际 cwd / `git rev-parse --show-toplevel` 为准，不得硬编码主目录来写文件。不得把 YAM-ABC 的机械臂控制代码同步进来替代本项目。

## Context OS 记忆规则

- 唯一记忆系统是 `AGENTS.md` + `docs/cache/`；不要新增 `.agents`、`.codex` 或其他平行缓存。
- 启动时依次读取本文件、`docs/cache/kernel.md`、`docs/cache/context_index.md`，然后按路由最多读取一个 mode；已由宿主注入的内容不重复读取。
- 稳定事实只保留一个 owner：规则归本文件，路由归 index，操作边界归 mode，详细事实归编号化 `docs/`，历史原因归 `docs/07_change_log.md`。kernel 只保留带来源的摘要，不独立维护第二份事实。
- 取消文档行数硬限制；长期信息完整保留，严格限制进入上下文的内容。记忆包最多 12,288 UTF-8 字节，同一活跃上下文累计最多 32,768 字节（含已加载记忆）；用 `skills/mlops-memory/scripts/memory_gate.py` 执行准入，超预算整段拒绝，不截掉必要条件。
- 每个仍保留历史的上下文使用同一预算账本 `docs/cache/runtime/`，不得靠换会话 ID 或多次读取绕过累计限制；真实压缩/新上下文后才重建并重新计入保留内容。完整请求 token 限制需宿主按实际 tokenizer、消息/工具封装和输出预留执行；字节预算不能宣称为完整 token 限制。
- 临时状态带观察时间并在使用前复核；经验先候选、再证据验证、再合并 owner；原始产物不因压缩而删除。通用 skill 源码在 `skills/mlops-memory/`，不承载项目记忆副本。设计与验收归 `docs/09_memory_system.md`。

## 编号化文档所有权

- `docs/00_handoff_index.md`：交接导航和当前/计划/历史边界。
- `docs/01_system_architecture.md`：代码与训练数据流架构。
- `docs/02_installation_and_environment.md`：服务器、Thor 环境、数据预检、路径和远端资源。
- `docs/03_training_and_evaluation.md`：训练、评估和 checkpoint gate。
- `docs/04_data_contracts.md`：YAM 数据格式、动作维度、单位待核项和 norm stats。
- `docs/05_inference_and_rollout.md`：训练后 policy 的 Thor 本地协议、Thor↔3588 网络通道和最小 smoke；不承载机械臂驱动说明。
- `docs/08_thor_edge_deployment.md`：Thor 官方系统、容器环境、Pi0.5 转换/加速和端侧验收；下属 `docs/reference/thor/` 为按阶段读取的安装操作冷手册，不承载机械臂驱动说明。
- `docs/09_memory_system.md`：记忆预算、证据生命周期、skill 接入与迭代验收。
- `docs/10_vla_platform.md`：新接入层的操作、模型/后端状态、Conda 工作流和模型接入验收；不是模型训练实现的第二份文档。
- `docs/06_openarm_research_plan.md`：历史 OpenArm/KAI0/Evo-RL 研究归档，不是当前 YAM 路线。
- `docs/07_change_log.md`：按日期记录原因和结果。
- `docs/reference/`：长篇技术参考或 legacy；默认入口不依赖其中的旧结论。

## 代码与目录

- `configs/`：模型选择、原生配置引用、系列环境、实验与机器人合同投影。
- `adapters/`：共享 LeRobot/OpenPI 入口；`packages/vla-platform/` 只做无模型依赖的调度和交接。
- `src/openpi/`：模型、策略、训练、数据 transform 和公共工具。
- `packages/openpi-client/`：通用机器人侧 WebSocket/IO 客户端；YAM 机械臂控制不在本次训练适配范围。
- `scripts/`：训练、数据准备、服务和审计入口。
- 非 YAM 的上游 `examples/` 和 ALOHA/LIBERO 子模块已清理；需要原作者示例时查固定上游版本或 Git 历史，不再作为默认目录。Pi 转换器在 `adapters/openpi/convert_jax_model_to_pytorch.py`。

## 环境与常用命令

新多模型接入层按模型系列使用独立 Conda prefix，不使用一个包含所有模型依赖的环境。控制层 `packages/vla-platform` 仅依赖 Python 标准库，不能导入 Torch/JAX/LeRobot。`environments/bootstrap.yml` 只创建 Python/pip，不代表已安装模型依赖或通过 GPU 审计。既有服务器环境和 Thor Pi 容器不在本地架构改造中自动更新。

服务器 module 入口为：

```bash
module load miniconda3/26.1.1
```

本项目使用独立的 `/home/wuyan/.conda/envs/condapi-yam`。使用 conda/pip 镜像和项目依赖，不使用 uv。Python 版本与完整依赖必须在 GPU 计算节点审计后才能宣称可训练；不要在登录节点执行长训练。

```bash
module load miniconda3/26.1.1
conda activate /home/wuyan/.conda/envs/condapi-yam
conda run -p /home/wuyan/.conda/envs/condapi-yam python -m pip install --no-build-isolation --no-deps -e .
conda run -p /home/wuyan/.conda/envs/condapi-yam python -m pip install --no-build-isolation --no-deps -e packages/openpi-client
```

验证和格式化：

```bash
ruff check .
ruff format .
conda run -p /home/wuyan/.conda/envs/condapi-yam python -m pytest --strict-markers -m "not manual"
```

## 当前 YAM 数据边界

- Pi 当前训练产物为 JAX/Flax checkpoint；新模型可以保留其 LeRobot/PyTorch 原生产物，不要求绕行 JAX。部署转换必须保留原始权重与 LoRA，精度验收依据见 `docs/08_thor_edge_deployment.md`，不得默认接受 BF16 转存或 FP8/NVFP4 量化。
- 双臂合同固定为 14D `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`；YAM 数据的具体物理单位必须由数据 metadata/audit 确认，不能擅自套用 OpenArm degree 或 ROS 弧度。
- 图像键固定为 `observation.images.top_rgb`、`observation.images.left_rgb`、`observation.images.right_rgb`；动作键为单数 `action`；状态键为 `observation.state`。
- 训练默认将每臂 6 个关节动作转为相对当前状态的 delta，夹爪维度保持 absolute；mask 为 `(6,-1,6,-1)`。
- `pi05_yam_lora` 是当前低显存首选配置；OpenPI 模型内部为 32D、action horizon 为 50，YAM policy 输出裁回 14D。
- Pi 的 YAM 训练使用 `LeRobotYamDataConfig`、`YamInputs`、`YamOutputs`；其他模型使用独立适配器并保持 YAM 原始数据语义，不能强制复用 Pi 的 padding、delta 或 norm。禁止把 OpenArm 16D 或 Piper 14D transform 当作 YAM 默认路径。

## 不可违反的边界

- 不提交凭据、token、私钥、服务器密码；不删除远端数据/权重/缓存，除非用户明确授权。
- 长训练使用 Slurm 作业或 tmux；端侧推理默认在 Thor 本地容器/进程执行，Thor↔3588 的直连以太网协议是生产数据通道。端口监听不等于推理可用，必须做真实本地推理和跨 IPC 直连 smoke。
- 本任务只改 Thor 侧；不得读取、修改、同步或替代 3588 的机械臂控制、相机采集和系统部署。
- Thor 推理默认使用 Docker + NVIDIA Container Toolkit，按模型系列隔离容器，Pi 系列共用一个服务，通过配置/checkpoint 选择模型；模型依赖安装在系列镜像内，容器规划与精度验收由 `docs/08_thor_edge_deployment.md` 持有。
- 数据转换只写新目录；原始 YAM 数据和现有下载任务不可覆盖、停止或删除。
- 2026-09-07 用户追加授权：确实损坏的 Lego episode 可修复或整条隔离排除；优先从固定上游 revision 恢复并逐帧验证，保留坏原件、哈希和修复记录，使用双写锁更新仅被修复文件的断点签名。不可把网络/权限/容量问题视为数据损坏；不随意裁帧，不改变其他数据和下载任务。非必要不永久删除。
- 任何 RTC 改动都必须保留旧推理路径，并可通过 `rtc_mode` 关闭或自动回退。

## Git 自动化

独立工作树改造期间只在当前功能分支提交并向已配置的两个远端备份同名分支；不自动合并、推送或部署 `main`，不自动同步 Thor/服务器代码。验收合并后才使用下面的 main 同步流程。用户允许不兼容的架构改造，不等于授权删除原始权重、实验证据或正在使用的远端环境。

完成请求后执行 `git diff --check`，代码改动再执行 Ruff/pytest，然后 `git add -A` 和中文 commit，例如 `git commit -m "接入YAM双臂训练配置"`。本地工作站提交后把同一 `main` 提交同步到 Gitea 和 GitHub；服务器无法连接 GitHub，只从 Gitea 同步代码：

```bash
git push -u origin main    # 本地 → Gitea，服务器的唯一代码来源
git push github main       # 本地 → GitHub，仅作本地侧备份
# 服务器：git fetch/pull origin main；不访问 github remote
```

Git 身份固定为 `wuyan_lyj <linyongjia@wuyanai.cn>`。本地推送后核对两个远端的 `main`；服务器只核对 Gitea。永远不要 force push，不要把凭据写进 remote URL 或提交历史，也不要推送到未明确配置的其他远端。
