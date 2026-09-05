# Repository Guidelines

## 项目定位

本仓库是 OpenPI 的 YAM 双臂训练适配分支，当前默认任务是使用 LeRobot 数据对 YAM（与 YAM-ABC 同硬件配置）进行 VLA 后训练，并把训练后 policy 部署到 NVIDIA Jetson AGX Thor 端侧推理。首选模型是 Pi0.5，首选低显存路线是 LoRA；训练仍在服务器 GPU 上，端侧推理默认不依赖远程 policy server。OpenArm、Piper 和独立 YAM-ABC-Reproduce 代码只作为 legacy/reference，不是本项目默认实现。

`/home/wuyan-lyj/YAM` 是外部 YAM 参考目录，只读查看训练数据合同和模型适配信息；本仓库自身始终是 `/home/wuyan-lyj/condapi`，不得把 YAM-ABC 的机械臂控制代码同步进来替代本项目。

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
- `docs/02_installation_and_environment.md`：服务器、环境、数据预检、路径和远端资源。
- `docs/03_training_and_evaluation.md`：训练、评估和 checkpoint gate。
- `docs/04_data_contracts.md`：YAM 数据格式、动作维度、单位待核项和 norm stats。
- `docs/05_inference_and_rollout.md`：训练后 policy 服务协议和最小 smoke；不承载机械臂驱动说明。
- `docs/08_thor_edge_deployment.md`：Thor 官方系统、容器环境、Pi0.5 转换/加速和端侧验收；不承载机械臂驱动说明。
- `docs/09_memory_system.md`：记忆预算、证据生命周期、skill 接入与迭代验收。
- `docs/06_openarm_research_plan.md`：历史 OpenArm/KAI0/Evo-RL 研究归档，不是当前 YAM 路线。
- `docs/07_change_log.md`：按日期记录原因和结果。
- `docs/reference/`：长篇技术参考或 legacy；默认入口不依赖其中的旧结论。

## 代码与目录

- `src/openpi/`：模型、策略、训练、数据 transform 和公共工具。
- `packages/openpi-client/`：通用机器人侧 WebSocket/IO 客户端；YAM 机械臂控制不在本次训练适配范围。
- `scripts/`：训练、数据准备、服务和审计入口。
- `examples/`、`third_party/`：平台示例或 vendored 依赖，除非任务明确要求，不改其 README 来表达本项目事实。

## 环境与常用命令

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

- 双臂合同固定为 14D `[左臂6关节, 左夹爪, 右臂6关节, 右夹爪]`；YAM 数据的具体物理单位必须由数据 metadata/audit 确认，不能擅自套用 OpenArm degree 或 ROS 弧度。
- 图像键固定为 `observation.images.top_rgb`、`observation.images.left_rgb`、`observation.images.right_rgb`；动作键为单数 `action`；状态键为 `observation.state`。
- 训练默认将每臂 6 个关节动作转为相对当前状态的 delta，夹爪维度保持 absolute；mask 为 `(6,-1,6,-1)`。
- `pi05_yam_lora` 是当前低显存首选配置；OpenPI 模型内部为 32D、action horizon 为 50，YAM policy 输出裁回 14D。
- YAM 训练只能使用 `LeRobotYamDataConfig`、`YamInputs`、`YamOutputs`；禁止把 OpenArm 16D 或 Piper 14D transform 当作 YAM 默认路径。

## 不可违反的边界

- 不提交凭据、token、私钥、服务器密码；不删除远端数据/权重/缓存，除非用户明确授权。
- 长训练使用 Slurm 作业或 tmux；端侧推理默认在 Thor 本地容器/进程执行，远程 WebSocket 只作兼容/调试回退。端口监听不等于推理可用，必须做真实本地推理或启用远程协议后的 WebSocket smoke。
- 数据转换只写新目录；原始 YAM 数据和现有下载任务不可覆盖、停止或删除。
- 任何 RTC 改动都必须保留旧推理路径，并可通过 `rtc_mode` 关闭或自动回退。

## Git 自动化

完成请求后执行 `git diff --check`，代码改动再执行 Ruff/pytest，然后 `git add -A` 和中文 commit，例如 `git commit -m "接入YAM双臂训练配置"`。本地工作站提交后把同一 `main` 提交同步到 Gitea 和 GitHub；服务器无法连接 GitHub，只从 Gitea 同步代码：

```bash
git push -u origin main    # 本地 → Gitea，服务器的唯一代码来源
git push github main       # 本地 → GitHub，仅作本地侧备份
# 服务器：git fetch/pull origin main；不访问 github remote
```

Git 身份固定为 `wuyan_lyj <linyongjia@wuyanai.cn>`。本地推送后核对两个远端的 `main`；服务器只核对 Gitea。永远不要 force push，不要把凭据写进 remote URL 或提交历史，也不要推送到未明确配置的其他远端。
