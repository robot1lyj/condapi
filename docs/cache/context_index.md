# 01 · Context Index

本文件只负责路由，不保存服务器地址、命令、指标或完整架构事实。

## Mode Packs

- `docs/cache/modes/02_code_change.md`：YAM 代码、配置、数据 pipeline、测试、文档写回和提交。
- `docs/cache/modes/03_deployment.md`：conda、SSH、远端训练/服务、GPU、checkpoint 和安全操作。
- `docs/cache/modes/04_model_ops.md`：YAM 数据、Pi0.5 全量微调、norm、训练评估和实验归因。

## Canonical Docs

- 乐高DAgger、人工接管纠正、首轮混采实验与论文依据 → [训练owner](../03_training_and_evaluation.md#2026-09-21--乐高-dagger-首轮方案) → [操作手册](../reference/lego_dagger_playbook.md)；字段归04，均须区分方案与实现。

- 任意模型训练看板、JSONL/CSV/Trainer-state、LeRobot 指标采集 → `docs/11_training_dashboard.md`
- 多模型薄接入层、Conda、工作树、Evo-1 接入状态 → `docs/10_vla_platform.md`
- Evo-1 独立环境安装与复现 → `docs/02_installation_and_environment.md#Evo-1 / LeRobot 独立环境`；实测安装证据 → `docs/reports/environments/evo1-20260908/README.md`
- MolmoAct2原生LeRobot/LoRA/双夹爪接入 → `docs/reference/molmoact2_integration.md`；环境状态 → `docs/reports/environments/molmoact2-20260908/README.md`
- Xiaomi-Robotics-1 / XR-1 checkpoint、服务器环境与 60D/14D 适配 gate → `docs/02_installation_and_environment.md#XR-1 服务器环境（2026-09-24）`、`docs/10_vla_platform.md`；完整哈希/版本/安装证据 → `docs/reports/environments/xr1-20260924/README.md`

- 交接入口 → `docs/00_handoff_index.md`
- 架构 → `docs/01_system_architecture.md`
- 服务器与环境 → `docs/02_installation_and_environment.md`
- 训练与评估 → `docs/03_training_and_evaluation.md`
- 数据合同 → `docs/04_data_contracts.md`
- 训练后 policy smoke → `docs/05_inference_and_rollout.md`
- Thor 端侧系统与 Pi0.5 部署 → `docs/08_thor_edge_deployment.md`
- 历史 OpenArm 研究归档 → `docs/06_openarm_research_plan.md`
- 变更历史 → `docs/07_change_log.md`
- 记忆架构、按需检索、账本迁移与证据记录 → `docs/09_memory_system.md`
- 旧方案指引 → `docs/reference/00_reference_index.md`

## Quick Route

- 从首次任务SFT模型开始DAgger/HIL纠正训练、3～5轮飞轮 → `dagger-flywheel`（源码 `skills/dagger-flywheel/SKILL.md`）；项目执行参数见03及其下属DAgger手册，数据字段归04。

- 新 Pi0.5 checkpoint / 权重就绪 / 转为 Thor 可推理 → 优先使用 `thor-checkpoint-deploy`（源码 `skills/thor-checkpoint-deploy/SKILL.md`）；操作 owner 为 `docs/reference/thor/12_checkpoint_handoff.md`。

- 多模型架构改造 → `02_code_change.md`，再读 `docs/01_system_architecture.md#多模型接入层`；执行命令和模型状态按需读 `docs/10_vla_platform.md` 的相关章节，不加载所有模型文档。

- 改代码/配置/文档、跑测试、提交 → `02_code_change.md`
- 远端环境、训练、服务、GPU、tmux → `03_deployment.md`
- Thor 系统盘、容器、JAX→PyTorch→TensorRT 和端侧 smoke → `08_thor_edge_deployment.md`
- Thor 逐步安装指导/烧录交接 → `docs/reference/thor/00_start_here.md`，只按当前阶段读取；版本、精度与当前状态仍归 `08_thor_edge_deployment.md`
- YAM 数据审计、SFT、Pi0.5 全量微调、评估 → `04_model_ops.md`
- 记忆维护与任务恢复 → 当前安装的 `mlops-memory/SKILL.md`（本项目源码为 `skills/mlops-memory/SKILL.md`）；先明确决策与缺口，再按需读取 `docs/09_memory_system.md` 和相关 owner 章节；账本迁移只展开 skill 的 `references/usage.md`。
- 只需了解项目 → 从 `docs/00_handoff_index.md` 的摘要和导航开始，只展开与当前问题相关的 owner 章节，不按编号加载全部文档。

## 问题与行动路由

只展开当前问题所需的一条，再按缺口查看关联；下列是导航，不是运行授权。

| 问题 / 别名 | 检查与前置条件 | 历史尝试 / 修复或工具入口 |
|---|---|---|
| 乐高接近但不抓、16 mm小颗粒补采、100000现场适配、控制时序 | [训练重设计](../03_training_and_evaluation.md#2026-09-16--训练重设计先复现乐高分拣再比较模型)；区分预测来源年龄、滤波响应、提交与反馈，先验收再适配 | [乐高分拣3原始证据/控制重放](../reports/training/redesign-20260916/README.md)；现场50+10集、原D1及Evo均为计划，未开训/切模型 |
| training-time RTC、基础权重、随机10小时子集 | [JAX实现与启动](../03_training_and_evaluation.md#2026-09-16--pi05-training-time-rtc基础权重与随机10小时)；前缀条件、新run、子集重算norm、完整episode清单 | `train_lego_full.py` 与 `select_lego_rtc_episodes.py`；18:15已授权启动2140，真实GPU更新通过；[运行/resume交接](../reports/training/rtc-base-10h-20260916/README.md)，首个保存/实际恢复与trained RTC部署待验收 |
| 采集期间训练Evo、4×4090、两阶段预算、梯度累积步数 | [四卡试验](../03_training_and_evaluation.md#采集期间的evo-1四卡试验)；与现场采集并行、同四卡训练串行，预算按优化器更新 | [固定版本落地检查](../10_vla_platform.md#四卡训练落地前的固定版本检查)；尚无真实GPU容量/吞吐，环境观察归02及只读快照 |
| 训练崩溃、Xid、illegal memory access、memcheck | [训练故障经验](../03_training_and_evaluation.md#故障诊断复用与重试条件)；先匹配快照、故障窗口和插桩范围 | 同节关联全量数据复核及两次超时；[故障恢复记录](../03_training_and_evaluation.md#2026-09-09-故障恢复与当前授权)持有历史启动器/续训保护，运行前重新核验当前授权和 checkpoint |
| 续训曲线缺段、重复 step、配置与 loss 不符 | [看板复用与验证](../11_training_dashboard.md#工具复用与配置实测边界)；区分显示参数与源日志 | 同节定位本地父日志拼接、测试入口；[loss 口径](../03_training_and_evaluation.md#loss日志与看板口径2026-09-08用户确认)解释统计窗口 |
| Thor 延迟、realtime-vla、位置偏移、时间广播 | [复用与重试](../reference/thor/10_acceleration_execution.md#realtime-vla-工具复用与重试条件)；先核对真实权重/输入合同/频率证据 | 同节关联离线审计、报告回放和无效归一化首轮；完整数值仍归其上方原复现记录 |
| W&B 配置、旧证据指纹失效 | [证据复核规则](../09_memory_system.md#工程经验的增量整理) | `docs/cache/records/yam-local-logging.json` 为待复验历史，不能当作当前 verified 事实 |

## 状态词

- **已实现**：仓库代码或可复核产物已经存在。
- **已验证**：有明确脚本、报告或 smoke 证据；不能只凭进程启动。
- **计划中**：需要后续数据、训练、服务器审计或用户确认。
- **历史/legacy**：保留用于追溯，不得作为当前 YAM 默认值。
