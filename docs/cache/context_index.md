# 01 · Context Index

本文件只负责路由，不保存服务器地址、命令、指标或完整架构事实。

## Mode Packs

- `docs/cache/modes/02_code_change.md`：YAM 代码、配置、数据 pipeline、测试、文档写回和提交。
- `docs/cache/modes/03_deployment.md`：conda、SSH、远端训练/服务、GPU、checkpoint 和安全操作。
- `docs/cache/modes/04_model_ops.md`：YAM 数据、Pi0.5 全量微调、norm、训练评估和实验归因。

## Canonical Docs

- 任意模型训练看板、JSONL/CSV/Trainer-state、LeRobot 指标采集 → `docs/11_training_dashboard.md`
- 多模型薄接入层、Conda、工作树、Evo-1 接入状态 → `docs/10_vla_platform.md`
- Evo-1 独立环境安装与复现 → `docs/02_installation_and_environment.md#Evo-1 / LeRobot 独立环境`；实测安装证据 → `docs/reports/environments/evo1-20260908/README.md`
- MolmoAct2原生LeRobot/LoRA/双夹爪接入 → `docs/reference/molmoact2_integration.md`；环境状态 → `docs/reports/environments/molmoact2-20260908/README.md`

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
| 训练崩溃、Xid、illegal memory access、memcheck | [训练故障经验](../03_training_and_evaluation.md#故障诊断复用与重试条件)；先匹配快照、故障窗口和插桩范围 | 同节关联全量数据复核及两次超时；[故障恢复记录](../03_training_and_evaluation.md#2026-09-09-故障恢复与当前授权)持有历史启动器/续训保护，运行前重新核验当前授权和 checkpoint |
| 续训曲线缺段、重复 step、配置与 loss 不符 | [看板复用与验证](../11_training_dashboard.md#工具复用与配置实测边界)；区分显示参数与源日志 | 同节定位本地父日志拼接、测试入口；[loss 口径](../03_training_and_evaluation.md#loss日志与看板口径2026-09-08用户确认)解释统计窗口 |
| Thor 延迟、realtime-vla、位置偏移、时间广播 | [复用与重试](../reference/thor/10_acceleration_execution.md#realtime-vla-工具复用与重试条件)；先核对真实权重/输入合同/频率证据 | 同节关联离线审计、报告回放和无效归一化首轮；完整数值仍归其上方原复现记录 |
| W&B 配置、旧证据指纹失效 | [证据复核规则](../09_memory_system.md#工程经验的增量整理) | `docs/cache/records/yam-local-logging.json` 为待复验历史，不能当作当前 verified 事实 |

## 状态词

- **已实现**：仓库代码或可复核产物已经存在。
- **已验证**：有明确脚本、报告或 smoke 证据；不能只凭进程启动。
- **计划中**：需要后续数据、训练、服务器审计或用户确认。
- **历史/legacy**：保留用于追溯，不得作为当前 YAM 默认值。
