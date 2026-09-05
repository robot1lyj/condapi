# 01 · Context Index

本文件只负责路由，不保存服务器地址、命令、指标或完整架构事实。

## Mode Packs

- `docs/cache/modes/02_code_change.md`：YAM 代码、配置、数据 pipeline、测试、文档写回和提交。
- `docs/cache/modes/03_deployment.md`：conda、SSH、远端训练/服务、GPU、checkpoint 和安全操作。
- `docs/cache/modes/04_model_ops.md`：YAM 数据、Pi0.5 LoRA、norm、训练评估和实验归因。

## Canonical Docs

- 交接入口 → `docs/00_handoff_index.md`
- 架构 → `docs/01_system_architecture.md`
- 服务器与环境 → `docs/02_installation_and_environment.md`
- 训练与评估 → `docs/03_training_and_evaluation.md`
- 数据合同 → `docs/04_data_contracts.md`
- 训练后 policy smoke → `docs/05_inference_and_rollout.md`
- 历史 OpenArm 研究归档 → `docs/06_openarm_research_plan.md`
- 变更历史 → `docs/07_change_log.md`
- 旧方案指引 → `docs/reference/00_reference_index.md`

## Quick Route

- 改代码/配置/文档、跑测试、提交 → `02_code_change.md`
- 远端环境、训练、服务、GPU、tmux → `03_deployment.md`
- YAM 数据审计、SFT、Pi0.5 LoRA、评估 → `04_model_ops.md`
- 只需了解项目 → 从 `docs/00_handoff_index.md` 开始，按编号阅读。

## 状态词

- **已实现**：仓库代码或可复核产物已经存在。
- **已验证**：有明确脚本、报告或 smoke 证据；不能只凭进程启动。
- **计划中**：需要后续数据、训练、服务器审计或用户确认。
- **历史/legacy**：保留用于追溯，不得作为当前 YAM 默认值。
