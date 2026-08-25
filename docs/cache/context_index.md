# 01 · Context Index

本文件只负责路由，不保存命令、指标、硬件地址或架构事实。

## Mode Packs

- `docs/cache/modes/02_code_change.md`：代码、配置、数据 pipeline、测试、文档写回和提交。
- `docs/cache/modes/03_deployment.md`：conda、SSH、远端训练/服务、checkpoint 和真机安全。
- `docs/cache/modes/04_model_ops.md`：OpenArm 数据、SFT/KAI0/Evo-RL、rollout 评估和实验归因。

## Canonical Docs

- 交接入口 → `docs/00_handoff_index.md`
- 架构 → `docs/01_system_architecture.md`
- 安装与资源 → `docs/02_installation_and_environment.md`
- 训练与评估 → `docs/03_training_and_evaluation.md`
- 数据合同 → `docs/04_data_contracts.md`
- 推理与 rollout → `docs/05_inference_and_rollout.md`
- 当前研究计划 → `docs/06_openarm_research_plan.md`
- 变更历史 → `docs/07_change_log.md`
- 旧方案指引 → `docs/reference/00_reference_index.md`

## Quick Route

- 改代码/配置/文档、跑测试、提交 → `02_code_change.md`
- 远端环境、训练、服务、GPU、tmux → `03_deployment.md`
- 数据采集、SFT、KAI0、Evo-RL、HIL、rollout → `04_model_ops.md`
- 只需了解项目 → 从 `docs/00_handoff_index.md` 开始，按编号顺序阅读。
