# 00 · 项目交接索引

本页是新同事的唯一入口。它只说明“哪类问题看哪份文档”，不复制命令、训练指标或完整实验计划。

## 推荐阅读顺序

`README.md` → `01_system_architecture.md` → `02_installation_and_environment.md` → `03_training_and_evaluation.md` → `04_data_contracts.md` → `05_inference_and_rollout.md` → `06_openarm_research_plan.md`。

## 文档所有权

| 编号 | 文档 | 唯一职责 |
|---|---|---|
| 00 | 本页 | 交接路由、当前/计划/历史边界 |
| 01 | 系统架构 | 代码模块、数据流、OpenArm/Piper 边界 |
| 02 | 服务器与环境 | conda、远端资源、数据预检、训练路径和安全要求 |
| 03 | 训练与评估 | config、训练命令、checkpoint 和评估 gate |
| 04 | 数据合同 | LeRobot、16D、单位、清洗、norm stats |
| 05 | 推理与 rollout | serve、WebSocket、HIL、RTC、安全 |
| 06 | 研究计划 | KAI0/Evo-RL/Hybrid 当前状态和下一步 |
| 07 | 变更历史 | 按日期记录原因、结果和事故 |

## 状态词

- **已实现**：仓库代码或可复核产物已经存在。
- **已验证**：有明确脚本、报告或真机 smoke 证据；不能只凭“服务启动”。
- **计划中**：需要后续采集、训练或用户确认，不能当作现状。
- **历史/legacy**：保留用于追溯，不得作为新 OpenArm 实验默认值。

## 不要从哪里开始

- 不从上游 README、`examples/` README 或 `third_party/` 文档推断本项目默认流程。
- 不从旧 Piper 文档拼接 OpenArm 命令；Piper 只看 `docs/reference/legacy/piper.md`。
- 不把 `docs/07_change_log.md` 当操作手册；它只保存已经发生的原因和结果。
