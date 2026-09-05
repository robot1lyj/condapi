# 00 · 项目交接索引

本页是新同事的唯一入口，只说明文档路由以及当前、计划、历史的边界，不复制完整命令或实验计划。

## 推荐阅读顺序

`README.md` → `01_system_architecture.md` → `02_installation_and_environment.md` → `03_training_and_evaluation.md` → `04_data_contracts.md` → `05_inference_and_rollout.md`。

`06_openarm_research_plan.md` 是历史 OpenArm 研究归档；变更原因和结果只看 `07_change_log.md`。

## 文档所有权

| 编号 | 文档 | 唯一职责 |
|---|---|---|
| 00 | 本页 | 交接路由、当前/计划/历史边界 |
| 01 | 系统架构 | YAM 训练模块、数据流和边界 |
| 02 | 服务器与环境 | conda、远端资源、数据预检、训练路径和安全要求 |
| 03 | 训练与评估 | YAM config、norm、训练、checkpoint 和评估 gate |
| 04 | 数据合同 | LeRobot、YAM 14D、图像/动作键、单位待核项和 norm |
| 05 | 训练后 policy smoke | policy 输入输出协议和最小服务验收，不承载机械臂控制说明 |
| 06 | 历史研究 | OpenArm/KAI0/Evo-RL 旧计划，仅供追溯 |
| 07 | 变更历史 | 按日期记录原因、结果和事故 |

## 状态词

- **已实现**：仓库代码或可复核产物已经存在。
- **已验证**：有明确脚本、报告或 smoke 证据；不能只凭进程启动。
- **计划中**：需要后续数据、训练、服务器审计或用户确认。
- **历史/legacy**：保留用于追溯，不得作为当前 YAM 默认值。

## 不要从哪里开始

- 不从上游 README、`examples/` README 或 `third_party/` 文档推断当前默认流程。
- 不把独立 `/home/wuyan-lyj/YAM/yam-abc-reproduce` 的控制/GUI 代码复制到本仓库；本项目只提取 YAM 训练合同。
- 不把 `docs/07_change_log.md` 当操作手册；它只保存已经发生的原因和结果。
