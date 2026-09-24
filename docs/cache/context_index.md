# Context Index · 按需路由

本页只在 owner 不明确时读取，不保存运行状态、配置值或授权。选中一条路由后只读相关章节；已知 owner 可跳过本页。

## Owner

| 任务 | 先读 | 仅在需要时展开 |
|---|---|---|
| 交接/架构 | 交接选 [00](../00_handoff_index.md)；架构选 [01](../01_system_architecture.md#多模型接入层) | 缺当前能力状态才读 [10](../10_vla_platform.md#当前状态) |
| 环境、服务器、checkpoint 下载 | [02](../02_installation_and_environment.md) | 对应 `docs/reports/environments/` 证据 |
| 训练、评估、恢复 | [03](../03_training_and_evaluation.md) | 对应 `docs/reports/training/` 运行报告 |
| YAM 数据、单位、分割、norm | [04](../04_data_contracts.md) | [10 模块化配置](../10_vla_platform.md#模块化配置后端-v02) |
| Thor 推理协议/部署 | 协议选 [05](../05_inference_and_rollout.md)；系统/转换选 [08](../08_thor_edge_deployment.md) | 当前 checkpoint 报告、`docs/reference/thor/` 阶段手册 |
| 多模型、XR-1、Conda 接入 | [10](../10_vla_platform.md) | [XR-1 环境证据](../reports/environments/xr1-20260924/README.md)、[MolmoAct2 参考](../reference/molmoact2_integration.md) |
| 看板 | [11](../11_training_dashboard.md) | 指标源报告 |
| 记忆维护 | [09](../09_memory_system.md) | `skills/mlops-memory/SKILL.md` 对应参考 |
| 历史原因/旧研究 | [07](../07_change_log.md)、[06](../06_openarm_research_plan.md) | 按关键词定位，不整篇预载 |

## 操作入口

- 代码/文档改动按需读 `modes/02_code_change.md`；远端环境/服务读 `modes/03_deployment.md`；YAM 训练/数据审计读 `modes/04_model_ops.md`。至多读取当前适用的一个 mode，且不替代 `AGENTS.md`。
- DAgger/HIL 轮次 → `dagger-flywheel` 技能、[03 方案](../03_training_and_evaluation.md#2026-09-21--乐高-dagger-首轮方案)、[操作手册](../reference/lego_dagger_playbook.md)；每轮配方确认边界仍归 `AGENTS.md`。
- 新 Pi checkpoint 转 Thor → `thor-checkpoint-deploy` 技能、[交接](../reference/thor/12_checkpoint_handoff.md)。Thor 安装按 [阶段入口](../reference/thor/00_start_here.md)选一份冷手册。
- Xiaomi XR-1 的 60D/14D 与末端语义 → [10 XR-1](../10_vla_platform.md#2026-09-24--xr-1-原生训练入口)；官方 checkpoint/环境身份见 [02](../02_installation_and_environment.md)。

## 问题检索

下表只给出处和复核条件；历史修复不是当前操作授权。

| 症状 | 当前检查与历史证据 |
|---|---|
| 乐高接近但不抓、16 mm 小颗粒、控制时序 | [03 训练重设计](../03_training_and_evaluation.md#2026-09-16--训练重设计先复现乐高分拣再比较模型)、[原始控制重放](../reports/training/redesign-20260916/README.md)；先区分预测年龄、滤波和反馈，再谈适配。 |
| training-time RTC、10h 子集 | [03 RTC 合同](../03_training_and_evaluation.md#2026-09-16--pi05-training-time-rtc基础权重与随机10小时)、[运行交接](../reports/training/rtc-base-10h-20260916/README.md)；复核子集 norm、保存/恢复状态。 |
| Evo-1 四卡预算/梯度累积 | [03 四卡试验](../03_training_and_evaluation.md#采集期间的evo-1四卡试验)、[10 固定版本](../10_vla_platform.md#四卡训练落地前的固定版本检查)；更新数与 microstep 分开。 |
| 训练崩溃、Xid、illegal memory access | [03 故障诊断](../03_training_and_evaluation.md#故障诊断复用与重试条件)；匹配故障窗口、范围及重试前提。 |
| loss 曲线缺段或重复 step | [11 看板边界](../11_training_dashboard.md#工具复用与配置实测边界)、[03 loss 口径](../03_training_and_evaluation.md#loss日志与看板口径2026-09-08用户确认)。 |
| Thor 延迟/位置偏移/realtime-vla | [加速执行与重试条件](../reference/thor/10_acceleration_execution.md#realtime-vla-工具复用与重试条件)；先确认权重、输入合同和频率。 |
| W&B/旧证据指纹失效 | [09 证据规则](../09_memory_system.md#证据与失效)、`records/yam-local-logging.json`；旧记录待复验。 |
