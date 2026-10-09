# Context Kernel · 按需恢复摘要

本页只在跨主题恢复或未知主线时读取；不是默认启动文件。用户约束以已注入的 `AGENTS.md` 为准，事实和未完成事项以各 owner 为准；不在这里保存设备、作业、checkpoint 或下载的实时状态。

- **多模型平台：** 控制层复用各模型原生入口，模型、环境及能力状态由 [架构](../01_system_architecture.md#多模型接入层)、[环境](../02_installation_and_environment.md)、[操作与待验收项](../10_vla_platform.md#当前状态)持有。替身/静态检查不代表真实 GPU 训练或 Thor 推理通过；XR-1 的 FK、统计与部署 IK 等 gate 从 owner 恢复。
- **YAM 数据与训练：** 单位、14D 双臂语义、模型独立适配及 norm 从 [04](../04_data_contracts.md)取；Pi 默认全量微调、RTC 与续训合同从 [03](../03_training_and_evaluation.md)取。50h RTC 运行历史见[既有交接](../reports/training/rtc-base-50h-20260921/README.md)，作业/进程状态使用前重新观测，不恢复旧监控为自动任务。
- **抓取残差学习：** PARTS/RLT 的实现、动作空间、待定奖励及数据条件见 [03](../03_training_and_evaluation.md#parts-左右抓取残差学习方案2026-09-30)、[05](../05_inference_and_rollout.md#parts-左右抓取的服务端与客户端合同2026-09-30方案)。旧夹爪、动作相位和左臂异常已由用户确认解决；当前方案仍须核对现场基线。客户端任务指令审核边界归 `AGENTS.md`。
- **Thor 推理：** 当前部署观察、checkpoint 报告、旧权重保留/清理和恢复条件见 [08](../08_thor_edge_deployment.md#1-当前决策)，输入与 RTC 协议见 [05](../05_inference_and_rollout.md)。操作仅限 Thor；现场及跨 IPC 验收不能由端口、历史本机测试或用户对旧故障的确认替代。
- **恢复方式：** 已知 owner 直接读必要章节；未知 owner 查 [路由](context_index.md)；失败按问题和重试条件查来源。来源检查、冷热预算和验收归 [09](../09_memory_system.md)，不全量加载历史或强制生成结构化记录。
