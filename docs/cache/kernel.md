# Context Kernel · 按需恢复摘要

本页只在跨主题恢复或未知当前主线时读取；不是默认启动文件。操作约束以 `AGENTS.md` 为准，详细事实以链接的 owner 为准。下面的设备/作业状态都是历史观察，使用前重新核验。

- **产品主线：** 多模型控制层已接模型、数据、算法组合配置、inventory/split 与独立 Conda 原生入口；首版是核心与 CLI，多人 API 尚未实现。Pi、OpenWAM、XR-1 的模块化训练路径均未获真实 GPU 验收，XR-1 还缺 YAM FK 末端标签和部署 IK；Evo-1/FastWAM/VLA-JEPA 不得当作已接通。见 [架构](../01_system_architecture.md#多模型接入层)、[状态与操作](../10_vla_platform.md#当前状态)。
- **YAM 合同：** 双臂原始 14D `[左6关节, 夹爪, 右6关节, 夹爪]`；物理单位须据数据审计。Pi 训练将关节改为 delta、夹爪保持 absolute，内部 32D/H50；不得迁移 Pi norm/padding 到别的模型。见 [数据合同](../04_data_contracts.md)、[训练](../03_training_and_evaluation.md)。
- **训练路线：** Pi 默认 `pi05_yam` 全量微调；训练在服务器获准 GPU 资源上执行，本地工作站禁止训练循环。历史 Slurm/下载状态不能当作当前状态。具体 checkpoint、子集、norm、续训合同见 [训练 owner](../03_training_and_evaluation.md)及其引用的运行报告。
- **Thor 与现场：** Thor 负责推理、3588 负责采集和控制，项目不操作 3588。2026-09-23 的 136000 Pi 服务虽有 Thor 本机精度/协议证据，用户反馈左臂异常，未获真机任务验收；操作前复核现场状态与异常输入，不能用端口或离线转换差异推定可驱动。见 [136000 报告](../reports/thor/rtc-20h-136000-20260923/README.md)、[Thor 部署](../08_thor_edge_deployment.md)。
- **恢复方式：** 已知任务直接读其 owner 的相关章节；未知 owner 查 [路由](context_index.md)；历史失败查当前问题的证据与重试条件，不全量加载 [变更历史](../07_change_log.md)。记忆流程与字节预算见 [09](../09_memory_system.md)。
