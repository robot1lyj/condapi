# 00 · Context Kernel

本文件是带来源的热摘要；详细事实只有对应 owner。启动按 AGENTS → kernel → index → 一个相关 mode，已注入内容不重读。长期文档不按行数删减；加载预算与执行器见 [记忆系统](../09_memory_system.md)。

## 当前默认

- YAM 双臂训练适配，首选 Pi0.5 LoRA / `pi05_yam_lora`；训练运行在服务器 GPU。训练关闭 W&B，使用本地日志、JSONL/CSV 指标与曲线，见 [训练](../03_training_and_evaluation.md)。
- YAM 为 14D `[左6关节, 左夹爪, 右6关节, 右夹爪]`；关节 delta、夹爪 absolute，mask `(6,-1,6,-1)`；模型内部 32D、horizon 50。物理单位仍须数据审计，不套用 OpenArm/Piper 合同，见 [数据合同](../04_data_contracts.md)。
- Thor 本地负责模型推理，3588 负责采集和控制，通过网线交换数据。本项目不读取或修改 3588 代码；JAX checkpoint 为转换源和数值参考，转换/精度路线与实机待验项见 [Thor 部署](../08_thor_edge_deployment.md)。
- 本仓库为 `/home/wuyan-lyj/condapi`，外部 YAM 参考只读；设备、Git、数据保护和 RTC 回退边界由 `AGENTS.md` 持有。

## 恢复任务

- 服务器入口、环境、原始数据路径和作业保护对象见 [环境](../02_installation_and_environment.md)。作业/下载状态是历史观察，使用前重新核验；没有实时证据就标记未知。
- Thor 安装介质、系统版本和下载验收状态见 [Thor 部署](../08_thor_edge_deployment.md)；官方 ISO 已下载并完成 SHA-256 校验，但系统盘制作、实机刷写和 IPC smoke 仍未完成。
- 新结果写回对应 owner，原因写入 [变更历史](../07_change_log.md)；候选经验、失效与预算协议见 [记忆系统](../09_memory_system.md)。
