# 00 · Context Kernel

本文件是带来源的热摘要；详细事实只有对应 owner。启动按 AGENTS → kernel → index → 一个相关 mode，已注入内容不重读。长期文档不按行数删减；加载预算与执行器见 [记忆系统](../09_memory_system.md)。

## 当前默认

- YAM 双臂训练适配，首选 Pi0.5 LoRA / `pi05_yam_lora`；训练运行在服务器 GPU。训练关闭 W&B，使用本地日志、JSONL/CSV 指标与曲线，见 [训练](../03_training_and_evaluation.md)。
- YAM 为 14D `[左6关节, 左夹爪, 右6关节, 右夹爪]`；关节 delta、夹爪 absolute，mask `(6,-1,6,-1)`；模型内部 32D、horizon 50。物理单位仍须数据审计，不套用 OpenArm/Piper 合同，见 [数据合同](../04_data_contracts.md)。
- 训练产物确定为 JAX/Flax checkpoint；Thor 按模型系列隔离容器，Pi 系列共用一个服务，3588 采集/控制，两 IPC 网线直连，本项目只操作 Thor。首版验证容器内原生 JAX，保留 LoRA；转换与量化须另过精度验收，见 [Thor 部署](../08_thor_edge_deployment.md)。
- 本仓库为 `/home/wuyan-lyj/condapi`，外部 YAM 参考只读；设备、Git、数据保护和 RTC 回退边界由 `AGENTS.md` 持有。

## 恢复任务

- 服务器入口、环境、原始数据路径和作业保护对象见 [环境](../02_installation_and_environment.md)。作业/下载状态是历史观察，使用前重新核验；没有实时证据就标记未知。
- Thor 已完成官方系统安装并从 NVMe 启动；2026-09-07 实机复核为 JetPack 7.2.1 / L4T 39.2.1。Wi-Fi 开机自动连接已配置、SSH 密钥登录可用；USB `192.168.55.1` 为临时管理/传输通道。HDMI 仍无显示，不能误记为安装未完成；详见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)。
- Pi0.5 原始 JAX 基础权重、分词器和 3 条完整 YAM 回放轨迹已放到 Thor 本地；测试不依赖服务器实时传输。ARM64 JAX 26.05 镜像已导入且 FP32/BF16 GPU 内核检查通过；OpenPI 依赖适配、完整模型推理和精度对照尚未完成，见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态) 与 [中文测试报告](../reports/thor/index.html)。
- Thor 已安装 NoMachine 9.8.3 并配置自启动，无屏 GNOME / 1920×1080 输出已核对；USB 连接 `192.168.55.1:4000`，账户 `wuyan-lyj`。为此停用 GDM 本地登录画面；恢复方法及许可边界见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)，客户端画面/重启回连不冒充已验收。
- 用户指定 **仅推理/测试期间**使用 MAXN（模式 0）+ 最高 CPU/GPU/EMC 频率；结束后恢复日常 120W、动态调频和自动风扇，不设 MAXN/锁频开机常驻。入口 `scripts/thor/maxn_session.py` 负责临时切换与退出恢复；每次核对温度/降频，见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)。
- 新结果写回对应 owner，原因写入 [变更历史](../07_change_log.md)；候选经验、失效与预算协议见 [记忆系统](../09_memory_system.md)。
