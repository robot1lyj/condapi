# 00 · Context Kernel

本文件是带来源的热摘要；详细事实只有对应 owner。启动按 AGENTS → kernel → index → 一个相关 mode，已注入内容不重读。长期文档不按行数删减；按需检索与计量边界见 [记忆系统](../09_memory_system.md)。

## 当前默认

- 2026-09-08 功能分支收敛为模型配置 + 共享后端：`configs/models/` 选择模型，`adapters/lerobot/` 共用原生入口、`adapters/openpi/` 保留 Pi；撤销按模型的 plugins 目录。系列环境仍独立 Conda，LeRobot 模型不重写 trainer/processor。控制层和合同测试已有，Evo-1/FastWAM/VLA-JEPA 尚未接通 GPU 训练或推理；共享入口不等于模型已支持。架构见 [01](../01_system_architecture.md#多模型接入层)，当前状态和 Evo 接入步骤见 [10](../10_vla_platform.md)。
- 2026-09-24 XR-1 官方 5B 后训练 checkpoint 与独立 Conda 环境已放在服务器；`pip check` 通过。原生 state 是关节/夹爪，action 是末端局部相对位姿/夹爪并含 waist/base 槽；YAM joint-space 14D 未投影，未运行训练或 GPU 验收。资产摘要和实测边界见 [环境报告](../reports/environments/xr1-20260924/README.md)，操作入口见 [02](../02_installation_and_environment.md#xr-1-服务器环境-2026-09-24) 与 [10](../10_vla_platform.md)。

- YAM 已实现路线为 Pi0.5 全量微调 / `pi05_yam`。2026-09-16用户暂停训练，完整基线100000；后确认现场16 mm小颗粒与ABC不同。14:51夹爪已能闭合；只读控制重放发现夹爪旧预测/关节融合与滤波响应需验收。现场计划先验控制，再采50条train+10条val，以100000权重另开SFT；用户随后提出采集窗口训练Evo，建议D1小预算两阶段与采集并行，四卡训练按时段串行。后续最新决定改为从官方Pi0.5 base随机10h训练RTC+全量微调，已接代码与子集norm；18:15获用户授权后Slurm2140/gpu001四卡已运行，随机468条10.020667h、记录至21次更新有限，首个保存/真实恢复待验收。接管见[运行报告](../reports/training/rtc-base-10h-20260916/README.md)；训练设计见[RTC方案](../03_training_and_evaluation.md#2026-09-16--pi05-training-time-rtc基础权重与随机10小时)。状态、预算和证据归 [训练](../03_training_and_evaluation.md#2026-09-16--训练重设计先复现乐高分拣再比较模型)，Evo版本/计数陷阱归 [10](../10_vla_platform.md#四卡训练落地前的固定版本检查)；不自动回退LoRA。
- YAM 为 14D `[左6关节, 左夹爪, 右6关节, 右夹爪]`；Pi 路径关节 delta、夹爪 absolute，mask `(6,-1,6,-1)`，内部 32D、horizon 50；这些模型内部规则不套用其他系列。物理单位仍须数据审计，不套用 OpenArm/Piper 合同，见 [数据合同](../04_data_contracts.md)。
- Pi 当前训练产物为 JAX/Flax；新模型保留原生 LeRobot/PyTorch 格式。既有 Thor Pi 部署按系列隔离容器，新接入层环境采用独立 Conda；3588 采集/控制，两 IPC 网线直连，本项目不操作 3588。转换与量化须独立验收，既有部署事实见 [Thor 部署](../08_thor_edge_deployment.md)，新架构见 [01](../01_system_architecture.md)。
- 主检出为 `/home/wuyan-lyj/condapi`，改造工作树以实际 cwd 为准，外部 YAM 参考只读；设备、功能分支提交、数据保护和 RTC 回退边界由 `AGENTS.md` 持有。
- **Thor Pi 当前状态（2026-09-23 15:22 CST复核）**：`pi05-rtc-infer` 在固定 `ws://192.168.250.1:8000` 运行20h136000训练型RTC，MAXN临时会话active；七步FP32权重＋TF32 TensorRT、quantile动作前缀、时间缓存/CUDA Graph、0.2rad/tick关节跳变拒绝。18项源SHA、无NaN/Inf、811张量转换、九例JAX/TRT同输入及上线后九例本机WebSocket通过，有限50×14绝对动作且前缀保持；最大关节差0.000747rad，上线后服务推理P50/P95约197.87/207.56ms，本机协议往返198.73/208.77ms。**未做3588端到端smoke或真机任务验收**。容器`--restart no`，停服后自动回120W，非开机常驻。旧124000大权重/引擎已删除，不能就地回退；OpenWAM试验暂停、已停止容器清理但原始权重/镜像保留。完整身份、合同来源和回执见[136000报告](../reports/thor/rtc-20h-136000-20260923/README.md)；前缀归一化事故仍见[事故报告](../reports/thor/rtc-prefix-quantile-incident-20260917.md)。
- **136000 真机安全待核（2026-09-23 15:28 CST）**：用户反馈左臂行为异常，暂不应继续驱动真机；同九例原生 JAX 124000→136000 左臂目标最大变化0.04418rad，约为136000自身TRT→JAX最大差0.000747rad的59倍。离线转换误差小不代表新checkpoint的现场行为可用；尚无故障请求三相机/状态/prompt/RTC前缀可重放，需对同一故障输入做 JAX/TRT 对照，见[136000报告](../reports/thor/rtc-20h-136000-20260923/README.md#1528-cst-用户现场反馈左臂行为异常待定位)。

## 恢复任务

- 100000 W/80 的既有 `pi05-infer` 曾在固定 `ws://192.168.250.1:8000` 完成本机smoke，MAXN单次约106ms；2026-09-17 已停止。RTC30000旧七步服务随后因前缀归一化事故停止；2026-09-18按用户要求用**新版**分位数前缀服务恢复固定8000，本机协议通过，跨3588/真机待验。推理/测试阶段MAXN、容器停止后恢复日常120W；当前运行状态使用前仍须重查。详见[事故报告](../reports/thor/rtc-prefix-quantile-incident-20260917.md#2026-09-18-更新新版服务与-thor-本机协议测试)，100000历史转换归[交接](../reference/thor/12_checkpoint_handoff.md)。

- 2026-09-09 用户授权本轮工作树合并，保留 main 的 Pi 全量微调与 5k 续训修复；服务器故障期间不部署。严禁本地训练循环（含 CPU/debug）；只允许轻量配置/协议和看板验证。合并流程归 [10](../10_vla_platform.md)，多模型指标协议归 [11](../11_training_dashboard.md)，操作约束归 `AGENTS.md`。记忆只按当前任务读取摘要和相关章节，必要时展开原文；无默认累计额度，完整证据和待办继续保留，见 [09](../09_memory_system.md)。
- 2026-09-08 Evo-1和MolmoAct2本地/服务器独立Conda环境均安装并通过CPU检查，两端分别106/108个包版本一致；MolmoAct2已加入同一LeRobot共享后端。普通MolmoAct2不含Think，LoRA与FP32动作专家参数的精度规则不套Pi结论；尚未验收真实YAM模型/GPU训练。环境与动态库配置见 [02](../02_installation_and_environment.md)，实测见 [Evo报告](../reports/environments/evo1-20260908/README.md) / [Molmo报告](../reports/environments/molmoact2-20260908/README.md)，模型特有事项见 [Molmo接入](../reference/molmoact2_integration.md)。

- 服务器入口、环境、原始数据路径和作业保护对象见 [环境](../02_installation_and_environment.md)。作业/下载状态是历史观察，使用前重新核验；没有实时证据就标记未知。
- Thor 已完成官方系统安装并从 NVMe 启动；2026-09-07 实机复核为 JetPack 7.2.1 / L4T 39.2.1。2026-09-23 工作站与 Thor 虽连同名`琶洲模方`，原先落在不同 AP/网段；Thor 新增工作站 AP 优先连接，当前 DHCP `10.18.10.89/23`、本机 SSH/NoMachine 端口可达，**非静态 IP、重启回连未验**。旧5 GHz/2.4 GHz配置保留回退；Thor↔3588直连当前 `192.168.250.1↔192.168.250.2`，连通实测通过；3588的Wi-Fi `wlan0` 当前 DHCP `192.168.110.55/23`，与Thor无线不在同一网段，但生产链路不依赖两者Wi-Fi互通。USB 管理口 `192.168.55.1`。HDMI 仍无显示，不能误记为安装未完成；详见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)。
- Pi0.5 原始 JAX 基础权重、分词器和 3 条完整 YAM 回放轨迹均在 Thor 本地。2026-09-08 已完成 18 组配置、3240 次正式调用。当前固定性能候选 W：非量化 TensorRT BF16/FP32 + 原 FP32 时间条件缓存 + CUDA Graph + masked 文本桶 80，P50 104.25 ms / P95 104.94 ms；对 JAX A 动作 MAE 0.002504、最大差 0.017282（数据集单位），不是任务精度保证。tokenizer 接口仍 200，保留全部真实 token、三相机/H50/去噪 10；长有效输入拒绝，显式使用原 V 的 200 桶（108.85 ms），自动路由未实现。FP32 填充诊断原严格门槛仅 8/9；同桶 wrapper 与旧采样器、同引擎图/非图分别零差异，不表示对原 JAX 零差异。保留 JAX A、V/I；当前约 100ms，严格 ≤100ms 未达到。基础模型无 LoRA，未来微调需独立合并/任务验收；结束恢复 120W。固定镜像/引擎/复测命令与证据见 [候选方案](../reference/thor/11_pi05_candidate_test_plan.md)、[执行记录](../reference/thor/10_acceleration_execution.md) 与 [中文报告](../reports/thor/index.html)。
- Thor 已安装 NoMachine 9.8.3 并配置自启动，无屏 GNOME / 1920×1080 输出已核对；USB 连接 `192.168.55.1:4000`，账户 `wuyan-lyj`。为此停用 GDM 本地登录画面；恢复方法及许可边界见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)，客户端画面/重启回连不冒充已验收。
- 用户指定 **仅推理/测试期间**使用 MAXN（模式 0）+ 最高 CPU/GPU/EMC 频率；结束后恢复日常 120W、动态调频和自动风扇，不设 MAXN/锁频开机常驻。入口 `scripts/thor/maxn_session.py` 负责临时切换与退出恢复；每次核对温度/降频，见 [Thor 当前状态](../08_thor_edge_deployment.md#7-当前状态)。
- 新结果写回对应 owner，原因写入 [变更历史](../07_change_log.md)；候选经验、失效与检索协议见 [记忆系统](../09_memory_system.md)。
- 2026-09-11 已在Thor复现Dexmal realtime-vla原版与独立修正版；本轮未优于既有W，保留W候选，不改变全量微调路线。有效/无效运行、精度与延迟边界见 [Thor执行记录](../reference/thor/10_acceleration_execution.md)，可视报告见 [对照页](../reports/thor/realtime_vla.html)。

- 工具复用、失败尝试和配置/实测差异先查 [问题路由](context_index.md#问题与行动路由)，只展开所需检查与重试条件；记录规范归 [09](../09_memory_system.md#工程经验的增量整理)。训练根因和未做的任务验收保持未知，不以历史启动或配置值替代实测。
