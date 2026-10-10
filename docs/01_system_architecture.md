# 01 · 系统架构

## 并行 RL 实验的所有权（2026-10-10）

用户明确多方案并行，本对话负责此前讨论的 PART 整轮残差专家；EXPO-FT 研究由另一条路线继续，不作为全仓唯一算法。实现隔离到独立工作树/功能分支，保留原生 Pi 与 RTC 服务。PART 当前恢复范围与未完成项归[03](03_training_and_evaluation.md#多方案并行与本对话-part-分工2026-10-10-用户更正)。不能把某个线程的方案切换解释为删除其他实验的授权。

## EXPO-FT 强化学习架构（2026-10-10，设计草案）

该并行研究路线的目标是乐高按颜色分拣与蓝牙耳机入充电盒的 RL 微调。**此前该线程退役旧实现的记录保留；本 PART 分支恢复的整轮 residual-rl 不受本节替代。新 EXPO-FT 学习器尚未实现、未训练、未部署。** 用户授权本仓库清理与设计；YAM 为只读参考，客户端方案尚未发送任务。普通 Pi/RTC、数据、权重及历史证据保留。详见 [论文与代码审读](reports/rl/expo-ft-20261010/README.md)、[训练设计](03_training_and_evaluation.md#expo-ft-学习与评估设计2026-10-10)、[数据核验](04_data_contracts.md#expo-ft-双任务数据合同2026-10-10)、[协议设计](05_inference_and_rollout.md#expo-ft-三方接口草案2026-10-10)。

[打开交互架构图](diagrams/expo-ft-architecture.html) · [编辑图源](diagrams/expo-ft-architecture.json) · [验收记录](diagrams/README.md#expo-ft-强化学习架构)

### 决策：一个学习闭环，三个运行位置

| 位置 / owner | 负责 | 不承担 |
|---|---|---|
| GPU 服务器 / condapi | 数据接收审计、Replay、EXPO-FT 更新、评估、原生 checkpoint、联合策略发布 | 逐 tick 控制、相机采集、机械臂 reset |
| Thor / condapi Pi 系列容器 | Pi 候选生成、轻量视觉编码器、Edit actor、Q 选择、模型版本装载、协议响应 | 梯度更新、SDK/CAN 写入、成功事实裁决 |
| YAM 客户端 / 外部仓库 | 三相机与 state 对齐、RTC 时间轴、唯一动作仲裁、人工接管、实际执行记录、终点标注、outbox | VLA/critic 训练、替服务器认定 training_ready |

上游把训练和推理放在 learner 侧、DROID 放在 actor 侧；我们保留算法所有权，**将上游推理职责移至 Thor**。远程服务器网络不进入30 Hz控制关键路径。图中“持久上传与接收”是逻辑传输通道：发送端在YAM、落盘接收端在服务器；无需在Thor转存全量视频。图中版本库发布整束至Thor，整束包含Pi、视觉编码器、Edit与Q，图内省略它们之间重复的装载连线。

初期采用**episode/batch 间更新 + 固定行为版本采集**，先排除在线换权重的混杂；待端侧时延和恢复验证后再研究低滞后的异步发布。首版不要求GPU服务器与机器人保持每步同步，也不为提高UTD阻塞现场控制。

### 算法组成与保留的原生实现

1. 复用作者 EXPO-FT 的 learner / Edit / Q / replay 抽样逻辑，固定源码后维护小范围 YAM 适配；VLA loss、参数树、优化器与 checkpoint 继续由 OpenPI 系列实现拥有。控制层只负责文件、身份、校验和子进程，不能导入JAX/Torch。
2. 每任务各有 `Pi + Edit + Q + critic encoder + norm + task/reward contract` 联合策略。Pi给出N个H50动作块，Edit/Q处理将实际执行的C步；Q同时比较基础和编辑候选。学习器在同一闭环内用执行经验更新VLA原生流匹配目标，不恢复“永远冻结Pi、等小专家成熟再另训”的旧路线。
3. 使用三路RGB的独立轻量视觉编码器，首选对照作者ResNet-50结构；不继续依赖冻结Pi缓存token、均值池化或旧RLT重建器。三视图共享/独立权重、输入拼接方式必须与固定版本训练/推理一致，端侧性能以实测为准。
4. 使用统一双臂14D动作与任务级价值。默认建模两臂和夹爪协同；不再将“单活动臂六关节 + 下探抓取资格”写死为RL定义。每维可编辑mask和物理边界是任务合同，允许首轮锁定夹爪等消融，但须显式区别于完整算法。
5. 标准EXPO-FT作为数值/离线对照；生产设计面向作者 **Real-Time EXPO-FT** 的慢候选、最新观测快编辑及延迟一致备份。它是2026-09的新论文，不能当成5月原文已有能力，也不能把原RTC加残差直接叫作已复现实时算法。

### 拟建模块（本轮只有设计，以下路径尚未创建）

| 拟建位置 | 边界与接口 |
|---|---|
| `adapters/expo_ft/` | 薄编排、指定Conda prefix、调用固定上游入口、适配 `LeRobotYamDataConfig/YamInputs/YamOutputs`；不复制第二套Pi训练循环 |
| 固定上游源码目录 | EXPO learner、real-time learner、Q/Edit/encoder；小范围补丁显式记录，不整包替换现有 `src/openpi` |
| `packages/vla-platform/.../rl_contracts.py` | 标准库 schema、task/行为包身份、manifest/上传完整性、拒绝未知单位 |
| `scripts/thor/` 下独立EXPO服务入口 | Pi系列同一容器；完整算法包加载和同版推理；旧普通/RTC入口可独立启动 |
| 服务端 ingest / replay assembler | 新目录接收、sha和episode闭合、标签审查、双时刻/实际执行拼接；不覆写原始数据 |
| YAM `hil/rl/`（客户端提案） | 任务会话、协议适配、journal/outbox、奖励标注；通过现有仲裁接口集成 |

环境规划：服务器新增独立的Pi-RL Conda prefix，依赖按固定EXPO/OpenPI兼容性审计确定，**不照搬上游uv、不修改正在训练的Pi环境**。Thor沿用Pi系列容器规划，具体JAX或PyTorch/ONNX执行路线须对齐数值后决定。跨框架转换若尚不可用，属于实现缺口，不以永久冻结VLA掩盖。新学习器恢复需带optimizer、target Q、target Pi、温度、随机状态、Replay水位和采样状态；推理包仅导出需要的组件。

### 实时设计中的关键适配

- 慢Pi使用采样时观测 `o_s` 与已承诺prefix；快速Edit/Q使用最新观测 `o_e`。两者都要记录，不能只在原请求上生成整块残差后称为实时反馈。
- `H=50`为Pi预测长度，`C`为执行/critic窗口，`d`为推理消耗的tick；三者分开。作者实现要求 `0≤d≤C` 且 `2C≤H`；本项目不能从现有d=10直接套用论文C=4/8。选C需先测双臂、三图、多候选的p99时延，窗口覆盖须满足 `d+C≤50`。
- YAM模型的关节delta相对于**该次观测state**。旧观测生成的归一化候选必须用旧state解回物理absolute，再以快速观测state重编码供Edit/Q；回包只给物理absolute14D。跨时刻直接复用归一化残差会产生参考点漂移。已承诺动作以物理值保存，不随norm或Pi版本重解码。
- 现场快路径含相机对齐、编码传输、Thor轻量编码/Edit/Q和回包；不能把“小网络”直接等同于33 ms以内。过期候选、超时、epoch变化不能执行；可用且同epoch的基础动作按既有合同降级，队列耗尽则由YAM HOLD。记录真实延迟及降级原因。
- RTC训练的时间变量约定不同：论文式子用clean prefix时间1，本项目/OpenPI实现以时间0表示clean；这是参数化差异，不能机械复制公式数值。需核对loss、inpainting和回放，而不是改现有RTC前缀语义。

### 分期落地与完成条件

| 阶段 | 交付物 / 退出条件 |
|---|---|
| 0（本轮） | 论文下载、源码审读、旧RL退役、双任务文件清单核验、服务端/客户端设计与架构图 |
| 1 | 新合同和assembler的纯数据验证；成功/失败/人工/缺帧/重传/换版本回放，不执行训练 |
| 2 | 获准GPU上的EXPO更新、保存恢复、YAM动作往返、同一数据的上游数值对照；具体训练配方另定 |
| 3 | Thor base-only→shadow，验证候选/Edit/Q同版装载、精度与延迟；YAM客户端改造先经用户审核 |
| 4 | 乐高任务固定版本小批实验，再耳机任务；成功率、人工率、时长与干预分开评价 |
| 5 | 同合同比较标准EXPO、实时EXPO、关Edit/关在线VLA更新，决定发布频率与共享模型策略 |

不得将架构图、文件齐全、端口监听、静态测试称为训练或真机就绪。客户端具体改造清单见 [待审核方案](reference/expo_ft_client_proposal.md)，本轮未发送。旧代码删除范围及可恢复基线见 [退役清单](reports/rl/expo-ft-20261010/retirement.json)。

交互架构图：[Archify 总览](diagrams/architecture.html) · [可编辑图源](diagrams/architecture.json) · [生成与验收记录](diagrams/README.md)。图示区分已实现入口、待验收模型和仓库外控制侧；不代表远端实时运行状态。

## 多模型接入层

旧PARTS/RLT服务端路线已于2026-10-10退役；现阶段RL能力为本页EXPO-FT设计草案，不能视为已有可运行后端。历史方案保留在03/04/05对应章节。

模型无关遥测使用标准库 `vla_platform.metrics.write_metrics`，训练器只需输出标量事件。LeRobot 共享入口采集原生结构化 tracker；Pi 保持原生日志；独立只读看板统一消费 JSONL/CSV/Trainer-state，不导入模型环境。指标合同、配置与局限归 [11](11_training_dashboard.md)。
2026-09-08 工作树改造：采用 **LeRobot 原生能力 + 薄接入层**，不是自研训练框架。LeRobot 负责其支持模型的网络、loss、数据集、处理器、优化器和 checkpoint；原版 OpenPI 是 Pi 的独立实现后端。RLinf 仅作为未来有具体 DAgger/RL 需求时的可选后端，不作为所有模型的强制依赖。

```text
scripts/vla.py / 安装后的 vla
  -> configs：模型选择、实验、Conda prefix、机器人合同
  -> packages/vla-platform：命令规划、运行记录、模型包完整性
  -> adapters/openpi 或 adapters/lerobot：按实现后端复用入口
  -> 独立 Conda 子进程：LeRobot 或 OpenPI
  -> checkpoint + 保存的 processors + 本模型 norm + 合同 + 参考样例
  -> Thor 模型侧适配与数值/性能验收
```

代码边界：控制层只用标准库；模型依赖留在系列环境；数据和权重在外部目录，不复制进 Git。跨环境使用文件/进程合同，不共享 Python 模型对象。`configs/robots/yam.toml` 是 [04 数据合同](04_data_contracts.md) 的机器可读投影，修改语义时必须同步 owner，不从 TOML 猜测未审计单位。

当前 Pi 模型声明选择 OpenPI 后端，连接现有 JAX 训练和 JAX/PyTorch 离线推理、ONNX 导出及回放入口。并未把 TensorRT 常驻服务迁入此控制层，也没有新建网络服务。请求中的图像路径只适用于本地离线调用，不是 Thor↔3588 的图像传输协议；生产协议仍归 [05](05_inference_and_rollout.md)。

Evo-1、FastWAM、VLA-JEPA 的模型声明共用 LeRobot 后端，不再为每个系列建一个 Python 插件目录。模型声明只选择 backend、policy_type、已接通的操作和机器人合同，不允许自行定义训练循环。LeRobot 原生 JSON 配置直接传给上游；本地不重写 trainer、processor 或 checkpoint 格式。共享入口存在与具体模型可用是两件事，三个新模型仍为 planned；具体状态、缺口和操作归 [10](10_vla_platform.md)。

目录职责：`configs/models/` 选择模型及后端，`configs/datasets/` 声明数据语义与来源，`configs/splits/` 保存分组分割配方，`configs/algorithms/` 声明原生算法和允许覆盖的参数，`configs/experiments/` 组合实验，`configs/environments/` 选择系列 Conda prefix，`configs/robots/` 持有机器人合同投影。`adapters/` 按后端组织代码，`packages/vla-platform/` 只做跨进程编排；已有 `src/openpi/` 和 `scripts/train.py` 是 Pi 原生实现，`scripts/thor/` 继续承载现有部署/评测，`docs/reports/thor/` 保存证据。

2026-09-24 控制层 0.2 增加可组合配置核心：`inventory.py` 从审核后的 episode/采集组/源文件清单生成源哈希，`splits.py` 以固定 seed 按组划分并发布只读成员清单，`composition.py` 交叉核对模型 IO、数据动作空间、相机顺序、算法后端、机器人合同及训练 split/推理模型包。`project.py` 将实验 schema 2 编译为既有 `Plan`，运行前再核对组件/源文件 SHA 和 split 或模型包；schema 1 旧实验继续可用。所有组件是 TOML/JSON 文件，解析器只用 Python 标准库；模型、数据读取和优化仍留在 OpenPI、LeRobot、OpenWAM、XR-1 各自环境。新 split 只记录 episode 成员，不移动或修改数据。实现和命令归 [10](10_vla_platform.md#模块化配置后端-v02)。

```text
源数据（只读） → episode inventory（文件哈希） → 分组 split（不可覆盖）
                                            ↓
模型声明 + 数据声明 + 算法声明 + 环境 + 机器人合同
                    ↓ 类型/能力/语义检查
             Plan（命令、源哈希、组件身份）
                    ↓ 运行前复核
        独立 Conda 子进程中的原生训练器 / 推理器
                    ↓
           原生 checkpoint + 模型包完整性/验收
```

首版是本地可复用核心和 CLI，不提供多用户服务 API。后续 API 应调用同一组合、分割和执行边界，另加身份、租户隔离、作业队列与持久状态；不能把 HTTP handler 直接放进模型进程，也不能绕过 `Plan`、数据 split 或模型包校验。服务设计不是当前已实现能力。

原型 `plugins/<family>/` 已撤销，项目配置升级为 schema 2；实验、环境和模型包用 `model` 字段替代 `plugin`，不提供旧控制层格式兼容。原始模型权重、历史报告、运行记录不迁移或改写。机器资源继续由 [02](02_installation_and_environment.md) 统一记录，当前执行器不自动 SSH 或复制服务器配置到各模型代码中。

2026-09-08 目录清理：删除非 YAM 的 `examples/` 示例以及未初始化的 ALOHA/LIBERO `third_party/` 子模块和配置，移除空 `plugins/` 目录。唯一仍用于部署的 JAX→PyTorch 转换器迁至 `adapters/openpi/convert_jax_model_to_pytorch.py`，Docker 和回归入口随之更新。`src/` 是实际 Pi 后端，`packages/` 是控制层/客户端，`artifacts/` 和 `docs/reports/` 是历史证据，`skills/` 是记忆工具源码，均不因目录数量多而删除。旧示例可由 Git 历史或原上游获取；历史文档提及的示例路径不代表当前可运行入口。

2026-09-22 新增 `adapters/openwam/` 独立后端，模型选择为 `openwam`。固定上游运行时快照在 `third_party/openwam/`（逐文件哈希归 `UPSTREAM.json`）；复用 OpenWAM 原生模型、Hydra/DeepSpeed trainer、归一化与 checkpoint，不经 LeRobot trainer，也不复制训练循环。YAM reader 只负责 LeRobot v2/v3 数据投影，微调/恢复入口和离线推理由独立 Conda 子进程承载。控制层仍无模型依赖；接口实现不表示 GPU 或 Thor 已验收，操作边界见 [10](10_vla_platform.md#2026-09-22--openwam-微调接入)。

2026-09-24 新增 `adapters/xr1/` 独立后端和 `third_party/xr1/` 固定上游运行时快照。控制层仅计划并调用独立 Conda 子进程；XR-1 保留其 Hydra/Lightning/DeepSpeed trainer 与 60D 原生动作。`configs/models/xr1-5b.toml` 当前只声明训练能力；YAM 的 14D joint 动作须先由可信 FK 产生末端目标标签、按 train split 重算统计，推理还需独立 IK/控制时序验收。兼容门槛见 [10 的 XR-1 入口](10_vla_platform.md#2026-09-24--xr-1-原生训练入口)，模块化组合见 [同页 v0.2](10_vla_platform.md#模块化配置后端-v02)。

以下数据流和 32D/H50、delta 配置描述的是 **Pi 后端**，不是对所有模型的统一要求。

## 当前训练数据流

```text
YAM LeRobot v3 数据
  -> LeRobotDataset / metadata task 映射
  -> YamInputs：三路 RGB + 14D state/action -> OpenPI 标准键
  -> DeltaActions：每臂 6 个关节转相对当前 state，夹爪保持 absolute
  -> norm stats -> Pi0/Pi0.5 模型输入
  -> 模型内部 32D action、50 步 horizon
  -> YamOutputs + AbsoluteActions -> YAM 14D action chunk
  -> checkpoint / norm assets
  -> 原 JAX policy golden + LoRA/精度审计
  -> Thor Pi 系列容器：原生 JAX 可行性验证，或经 FP32 转换审计的未量化后端
  -> 需要加速时独立评估 TensorRT/FlashRT；量化须额外通过 gate
```

本仓库只负责训练数据、模型 transform、Thor 端侧模型部署和训练后 policy 输出；生产系统由两台 IPC 组成：Thor 负责本地模型推理，3588 负责相机信号采集、机械臂驱动、CAN、GUI、home pose 和控制频率。两者通过网线直连交换 observation/action；本仓库不读取、修改或同步 3588 的代码。

## 代码模块

| 层 | 主要位置 | 责任 |
|---|---|---|
| 模型 | `src/openpi/models/`、`src/openpi/models_pytorch/` | Pi0、Pi0.5、视觉/语言 backbone 和 action sampling |
| 配置/训练 | `src/openpi/training/`、`scripts/train.py`、`scripts/train_pytorch.py` | `TrainConfig`、数据 loader、优化器、checkpoint |
| 数据变换 | `src/openpi/transforms.py`、`src/openpi/training/config.py` | YAM 输入、delta action、prompt、norm 和模型输入 |
| YAM policy | `src/openpi/policies/yam_policy.py` | 校验 14D 合同、映射三路图像、裁掉模型 32D padding |
| 部署 | `docs/08_thor_edge_deployment.md`、Thor 容器/engine | 训练后 policy 的端侧转换、加速和本地推理 |
| IPC 直连服务 | `src/openpi/serving/`、`scripts/serve_policy.py` | Thor 上加载 checkpoint，通过直连以太网向 3588 提供 observation/action；这是两 IPC 的数据通道，不是远程模型推理 |
| 客户端 | `packages/openpi-client/` | 通用请求/响应协议；不实现 YAM 机械臂控制 |
| 数据工具 | `scripts/compute_norm_stats.py`、审计脚本 | norm、metadata、视频和 loader 预检 |

## YAM 配置路径

YAM 配置集中在 `src/openpi/training/config.py`：

- `LeRobotYamDataConfig`：默认双臂、14D、`assets/yam`，动作序列键为单数 `action`。
- `YamInputs`：接收 `observation.images.top_rgb`、`left_rgb`、`right_rgb`、`observation.state`、`action` 和 prompt。
- `YamOutputs`：把模型至少 32D 的 action chunk 裁回 14D；不足 14D 直接报错。
- `pi05_yam`：当前默认全量微调配置，正式入口为 `scripts/train_lego_full.py`；`pi05_yam_lora` 仅保留为显式可选配置，不是默认路线。

YAM policy 不复用 OpenArm 或 Piper 的 transform。单臂 7D 只作为配置类的显式兼容选项，当前项目默认始终是双臂 14D。

## 数据加载兼容

当前仓库兼容 LeRobot 新旧 import 路径：优先使用 `lerobot.datasets.lerobot_dataset`，旧版本才回退到 `lerobot.common.datasets.lerobot_dataset`。LeRobot v3 的 `meta.tasks` 可能是 DataFrame，loader 会先归一化为 `task_index -> prompt` 映射。

LeRobot 原始键直接在 YAM policy boundary 处理，因此本仓库没有把独立 YAM-ABC 项目的数据转换器或机械臂控制层复制进来。若实际数据不是上述键，先写独立转换/审计产物，再接入训练配置。

## 模型与机器人边界

- OpenPI 模型统一需要标准 `image/state/actions` 结构和 32D padding；YAM 的真实合同只在 `YamInputs/YamOutputs` 边界出现。
- norm stats 在训练输入经 delta transform 后计算，checkpoint 内保存到 `assets/yam/norm_stats.json`；不能跨单位或跨数据版本复用。
- Thor 部署保留 JAX checkpoint 作为数值基线，转换后的 PyTorch/TensorRT 产物必须在同一 YAM 样本上比较；官方 `pi05_libero` 的 10 步/7D benchmark 不能替代 YAM 的 50 步/14D gate。
- 端侧 policy 返回 50 步、14D YAM 动作；任何单位转换、限幅、执行和安全检查必须由已核实的机器人侧系统负责。

## 历史边界

OpenArm 16D、Piper transform、KAI0/Evo-RL/HIL 方案只在变更记录和历史归档中保留背景；对应的当前专用代码已从默认训练树清理。它们不能改变当前 YAM 默认配置，也不能把 OpenArm 的单位或动作顺序套到 YAM。
