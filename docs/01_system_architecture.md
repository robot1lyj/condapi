# 01 · 系统架构

## 多模型接入层

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

目录职责：`configs/models/` 选择模型及后端，`configs/experiments/` 组合实验，`configs/environments/` 选择系列 Conda prefix，`configs/robots/` 持有机器人合同投影。`adapters/` 按后端组织代码，`packages/vla-platform/` 只做跨进程编排；已有 `src/openpi/` 和 `scripts/train.py` 是 Pi 原生实现，`scripts/thor/` 继续承载现有部署/评测，`docs/reports/thor/` 保存证据。不为凑目录树新建空的 deployment/evaluation 层，也不移动现有实机脚本导致远端路径失效。

原型 `plugins/<family>/` 已撤销，项目配置升级为 schema 2；实验、环境和模型包用 `model` 字段替代 `plugin`，不提供旧控制层格式兼容。原始模型权重、历史报告、运行记录不迁移或改写。机器资源继续由 [02](02_installation_and_environment.md) 统一记录，当前执行器不自动 SSH 或复制服务器配置到各模型代码中。

2026-09-08 目录清理：删除非 YAM 的 `examples/` 示例以及未初始化的 ALOHA/LIBERO `third_party/` 子模块和配置，移除空 `plugins/` 目录。唯一仍用于部署的 JAX→PyTorch 转换器迁至 `adapters/openpi/convert_jax_model_to_pytorch.py`，Docker 和回归入口随之更新。`src/` 是实际 Pi 后端，`packages/` 是控制层/客户端，`artifacts/` 和 `docs/reports/` 是历史证据，`skills/` 是记忆工具源码，均不因目录数量多而删除。旧示例可由 Git 历史或原上游获取；历史文档提及的示例路径不代表当前可运行入口。

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
- `pi05_yam_lora`：当前首选低显存配置；对应 `gemma_2b_lora` 和 `gemma_300m_lora`，关闭 EMA。

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
