# 10 · 多模型接入层操作

## 当前状态

观察日期：2026-09-09。本轮按用户授权将多模型工作树与 main 的 Pi 全量训练/续训修复整合，主检出通过快进接收合并结果；不部署服务器或 Thor。实际提交/远端同步以 Git 为准。架构 owner 为 [01](01_system_architecture.md#多模型接入层)，设备信息归 [02](02_installation_and_environment.md)，模型精度与 Thor 验收归 [08](08_thor_edge_deployment.md)，模型无关训练看板归 [11](11_training_dashboard.md)。

| 部分 | 状态 |
|---|---|
| 控制层 | 标准库实现；模型与后端分离、计划、Conda 子进程、运行记录、模型包哈希与离线 IO 检查 |
| Pi | 现有 OpenPI 训练/参考推理/回放/ONNX 导出入口已接线；新环境 GPU 执行未验证 |
| LeRobot 后端 | 共用原生训练 launcher 已实现并用替身测试；无自建 trainer/processor；尚无通用离线推理入口 |
| Evo-1 | 本地/服务器专用环境已安装并通过CPU检查；YAM真实训练/推理仍待接入 |
| MolmoAct2 | 原生LeRobot共享后端已注册；两端独立环境CPU检查通过、108包版本一致；仅普通版，不含Think，真实YAM/GPU仍待验收 |
| FastWAM、VLA-JEPA | 注册 planned；不得运行或报告已支持 |
| Conda | Evo-1已有独立环境规格和104个wheel的锁；Pi等bootstrap仍不代表模型环境已安装 |
| Thor | 原 Pi 容器、TensorRT 引擎、报告保持原状；没有部署此次改造 |

当前 CPU 回归：平台测试52项；与 `scripts/thor`、`skills/mlops-memory/tests` 和下载完整性测试合跑170项通过。测试命令为 `python -m pytest -q packages/vla-platform/tests scripts/thor skills/mlops-memory/tests scripts/conda/fetch_locked_wheels_test.py`。框架测试替身不冒充模型运行；独立Evo环境实际导入/processor检查见 [环境证据](reports/environments/evo1-20260908/README.md)，MolmoAct2的原生精度/数据差异见 [接入说明](reference/molmoact2_integration.md)。

不复制 LeRobot 的 registry、trainer、processor 或 dataset 实现；`configs/models/*.toml` 只选后端、policy_type 和已接通能力，入口集中在 `adapters/<backend>/backend.toml`。具体模型在子进程中调用上游；Pi 调用既有 OpenPI。原作者代码用于对照，不强制每个模型维护双实现。RLinf 的 DAgger/RL 接入不是当前范围。

## 工作树与 Git

主目录 `/home/wuyan-lyj/condapi` 的 main 与当前工作树分支是独立检出。用 `pwd`、`git branch --show-current`、`git worktree list` 确认当前目录。代码和版本化记忆随分支变化；未合并前主目录不会看到这里的更新。

```bash
git status --short
git diff
python3 scripts/vla.py models
```

默认只提交/备份当前分支；明确授权合并时，先把最新 main 合入功能分支解决冲突并检查，再将干净的主检出 fast-forward 到验收提交，向两个已配置远端推送，不能 force push。不自动同步 Thor/服务器。工作树合并不是搬文件或删除目录，分支和工作树可保留。共享权重、Conda prefix、外部数据并不受工作树隔离保护。

当前 Pi 正式全量入口仍为 `scripts/launch_lego_full.sh` → `scripts/train_lego_full.py`，每 5k 保存与 committed checkpoint 续训保护保留。`configs/experiments/pi-train.toml` 是通用全量配置示例，不替代已有 Lego 正式启动参数。服务器故障未解决，本地禁止运行训练循环，包括 debug/CPU smoke；仅做静态、配置和看板验证。多模型指标接入不提升尚未 GPU 验收的模型状态。

文件记忆读取当前工作树版本，只按当前任务展开摘要与相关章节；检索与旧账本迁移规则归 [记忆系统](09_memory_system.md)。可选 `docs/cache/runtime/` 账本保留检索计量与去重历史，无默认累计额度，不把会话 ID 固化为长期操作步骤，也不将账本字节当作当前上下文占用。

## 命令与环境

所有命令在工作树根目录运行。控制层 Python 3.11+，模型环境独立。可选安装：`python -m pip install --no-deps -e packages/vla-platform`，随后使用 `vla`；不需要安装根目录 OpenPI 大依赖。

```bash
python3 scripts/vla.py models
python3 scripts/vla.py backends
python3 scripts/vla.py plan configs/experiments/pi-reference.toml infer --run-id pi-plan-001
python3 scripts/vla.py env plan configs/environments/evo1-workstation.toml
```

`plan` 只输出命令，不下载/安装/运行模型。先替换实验配置中的 `/path/to`、模型版本和 action_dt_s；0.1 只是示例，不是对 YAM 频率的确认。

`env create` 显式创建不存在的 prefix，不更新现有环境；`env audit` 只列包，不证明 CUDA 可用。现有 `condapi-yam` 服务器 prefix 仅引用，不自动迁移。真正的模型环境还需锁定源码、Torch/CUDA/Transformers 和处理器依赖，x86 服务器与 ARM Thor 分别核对。

```bash
python3 scripts/vla.py run configs/experiments/pi-reference.toml infer --run-id pi-infer-001
```

`run` 是本机执行，不会自动 SSH。服务器训练在已审计计算节点的 Slurm allocation/tmux 内启动，框架不是 Slurm 提交器。Thor profile 必须在实机运行；infer/benchmark 包装现有 MAXN session，退出恢复 120W；不设置开机 MAXN。Conda 缺失或 profile 未安装时直接失败。

输出位于 `runs/platform/<run-id>/`：plan、started、console.log、finished 以及模型产物。目录不可覆盖；成功退出标为 `command_succeeded_not_model_accepted`。运行元数据是命令与源码追溯，并非完整模型验收清单，数据/训练参数指纹仍由模型侧产物补齐。

Pi `infer` 是一次本地请求的参考路径，图像为本机 RGB 文件，示例见 `configs/requests/pi-example.json`。每次新进程，不实现常驻缓存或网络服务。时间包含首调用编译，不能拿它与旧 TensorRT 稳态 104ms 比较。未合并 LoRA 的 PyTorch 推理会拒绝；JAX 保持 checkpoint 原始加载精度。

## 模型包

模型包保留原格式，不强制转换成 JAX。recipe JSON 必须包含 schema_version=1、model、model_version、code_revision、contract_id、format、precision.storage/compute，以及 files 列表（role/path）。必须包含 weights、model_config、preprocessing、normalization、contract、reference 六类文件；contract 为本版本 TOML。目录权重要逐个列文件，不能只记录目录名。LoRA adapter 还需 base_model_sha256。旧原型 plugin 字段不再接受；不重写已经留存的历史记录。

```bash
python3 scripts/vla.py bundle seal /path/to/package/recipe.json --output /path/to/package/manifest.json
python3 scripts/vla.py bundle check /path/to/package/manifest.json
```

路径只能在包内；检查每个文件 SHA256。seal 不复制、裁剪、量化或合并权重，也不覆盖 manifest。完整性通过不等于格式能加载、参考输出一致或可部署。模型精度批准记录与模型包分离。

## Evo-1 接入判断

已检查 LeRobot commit `2774d9bddcbbda50e697e162e89e7eaada8d7105` 的 `configuration_evo1.py`、`modeling_evo1.py`、`processor_evo1.py`；这个 commit 是 API 调研固定点，不是已经验收的依赖锁。

- 配置支持 max_views=3，默认图像 448×448，内部 state/action padding 为 24D，chunk_size 默认 50。模型预测 padded chunk，必须经过原生后处理器裁回真实动作维度；不能直接把内部24D给控制侧。
- 原生 processor 已做 padding、归一化、反归一化、动作裁剪；保存/重载原生 processors，不重新写一套。不要把 Pi norm 资产套给 Evo，也不要把内部24D改成32D来模仿 Pi。
- YAM 需显式配置真实 state/action feature 为14D和三路相机。必须关闭 LIBERO 单夹爪二值化；不能只设置 gripper_index=6 而遗漏右夹爪13。
- 动作表示属于训练产物合同。若用 YAM absolute 原始 action，则训练和部署都保持该语义；若实验选择 delta，必须明确 processor、转换顺序及对应统计，不能暗中沿用 Pi 的 delta norm。
- stage1 默认冻结 VLM、训练动作头；stage2 默认解冻相关 VLM 分支，不是 LoRA。stage2 会重新应用阶段默认值，不能只沿用 stage1 checkpoint 的冻结状态。
- 训练 FP32 主参数和 BF16 autocast 需检查真实配置；`vlm_dtype`、`use_amp` 是不同开关。当前未在本地跑 Evo backward 或推理。
- LIBERO checkpoint 的7D语义不是 YAM14D。修改配置/裁剪输出不能把7D模型变成已训练的YAM策略；必须进行正确机器人适配和微调。

接入顺序：固定 LeRobot 实现 → 专用 Conda 依赖审计 → 检查 checkpoint config/processors → YAM batch 经原生 processor → 一次真实前向/反向和保存重载 → 开放模型声明中的对应操作 → Thor 原生推理计时。后续 FastWAM/VLA-JEPA 复用同一个 LeRobot 后端，但视频帧采样、文本编码和动作头仍用各自原生实现。

## 接入一个 LeRobot 模型

1. 在 `configs/models/` 声明模型的 `backend = "lerobot"` 和上游 `policy_type`，不新增 `plugins/<model>/` 或复制训练脚本。
2. 选择该系列的 `configs/environments/`，固定其源码和依赖。共用后端不要求共用 Conda；不同系列可以固定不同的上游版本，接口变更时需重测共享入口。当前 backend 中的 revision 只是 API 参考，运行器未强制校验安装版本。
3. 实验用 `model`、`environment`、`contract` 加 `[parameters].native_config` 指向上游原生训练 JSON；数据、图像采样、stage、精度、优化器和模型参数都留在原生配置内，不再翻译为自定义统一模型配置。路径相对项目根目录解析，文件哈希写入运行计划。
4. `adapters/lerobot/train.py` 检查 policy.type，然后直接调用 `lerobot.scripts.lerobot_train`。只覆盖输出目录、禁用 W&B、禁用最终/中途 Hub 上传并限定本机执行，不修改 dtype、归一化或 loss。当前仅支持新运行；resume、分布式启动器和远端提交暂走独立原生工作流，不伪装为已接入功能。
5. YAM 样例与保存重载验证后再开放该模型的 `train`；`infer` 需另行接通共享原生 policy + processor 路径，不能因 train launcher 存在就标为可推理。原生权重和 processors 是部署交接物，seal 仅补充哈希，不创造另一套权重格式。

现有 `evo1-yam.toml` 是待替换路径的实验骨架，仍会明确拒绝执行；不是可直接训练的 YAM 配置。Evo环境安装与模型capability分开：安装位置、依赖和CPU检查见 [02](02_installation_and_environment.md#evo-1--lerobot-独立环境)，没有下载权重、发起训练或改变 Thor 服务。

训练入口 API 依据固定源码：[原生训练入口](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/scripts/lerobot_train.py)、[训练配置](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/configs/train.py)。

源码来源：[Evo config](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/configuration_evo1.py)、[模型](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/modeling_evo1.py)、[处理器](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/evo1/processor_evo1.py)。
