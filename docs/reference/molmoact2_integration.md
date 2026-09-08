# MolmoAct2：LeRobot 原生接入

观察日期：2026-09-08。环境位置/安装事实归 [02](../02_installation_and_environment.md)，控制层归 [10](../10_vla_platform.md)。本页只记录模型特有差异和接入步骤。

## 选择的实现

使用官方 Hugging Face LeRobot 的普通 MolmoAct2，固定源码 `2774d9bddcbbda50e697e162e89e7eaada8d7105`，与本次 Evo-1 使用同一份源码wheel，但安装在两个独立Conda环境中。该版本有 `molmoact2` policy、处理器、优化器和checkpoint加载代码，不需要复制作者的训练框架。MolmoAct2-Think 尚不在该policy支持范围。

- [固定版本官方文档](https://github.com/huggingface/lerobot/blob/2774d9bddcbbda50e697e162e89e7eaada8d7105/docs/source/molmoact2.mdx)
- [固定版本模型实现](https://github.com/huggingface/lerobot/tree/2774d9bddcbbda50e697e162e89e7eaada8d7105/src/lerobot/policies/molmoact2)
- [原作者训练代码](https://github.com/allenai/molmoact2/tree/main/experiments)

模型声明为 `configs/models/molmoact2.toml`，`policy_type=molmoact2`；复用 `adapters/lerobot/train.py`，原生JSON直传，不重写trainer、processor、优化器或训练精度。不额外增加一套模型目录。服务器profile和实验占位在 `configs/environments/molmoact2-server.toml`、`configs/experiments/molmoact2-yam.toml`。

## 环境与复现

工作站：`/home/wuyan-lyj/.conda/envs/vla-molmoact2-dev`。

服务器：`/home/wuyan/.conda/envs/vla-molmoact2-train`。

依赖入口：`environments/molmoact2.yml` + `molmoact2-requirements.txt`；106个wheel的版本与SHA256见 `molmoact2-wheels.lock.json`，包含额外PEFT 0.20.0和SciPy 1.18.1。这是Linux x86_64/Python3.12环境，不可直接用于Thor ARM。

本次复用已校验的Python/FFmpeg conda-pack bootstrap和已有wheel缓存；新prefix独立解包、`python <prefix>/bin/conda-unpack`修复前缀，再按锁安装 `lerobot[molmoact2,training]`。不往Evo环境追加Molmo依赖，不升级系统CUDA/驱动。激活新环境后需让 `LD_LIBRARY_PATH` 包含该prefix的 `lib`，使TorchCodec找到Conda FFmpeg；用环境专属 `conda env config vars set -p <prefix> LD_LIBRARY_PATH=<prefix>/lib`，不写全局shell配置。

服务器项目检出没有同步本分支。安装用源码wheel、bootstrap、审计脚本与锁单独放在 `/home/wuyan/lyj/evo1-install-2774d9b/`（沿用先安装Evo的缓存目录名，不表示环境共用）；依赖wheel临时缓存为 `/tmp/evo1-wheels-wuyan-2774d9b/`。在尚未检出本分支的服务器上，审计脚本路径用该独立目录中的 `audit_molmoact2_environment.py`，不要修改正在训练的项目来运行下面的相对路径示例。

本次服务器离线写入分Torch/CUDA类21个wheel和其余85个wheel，两组无wheel文件交集，各自 `pip install --no-deps --no-index --no-compile`，最高两并发、nice15；记录分别为缓存目录下的 `molmoact2-pip-heavy.json`、`molmoact2-pip-base.json`。完整性来自事先106个wheel的SHA256检查，完成后再统一检查依赖和两端版本。普通复现也可按同一锁单次串行安装；不对已有环境并发升级。

CPU检查命令（不会下载模型或使用GPU）：

```bash
conda run -p /home/wuyan/.conda/envs/vla-molmoact2-train python -m pip check
conda run -p /home/wuyan/.conda/envs/vla-molmoact2-train python -B scripts/conda/audit_molmoact2_environment.py --output /path/to/new-audit.json
```

检查仅覆盖真实模块导入、原生config、合成14D归一化和一个标量FP32优化器步骤。没有用伪输出充当模型前向。实测见 [环境报告](../reports/environments/molmoact2-20260908/README.md)。

## 精度与训练方式

| 配置 | 实际含义 |
|---|---|
| `train_mode_vlm=lora` | VLM训练LoRA；动作专家仍全量训练，不是整模型只训LoRA |
| `dtype=bfloat16` | 大型VLM矩阵以BF16保存；动作专家、LoRA和敏感参数以FP32保存；算子由BF16 autocast按类型选择计算精度 |
| `dtype=float32` | 完整模型FP32，关闭该低显存autocast路径；显存成本更高 |
| `compile_model=true` | 此版本只编译动作专家块；VLM等保持eager，不能据此声称整个模型编译等价 |

这里“FP32动作专家”指参数和优化器状态，不表示其所有运算都用FP32：固定源码 `_apply_bfloat16_parameter_policy` 明确说明动作专家算子仍可在autocast下执行BF16。输出转为FP32也不表示整个前向为FP32。

原作者训练保留FP32主权重并使用BF16 AMP；LeRobot低显存FFT路径的大型VLM参数和Adam状态为BF16，并使用BF16更新补偿张量，**不等价于FP32主权重/优化器状态**。不能把Pi转换误差结论直接套到本模型。后续以原生LoRA + FP32动作专家参数作为低显存候选，在空闲GPU验证实际存储/计算dtype、梯度和数据输出后再定训练配置；本次没有测吞吐或任务效果。

使用原作者HF权重时设 `policy.checkpoint_path`；使用LeRobot训练产物时设 `policy.path`，保留checkpoint已有FFT/LoRA结构，不能随意用配置覆盖拓扑。普通连续动作训练选择 `action_mode=continuous`；离散/联合动作另需FAST tokenizer，不能把它与基础文本tokenizer混同。

## YAM接入剩余步骤

1. 选定本地checkpoint、任务数据和原生训练JSON。三相机顺序显式使用top/left/right，14D顺序沿用YAM合同；图像/状态/动作维度从metadata核对。
2. 明确实际动作语义与单位，再填写 `setup_type`、`control_mode`；不要自动套用Pi的关节delta变换、50步horizon或LIBERO的末端位姿。Molmo默认chunk为30、内部最大动作维度32，这不是YAM频率结论。
3. 使用YAM自己的q01/q99等统计；不要套LIBERO/DROID统计。`normalize_gripper=false`时，用名称 `left_gripper`、`right_gripper` 或显式mask标记索引6/13，且原生代码要求跳过归一化的夹爪范围在[-1,1]。如果真实范围不符，应选择明确的夹爪归一化方案；不静默裁剪或改原数据。
4. 空闲计算节点做原生checkpoint加载、一个真实batch前向/反向及保存重载检查，再开放项目的YAM训练能力。当前模型声明planned只表示这一数据/模型验收未完成，不表示LeRobot没有MolmoAct2。

这轮不启动GPU任务、不改服务器正在训练的检出，也不部署Thor或操作3588。Thor推理和端侧加速仍是后续独立工作。
