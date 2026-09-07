# 03 · 训练与评估

本页只描述当前 YAM 训练路线。服务器和环境先看 [02 · 服务器与环境](02_installation_and_environment.md)，动作/图像合同看 [04 · 数据合同](04_data_contracts.md)。

## 当前默认路线

服务器训练关闭 W&B（`wandb_enabled=False` / CLI `--no-wandb-enabled`），不依赖 W&B 在线或离线运行。训练入口已有 `LocalMetricLogger`，将指标写入实验目录的 `metrics/metrics.jsonl`、`metrics/metrics.csv` 和 `metrics/plots/`；结合 Slurm stdout/stderr 和 checkpoint 作为记忆证据。日志存在不等于 checkpoint/部署 gate 通过。

首选配置为 `pi05_yam_lora`：

- Pi0.5，`gemma_2b_lora` + `gemma_300m_lora`；
- 冻结规则由对应 `Pi0Config.get_freeze_filter()` 生成，LoRA 关闭 EMA；
- YAM 双臂真实动作 14D，模型内部 padding 为 32D；
- action horizon 为 50；
- 初始保守默认值为 batch 4、workers 2、每 1000 步保存、保留周期 5000、总步数 30000；这些是新服务器上的起始值，不是 GPU 性能结论。

`pi0_yam`、`pi0_yam_lora`、`pi05_yam`、`pi05_yam_lora` 均已注册在 `src/openpi/training/config.py`。默认 `repo_id=local/yam_bimanual` 只是占位符，正式训练必须用 CLI 或复制配置覆盖为已审计的数据路径。

阶段顺序固定为：第一阶段用已审计的乐高分拣示范做 Pi0.5 LoRA SFT；第二阶段再接入 DAgger，用人工纠正/回放数据建立独立数据版本和实验名。当前仓库只提供 YAM 的输入输出合同和通用训练入口，DAgger 的采集、纠正合并和安全 rollout 尚未宣称完成，不得把普通 SFT 结果写成 DAgger 结果。

## 训练前 gate

按以下顺序执行，任一步失败都不启动长训：

1. 确认当前 commit、数据版本、episode split、config 和初始化 checkpoint。
2. 检查 LeRobot metadata、`action`/`observation.state` 的 14D、三路图像、task/prompt、视频首中尾解码。
3. 按同一训练数据版本计算 norm stats，保存为 `assets/yam/norm_stats.json`。
4. 用真实 dataset loader 取样，确认 transform 后 state/action 能进入模型的 32D spec。
5. 在 Slurm GPU 分配内做短步数 smoke，再启动正式训练。

基础代码测试：

```bash
"$PYTHON" -m pytest --strict-markers -m "not manual" -q
```

## Norm stats

YAM 的 norm 必须在 `YamInputs` 和 delta action transform 后计算；不要复用 OpenArm、Piper 或其他单位合同的 stats。示例：

```bash
DATASET=/home/wuyan/lyj/YAM/YAM_data/audited/yam_lerobot_v001
"$PYTHON" scripts/compute_norm_stats.py pi05_yam_lora \
  --repo-id="$DATASET"
```

实际输出根目录和资产目录以脚本日志为准，完成后确认 `yam/norm_stats.json` 可读且与训练数据版本一致。`compute_norm_stats.py` 不会替代数据结构审计。

YAM 的脚本默认输出已对齐训练读取目录：默认配置写入
`assets/pi05_yam_lora/yam/norm_stats.json`，不是 dataset 根目录。
自定义 `--output-dir` 时必须同时使训练的 assets 配置指向同一父目录。
每个正式数据版本使用独立 assets 目录，避免重算 norm 覆盖另一实验的统计量。

## 单机 smoke 和正式训练

先在已分配 GPU 的节点运行 10～20 步，使用新实验名和独立输出目录：

```bash
CONFIG=pi05_yam_lora
DATASET=/home/wuyan/lyj/YAM/YAM_data/audited/yam_lerobot_v001
EXP_NAME=yam_pi05_lora_smoke

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/train.py "$CONFIG" \
  --data.repo-id="$DATASET" \
  --exp-name="$EXP_NAME" \
  --num-train-steps=20 \
  --batch-size=1 \
  --num-workers=0
```

smoke 通过后再用保守默认值启动正式训练；长任务使用 Slurm/tmux：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  "$PYTHON" scripts/train.py pi05_yam_lora \
  --data.repo-id="$DATASET" \
  --exp-name=yam_pi05_lora_v001 \
  --num-train-steps=30000
```

当前使用 JAX 训练入口；PyTorch LoRA 支持须另行审计和验证，不属于本轮已验收路径。
续训只从完整 checkpoint resume，不能覆盖旧实验。

## 参数和实验隔离

2026-09-07 源码审查：当前继承 CosineDecaySchedule（warmup 1000 步，peak LR `2.5e-5`，
decay 30000 步，终点 `2.5e-6`），AdamW（b1=0.9、b2=0.95、eps=1e-8、weight decay=1e-10、
global gradient clip=1.0）。LoRA 的 PaliGemma rank/alpha=16/16，action expert=32/32。
冻结过滤器冻结对应 LLM 主干并排除 LoRA；不能将其表述为全模型“只有 LoRA 可训练”，
其他未命中过滤器的参数仍可训练。EMA 关闭，W&B 关闭；正式开跑前记录实际 trainable parameter 数和显存。

30 FPS 数据上 horizon 50 对应约 1.67 秒预测窗口，不等于机器人每次必须执行 50 步。
batch 4 × 30000 steps 约采样 120000 个训练窗口；manifest 声明数据约 97.586 小时（含验证集），
因此 30000 步只是初始预算，不是完整 epoch 或已验证的收敛方案。
完成清洗后按实际 train 帧数记录 `steps * global_batch / train_frames`，结合固定 holdout 决定训练长度。
20 步 smoke 位于 warmup 初段，只验证训练链路，不作学习效果结论。

当前首选是 `scripts/train.py` 的 JAX 路径；不要把该 LoRA 配置直接视为 PyTorch 入口已验证支持。
上传子集的转换/发布流程见 [数据合同](04_data_contracts.md#abc-乐高子集上传期间的清洗流程)。

| 参数 | 入口 | 规则 |
|---|---|---|
| 数据路径/split | `--data.repo-id` 或独立配置 | 变化就新建数据版本并重算 norm |
| LoRA/全量 | config 的 model/freeze filter | 首轮优先 `pi05_yam_lora`，不要混用 checkpoint |
| batch/workers | config 或 CLI | 先以 GPU smoke 测定，OOM 后降低 batch/worker |
| horizon/action dim | model config + YAM contract | 当前为 50/32 内部、14D 外部；不能随意改一端 |
| 保存/步数 | config 或 CLI | 输出目录包含 config、实验名和 step |

每个实验至少记录 git commit、config、repo id、数据版本、split、norm 路径、base checkpoint、LoRA 设置、batch、workers、step 和 seed。普通 SFT、不同 LoRA 设置和后续评估必须使用独立实验名，避免结果无法归因。

面向跨会话记忆的产物身份、指纹和本地 run manifest 合同见 [09 · 记忆系统](09_memory_system.md)。该合同用于记录与验收；训练入口尚未自动生成其全部字段，不得把文档规范写成已经实现的采集器。

## Checkpoint gate

训练日志显示完成不代表 checkpoint 可用。部署或评估前检查：

- step 目录的 Orbax 参数元数据完整；
- `assets/yam/norm_stats.json` 存在且与 config/data 绑定；
- config、git commit、数据版本和训练日志可追溯；
- 用 [05 · 训练后 policy smoke](05_inference_and_rollout.md) 验证输出为有限 `(50,14)`；若部署 Thor，还要按 [08 · Thor 端侧部署](08_thor_edge_deployment.md) 保留 JAX reference 与转换后 engine 的验证报告。

半写入数字目录、缺少 norm 或只有单独 `params/` 的目录不得部署。

## 评估与结果记录

离线 loss、动作误差和 chunk 连续性只能作为诊断，不能直接等同于 YAM 真机成功率。固定 holdout 上比较时必须保持数据版本、prompt、horizon 和 norm 一致；真机或服务结论另行记录 checkpoint、版本和 smoke 证据。结果原因写入 `docs/07_change_log.md`，不把历史 OpenArm KAI0 计划混入当前 YAM 结论。

当前评估入口示例：

```bash
"$PYTHON" scripts/evaluate_checkpoint.py \
  --config=pi05_yam_lora \
  --checkpoint-dir=/path/to/checkpoint \
  --dataset="$DATASET" \
  --val-split=80:100
```

该入口只做数据/动作合同、首步误差和 chunk 连续性的离线诊断；DAgger 只有在纠正数据、采集协议和安全 rollout 均单独留痕后才进入第二阶段。
