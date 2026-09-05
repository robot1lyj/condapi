# 03 · 训练与评估

本页只描述当前 YAM 训练路线。服务器和环境先看 [02 · 服务器与环境](02_installation_and_environment.md)，动作/图像合同看 [04 · 数据合同](04_data_contracts.md)。

## 当前默认路线

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

若使用 PyTorch 训练入口，仍必须使用同一个 YAM config、数据版本和 norm stats；JAX/PyTorch 不能共用未记录的输出目录。续训只从完整 checkpoint resume，不能覆盖旧实验。

## 参数和实验隔离

| 参数 | 入口 | 规则 |
|---|---|---|
| 数据路径/split | `--data.repo-id` 或独立配置 | 变化就新建数据版本并重算 norm |
| LoRA/全量 | config 的 model/freeze filter | 首轮优先 `pi05_yam_lora`，不要混用 checkpoint |
| batch/workers | config 或 CLI | 先以 GPU smoke 测定，OOM 后降低 batch/worker |
| horizon/action dim | model config + YAM contract | 当前为 50/32 内部、14D 外部；不能随意改一端 |
| 保存/步数 | config 或 CLI | 输出目录包含 config、实验名和 step |

每个实验至少记录 git commit、config、repo id、数据版本、split、norm 路径、base checkpoint、LoRA 设置、batch、workers、step 和 seed。普通 SFT、不同 LoRA 设置和后续评估必须使用独立实验名，避免结果无法归因。

## Checkpoint gate

训练日志显示完成不代表 checkpoint 可用。部署或评估前检查：

- step 目录的 Orbax 参数元数据完整；
- `assets/yam/norm_stats.json` 存在且与 config/data 绑定；
- config、git commit、数据版本和训练日志可追溯；
- 用 [05 · 训练后 policy smoke](05_inference_and_rollout.md) 验证输出为有限 `(50,14)`。

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
