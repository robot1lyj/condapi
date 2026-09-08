# 03 · 训练与评估

## 正式运行：2026-09-08 Lego全量微调

用户已授权并于北京时间14:03启动，首个更新14:05:36完成。运行在Slurm2064的步骤2064.33、
gpu001、tmux `yam-lego-full`；这些是启动观察，巡检必须重新查询。固定代码快照
`/home/wuyan/lyj/YAM/env-transfer/lego-full-f93a792`（提交f93a792），从Gitea获取。
batch64/FSDP4/EMA关闭、40k阶段终点、每20k完整保存，LR周期162097，详见
[启动证据](reports/training/pi05-full-launch-20260908/README.md)与其中`initial_config.json`。

- 正式run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64`。
- 控制日志：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64/train.log`；配置和恢复测试证据同目录。
- 启动器：固定快照的`scripts/launch_lego_full.sh 2064`，通过tmux启动；锁保护避免重复launcher。
- 仅在无活跃训练、完整checkpoint和当前资源已核验后，使用同一快照启动器加`--resume`。
  `--steps`表示累计停止点，不是额外步数；不得自动超过40k。不要使用overwrite或base权重冒充续训。
- 全尺寸Pi0.5 checkpoint恢复30→31、GPU新进程训练入口恢复2→4并再次保存均已通过。
  CPU同进程测试在第二次JAX调用Aborted仍未定位，不能写成已修复；正式运行/恢复使用GPU独立进程。
- 已核对远端与看板步数1→11→21→31→41，loss和梯度有限，近期约3.37秒/步；
  瞬时四卡利用率100%。只是启动验收，不是训练完成或任务效果验收。
  第一份正式checkpoint须到20k才产生，此前只有恢复测试产物，不能混淆。

## 2026-09-08 四张 4090 全参数短测

`pi05_yam` 在Slurm2064/gpu001的4×4090上通过全参数容量测试，33.53亿参数全部可训练。
先分片加载checkpoint修复初始化OOM；随后预分配显存消除了batch32按需增长时的失败。
固定FSDP4、EMA=None、原AdamW、三相机/文本200/H50/32D、JAX92%显存预算：
global batch92通过3步、相邻96与128 OOM；这不是改变精度/卸载/预算后的硬件绝对上限。
推荐batch64作为长训候选：固定batch10步通过，活跃峰值18.54GiB，3.3385秒/步；
全部训练集随机取数、8worker的30步测试平均3.3895秒/步（去掉首步），平均取数等待0.0328秒。
这是短测，不保证长期稳定或任务收敛；正式长训未启动，默认LoRA配置仍保留。
用户随后确定阶段计划：batch64、原AdamW峰值2.5e-5、warmup1000、约162097步余弦至2.5e-6，
首阶段累计40k，每20k完整保存，后续评估后续训向约一遍数据推进。不是每40k重置学习率。
按3.3895秒/步，40k纯训练37.7小时（排期40–48小时），一遍约152.6小时，不含停机/评估。
此计划替代短测报告中的最初30k/60k方案；正式运行见本页顶部，验证/早停尚未接入循环。
限定条件、保存证据、参数/步数解释与时间外推见[batch测试报告](reports/training/pi05-batch-limit-20260908/README.md)；
初始化修复及最初batch4证据见[首轮报告](reports/training/pi05-full-20260908/README.md)。
全量train-only norm见[数据合同](04_data_contracts.md#2026-09-08-全量发布与归一化完成)。

## 训练看板与每小时巡检（2026-09-08）

- 页面：[training_dashboard.html](../scripts/training_dashboard.html)，本地入口 `http://127.0.0.1:8765/`。
  W&B风格的只读工作台，不使用W&B上传或外部CDN。主图loss，附验证loss、LR、梯度/参数范数、
  单步耗时、吞吐、显存、GPU利用率、取数等待；日志未提供的指标明确留空。
- 服务：[training_dashboard.py](../scripts/training_dashboard.py)，仅Python标准库、仅监听loopback。
  10秒SSH拉取指定JSONL并原子替换本地缓存，页面10秒轮询；两者均不调用模型，不消耗模型token。
  SSH失败保留上次数据并提示；忽略未完成的尾行、报告坏行、处理step回退，不读取/修改checkpoint。
  单文件上限32MiB，最多展示最近20000条日志记录，超限明确提示；这不是训练步数上限。
- 本机用户服务 `training_dashboard.service` 已启用，文件位于 `scripts/`；随用户服务管理器启动，
  异常退出10秒重启。查看/恢复：`systemctl --user status training_dashboard.service` /
  `systemctl --user restart training_dashboard.service`。本机休眠、断网或用户服务未运行时不能保证刷新。
- 本地缓存 `artifacts/training_dashboard/live/metrics.jsonl`；预留远端
  `/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64/metrics/metrics.jsonl`。
  已绑定正式run并核对多次真实步数增长；不能仅凭页面在线断言训练健康。
  参数面板显示讨论计划，不冒充实际配置验真。原生LocalMetricLogger已有loss/grad_norm/param_norm/LR；
  正式入口已补记时间、近期步速/吞吐与JAX活跃分配（不是nvidia-smi显存或峰值）；验证loss和GPU利用率
  仍需独立采集，留空不伪造。10秒刷新不等于每10秒新增loss，log_interval=10时约34秒更新一组。
- 启动例：`python3 scripts/training_dashboard.py --metrics /absolute/run/metrics/metrics.jsonl`；
  远端镜像加 `--remote yam-server --remote-metrics /absolute/remote/metrics.jsonl`。
  服务内的参数与路径通过CLI设置；默认stage40k/total162097/save20k，不启动训练。
- 每小时heartbeat `pi0-5`（“每小时巡检Pi0.5全量训练”）已设ACTIVE，无终止日期；
  只在异常/修复/重要进展时通知。尚未启动训练时不自行首次启动；40k阶段结束不自行越过停止点。
  巡检现场Slurm、计算进程、步数增量、有限loss/梯度、GPU、磁盘/NFS及完整checkpoint。
  故障恢复先排除编译/保存/排队，确认无重复写进程、资源有效、配置/数据一致和恢复链路通过，
  再用该run记录的启动器续训，验证多个新步数；不改batch/LR/精度/预算。同一修复两次失败停止盲重试。
- 用户允许跳过部分经审计证实损坏的样本：保留原件和源ID/原因/哈希/排除清单，优先整轨迹隔离到
  新数据版本、保持三相机/state/action对齐；自动修复单轮最多5条、累计不超过原train的1%，超出请确认。
  不能把OOM、网络、存储或解码依赖故障当作数据损坏。数据集合改变需记录版本分支、验证采样/续训位置，
  按合同重算验收norm，不能宣称原序列无缝续接。无法验证则保留现场报告。无限巡检不等于无限重启。
- 验证：13项pytest覆盖日志追加/半行、非法值、续训回退、读取上限、HTTP路径隔离、镜像失败保留及成功发布；
  JS语法检查、实际HTTP200与等待远端日志API通过。未宣称浏览器交互或正式训练的端到端验收。

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
