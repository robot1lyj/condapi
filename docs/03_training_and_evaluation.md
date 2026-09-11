# 03 · 训练与评估

> 当前路线：用户于2026-09-08明确切换为 **Pi0.5全量微调**，正式训练已启动。
> 旧LoRA低显存默认建议不再适用，不自动回退；保留的LoRA代码和历史记录不代表当前实施方案。
> 当前正式续训配置为全局batch32/FSDP4/EMA关闭，首阶段目标80k、总体目标324,194步、每5k保存。训练集有10,374,181个有效帧，`ceil(10,374,181 / 32) = 324,194`，即完整一轮数据的累计步数。余弦学习率周期仍为162097，详见正式运行章节。

## Loss日志与看板口径（2026-09-08用户确认）

- 正式训练保持`log_interval=10`，不改成逐步记录。每一步先对batch、动作时间窗口和模型动作维度
  求平均得到该步loss；日志再对最近10步的loss取算术平均，首条记录例外，仅包含第1步。
- 当前JSONL使用已完成步数：step1对应第1次更新的loss；step11对应第2–11次更新的平均，
  step21对应第12–21次更新的平均，以此类推。这是训练loss，不是关节物理误差或任务失败率。
- 看板“原始曲线”指未经页面平滑的**日志均值**，不是逐步原始loss；平滑线是在这些日志均值上
  再做指数平滑，仅影响显示，与模型权重EMA无关。看板10秒刷新不改变日志统计窗口。
- 用户讨论论文中密集原始点形成的“厚毛边”后，明确决定无需调整记录方式，只需说明上述口径。
  不人为添加噪声，不把线条厚度自动解释为置信区间，不尝试还原已被平均掉的历史逐步波动。
- 依据：`src/openpi/models/pi0.py::compute_loss`、`scripts/train.py::train_step/main`及
  `scripts/training_dashboard.html`的绘图逻辑；运行版本f93a792与启动证据见本页下文。

## 2026-09-09 故障恢复与当前授权

**17:10最新授权与启动：**用户转达管理员允许直接开训、遇问题再处理，并明确要求现在开始，取代下方17:04暂不放行结论；ECC不再作为本轮启动前置条件，根因未解决的事实保留。已核对2064有效、四卡空闲、无训练/诊断进程，r2无完整checkpoint；保留旧现场，从base新建`lego_full_b64_r3_20260909`。17:10:35通过tmux `yam-lego-full-r3`派发Slurm2064.81，PID1061106，沿用不可变a53bb00快照，batch64/FSDP4/40k/每5k保存、LR/精度/数据均不变。看板切到独立r3缓存。启动验证见[r3记录](reports/training/pi05-r3-20260909/README.md)。

**17:04用户要求先排查再训练：**正式训练暂不放行。本轮全量4458个Parquet/10,374,181帧哈希与norm来源一致，14D state/action全部有限，19项轻量合同测试通过；未运行训练或训练smoke。17:03四卡空闲，CPU CE137754、UE0，持续增长；根因仍未知，不能归咎数据/算法或反向断言硬件。当前小时巡检为PAUSED。范围和恢复门槛见[本轮复核](reports/training/pi05-crash-audit-20260909/README.md)。此状态优先于下方历史恢复授权。

**11:42状态更新：**NCCL定向诊断2064.73在10:58:40 TIMEOUT，进度条到6步，约130秒/步；
日志没有捕获非法访问或Internal Sanitizer Error，也无最终ERROR SUMMARY，不能记为通过。
正式r2仍停在271步历史日志，没有完整checkpoint；当前没有训练/诊断进程，四卡空闲于allocation2064。
gpu002仍全部分配；CPU CE131367、UE0。看板服务与镜像正常，仅展示历史值。
两次有限插桩均未给出根因，不重复同类耗时测试或原样重开第三轮；后续优先复核管理员对
CPU0_DIMM_B1和GPU1/Xid13/43的检查、健康替代节点或新的可验证软件修复证据，再恢复训练。

**10:43诊断更新：**2064.71完整memcheck在10:13:40 TIMEOUT，只观察到step1；工具多次提示
无法分配插桩内存、部分kernel未检查，因此既非通过，也不能解释为原训练OOM。CPU CE增至131365、UE0。
启动独立`lego_ncclcheck_20260909`短测（tmux `yam-lego-ncclcheck`）：同快照、20步、15分钟上限，
仅`--kernel-name kns=nccl`检查通信内核，`--force-synchronization-limit 100`限制诊断积压，
XLA显存池从0.92降至0.85给工具留空间；batch64/精度/优化器不变。这不是正式第三轮。
日志位于 `/home/wuyan/lyj/YAM/training-runs/control/lego_ncclcheck_20260909/diagnostic.log`。
该过滤诊断即便通过也不能排除未插桩XLA计算kernel/硬件或长时问题；不要直接据此宣称全量训练恢复。

**09:41再次核验：r2已于09:33:21崩溃，下面09:20启动成功不代表当前在训。**
最后logged step271，进度条278，SIGSEGV；NCCL报告illegal memory access，内核GPU1（PCI52:00.0）
Xid13 Out Of Range Address，继发Xid43。CPU corrected ECC计数127141，UE0，不能直接归因硬件。
gpu002四卡均已分配，无空闲替代四卡节点，不取消其他作业。原样重启已复现，不再盲重开正式run。
09:43:27启动独立诊断 `lego_full_memcheck_20260909`、tmux `yam-lego-memcheck`，同a53bb00快照，
使用`/usr/local/cuda-13.2/bin/compute-sanitizer --tool memcheck --error-exitcode 86`包裹正式入口，
batch64/FSDP4不变、仅20步、Slurm限时30分钟，NCCL_DEBUG=INFO；不是40k正式训练或已修复。
诊断日志 `/home/wuyan/lyj/YAM/training-runs/control/lego_full_memcheck_20260909/diagnostic.log`。
下一次巡检先查诊断结果/实际步骤与退出码，再决定有证据支持的代码或环境修复，不重复启动诊断。
插桩额外显存/超时不能当原训练OOM或训练健康验收；需管理员排查时提供GPU1 Xid13/43和DIMM_B1证据。

以下取代旧运行的保存周期与“无checkpoint等待确认”边界：用户已授权主动排障、修复并恢复训练，
保存/保留间隔均为5000步；batch64/FSDP4、LR、精度、数据和40k阶段终点不变。
有完整checkpoint优先真续训；没有则允许保留旧现场，在新run从base重开，不能混接旧步数或loss。
反复原样重试不构成修复，驱动/硬件需管理员权限时提供证据并寻求安全替代，不擅自升级系统驱动。

9月9日09:06现场核验：旧步骤2064.33于05:21:05以SIGSEGV退出，最后日志16031步，loss0.01768966，
旧run仅metrics、没有正式checkpoint。四卡allocation仍有效、GPU空闲。core虽然journal访问受限，
其文件ACL允许本用户读取，已解压并通过GDB检查；PC `0x7fee0e2787c7` 落在
`/usr/lib64/libcuda.so.595.45.04` 映射内。core截断为1GiB，栈页缺失，不能据此宣称已定位最终根因。
EDAC累计CE127134、UE0，并有CPU0_DIMM_B1单bit corrected ECC记录；未建立与05:21退出的因果关系。
无已见OOM/磁盘满证据。现场证据与管理员建议见
[故障记录](reports/training/pi05-recovery-20260909/README.md)。

本次恢复目标run为 `lego_full_b64_r2_20260909`，与旧run隔离。入口新增 `LEGO_RUN_NAME`，
原生崩溃启用 `PYTHONFAULTHANDLER=1`，`--resume` 无已提交checkpoint时强制拒绝。
看板service同步新run和独立缓存 `artifacts/training_dashboard/recovery_20260909/metrics.jsonl`，
显式传入save-interval5000；每10秒刷新不调用模型。每小时巡检`pi0-5`已更新为主动恢复模式。
9月9日09:16:39已提交新进程：Slurm2064.63、PID3091182、tmux `yam-lego-full-r2`，
固定Gitea快照 `/home/wuyan/lyj/YAM/env-transfer/lego-full-a53bb00`（a53bb001e4acb2cce486f2da83d6d8256439a188）。
启动命令：`LEGO_RUN_NAME=lego_full_b64_r2_20260909 bash scripts/launch_lego_full.sh 2064`。
恢复时使用同一快照/环境/run，加`--resume`，先核验完整checkpoint与无重复进程。
实际`initial_config.json`已核对save_interval=keep_period=5000、batch64、steps40000、resume=false。
新控制目录 `/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64_r2_20260909`，
新run目录 `/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64_r2_20260909`。
09:20启动验收：已核对真实日志21→31→41，step41 loss0.04490658、grad_norm0.28559217、
3.364秒/步；四GPU100%、约22828MiB/卡、56–61°C。看板API已同步41步、save_interval5000、
无同步错误。此为启动验收，首份完整checkpoint与长期稳定性仍须后续验证。

## 故障诊断复用与重试条件

此节整理 2026-09-09 的历史诊断，不报告当前进程或改变后续授权。当前节点、驱动状态和首个非法读写算子均需新证据；本轮记忆维护未查询服务器。

| 问题与原因假设 | 实际尝试与结果 | 判断、局限与重试条件 |
|---|---|---|
| r2 出现 Xid13/43、NCCL illegal memory access；完整插桩能定位首个非法访问 | 同 a53bb00 快照、batch64/FSDP4，20步/30分钟 memcheck；只观察 step1，插桩内存不足、部分 kernel 未查，TIMEOUT | inconclusive；工具内存不足不是原训练 OOM。只有可用插桩资源/时限、缩小且能触发问题的案例或诊断策略发生变化才值得重做；新案例须另存证据 |
| 通信内核可能是非法访问来源 | NCCL-only 20步/15分钟；同步积压限100、XLA池0.85；到6步 TIMEOUT，无最终 ERROR SUMMARY | inconclusive；过滤掉计算 kernel 且改变池配置，未捕获不等于排除通信/硬件。得到新的通信线索或足够覆盖的诊断窗口后再重试，不原样重复耗时测试 |
| 数据损坏/非有限输入可能导致失败 | [全量复核](reports/training/pi05-crash-audit-20260909/README.md)：4458个Parquet/10,374,181帧哈希匹配，14D state/action有限，19项轻量测试通过 | 已检查范围未见异常；未全量解码视频、未复验基础权重完整哈希，也不证明GPU算法正确。数据/变换/权重版本变化或有具体异常样本时重查对应部分 |

前两项实际参数、远端日志路径与退出观察见 [故障恢复记录](#2026-09-09-故障恢复与当前授权)；本地报告只保存其结果摘要，完整诊断日志本轮未重新取回，不能补写未知错误栈或声称根因已定位。CPU CE增加与GPU错误的因果关系仍未知。恢复训练是授权下的操作，不是修复有效性的证明。

复用检查时先区分：配置要求 batch/FSDP/保存周期、`initial_config.json` 的启动记录、实际日志步数/退出码，以及完整 checkpoint 的独立进程恢复结果。配置为每5k保存不证明文件已完整提交；看板有曲线不证明进程还活着。历史命令中的 Slurm ID、快照和阶段终点不能直接作为当前运行参数。本地仅运行轻量合同检查，训练和插桩执行须在获准计算资源上。

## 正式运行：2026-09-08 Lego全量微调（历史启动记录）

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
