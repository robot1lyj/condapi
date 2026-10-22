# Pi0.5 服务器反复崩溃调查与现场采集（2026-10-22）

## 结论与适用范围

本轮只读调查 `yam-server → gpu001`，观察时间为北京时间2026-10-22 14:19–14:24；未提交训练、改变配方/环境、停止作业、重启节点或修改自动恢复。原始源日志保留，本地完整副本位于 `evidence/raw/`；Git保存来源/哈希、关键摘录和统计。仅涉及服务器训练，不涉及Thor/3588。

**当前最强线索是 GPU1 / PCI `0000:52:00.0` 的重复设备异常，同时存在独立的主机内存纠错异常。尚未定位到硬件、驱动、XLA生成代码或应用内存错误的唯一根因。** NCCL报错和最终SIGABRT可能是前面CUDA故障的后续表现，不能根据最后一行定责。

Slurm窗口自10月20日00:00至14:19采样，共44个 `pi05-earbuds-50h` 作业为FAILED，另1个RUNNING；按退出日期分别为20日21次、21日18次、22日5次。包含19日晚开始、20日结束的2466。失败时长中位数34.35分钟；44并不等于44次已证明同根因的GPU故障。[Slurm原始结果](evidence/sacct.txt)、[逐作业关联](evidence/job-correlation.json)。

14:19时作业2562在gpu001从本run完整1000恢复，目标本地86000；14:20日志已推进1160，仍须动态复核。不能把旧114000参数初始化后的新优化器步数与旧run直接混成同一个优化器轨迹。

## 直接证据

| 作业 | 结束时间（+08） | 调度器ExitCode | 应用/内核观察 |
|---|---|---|---|
| 2542 | 10-21 17:19:56 | 1:0 | 52:00 Xid13/43；Out Of Range Address |
| 2544 | 10-21 18:36:33 | 1:0 | 同卡Xid13/43 |
| 2546 | 10-21 19:51:55 | 1:0 | 同卡Xid13/43 |
| 2550 | 10-21 20:50:01 | 0:11 | SIGSEGV、NCCL illegal access；同卡Xid13/43 |
| 2552 | 10-22 11:00:29 | 6:0 | `loop_convert_fusion_19` launch illegal address，同时NCCL报错；同卡Xid31 MMU fault/GPC8 |
| 2554 | 10-22 11:24:11 | 1:0 | 4秒启动失败：预创建的training_contract.json触发FileExistsError；未进入训练 |
| 2556 | 10-22 12:03:40 | 11:0 | SIGSEGV；同卡Xid13/43 |
| 2558 | 10-22 13:13:03 | 11:0 | SIGSEGV；同卡Xid13/43 |
| 2560 | 10-22 14:04:57 | 6:0 | 日志聚合将JAX数组转NumPy时暴露illegal address，随后释放资源失败/abort；同卡Xid13/43 |

证据为[应用首错与栈摘录](evidence/failure-excerpts.txt)、[28条Xid原文](evidence/kernel-xid-window.txt)。8组内核故障均在对应Slurm结束前约1–5秒，28条全在 `52:00`；7组Xid13的warp异常全为 `(GPC 8, TPC 0, SM 0)`，另1组Xid31也指向GPC8。不是28次独立崩溃，不把“其余作业没有留存Xid”解释为“没有GPU错误”。

GPU1 UUID：`GPU-7a9a29a1-9059-b0ef-d7b4-53c298565ca1`。同一物理卡/子单元重复是优先做隔离对照的理由，但固定rank、shard、代码路径也可能固定触发同一设备；不能直接宣布坏卡。Xid13/31可由应用非法访问、系统软件或硬件问题产生，Xid43常是故障后的通道终止。[NVIDIA Xid说明](https://docs.nvidia.com/deploy/xid-errors/archive/index.html)。

节点本次开机为10月21日14:26:30；当前内核环形缓冲最早仅到17:05:10，当时大量EDAC记录占据缓冲，旧现场已不可从当前dmesg恢复。普通用户读journal/coredump journal被拒绝，`sudo -n`提示需要密码；**这是访问缺失，不是系统无错误/无core**。完整dmesg已本地保存，哈希见[kernel-manifest](evidence/kernel-manifest.json)。

主机EDAC累计CE为 **3,781,237**，全部在 `mc0/dimm2`（label `CPU_SrcID#0_MC#0_Chan#1_DIMM#0`），UE为0。这是本次计数器观察值，不是两天新增量，也不是GPU显存ECC。应由管理员核对物理DIMM槽位、BMC/rasdaemon记录与计数增量；已纠正错误不能直接证明本次CUDA错误由RAM造成。[计数与进程现场](evidence/node-process-and-edac.txt)、[EDAC摘录](evidence/host-memory-excerpts.txt)。

14:20四卡温度62–65°C；14:21完整快照的热降频未激活、热降频计时为0。未取得每次崩溃前连续温度/供电序列，不能据事后温度排除瞬态问题。显存占用约22.8GiB/24GiB，XLA预分配0.92；占用接近满不等于OOM。已取首错是非法地址，未取得与上述8次对应的OOM-kill证据。[节点快照](evidence/node-health.txt)。

## 软件与已有记录的缺口

- 真实前缀：Python3.12.12、JAX/jaxlib/PJRT/plugin0.5.3、Flax0.10.2、NCCL包2.26.2、CUDA runtime12.6.77、cuDNN9.5.1.17、Torch2.7.1；驱动595.45.04。具体见[包metadata](evidence/environment-versions.txt)。`nvidia-smi`显示的CUDA支持版本不等于训练进程实际加载的runtime。
- `/proc/1703839/maps`确认NCCL/cuDNN/cuBLAS来自condapi-yam前缀，`libcuda.so`来自系统595.45.04。保留实际加载路径，不能只存pip列表。
- 2562进程已设置`PYTHONFAULTHANDLER=1`、`PYTHONUNBUFFERED=1`；未设置`NCCL_DEBUG`、`NCCL_DEBUG_FILE`、`XLA_FLAGS`、`CUDA_ENABLE_COREDUMP_ON_EXCEPTION`。已经有Python线程栈，但缺原生/GPU首错上下文。
- 旧wrapper `pi05_earbuds_50h.py`通过`subprocess.run(check=True)`启动训练，子进程信号会包装为异常；`flock`也可能把子进程信号转为普通退出值。`1:0`、`11:0`和`0:11`不能混写成同一层的退出信号。记录调度器状态、原生子进程returncode/signal和日志签名三个字段。
- `np.stack`处的错误不证明NumPy实现有问题；JAX异步执行在host取值时才同步，错误可能更早发生。`loop_convert_fusion_19`也只是首批报告位置，需core/sanitizer确定非法访问起点。[JAX异步执行](https://docs.jax.dev/en/latest/async_dispatch.html)。
- 重启前“健康检查”只有列出四卡和小张量`pmap(psum)`，典型2–3秒；它证明当时基本初始化/通信可用，不证明训练规模计算、显存和长时通信稳定。
- 数据排除158/556后仍有2550/2552/2556/2558/2560故障，所以这两条episode不能独立解释持续崩溃；本轮未重新做数据全量解码。2548的`KeyError: state`是norm完成后的检查脚本层错误，2554是启动合同冲突，应各自分类，不能计成CUDA故障。

## 怎么记录：每次attempt一个不可覆盖目录

推荐服务器实验目录结构：`control/<run>/incidents/<jobid>-<start-UTC>/`。同一run恢复也创建新attempt目录，并在run索引关联父checkpoint；原始train.log留存完整副本，另做首错窗口便于阅读。禁止只有`tail -100`，否则大量CUDA清理错误会挤掉首错。

| 层次 | 必须保存 | 为什么 |
|---|---|---|
| 作业身份 | run/job/step ID、开始结束UTC及时区、node/boot_id、实际PID/父PID、原始退出码与信号 | 关联跨进程、跨重启时间线 |
| 复现合同 | 固定代码commit/dirty差异、提交时脚本副本、初始/恢复配置、checkpoint与优化器恢复方式、数据/选集/norm身份、seed/loader位置 | 防止混合不同重试配方 |
| 资源/版本 | GPU UUID+PCI+逻辑设备映射、driver/kernel、包metadata、实际加载.so、诊断环境变量白名单、cgroup OOM事件、磁盘/inode/共享内存 | 分离OOM、依赖、设备和存储问题 |
| 过程时间线 | 每5–10秒GPU利用率/显存/温度/功耗、CPU内存、EDAC增量；持续内核日志 | 单次重启后的健康快照补不回故障前状态 |
| 故障 | 完整stdout/stderr、NCCL每进程文件、Xid/AER/MCE/OOM、CPU/GPU core、XLA编译产物 | 保住第一个异常与执行上下文 |
| 判定 | 已知事实、候选原因、反证/缺证、干预、验证覆盖、恢复点、后续条件 | “重新跑起来”不自动成为“已修复” |

新工具 [collect_training_incident.py](../../../../scripts/collect_training_incident.py) 是标准库只读快照工具：采集GPU、内核、Slurm、EDAC、core目录查询、指定PID白名单环境/加载库、包metadata和显式日志/配置；每项保存退出码、错误与时间，拒绝复用输出目录。它不训练、不发信号、不sudo、不改GPU参数，也不自动部署到服务器。复制增长日志只读开始时的文件长度，并注明复制期间是否变化。

部署该工具后，在**故障计算节点**执行，变量须由本次实际作业填写：

```bash
python3 scripts/collect_training_incident.py \
  --output "$INCIDENT_DIR" --since "$INCIDENT_START" \
  --job-id "$JOB_ID" --pid "$TRAINER_PID" \
  --python /home/wuyan/.conda/envs/condapi-yam/bin/python \
  --log "$TRAIN_LOG" --log "$TRAINING_CONTRACT"
```

这是一次快照，不是持续采集器；需要在启动前、首错后各做一次。进程退出后`/proc`证据会消失；必须在启动时存PID/库/环境，不能依赖退出trap。收集器不自动保存任意cmdline/全环境以避免凭据；提交脚本/日志须检查后再共享。现有本地原始日志和core不进Git。

连续采集建议放在受控的节点日志服务中：持久journal/rsyslog保留每次boot；`dmesg --follow`仅为普通用户可用的临时补充，不清空缓冲。训练启动时启动限定生命周期的GPU遥测；退出先冻结事故包，再评估恢复。退出trap处理不了SIGKILL、节点掉电、OOM杀死整个cgroup，需登录节点依据Slurm结束事件做独立补采，节点持续日志与训练作业解耦。当前还未安装这些自动采集/恢复熔断逻辑。

## 更细的CUDA/XLA/NCCL现场：按层启用

以下是诊断设计，不是新一轮训练配方或已运行命令；本轮未修改任何训练作业。涉及新的训练/插桩实验先明确配方、GPU分配、时限和输出目录；DAgger轮次遵循逐轮确认规则。

### 第一层：下一次受控启动加入NCCL日志和有限XLA dump

新attempt目录必须先创建；在导入JAX/初始化CUDA之前设置。保留原有XLA_FLAGS，避免覆盖既有行为。

```bash
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL,P2P,SHM,NET
export NCCL_DEBUG_FILE="$INCIDENT_DIR/nccl.%h.%p.log"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_to=$INCIDENT_DIR/xla --xla_dump_hlo_as_text"
```

NCCL文件名必须每job/进程唯一，否则会覆盖；单进程四GPU可能只有一个文件，不能强行假设四rank四文件。先INFO，必要时短窗口TRACE；同步收集stdout/stderr和内核时间线。当前NCCL2.26.2归档文档没有新的时间格式变量，不能把最新版选项无核对照搬；默认依靠原始时间/PID和采集时间对应。[对应版本NCCL说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2262/user-guide/docs/env.html)。

XLA dump保留优化前后HLO及可用后端产物，帮助把fusion名对应到编译图；不是运行时内存快照。先不启用所有pass或逐步HLO snapshot，避免海量输出。JIT符号/行号可能不足，能否得到PTX/LLVM和源码映射须按已安装XLA验证。缓存命中可能不产生本次编译dump，不应删除共享缓存；诊断缓存隔离需单独记录。[OpenXLA HLO dump](https://openxla.org/xla/hlo_dumps)。

### 第二层：保留CPU原生core和CUDA GPU core

当前trainer实际core limit已经unlimited，但`core_pattern`指向systemd-coredump，用户读journal被拒。首先请管理员按故障时间/PID导出已有`coredumpctl info/dump`和栈；核对Storage、ProcessSizeMax、ExternalSizeMax/配额。**仅再加`ulimit -c unlimited`不能保证拿到core。** CPU core用于原生线程栈/加载库，GPU core用于异常warp、指令/寄存器/显存上下文，两者不相互替代。

下次受控诊断可加入：

```bash
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_FILE="$INCIDENT_DIR/cuda.%h.%p.%t.nvcudmp"
```

GPU core可能很大，启用本身也可能影响性能；预留空间/配额并记录开关。缺少GPU global memory会损失指针追溯能力，首轮不默默裁掉；若为容量而裁剪，必须声明。不要同时挂cuda-gdb或Compute Sanitizer然后假设同一core机制仍有效，按工具限制择一路径。离线可用`cuda-gdb`的`target cudacore <文件>`，检查`info cuda kernels`、异常线程及backtrace；CPU core由gdb检查`thread apply all bt full`。core含进程内存，不提交公开仓库。[CUDA-GDB core说明](https://docs.nvidia.com/cuda/cuda-gdb/index.html#gpu-core-dump-support)。

本机工具不在默认PATH，但已实测存在 `/usr/local/cuda-13.2/bin/compute-sanitizer`（2026.1.0.0）和`cuda-gdb`，无需据`command not found`断言未安装。

### 第三层：有界的Compute Sanitizer复现

目标是捕获第一个非法访问的kernel、指令偏移、读/写地址、线程/block、主机/设备栈。拟用memcheck、`--target-processes all`、`--show-backtrace yes`、`--error-exitcode 99`和唯一`--log-file`；先核对现场工具`--help`，再组装获批的复现入口。`--generate-coredump`在新旧版本参数形式有差异，不照抄旧的`yes`写法。[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)。

本仓库9月9日全量插桩曾因资源/时限不足而只跑到step1，NCCL-only过滤跑到6步TIMEOUT，均无有效通过结论。不能原样重跑20步并宣称解决。先用core/HLO/首错范围建立能触发问题的最小案例，保留计算与通信覆盖；只过滤NCCL会漏掉XLA计算内核的越界。`loop_convert_fusion_19`是线索，不足以仅过滤该kernel后排除其余路径。

`CUDA_LAUNCH_BLOCKING=1`或增加`block_until_ready()`可做独立同步定位实验，但会改变并发/时序；不要默认长期打开。sanitizer显存开销、降低显存池/批大小、关闭图或换GPU数都算诊断条件变更，分别记录。无错误但超时、工具内部错误、未插桩kernel、未到故障窗口必须写`inconclusive`。

## 排查优先级与恢复门槛

1. 先保住现场；建议暂停“几秒psum通过即无限恢复”的策略，重复同类Xid进入人工诊断。当前自动恢复配置未在本轮修改。
2. 管理员检查GPU1/52:00的硬件、插槽/供电/链路和主机mc0/dimm2，导出本次及前一次boot的内核/BMC/core记录。GPU1和DIMM问题分别建条目，不先假定共同因果。
3. 在独占的获批资源上，用同一小型复现对比GPU UUID并改变逻辑映射，观察错误跟随物理卡还是rank/代码路径；再区分单卡计算、双卡通信、原四卡训练。不能随手把FSDP4改成3卡继续原run。若单节点同卡异常持续，优先硬件/节点诊断，不先泛化为跨节点网络问题。
4. 若物理卡对照不能解释，再在隔离环境做驱动/JAX/NCCL的一次一变量对照；保留595.45.04/JAX0.5.3/NCCL2.26.2基线，不原地升级训练前缀。
5. 验证覆盖必须超过已观察故障窗口并有多次对照，同时核对checkpoint完整性和有限值；一次短测通过、loss有限、从checkpoint恢复成功都不能替代稳定性验收。

本轮完成的是定位调查方向与采集方案，不是根因修复。验证：收集器4项离线测试通过（凭据白名单、拒绝覆盖、失败/缺工具/超时保留、完整日志快照），CLI帮助通过；没有运行训练或训练smoke。新增工具尚未部署至服务器，NCCL/XLA/core/持续遥测方案尚未接入当前作业。
