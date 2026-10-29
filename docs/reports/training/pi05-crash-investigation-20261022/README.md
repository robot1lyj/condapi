# Pi0.5 服务器反复崩溃调查与现场采集（2026-10-22）

> 2026-10-29保留状态更新：用户明确要求删除27份历史CUDA崩溃转储，已执行，含本报告2598/2600原生GPU core。已有分析、散列与其他日志仍保留，服务器原转储路径已不可复读；当前2637 incident目录未删。精确删除清单及回执见[存储报告](../../server-storage-20261029/README.md)。

## 2600复核与2598累计对照（2026-10-23）

用户要求检查2600并累计两次原因后联系管理员。本次只读分析、保存证据，不提交训练、不取消作业、不发跨聊天或管理员消息；16:29观测队列已有2604在gpu001运行，未干预。两次训练代码均为`4959866c2eb51538fa92f1a4aed0d2110d2f7b32`，冻结training_contract、包metadata和驱动快照逐字节相同，均从本run完整7000恢复。[环境与合同对照](evidence/jobs-2598-2600-environment-comparison.json)。

**两次均已抓到可解析GPU core，直接故障特征相同：同一物理GPU1、同一硬件子单元、同一个CUTLASS BF16 GEMM、同一条偏移+2720的LDGSTS指令发生Warp Out of range Address，随后SIGABRT。** 这比只比较Slurm的6:0有明确得多的定位；共同根因仍需物理卡/节点或软件对照确认，不能把重复签名直接写成已证明坏卡。

| 对照项目 | job2598 | job2600 |
|---|---|---|
| 运行时间（10月23日+08） | 07:20:46–07:51:02 | 08:20:53–14:14:33 |
| 内核异常时间 | 07:41:16.593 | 14:04:55.623 |
| 最后10步汇总completed指标 | 7301 | 14411 |
| 逐步检查实际最后完成 | 7307 | 14415 |
| 实际失败尝试步 | **7308** | **14416** |
| 此attempt正常完成更新数 | 307 | 7415 |
| 故障卡 | GPU1 / PCI52:00.0 | GPU1 / PCI52:00.0 |
| 内核Xid位置 | Xid13 / GPC8/TPC0/SM0 | Xid13 / GPC8/TPC0/SM0 |
| CUDA-GDB逻辑位置 | device1 / SM12 / warp0 | device1 / SM12 / warp0 |
| 故障kernel | 下述同一CUTLASS kernel | 下述同一CUTLASS kernel |
| errorpc函数偏移 | **+2720 / 0xaa0** | **+2720 / 0xaa0** |
| 错误指令 | 下述同一LDGSTS指令 | 下述同一LDGSTS指令 |
| 焦点block | (18,1,1) | (39,1,1) |
| grid / block维度 | (256,3,2) / (128,1,1) | (256,3,2) / (128,1,1) |
| 原生退出 / Slurm | -6、SIGABRT / FAILED6:0 | -6、SIGABRT / FAILED6:0 |
| GPU core字节数 | 103,996,497,694（约104GB） | 98,597,930,094（约98.6GB） |

疑似设备UUID为`GPU-7a9a29a1-9059-b0ef-d7b4-53c298565ca1`。两份core均由CUDA-GDB13.2离线打开，没有attach活动进程；两次同一kernel和指令如下：

```text
cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>
+2720: LDGSTS.E.BYPASS.LTC128B.128 [R79][R14.64],P0
```

2600的errorpc为`0x7f7ac1536ca0`，停止时pc为`0x7f7ac1536de0`（+3040）；2598的errorpc为`0x7f66d652ffa0`。跨进程绝对地址不同，比较的是函数内相同偏移，不能只比较绝对PC。两次core的其他三张卡均在`ncclDevKernel_AllReduce_Sum_f32_RING_LL()`，报异常的GPU1均在计算kernel。2600的Python栈位于`numerical_diagnostics.py:284 → inspect:162 → jax.device_get`，这是读取梯度统计的同步位置；core实际指向CUTLASS计算，不能把host报错位置直接认作统计器自身发生越界。[2600 core解析/寄存器/反汇编](evidence/job-2600-cuda-gdb.txt)、[2598 core解析](evidence/job-2598-cuda-gdb.txt)、[2598故障指令](evidence/job-2598-cuda-gdb-fault-instruction.txt)、[两作业Slurm原文](evidence/jobs-2598-2600-slurm.txt)。

### 2600的数值与输入对照

14:04:50.469，14415的完整更新检查通过：loss0.00756020、原生grad_norm0.08314363、稳定范数0.08314364，620个参数/optimizer物理分片和204个updates分片有限；7001–14415全部7415次更新未记录非有限值或副本统计不一致。14:04:50.594，14416输入40个分片检查通过，随后没有该步`gradients_checked`/`update_finished`，该步optimizer尚未调用。原生abort没有回到Python异常导出逻辑，因此没有`numerical/failure-step-*`，不能声称已有14416完整独立NumPy梯度包。[数值日志全量扫描/末步身份](evidence/job-2600-numerical-summary.json)。

14416的32条样本身份、loader SHA`09a8250015109b920544db72b4a653090c1015644f5580d5d5667fa3f77413b1`和RNG`[1202269147,2537917816]`均已保存。另一个关键对照是：**2598崩溃的7308，在2600于08:43:01正常完成；记录的batch身份、输入SHA和RNG完全相同。** 该次loss0.00631518、grad_norm0.07670385，输入/梯度/更新/完整state都通过。这削弱“固定坏输入每次必崩”的解释，但没有逐值比较两次完整模型/optimizer，因此不宣称所有执行状态完全相同，也不能排除时序相关软件问题。[7308逐项对照](evidence/jobs-2598-2600-step7308-comparison.json)。

2600四卡各4226次GPU采样，最高温73/74/73/72°C，GPU1最高功耗326.16W；SW/HW热降频累计值在作业前后不变。主机EDAC的mc0 CE仍3,781,237、全部UE仍0，本窗口无增量。5秒采样不能排除瞬态，现有记录不足以把过热或CPU纠错事件作为两次直接原因。core没有可用源码行号/参数符号，尚未唯一映射到模型层。

2600的NCCL原件为19,787,134,857字节，使用服务器可用的grep对整个文件扫描`WARN|ERROR|Cuda failure|illegal|abort`，返回1（无匹配）、stderr为空；没有把抽样tail当全量结果。该19.8GB原件没有另做全量SHA，仅记录大小/mtime/首64KiB哈希，保留服务器原件。XLA目录1879个文件，主机4240次采样，作业窗口内核仅记录上述三条Xid13。原生PID3624406在14:14:28.540以-6/signal6退出，wrapper未收到取消信号；core最后修改时间为14:14:25.371。详细[系统与NCCL扫描结果](evidence/job-2600-system-summary.json)可与两份core交叉核对；未报告NCCL错误不等于排除所有通信静默问题。

### 给管理员的结论与现场位置

建议表述为：**gpu001的GPU1已在两次独立训练进程中重复发生完全一致的设备/内核/指令级异常，优先申请硬件检查和同型号卡替换对照；硬件稳定性为首要怀疑，驱动/计算库缺陷仍未排除。** 请检查该卡本体、供电/插槽和BMC记录；保持四张卡与原训练合同做替换或换节点验证，观察故障是否跟随该物理卡。操作前协调停训窗口、保留两份core；本次没有自行执行换卡、GPU reset、驱动升级或新训练。

两次现场位于同一control目录下的`incidents/job-2598`和`incidents/job-2600`：

```text
/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/incidents
```

2600的完整逐步日志758,704,816字节已全量扫描并计算SHA256；完整原件留在服务器，没有截断或删除。约21.9MB的配置/系统日志/训练日志/分片schema已复制本地忽略目录，归档全量SHA归[原件清单](evidence/job-2600-raw-manifest.json)。98.6GB core和19.8GB NCCL原件不复制到Git，core仅记录大小/mtime/首64KiB哈希并验证可解析，不冒称做了core全量哈希。[2600 Xid原文](evidence/job-2600-kernel.txt)、[末次汇总指标与原生栈](evidence/job-2600-failure.txt)。

## 2598 GPU原生异常：已取得可解析core（2026-10-23复核）

用户报告“job2598在step7301后以exit6:0失败”后，只读核查该attempt并保存证据，未重启/取消训练、修改配方或发送跨聊天指令。09:48查询时另有2600在gpu001运行，本次没有干预。2598实际源码仍为`4959866c2eb51538fa92f1a4aed0d2110d2f7b32`，诊断/缓存保护/四GPU自检生效，从完整7000恢复。

**本次已定位为尝试完成7308时的GPU原生越界异常，退出为SIGABRT；不是已记录到的NaN/Inf保护退出。** Slurm显示07:20:46–07:51:02、FAILED、`6:0`；子进程PID1690708的`exit.json`明确`returncode=-6, signal=6, wrapper_received_signals=[]`。7301是最后一条每10步汇总的completed指标；逐步日志完整记录7001–7307共307次更新，输入/梯度/实际updates/全部参数和optimizer均有限。7308的输入已通过检查，未出现该步`gradients_checked`或`update_finished`；原生栈停在`numerical_diagnostics.py:283`的gradient_function调用内，故障步尚未调用optimizer。

| 时间（2026-10-23 +08） | 现场事实 |
|---|---|
| 07:41:11.712–11.862 | 7307梯度与更新后状态检查完成；loss0.00818456、原生grad_norm0.09573331、稳定范数0.09573330；620个state物理分片有限 |
| 07:41:11.928–11.956 | 7308开始，32条真实样本身份、输入SHA、RNG落盘，40个输入分片有限 |
| 07:41:16.593 | 内核记录GPU1/PCI52:00的Xid13，GPC8/TPC0/SM0，Out Of Range Address / Multiple Warp Errors |
| 07:50:57.436 | GPU core文件最后修改时间；不能把文件名中的时间字段当作本次异常发生时刻 |
| 07:51:00–07:51:02 | 原生进程SIGABRT，wrapper完成after补采，随后Slurm结束；故障至退出约10分钟与core生成窗口相符，不是继续正常训练 |

### core给出的直接定位

本次保留`103,996,497,694`字节（约104GB/96.9GiB）的`.nvcudmp`，使用节点已有CUDA-GDB13.2以`target cudacore`离线成功打开，没有attach当前训练进程。得到：

- 故障设备1，PCI`52:00.0`，UUID仍为`GPU-7a9a29a1-9059-b0ef-d7b4-53c298565ca1`；异常为`Warp Out of range Address`。
- 内核为`cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_tn_align8>`，属于BF16 GEMM计算；grid`(256,3,2)`、block`(128,1,1)`。调试器焦点为block`(18,1,1)`、SM12/warp0/lane0；这里的CUDA逻辑SM编号与内核日志的GPC/TPC/SM层级分别原样保留，不自行换算。
- 异常PC=`0x7f66d652ffa0`，函数偏移`+2720`，指令为`LDGSTS.E.BYPASS.LTC128B.128 [R79][R14.64],P0`。停止时PC另为`+3072`，已分别保留errorpc与pc，不能混用；寄存器与相邻反汇编已保存。
- 其他三卡当时的活动kernel为`ncclDevKernel_AllReduce_Sum_f32_RING_LL()`。这支持把GPU1计算异常作为本次直接定位，不能由其他卡在collective中等待反推NCCL是首因。

[CUDA-GDB完整解析](evidence/job-2598-cuda-gdb.txt)、[故障指令/warp现场](evidence/job-2598-cuda-gdb-fault-instruction.txt)。当前core没有可用源码行号和参数符号，尚未映射到唯一模型层/HLO矩阵；第二批gdb命令的exit1来自13.2不支持`info cuda kernel_launch_backtrace`，之前core/寄存器/反汇编均成功。NVIDIA将该异常定义为warp访问局部或共享内存有效范围之外；该定义说明故障类型，不证明软件或硬件中的唯一根因。[CUDA-GDB异常说明](https://docs.nvidia.com/cuda/cuda-gdb/index.html#cuda-exception-codes)。

### 数值、系统证据与缺口

完整逐步日志31,463,147字节；7308 batch SHA为`12ba06607f6f4a53b8c41d38e00224a64c5795df464527dd3fe16de4e2431d8c`，step RNG为`[4052012641,2720522280]`，全部episode/frame/index已保留。[末步身份与数值](evidence/job-2598-numerical-summary.json)。此次原生abort没有返回Python异常处理，因而没有`numerical/failure-step-*`张量导出包；GPU core、样本身份/hash和此前分片统计已经留下，不能声称有故障步的完整独立NumPy梯度/模型包，也不能由输入有限推定所有内部激活健康。

- NCCL原件817,293,594字节、4,170,320行，全量扫描WARN/ERROR/Cuda failure/illegal/abort为0；已存完整SHA256，源文件未改动。这仅说明NCCL没有报告错误，不证明所有通信数据一定正确。
- 四卡各361次采样，最高温分别77/78/77/76°C；本次前后SW/HW热降频累计计数完全相同，未见本窗口新热降频。GPU1最高功耗338.49W。5秒采样不能排除瞬态供电/温度问题，不能直接把78°C当过热根因。
- 362次主机采样中，mc0 CE始终3,781,237、所有UE始终0，无本窗口新增EDAC。内核本作业窗口只有上述三条Xid13，没有新增OOM/MCE记录；after快照/home尚余2.7TiB。CPU journal/core仍因权限不可读。
- XLA目录596个文件，原始HLO/PTX等保留。core仅记录路径/大小/mtime和首64KiB哈希，明确**未对104GB文件计算全量哈希**，避免在另一训练运行时额外全盘扫描。约39.9MB的完整小型日志/配置/逐步记录已复制本地忽略目录并计算归档全量SHA；大core、NCCL及XLA原件仍在服务器。

[系统/NCCL扫描汇总](evidence/job-2598-system-summary.json)、[Xid原文](evidence/job-2598-kernel.txt)、[末次指标与原生栈](evidence/job-2598-failure.txt)、[原件归档清单](evidence/job-2598-raw-manifest.json)。现场根目录为：

```text
/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/incidents/job-2598
```

**当前判断：同一物理GPU1/同一GPC8子单元持续重复异常，且这次捕获到了具体计算kernel与出错指令，优先排查GPU1硬件稳定性及其驱动/计算执行路径。** 尚不能直接定为坏卡，也不能因为kernel名含CUTLASS就判定CUTLASS实现有bug。下一步最有区分力的是保留四卡/FSDP4合同，做物理卡替换或节点/设备映射对照，观察异常是否跟随该UUID；再围绕已捕获GEMM做有界复现与软件版本对照。本次未执行这些新实验，也未改动当前2600。

## 数值现场实现与重新启动授权

用户要求“记录代码写好，然后开始重新启动训练”，追加“确保写全写好”。本次沿同一run和已确认训练合同增加诊断，不改变batch32/FSDP4/workers2、选集/norm、LR/RTC、目标本地86000或checkpoint保存合同。启动前动态确认唯一活动作业、最新finalized checkpoint和norm一致性，保留旧现场后串行切换；不新增聊天监控或自动恢复。

实现入口：`scripts/train.py`复用原生loss/gradient及optimizer函数，诊断关闭仍为原来单次JIT路径；开启`OPENPI_NUMERIC_DIAGNOSTICS_DIR`后在原生循环内分开gradient与update，以host guard确保失败步不会继续调用optimizer。编译/同步和显存生命周期会变化，属于诊断开销，不把该路径的稳定性直接当原路径的修复证明。

- 每步的`attempt_started`记录精确尝试步、真实32条dataset/index/episode/frame/task/timestamp、loader实际张量SHA256，以及按step fold后的RNG；metadata在变换前保留、在输入模型前移除，不改变模型输入。SHA明确对应Observation.from_dict之前；异常导出的是实际进入forward的batch，包含图像/state/action/token/mask。
- `input_checked`、`gradients_checked`、`update_finished`按实际addressable shard记录NaN/+Inf/-Inf数量、最大有限绝对值、首异常和最大值的局部索引/数值、稳定缩放平方和；schema固定参数路径、全局/局部shape、dtype、分片全局范围、device ID/replica ID。设备ID与外部GPU UUID/PCI快照关联。完整统计先在各设备本地算，复制小表到CPU；不把整模型all-gather到单张卡。
- 原生grad_norm与CPU float64组合的稳定范数并存；复制分片只在逻辑范数中计一次，但每一物理副本的异常独立保留，不能被去重掩盖。输入/梯度非有限及原生范数非有限均在optimizer调用前停止。优化器的实际updates及所有参数/optimizer叶子在更新后再验，不只看旧param_norm筛选的kernel。
- 故障先原子保存`failure.json`并fsync，再逐叶/逐设备写实际batch、根RNG、梯度（若尚在）、参数、optimizer和updates（若已有）；`.npy`禁止pickle，BF16按uint16原始位存并注明编码。每个文件有shape、全局分片索引、原dtype、字节数、SHA256。`tensor-manifest.json`逐文件更新；只有全部写完才有`COMPLETE`。这是故障包，不是正式checkpoint，带`NOT_A_TRAINING_CHECKPOINT`，不自动恢复。
- **边界明确：** pre-update异常保留实际更新前state；post-update或optimizer运行期异常可能已donate旧buffer，记录明确phase，不能声称保留旧state。SIGKILL/掉电/设备不可读也可能阻止张量导出，manifest保留partial/error；外层CUDA/Xid/core及退出采集继续提供证据。不承诺抓到模型内部每一层激活，先用输入、实际梯度及更新定位坏叶子后再细化反向算子。
- `steps.jsonl`逐事件flush/fsync。诊断作业禁用JAX持久编译缓存，保留进程内JIT复用与完整新编译图；不删除共享缓存。self-check与真实训练XLA目录独立。外层CUDA/NCCL/内核/GPU遥测仍工作。

本地只做纯数组、序列化与stub协议检查：17项通过，包含稳定大数范数、坏副本不被隐藏、身份/内容hash、严格JSON、拒绝覆盖、失败/超时留痕、输入/梯度异常时optimizer绝不被调用、post-state语义与正常路径各调用一次；Ruff通过。未在本地运行模型前向、梯度或训练smoke。新Slurm作业内先运行`check_numerical_diagnostics.py`，仅用四GPU小数组验证正确故障卡/分片定位、有限大数的范数溢出、复制去重、真实异常张量导出/读回，以及实际Orbax保存/恢复后的分片设备绑定；该检查不创建模型或执行优化器，通过后才进入原生正式续训。`numerical-self-check/failure-step-1001`是自检故意注入的异常包，不能当作正式训练故障。

### 诊断启动记录与交接

固定诊断版本为`4959866c2eb51538fa92f1a4aed0d2110d2f7b32`，从Gitea取得并在独立稀疏checkout `/home/wuyan/lyj/YAM/code-snapshots/pi05-numerical-evidence-v3-20261022`执行。它是用户明确批准的数值取证版本，训练配方沿原合同不变；不能仅因训练源码不再是旧`b1446ee`自动取消。

- 原2565已发布完整2000（params/train_state/assets齐备，norm JSON与冻结选集资产逐值一致）；16:10授权切换时保留原现场再取消。
- 首版2566（`58578bea`）在恢复后的初始统计处退出，尚未执行新优化器更新：Orbax恢复的未绑定buffer被统计JIT错误派发至cuda:0。这是新增诊断代码的启动缺陷，**不是原训练再次发生硬件/数值故障**。`bdccf199`为每个实际设备固定统计JIT，并增加真实Orbax保存/恢复自检；该修正仍未覆盖持久缓存命中，见2572/2573结果。
- 2567使用修复版，四GPU及Orbax恢复自检通过，但16:23:52被另一监控聊天按旧代码规则取消，尚未进入更新。之后该聊天另起2571。用户选择自行通知其暂停，并明确“他已经停止了，你直接开始吧”；本聊天没有向其他聊天发送指令。
- 2571在原代码上运行到2000之后，16:31:24由本聊天按切换授权取消；完整交接现场保存在同control的`incidents/handover-job2571-numerical-20261022`。已确认extern结束、队列清空、launch.lock可取得后才提交2572。未删除旧现场/数据/权重。
- 2572于16:32:00 +08提交，gpu001，训练PID3943158，同run从2000恢复。小数组/Orbax自检通过，但16:35:49真实恢复状态的统计仍报cuda:1 buffer与cuda:0 executable不匹配；退出1，未执行新更新。此前“固定device即可修复”的假设不足。
- **2573为独立6秒纯数组对照，无模型/优化器：** 将缓存写入阈值设为0后，旧采集器在持久缓存开启时稳定复现同一错误（rc1），关闭时四GPU/Orbax/导出读回全部通过（rc0）。失败目录内仅有一个`jit_local_statistics`缓存条目，现场JAX 0.5.3的GPU缓存键会剥离device assignment；设备绑定复用是此次采集器启动缺陷的已复现机制，不把它当成原训练Xid/Inf的根因。新版本在wrapper与诊断入口明确禁用磁盘缓存，保留内存JIT复用，不改共享缓存文件；自检主动启用缓存/阈值0再验证保护生效。[对照原始输出](evidence/cache-placement-regression.txt)。

[首次启动回执](launch/numerical-submission.json)、[首次启动过程](launch/numerical-startup-progress.json)、[修复版2567回执](launch/numerical-v2-submission.json)、[四GPU与Orbax自检](launch/numerical-v2-self-check.json)、[作业分类原始记录](evidence/numerical-attempts.txt)、[2572提交回执](launch/numerical-v2-retry-submission.json)。完整失败原件保留在服务器各job目录及本地忽略的`evidence/raw/`，不把core或模型张量推送Git。

**最终启动验收：2574，gpu001，训练PID3976967。** 16:41:55提交，16:49:55 +08核验仍RUNNING，从2000连续完成2001–2041共41次真实更新；不是只检查进程存在。每步32条身份/输入hash/RNG齐备，输入40、梯度204、更新量204、全部参数/optimizer620个物理分片均有限，未记录坏副本。原生与稳定梯度范数最大相对差约1.46e-7；初始编译后平均更新间隔2.52秒（早期观察，不是长期吞吐保证）。本次恢复的全部state也通过有限值检查。

训练源码为上述`4959866`，两远端已核对，服务器只从Gitea部署。自检新增缓存保护通过；真实进程环境确认`JAX_ENABLE_COMPILATION_CACHE=false`、NCCL INFO及CUDA core开关生效。XLA目录已有596个文件，含真实梯度/更新编译的HLO、LLVM、PTX等206个产物，文件名/大小/SHA归验收JSON；不再只有小数组图。原日志保留了编译期lexer和rematerialization警告，随后实际更新正常；不把警告改写成不存在。kernel-follow虽含约1.2MB历史缓冲，末条仍是14:04:54旧Xid，验收时无新内核记录。CPU journal/core权限及原生异常时张量可能不可读的限制仍存在。

新现场根目录（服务器）：

```text
/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/incidents/job-2574
```

交接时优先读`numerical/steps.jsonl`的最后完整事件，关联`trainer.log`、`events.jsonl`、退出时的`exit.json`。数值异常看`numerical/failure-step-*/failure.json`和`tensor-manifest.json`；只有`COMPLETE`才表示全部列出的张量导出成功。CUDA/NCCL/XLA及GPU/主机/内核证据在该根目录。`numerical-self-check/`是故意注入异常的自检，不能误判正式训练崩溃。监控由用户安排，本聊天验收后停止轮询，不自动重启、不发送跨聊天指令。当前完整恢复点仍为2000；2041是已完成更新，不是新checkpoint，也不证明之前的训练根因已修复。

[2574提交回执](launch/numerical-v3-submission.json)、[完整启动验收](launch/numerical-v3-startup-verified.json)、[真实运行manifest](launch/numerical-v3-manifest.json)、[加强后的GPU自检](launch/numerical-v3-self-check.json)、[原件快照哈希](evidence/job-2574-raw-manifest.json)。

## 2563失败复核（2026-10-22 15:21起，只读）

用户报告再次崩溃后，按本次请求读取事故包；没有启动训练、修改当前配方、恢复自动化或向其他agent发消息。15:21队列为空。**本次直接退出原因已确定为数值保护：completed step1491检测到聚合`grad_norm=inf`，抛出FloatingPointError；原生trainer退出码1、signal=null，包装器未收到终止信号。** Slurm于15:02:58结束，elapsed22:31。该事件不能继续归类为SIGSEGV或已证明的CUDA非法地址。

异常指标为`grad_norm=inf, loss=0.00729946, param_norm=1803.7782`。最后正常结构化指标在completed1481；日志每10步聚合一次，因此首个异常更新只能限定在 **completed1482–1491**，不能把1491直接认作唯一坏batch。原始日志与退出记录已完整保存，见[失败摘录](evidence/job-2563-failure.txt)、[汇总](evidence/job-2563-summary.json)、[原件清单与哈希](evidence/job-2563-raw-manifest.json)。

### 新现场支持与不支持什么

| 证据 | 本次观察 | 能支持的结论 |
|---|---|---|
| 内核follow与退出后dmesg | 本次14:40:27启动后无新增记录；末条仍为14:04:54旧作业Xid | 本次没有记录到新Xid/MCE/OOM；不能借旧Xid认定本次是GPU1故障 |
| NCCL完整日志 | 1,305,029,845字节、6,655,548行，全量扫描WARN/ERROR/Cuda failure/illegal/abort匹配0；通信连接记录为SHM/direct/direct | 没有NCCL报告的运行错误；日志不校验通信数据的数值正确性，不能排除静默错误 |
| GPU遥测 | 每卡269次采样；四卡最高温分别69/70/69/68°C，功耗最高321.19/318.69/324.38/309.93W；热降频计数0 | 没有采样到过热或热降频；5秒采样不覆盖所有瞬态 |
| CPU内存/EDAC | 270次采样，CE一直3,781,237、UE一直0；末次MemAvailable约479GiB | 本次窗口无新报告的主机纠错事件，内存容量充足；既有DIMM问题未被消除 |
| 存储/共享内存 | /home剩余3.9TiB、/dev/shm仅约504KiB已用 | 无空间耗尽证据；1.3GB NCCL日志尚未造成容量故障 |
| GPU core与XLA | 无`.nvcudmp`；170个XLA文件共140,012字节，仅17个小型模块，没有完整train_step模块 | 普通浮点Inf不等于CUDA设备异常，无GPU core符合本次退出类型；当前dump不足以定位反向计算。训练入口启用共享持久编译缓存，缓存命中是缺失主训练图的候选解释，未从日志直接验证命中 |

[NCCL扫描/文件清单/哈希](evidence/job-2563-nccl-scan.json)绑定服务器保留的1.3GB原件；本地保存小型XLA原件归档，不删除或压缩替换服务器源文件。新增采集确实区分出了失败类型，尚未保存故障步的完整输入、原始梯度或优化器状态。

### 为什么还不能直接断言“梯度爆炸”或“坏卡”

核对固定训练源码b1446ee：`train_step`先调用优化器更新，再返回`optax.global_norm(grads)`；这个指标针对**裁剪前原始梯度**。`param_norm`只覆盖筛选后的二维及以上kernel参数，排除了bias、scale、pos_embedding和input_embedding。因此loss与param_norm有限，既不证明所有梯度有限，也不证明所有参数/优化器状态完整健康。

现场Optax0.2.4的`global_norm`为`sqrt(sum(sum(abs_sq(x))))`，并非避免溢出的缩放范数。因而`inf`至少有两种未区分情形：原始梯度叶子已经含Inf；或者叶子仍有限但平方/求和等归约发生溢出。还缺少每叶finite计数、最大绝对值及稳定范数对照，不能把其中一种补写成已证原因。10步均值归约本身也未保存输入，无法仅凭聚合标量定位首个坏算子。

上一作业2562同样从1000恢复，日志printed1490（completed1491）的`grad_norm=0.0849、loss=0.0073`，并继续到1651；[对照摘录](evidence/job-2563-prior-comparison.txt)。这不支持“只要到1491必崩”的确定性解释，但没有逐batch身份/增强结果留存，不能把同合同等同于实际输入逐值相同。新增诊断开关和执行时序也是本次运行差异。

进一步回查此前完整旧train.log，在10月20日起按前一正常指标时间归属的窗口内，已有 **10条数值保护异常**：5条只有grad_norm为Inf，5条grad_norm/loss/param_norm为NaN。旧108351就出现`grad_norm=inf、loss=0.00896384`。因此这种症状在本次新增采集之前已经存在；[历史证据](evidence/job-2563-historical-numerical.json)。这不证明所有数值异常共享一个原因，但必须修正此前将大量失败笼统解释为原生GPU异常的倾向。

当前判断分两条保留：历史8组同GPU1/GPC8的Xid仍支持优先做设备/软件路径隔离；2563属于数值异常分支，尚不能在算法数值不稳定、编译/归约问题、静默计算/通信错误之间定责。没有本次新增Xid并不证明硬件健康，也不能反向证明硬件导致这次Inf。

### 下一步需要补的数值证据

在获准的下一次诊断中，沿原生训练器增加可关闭的数值取证：每步保存step/RNG/batch身份；在optimizer更新前计算各梯度叶子的finite计数、最大绝对值、dtype和按设备分片摘要；增加稳定范数对照，首异常标记精确参数路径和首个坏step。验证输入state/action与相关激活的finite情况，并区分clip前/后梯度、参数和optimizer状态，故障样本/状态写独立诊断目录，不污染正式checkpoint。主机日志每10步聚合不是逐步异常检测的替代。

为取得完整训练HLO，可在另一个诊断缓存目录中编译同一固定图，不删除共享缓存。再根据首坏叶子/设备决定最小数值复现或物理GPU对照。不能用降低LR、跳过异常更新、关闭finite保护或只增加NCCL verbosity替代定位。本次没有生成新的训练配置/命令，也没有运行训练或本地训练smoke。

最新完整checkpoint仍为1000，未生成2000；故障时内存中的状态没有作为新恢复点保存。当前事故包、旧原件和checkpoint均保留，后续监控/重启仍由用户另行安排。

## 接管后交接（2026-10-22 14:43 +08，历史启动快照）

用户先明确授权“接管服务器，可以取消目前的训练，然后用你有现场采集工具的开始训练”，随后明确“不需要你来监控，我让别的agent监控的，你只需告诉我进程即可”。已停止后续主动轮询，没有创建新监控/自动恢复，也没有向其他聊天发送任务。

- **Slurm2563，gpu001，训练PID1594045**；14:40:27启动，14:43单次交接核查为RUNNING。当时尚未出现首个新训练指标或NCCL/XLA输出，不能宣称已完成恢复/编译或稳定性验收；其他agent需按实际日志继续核验。
- 原2562于14:39:21取消，末次指标1651；交接前完整现场保存在同control的`incidents/handover-job2562-20261022`。Slurm batch以SIGTERM结束，extern已结束、队列清空、launch.lock可取得。分配释放后SSH到gpu001被pam_slurm_adopt拒绝；新作业before快照的GPU进程表为空，未见残留计算进程。
- 保持同run从完整1000恢复，保留参数/优化器/loader合同，目标本地86000、逻辑偏移114000。checkpoint norm和选集norm的JSON逐值相等，序列化字节哈希不同已核对；原训练norm SHA仍05f65ada…，未改batch32/FSDP4/workers2/RTC/LR/选集/代码。
- 原训练代码固定`b1446eefdf8c3cd347689bb41cc77fa6af251aaa`；诊断代码固定`43e9bc0daa8a9a0c1c0e2d219442b19d54e93b9c`，从Gitea经验证主机密钥的SSH隧道取得。诊断checkout为`/home/wuyan/lyj/YAM/code-snapshots/pi05-incident-tools-sparse-20261022`，训练仍在原checkout执行。全量worktree首次checkout较慢，改用三文件稀疏checkout完成；旧目录保留。
- GPU遥测PID1594043、内核日志PID1594044是作业内采集子进程；每5秒记录GPU/主机/EDAC，每60秒保存真实trainer环境/加载库。原生退出单独写`exit.json`并补采，不自动重启。它们属于故障取证，不是新的聊天监控任务。
- 原自动化 **Pi0.5耳机训练监控（pi0-5-50h）已PAUSED**，避免其旧恢复命令重复拉起。用户安排其他agent监控，本聊天不恢复旧自动化、不新建自动化。

现场目录（服务器）：

```text
/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/incidents/job-2563
```

其中`trainer.log`为完整训练日志，`events.jsonl`/`exit.json`为生命周期，`gpu-telemetry.log`、`host-telemetry.jsonl`、`kernel-follow.log`、`process-snapshots.jsonl`为连续证据；NCCL写`nccl.<host>.<pid>.log`，XLA写`xla/`，GPU异常触发`cuda.<host>.<pid>.<time>.nvcudmp`。实际环境已确认启用这些开关，但没有人为制造GPU异常来验证core生成。退出/失败后先读现场，再决定是否恢复。

节点本地`sacct`连接localhost:6819失败，需从`yam-server`登录节点读取；当前作业已另取登录节点Slurm回执。节点journal/coredump访问权限仍未解决，首版收集器还遇到Rocky journal时间参数不接受ISO的T分隔符；当前源码已修正，未修改正在运行的固定43e9bc0。当前核心内核证据由已实际写出的dmesg快照和follow提供。未来节点掉电/SIGKILL仍可能跳过after补采，需要监控agent在登录节点补取调度器结果。

[提交回执](launch/submission.json)、[单次进程/环境核验](launch/startup-verified.json)、[原作业取消](evidence/handover-cancel.txt)、[固定作业文件](launch/diagnostic-resume.sbatch)。7项离线合成进程/收集测试及Ruff通过，未在工作站运行训练。

## 结论与适用范围

初始只读调查 `yam-server → gpu001`，观察时间为北京时间2026-10-22 14:19–14:24；该调查阶段未改变运行。随后用户授权接管、取消旧训练并带现场采集续训，实际操作见下节。原始源日志保留，本地完整副本位于 `evidence/raw/`；Git保存来源/哈希、关键摘录和统计。仅涉及服务器训练，不涉及Thor/3588。

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

新工具 [collect_training_incident.py](../../../../scripts/collect_training_incident.py) 是标准库只读快照工具：采集GPU、内核、Slurm、EDAC、core目录查询、指定PID白名单环境/加载库、包metadata和显式日志/配置；每项保存退出码、错误与时间，拒绝复用输出目录。它不训练、不发信号、不sudo、不改GPU参数，也不自动部署到服务器；本次部署是后续用户明确授权的操作。复制增长日志只读开始时的文件长度，并注明复制期间是否变化。

部署该工具后，在**故障计算节点**执行，变量须由本次实际作业填写：

```bash
python3 scripts/collect_training_incident.py \
  --output "$INCIDENT_DIR" --since "$INCIDENT_START" \
  --job-id "$JOB_ID" --pid "$TRAINER_PID" \
  --python /home/wuyan/.conda/envs/condapi-yam/bin/python \
  --log "$TRAIN_LOG" --log "$TRAINING_CONTRACT"
```

这是一次快照，不是持续采集器；需要在启动前、首错后各做一次。进程退出后`/proc`证据会消失；必须在启动时存PID/库/环境，不能依赖退出trap。收集器不自动保存任意cmdline/全环境以避免凭据；提交脚本/日志须检查后再共享。现有本地原始日志和core不进Git。

连续采集建议放在受控的节点日志服务中：持久journal/rsyslog保留每次boot；`dmesg --follow`仅为普通用户可用的临时补充，不清空缓冲。训练启动时启动限定生命周期的GPU遥测；退出先冻结事故包，再评估恢复。退出trap处理不了SIGKILL、节点掉电、OOM杀死整个cgroup，需登录节点依据Slurm结束事件做独立补采，节点持续日志与训练作业解耦。当前已接入作业内采集，未安装登录节点独立补采服务；后续交由用户指定agent监控。

## 更细的CUDA/XLA/NCCL现场：按层启用

以下保留分层诊断设计；按用户后续明确授权，第一/二层采集已加入2563，第三层sanitizer未运行。涉及其他训练/插桩实验先明确配方、GPU分配、时限和输出目录；DAgger轮次遵循逐轮确认规则。

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

1. 先保住现场；建议暂停“几秒psum通过即无限恢复”的策略，重复同类Xid进入人工诊断。旧自动恢复已在后续用户授权接管时暂停，详见交接。
2. 管理员检查GPU1/52:00的硬件、插槽/供电/链路和主机mc0/dimm2，导出本次及前一次boot的内核/BMC/core记录。GPU1和DIMM问题分别建条目，不先假定共同因果。
3. 在独占的获批资源上，用同一小型复现对比GPU UUID并改变逻辑映射，观察错误跟随物理卡还是rank/代码路径；再区分单卡计算、双卡通信、原四卡训练。不能随手把FSDP4改成3卡继续原run。若单节点同卡异常持续，优先硬件/节点诊断，不先泛化为跨节点网络问题。
4. 若物理卡对照不能解释，再在隔离环境做驱动/JAX/NCCL的一次一变量对照；保留595.45.04/JAX0.5.3/NCCL2.26.2基线，不原地升级训练前缀。
5. 验证覆盖必须超过已观察故障窗口并有多次对照，同时核对checkpoint完整性和有限值；一次短测通过、loss有限、从checkpoint恢复成功都不能替代稳定性验收。

本轮完成了调查、采集工具和授权后的诊断续训启动，不是根因修复。收集器/包装器共7项轻量测试和Ruff通过；只在服务器Slurm中启动原生训练，未在工作站运行训练或训练smoke。后续监控交由用户指定agent。
