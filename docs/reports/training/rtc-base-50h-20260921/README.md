# 50h Pi0.5 RTC 全量微调（2026-09-21）

状态更新（2026-09-28）：本run已在服务器运行并进入自动监控；当前作业状态须每次动态查询。沿用上一轮用户选择，从官方base重新训练，不继承20h模型。最新运行证据与恢复点记录在文末。

## 数据与训练合同

seed42，从原train选择完整episode；保留20h的941集，新增1396集，共2337集、5,400,685帧@30fps=50.006343h。原20h包含关系及源metadata哈希一致性已验证，原val不参与抽样。清单随机，不代表已逐集人工验收。

global batch32、4卡FSDP全量微调、H50、RTC d=0..10，关节delta/夹爪absolute；EMA关闭、Adam eps1e-6、seed42。沿用1e-5峰值、1e-6末端、warmup1000，cosine decay_steps固定338000，分段/恢复不重置。

两轮需要ceil(5400685×2/32)=337543步，设总目标338000，约2.0027等效轮。按训练帧起点计数，不能乘H50。独立50h norm由首次Slurm作业内计算并校验loader等价，不沿用20h统计。

旧入口324194硬上限不足以支持50h两轮，现改为正整数步数校验；此运行脚本额外限定最多338000。使用独立代码worktree `rtc-50h-20260921`，不改当前训练快照。

## 启动和中断恢复（服务器执行）

control：`/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921`。
预计2.28秒/步，全程约214小时/8.9天，实际包含存盘、编译和排队会更长。Slurm分区单次最多7天，本脚本申请6天，分两段正常收尾；不自动提交后继作业。

第一段，当前训练完成后手动提交，累计目标170000：
```bash
sbatch /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/run.sbatch start 170000
```
第一段中断，确认旧进程已退出、有完整checkpoint：
```bash
sbatch /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/run.sbatch --resume 170000
```
第一段正常完成后，第二段累计到338000（不是再训练338000）：
```bash
sbatch /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_50h_20260921/run.sbatch --resume 338000
```
第二段中断则重复第二段命令。只改变累计停止步数，学习率、norm、subset及优化器合同保持不变。恢复从最新完整checkpoint加载参数、优化器和step；不是从base重来。每1000步保存、长期保留每5000步及最新。若6天超时，同样检查最后完整保存后恢复，最多重做未保存更新。

首份checkpoint前中断没有可恢复进度；保留记录、查明原因后用新run重新准备，不删除合同绕过保护。NaN或数据错误需先诊断，不盲目重试。启动器flock阻止同run并发。

资产：`/home/wuyan/lyj/YAM/training-assets/lego_rtc_50h_20260921`；权重/metrics：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921`；日志在control。

## 验证与评估

准备阶段本地只做静态和轻量测试，没有训练循环；下述服务器运行及恢复是在获准GPU作业中执行。170k约一轮、338k约两轮，使用相同提示词、颜色格规则及部署控制版本比较抓取成功率和成功抓取后的分类正确率。扩数据不保证解决抓取偏差或语言/格子歧义。不同数据规模的LR衰减预算不同，属于整体训练方案比较，不是严格单变量数据量消融。

## 2026-09-28 · 恢复、故障证据与持续监控

### 服务器连接

已验证的 SSH 入口是 `yam-server`，即 `wuyan@10.18.31.234:22`，主机名 `rocky-login.hlink.local`。此前把地址误写为 `10.18.31.233`，因此该次连接失败；不是网页终端或服务器整体不可用。用户已在账户中安装长期受限公钥，当前非交互 SSH 已成功。之后从本机执行远程只读检查和 Slurm 操作使用 `ssh yam-server`，不得再尝试 `.233`。

### 前三次作业结果

- 2175：步 53,081 出现 NaN（grad norm/loss/参数范数非有限），FAILED。
- 2176：从完整 53,000 检查点恢复，步 68,021 的 grad norm 为 Inf，FAILED。
- 2177：从 68,000 检查点恢复；步 69,771 前指标仍有限（loss 0.01065、grad norm 0.08636、LR 9.1065e-6），随后 Python segmentation fault，并伴随 NCCL `ncclGroupEnd()` CUDA illegal memory access，Slurm 退出码 11。保存日志显示这是 CUDA/进程级故障；当时没有温度或 Xid 证据可确认热故障根因。

用户指出假期期间机房空调可能异常、服务器可能过热，目前环境预计改善。该信息是重要的排查假设，不是已证实根因；训练日志里的约 9.4 GiB GPU 占用也不是温度证据。重启后尽可能查看 GPU 温度/功耗、内核 Xid/CUDA 日志和训练宿主状态；若这些遥测不可用，要明确记录未观测，不能把用户的判断写成硬件诊断结论。

### 69,000 检查点与新恢复作业

2177 失败后作业队列与训练进程为空，`launch.lock` 无持有者。最新数值检查点是 `run/69000`，提交时间 2026-09-24 18:13:51 +08，早于 69,771 步故障；目录含 Orbax 根提交元数据、`params`、`train_state`（包括 optimizer state 与 step 元数据）及 assets，没有 `.orbax-checkpoint-tmp-*`。最新 metrics 行是 69,771。检查点 norm 与合同源 norm 的 JSON 值完全一致，文件哈希不同仅因检查点副本省略了末尾换行；源 norm SHA256 与合同一致。

启动前重新通过了 `selection.sha256`、2337 集数、RTC delay=10、batch32、固定学习率/优化器合同、`train_lego_full.py` pinned SHA256 `bfe1dd3347ee2cb2d392468a10cd0e87fe5f5f3a8e20f05343eb104e55f82afd`、`run.sbatch` SHA256 `b64ca9f0e8b004fa3837ca6e1121ef52c436ce3c2e8214716c1179e80f4061c2`，并确认无训练进程、无同名排队作业及无活动锁。用户重新明确要求从最新检查点续训后，已用原命令 `sbatch --parsable run.sbatch --resume 338000` 提交同一 run，作业 2183 于 2026-09-28 10:20:27 +08 在 `gpu001` 开始运行。该 338,000 是累计停止步数，不是新增步数；不得更改 optimizer、LR schedule、数据子集、norm、batch 或代码合同。

10:24:57 +08 再查时 2183 仍 RUNNING，恢复已成功进入训练：远端 metrics 已到 step 69,021，loss 0.00939、grad_norm 0.08554、LR 9.1252e-6，均有限；训练日志继续更新。登录节点未取得 GPU 温度或内核 Xid 数据。看板本地服务处于 active，SSH 镜像成功；刚同步时本地缓存为 step 69,011，远端为 step 69,021（约一个10秒轮询间隔），需按源 mtime 识别短暂滞后。

### 监控和恢复边界

复核 Slurm 作业 ID/状态、metrics 最新时间与 step，每 5 分钟一次；每小时检查作业日志/退出码、最近完整数值检查点和 Orbax 临时目录、GPU 温度/显存/利用率（可用时）、NCCL/CUDA/Xid、NaN/Inf、SIGSEGV/SIGBUS、`/dev/shm`、NFS/I/O、episode 读取与视频解码。健康有进展时保持安静，仅在恢复、完成、故障或需要用户行动时通知。

若 2183 失败，先诊断并确认无活跃训练进程/锁/保存，再从最新完整且可验证的数值检查点按原合同恢复；从本次用户重新授权后的恢复起连续两次失败，停止自动启动并报告证据。任何时候合同不匹配、检查点不完整/无法验证或无法确认保存静止，都先停启并通知。保留所有故障日志、检查点、临时目录和数据，不删除、不改写。目标 338,000 达成后不提交下一轮。
