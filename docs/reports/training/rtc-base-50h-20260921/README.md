# 50h Pi0.5 RTC 全量微调（2026-09-21）

状态：配置与选集已准备，未启动、未停止当前训练。沿用上一轮用户选择，从官方base重新训练，不继承20h模型。

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

本地只做静态和轻量测试，没有训练循环。此轮norm、GPU训练与真实resume尚未执行。170k约一轮、338k约两轮，使用相同提示词、颜色格规则及部署控制版本比较抓取成功率和成功抓取后的分类正确率。扩数据不保证解决抓取偏差或语言/格子歧义。不同数据规模的LR衰减预算不同，属于整体训练方案比较，不是严格单变量数据量消融。
