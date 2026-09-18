# Pi0.5 base + 20h RTC 训练配置（2026-09-18）

状态：已准备选集与启动脚本，未提交训练，不会自动在22点切换或停止旧作业。
用户最终决定从官方base重训，保留原10h，补足20h，batch32，至少两轮。

## 固定配置

- run：`lego_pi05_rtc_base_20h_20260918`，与旧实验隔离。
- 数据：原train、seed42、完整episode随机选取；941集、2,160,813帧、30fps，20.007528h。包含原468集，新增473集；原val不参与抽样。源metadata哈希与10h一致；清单见本目录JSON。
- 初始化：`/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`。不加载60k参数或优化器。
- 代码复用已经进行10h训练的固定快照 `3dd9c82046eef4edfda776771290ba9994da30ae`，不引入另一套训练循环。
- Pi0.5全量微调，global batch32、FSDP4、seed42、H50、RTC随机前缀d=0..10；14D YAM，关节delta、夹爪absolute。EMA关闭、Adam eps=1e-6，保持10h条件。
- 总步数136000；2160813/32=67525.406步/等效轮，136000步约2.014轮。按样本起点计量，不乘H50；shuffle/drop_last使“轮”是近似计量。
- LR cosine：warmup1000，peak1e-5，decay_steps136000，末端1e-6。resume时不重置调度。
- norm在首次Slurm分配中仅基于这941集计算，保存在独立20h资产目录；包含真实loader等价检查和provenance。当前尚未生成，不沿用10h统计。
- 每1000步保存，长期保留每5000步及最新；训练合同约束subset、norm、优化器、LR等。恢复使用完整checkpoint的参数、优化器和step。
- Slurm四卡、64CPU、480G、五天上限；已确认分区最大七天。约2.28s/步估算86.1小时，未含排队、编译、存盘或中断。

## 服务器启动与恢复

所有命令在服务器执行。control路径：
`/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_20h_20260918`

今晚22点先检查旧作业和保存状态，确保所需四卡可用；不要为切换直接杀掉正在保存的进程。本配置不会取消旧作业，也没有自动调度切换。

首次启动：
```bash
sbatch --parsable /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_20h_20260918/run.sbatch start
```

中断后，确认本run已停止且至少存在一份完整checkpoint，再执行：
```bash
sbatch --parsable /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_20h_20260918/run.sbatch --resume
```

首次启动前会检查episode清单哈希。resume保留同一清单、norm和学习率计划，自动选择最新完整checkpoint；不是从base重来。第一份checkpoint前中断没有可恢复进度，应保留失败记录、查明原因后用新run重新准备，不能删除合同绕过门禁。脚本以flock防止同一run并发。

权重和指标：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_20h_20260918/`。
日志：control下`train.log`、`norm-<jobid>.log`、`slurm-<jobid>.log`。
资产：`/home/wuyan/lyj/YAM/training-assets/lego_rtc_20h_20260918/`。

## 如何判断20h是否更好

保留60k（相同更新次数的参考）、70k（约1.04轮）、135k（约2.00轮）的检查点。固定现场测试布局、颜色规则和推理控制配置，分别记录抓取成功率、正确投放率、整任务成功率。原10h的60k约1.77轮，20h的60k只有0.89轮，不能把它们叫作同轮数比较。

本轮LR衰减跨度与旧10h不同，因此结果检验的是“20h训练方案”的收益，不能严格归因于仅增加数据。要做严格数据量消融，需要另开匹配学习率与训练预算的10h对照。本轮不为此额外启动实验。

验收范围：选集包含关系、源哈希、脚本语法、服务器依赖路径和Slurm提交预检；尚未进行本轮GPU训练、norm计算或实际恢复。旧10h运行通过不等于本轮已启动。
