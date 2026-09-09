# 2026-09-09 · Pi0.5 r3正式训练

用户明确转达管理员允许直接开训、遇到问题再处理，并要求现在开始。此授权取代此前暂不放行建议；未宣称Xid或ECC根因修复。本轮未改系统驱动、模型、数据或数学参数。

- 17:10:23预检：2064有效、四GPU各1MiB/0%，无训练/诊断进程，r2只有metrics无完整checkpoint。
- 17:10:35启动：tmux `yam-lego-full-r3`，Slurm2064.81，训练PID1061106，gpu001。
- 快照：`/home/wuyan/lyj/YAM/env-transfer/lego-full-a53bb00`，提交`a53bb001e4acb2cce486f2da83d6d8256439a188`；未同步本地架构合并到服务器。
- 命令：`LEGO_RUN_NAME=lego_full_b64_r3_20260909 bash scripts/launch_lego_full.sh 2064`，由上述快照目录的tmux执行。
- run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64_r3_20260909`，control：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64_r3_20260909`。
- 从pi05_base开始，resume=false；batch64/FSDP4、40k终点、每5k保存/保留、EMA=None、seed42、原LR/精度/数据，见[实际配置](initial_config.json)。旧run完整保留，不混接旧步数。
- 17:11:25真实完成step1，随后step11，loss分别0.08078732/0.07457335，梯度有限，稳态3.3797秒/步。四GPU100%、约22830MiB、53–56°C。
- 看板service已切到r3，独立缓存`artifacts/training_dashboard/r3_20260909/metrics.jsonl`，API实际同步step11且sync.error=null，每10秒镜像。原暂停的小时巡检未擅自恢复，不承诺无人值守自动排障。
- 看板16项pytest通过，未在本地运行训练或训练smoke；service启动与API验证通过。

后续复核step1→11→21连续增长，step21 loss0.06119535、grad_norm0.71701068、3.3567秒/步；见[启动指标快照](launch_metrics.jsonl)。

这只证明正式训练开始且产生新更新，不代表长期稳定或完整checkpoint已验收。若再次故障，保留现场并据新日志定位；不把重复开训描述为根因修复。
