# Pi0.5 base + 随机10h training-time RTC：运行交接

用户于2026-09-16明确授权启动，并要求支持中断恢复，后续监控由用户另行安排agent。本报告不创建监控自动化，也不向其他任务发消息。

## 固定运行合同

- Run：`lego_pi05_rtc_base_10h_20260916`。
- 初次Slurm作业：`2140`，2026-09-16 18:10:10+08在`gpu001`开始；实际4×RTX4090，每卡24564MiB，启动前仅1MiB占用。
- 代码：`3dd9c82046eef4edfda776771290ba9994da30ae`，独立detached worktree `/home/wuyan/lyj/YAM/code-snapshots/rtc-10h-3dd9c82`。服务器从Gitea取代码；直连本地Gitea不可达时使用SSH临时反向转发，未从GitHub拉取、未改原remote。
- 初始化：官方`/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`，不是100000。
- 数据：已发布原train中seed42随机468条完整episode，1,082,232帧@30fps=10.020667h；全部任务metadata为`sort the legos into containers by color`。保持原val独立；metadata同任务不保证每条视觉质量或成功。
- norm：只用该468条、H50、关节delta/夹爪absolute；耗时65.03秒；12个真实LeRobot抽样与统计动作路径逐元素相同。norm SHA256 `606d5c69e56aadb273ed3882ba9b62e11978a4222d8ff1e827e541bddd112893`。
- batch32、FSDP4、全量微调、RTC d∈{0,…,10}、LR1e-5→1e-6、warmup1000、decay30000，初始停止10000次更新。
- 每1000步保存，长期保留每5000步及最新检查点。保存包含参数、优化器、step和norm；数据起点由既有确定性采样恢复逻辑恢复。

## 服务器位置

连接：`ssh yam-server`（`wuyan@10.18.31.234`）。

- 调度脚本与日志：`/home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_10h_20260916/`
- 训练日志：上述目录`train.log`（恢复追加，不截断）。
- Slurm日志：`slurm-2140.log`，恢复后对应新job ID。
- norm日志：`norm-2140.log`；统计完成标记`NORM_COMPLETE`。
- 权重/训练状态：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/<step>/`
- 指标：上述run目录`metrics/metrics.jsonl`。
- 数据清单与norm：`/home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/`。
- 退出回执：control目录`exit-<jobid>.txt`；节点断电/SIGKILL时trap不保证执行，需结合Slurm状态。

## 交给监控agent的恢复操作

先核对`squeue -u wuyan`与本run日志：不要在原训练仍运行时重复提交。批处理脚本有`flock`防止同一run并发。Slurm作业不依赖SSH或本对话存活。

若进程已停止，且已有**完整提交**的检查点，在服务器执行：

```bash
sbatch --parsable /home/wuyan/lyj/YAM/training-runs/control/lego_pi05_rtc_base_10h_20260916/run.sbatch --resume
```

新Slurm作业会申请四卡，使用同一代码快照、base来源、norm、episode清单与学习率计划，由Orbax选择最新完整检查点；不是从base重来。不要手动把未完成临时目录当作可恢复检查点。训练入口核对`training_contract.json`；若不一致，先查明配置/数据变化，不能删除合同绕过检查。

- 第一份1000检查点完成前若中断，**没有可恢复进度**。入口会拒绝假resume；保留失败记录，评估后以新的run名从base重启，不能删旧目录强行重来。
- 若因NaN/Inf、设备错误或损坏数据中断，先定位原因及最后有效保存点，不要无条件自动重试。用户授权训练并不表示可改数据/学习率后继续同一run。
- 本批处理固定停在10000；到达目标后属于正常结束。将来若延长到20000/30000，需要明确新的总步数与调度脚本版本；维持decay30000，不能把正常完成误判成失败。
- 不恢复/停止其他实验、下载、Thor服务或3588进程。不会在退出后自动无限申请资源。

## 证据与验收范围

选集、norm来源与启动合同见同目录JSON；调度脚本的本地副本为[run.sbatch](run.sbatch)。此前41项轻量测试通过；本轮GPU运行进度以下方最新快照为准。代码测试不等于现场任务成功率，trained RTC的Thor采样/导出仍未接入。

## 已运行快照：2026-09-16T18:15:21.951858+08:00

已完成真实GPU前后向，metrics记录到第21次更新：loss=0.063918，grad_norm=0.921431，约2.270秒/步，全部有限。相邻记录第1/11/21次更新连续递增。四卡nvidia-smi均100%利用率、占用约22824MiB；JAX指标bytes_in_use约9.40GiB与驱动预分配显存口径不同。初次编译有rematerialization内存预算警告，但随后更新成功；不把此警告描述为OOM。

截至该快照尚未产生首个1000检查点，**真实保存/恢复尚未验收**。恢复功能已实现且合同测试通过，但不能把启动通过说成实际恢复测试通过。预计约2.27秒/步只来自短区间，不保证长期吞吐或结束时间。用户可将本报告直接交给监控agent；实时情况以服务器为准。
