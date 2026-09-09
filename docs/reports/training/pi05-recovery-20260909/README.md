# 2026-09-09 Pi0.5 全量训练中断诊断

## 09:41补充：第二轮再次崩溃

原始日志保存在[r2_crash.log](r2_crash.log)。2064.63于09:33:21以0:11退出，最后logged271步，
loss0.02552907；09:33:17 NCCL ncclGroupEnd报告CUDA illegal memory access。
09:33:23内核GPU1（PCI52:00.0）Xid13 Out Of Range Address / Multiple Warp Errors，继发Xid43。
因此崩溃涉及GPU非法地址访问，不再只有CPU栈线索；不能直接宣称是驱动自身bug或坏显卡。
根据[NVIDIA Xid13排查说明](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)，
先使用Compute Sanitizer定位非法读写；这类错误也可能来自应用，不能单凭Xid诊断硬件。
09:43:27启动独立20步memcheck诊断，Slurm限时30分钟，正式两轮目录保留。
诊断使用原数学参数，插桩若引入显存不足需单独识别；20步无错误也不足以排除长时偶发问题。
gpu002现已分配全部4卡，未申请抢占、未取消他人作业。CPU CE计数127141、UE0，DIMM_B1持续有corrected报警。

观察时间：2026-09-09 09:06–09:16 CST。只记录有证据的定位，不宣称根因已解决。

- Slurm2064.33：2026-09-08 14:03:50 → 2026-09-09 05:21:05，15:17:15，ExitCode `0:11`。
- train.log尾部：`srun: error: gpu001: task 0: Segmentation fault (core dumped)`。
- 最后metrics：step16031，loss0.0176896583288908，grad_norm0.06273408234119415，
  step_seconds3.4065351199998988，time1788902458.8154263。
- 旧run `/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64` 仅metrics目录，无正式checkpoint。
- 未达到20k首次保存，不能用旧容量测试checkpoint冒充这次16031步恢复。
- core原件：`/var/lib/systemd/coredump/core.python.1615000055.b2511371e15a4c84afdaa153c77c78ca.844605.1788902461000000.zst`（gpu001）。
- 解压诊断副本：`/home/wuyan/lyj/YAM/training-runs/control/core-inspect-20260909.dWPJuW/core`。
  文件ACL可读，无提权。systemd journal仍不可读。
- GDB：`Program terminated with signal SIGSEGV`，`#0 0x00007fee0e2787c7`；
  `Backtrace stopped: Cannot access memory at address 0x7fed30ff9c00`。
- 映射 `0x00007fee0e162000..0x00007fee0f23b000` 属于 `/usr/lib64/libcuda.so.595.45.04`。
  可定位崩溃发生在驱动库内，不能区分驱动自身bug、调用者内存损坏或硬件因素。
- core恰为1073741824字节，GDB提示segment超出EOF；不是完整可恢复训练状态。
- EDAC mc0：CE127134、UE0。内核有CPU0_DIMM_B1单bit corrected ECC，07:31仍有记录；
  读取05:15–05:25内核窗口为空，无法据此认定崩溃原因，也无法排除因日志滚动丢失。
- 09:07四GPU均1MiB、0%利用率、36–37°C。磁盘7.8T可用；没有已见OOM证据。

## 处置与边界

用户授权从base新run重开，保存/保留每5k，首阶段40k，其余数学参数不变。
保留旧训练目录和全部原始数据；开启Python faulthandler为下次原生崩溃补栈。
不做无证据降batch、删样本或升级系统驱动。若复现同类驱动内崩溃，需管理员导出完整core/系统日志，
检查CPU0_DIMM_B1及驱动稳定性，优先健康节点；不能用重复从base重开掩盖持续失败。

代码验证：配置检查、无checkpoint拒绝resume/缺目录不创建、已有checkpoint允许resume的mock测试，
加看板13项测试共17项通过。此测试不替代新运行GPU/完整checkpoint验收。

## 新一轮启动验收（09:20 CST）

- 快照a53bb001e4acb2cce486f2da83d6d8256439a188，从已配置Gitea获取；本地同提交已核对双远端。
- 09:16:39启动新run `lego_full_b64_r2_20260909`，Slurm2064.63，PID3091182，tmux yam-lego-full-r2。
- [实际配置](initial_config.json)与[早期指标快照](launch_metrics.jsonl)从新run只读复制，不是实时日志。
- 21→31→41步，loss0.06119535→0.04840846→0.04490658，梯度有限，近期约3.36秒/步。
- 四GPU100%、显存约22828MiB，温度56–61°C；本地API同步41步、保存间隔5000、无同步错误。
- 仅短时运行恢复已验证，不代表驱动问题根治或40k已完成；首份5k checkpoint仍待产生并验收。
