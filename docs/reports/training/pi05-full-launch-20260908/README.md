# 2026-09-08 · Lego Pi0.5正式训练启动验收

## 运行身份与边界

用户授权40k首阶段、batch64、每20k保存，后续讨论续训至约一遍数据。2026-09-08北京时间
14:03派发正式训练，14:05:36完成首个更新。Slurm2064.33/gpu001、启动PID844605、tmux
`yam-lego-full`均为启动时观察，使用前重查。运行快照提交
`f93a7925f279be5f907a2897c2f5a378c7b6d2c0`，目录
`/home/wuyan/lyj/YAM/env-transfer/lego-full-f93a792`；运行中不改快照。

正式run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_full_b64`。
控制目录：`/home/wuyan/lyj/YAM/training-runs/control/lego_full_b64`，包含启动器锁、配置和日志。
本目录的日志是定点证据副本，不是实时数据源；`SHA256SUMS`固定本次归档指纹。

## 固定参数

完整配置见`initial_config.json`。Pi0.5 base原始权重启动，不沿用容量测试更新后的权重；
全参数微调，FSDP4，global64，workers8，EMA=None，seed42，原AdamW，峰值LR2.5e-5、
warmup1000、cosine162097到2.5e-6，阶段终点40000；save_interval/keep_period均20000。
PREALLOCATE=true、MEM_FRACTION=0.92；BF16计算/FP32状态。log_interval=10，每组约34秒。
不改YAM三相机/14D数据合同、H50/内部32D。当前没有验证loss或闭环成功率，不能以训练loss代替。

norm SHA256 `c496b738470e432b9da02b22a45c8c309b3db8412d73723944a7cbdc62305be9`；
train info SHA256 `ba88887ff84544eb130708cdf8c0ce965b53f099fdad07286a57f41883211851`，启动前远端重查一致。
未排除任何样本，未改原数据/norm/base。用户允许的损坏样本隔离边界见训练owner。

## 恢复验证与未解决项

- `restore-full.log`：真实Pi0.5全尺寸参数和AdamW状态恢复30→31，loss0.04792669、grad_norm0.349032，
  有限；未覆盖源checkpoint，也不将此权重作为正式起点。
- `gpu-main-resume-2.log` / `gpu-main-resume-3.log`：GPU训练入口先保存2，然后独立新进程恢复到4，
  再次完整保存并输出`GPU_TRAIN_MAIN_RESUME_VERIFIED`。测试产物位于verification目录，非正式run。
  小模型首次GPU测试batch2不整除4卡，第二次测试debug默认overwrite与resume冲突；修正测试参数后验证。
  正式配置从始至终batch64、overwrite=False，不受这两项测试设置影响。
- `resume-test.log` / `resume-test-2.log`：CPU同进程重复调用train.main在第二次JAX执行Aborted，
  分别8/32 CPU配额，原因未定位，不能记作通过；没有无限重试。独立GPU恢复证据用于正式GPU流程验收。
- 原生保存管理器会清理未保留的测试旧步骤，测试step2已由测试step4替代；未删除用户原始权重或正式checkpoint。
- 正式20k checkpoint尚未产生，完整长期恢复/40k完成仍待巡检验收。当前采用进程级锁防重复启动；
  断电可能回退到最近20k保存点，缺少首个checkpoint时不能假装已经具备完整续训状态。

## 启动观察

远端及本地API实际观察step 1、11、21、31、41连续增长，loss分别约0.08079、0.07457、0.06120、
0.04841、0.04491，梯度均有限。近期3.37秒/步、约19 samples/s。一次nvidia-smi采样四卡均100%，
占用22826–22848MiB/24564MiB；这不是持续峰值。看板的gpu_memory_gib约9.41是更新后JAX活跃分配，
不等于池预留或反向峰值。10秒自动同步已核对到多组新指标，不调用模型。

按短测和初始稳态，40k纯训练约37.5小时，排期仍预留40–48小时（约9月10日06–14时）；
不保证持续无故障，未包含所有未来评估/停机影响。小时巡检已切换到实际run，40k结束不自行延长。

## 验证与可复现入口

启动器：固定快照的`bash scripts/launch_lego_full.sh JOB_ID`；已存在正式训练禁止重复执行。
续训：确认资源/锁/完整checkpoint及配置数据一致后加`--resume`，同一累计终点和LR周期不重置。
完整日志及配置归档在本目录。Ruff针对变更通过（train.py既有E402初始化顺序单独排除），
13项看板pytest及JS语法检查通过；GPU恢复验证如上，CPU回归不计入通过项。
