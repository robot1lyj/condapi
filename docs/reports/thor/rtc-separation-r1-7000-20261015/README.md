# Pi0.5 RTC：240000＋专项7000，下载准备

## 当前状态与授权

2026-10-15用户先称“20w＋7k dagger”，核对服务器训练合同和启动配置后明确确认实际目标为**原50h 240000初始化＋专项run第7000步**，不是200000初始化，也不是原50h run的247000。最新完整7000保存于13:33+08；本次固定拉取这个保存点，不因训练继续而静默改取8000等更新点。

**14:13:45+08下载和完整性校验完成**：[下载回执](download-receipt.json)确认20个文件、12,440,547,840 bytes（约11.59GiB），[源SHA清单](source.sha256)20项全部通过。14:01:11+08启动[下载专用脚本](download-only.sh)，Thor直接从服务器六路rsync，只取params、assets、Orbax完成元数据及同run训练合同/启动配置，不取train_state。没有转换、参数有限性审计、GPU推理或上线。用户“拉取下来先，准备测试”不解释为暂停PLD或自动切换Pi。

14:13:53+08复核：`pld-rl`仍运行、`pi05-rtc-infer`明确Running=false，保留的服务命令仍指向原50h240000；下载后可用528,306,700,288 bytes，约492GiB。10月14日“24万在线/PLD停止”是历史状态，不能据此覆盖当前服务。本次不停止或启动任何容器、不改电源模式、不操作3588、不删除模型或数据、不发客户端任务。

### 传输恢复记录

[首次传输日志](download-attempt.txt)保留原SSH会话超时及配置文件rsync失败（code12），不宣称首次脚本退出0。全部六路大权重已收齐；唯一卡住的是本任务读取小`training_contract.json`的SSH认证子进程3991596。复核精确cmdline/父PID后仅结束这个认证子进程，不终止大权重传输或其他任务。新的[收尾脚本](finalize-download.sh)等待原下载锁释放，以新会话补齐合同和配置，再对已有大权重及小文件完整SHA校验，14:13:45退出0，见[最终校验回执](finalize-receipt.txt)。没有重复下载大权重；冻结原脚本/失败证据保留，不覆盖。

## 身份与路径

- run：`lego_pi05_rtc_separation_r1_10k_20261014`，checkpoint `7000`。
- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_separation_r1_10k_20261014/7000`。
- Thor目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/7000`；独立新目录，不覆盖原50h权重。
- Thor工具/日志：`/home/wuyan-lyj/thor/pi/probes/rtc-separation-r1-7000-20261015-r1`；agent只随本次SSH传输会话临时转发，不复制私钥。
- [训练合同](training_contract.json)、[实际初始配置](initial_config.json)均明确parent为原50h `240000/params`，Pi05全量、H50/32D、RTC dmax10；norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`。
- 训练来源是已冻结专项100集＋原数据回放100集，用户称DAgger模型，但不能仅据称呼宣称新增在线专家纠正数据或已经实现完整DAgger。训练配方/审批归[训练报告](../../training/lego-separation-20261014/replacement-10k/README.md)，本次没有改配方或启动训练。

## 后续测试边界

下载完整性通过只证明文件传输一致，不证明参数有限或模型行为。后续获准接入时沿用[训练时RTC手册](../../../reference/thor/13_trained_rtc_inference.md)：FP32保真转换、checkpoint自身quantile norm、独立七步JAX/Torch/TF32 TensorRT同输入对照及真实d=0/1/10协议。旧24万引擎或参考不能替代本次7000验收；保留0.2rad/tick拒绝、完整RTC语义和固定服务地址。

GPU推理/预热/测试均MAXN＋锁频；开始独占GPU前先确认当时PLD/其他服务状态和暂停授权。本轮不转换、不测试、不切服务，下载后等待用户指示。
