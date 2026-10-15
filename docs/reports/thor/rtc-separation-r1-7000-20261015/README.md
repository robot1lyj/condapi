# Pi0.5 RTC：240000＋专项7000，转换与上线测试

## 当前状态与授权

**2026-10-15 16:05:55+08上线完成，16:06:09+08复核正常。** 用户在下载后另行授权转换并上线测试，确认DAgger与Pi05共用系列容器；本次固定7000，不改训练。新模型已由同名`pi05-rtc-infer`提供固定`ws://192.168.250.1:8000`，healthz OK，`thor-pi-maxn-separation7000-20261015.service` active、MAXN/GPU1575MHz；七步FP32权重＋TF32、quantile前缀、缓存80/CUDA Graph和0.2rad/tick拒绝不变。独立JAX/Torch/TRT和临时/正式两轮各九例真实协议通过。

15:49:23 CPU准备退出0：20项源SHA、[原参数审计](rtc-separation-r1-7000-audit-20261015-r2.json)非有限0、[转换审计](conversion_audit.json)811张量/3,353,433,872元素FP32映射逐位一致、[RTC manifest](rtc_manifest.json)及[真实九例](cases.json)完成。权重SHA `223b7dc80c1ce6458787412e478848ffd297072cceebb9b79b983d686bab2e95`；训练norm逐字节相同。映射器打印的基础配置`rtc_training_max_delay=0`不代表普通路线，实际合同/manifest/采样器均为训练时RTC dmax10。

按Pi模型系列复用既有镜像，不新建DAgger镜像/独立长期服务。转换/验证工具容器为临时`--rm`；正式仍只有一个Pi推理服务，替换checkpoint配置，旧停止容器仅作回退。入口为[CPU准备](prepare-20261015.sh)、[GPU验收与上线](deploy-20261015.sh)，执行前冻结，不覆盖运行脚本；异步启动等待和按容器唯一ID监控已吸收上次交接恢复，不复用旧失败逻辑。

15:44原Pi/PLD推理均停止、120W/GPU空闲，仅CPU数据审阅`pld-closing8-review-r06`运行（runc、无GPU请求）。本次保留该页面；上线后仅Pi占GPU，PLD推理继续停止，旧24万停止保留为`pi05-rtc-50h240000-preserved-20261015`，所有旧权重/引擎保留，无删除。未操作服务器源、训练、3588或发送客户端任务；用户实机效果与跨IPC全链路仍待测试，本机通过不外推任务成功率。

## 精度与延迟

同输入、同固定噪声、七步采样，[JAX参考](reference_manifest.json)、[Torch/ONNX导出](export_report.json)及[TensorRT验证](validation.json)绑定本次权重/norm/真实九例，覆盖episode95/96/97早中晚、d=0/1/10。原JAX/FP32诊断关闭TF32，候选引擎显式开启TF32，无BF16/FP8量化。

- JAX/Torch最大物理数值差9.99213e-6，归一化最大1.99825e-5，九例均通过1e-4门槛；缓存wrapper门槛1e-5也通过。混合14D数值不能统一标成rad，缓存优化也不宣称逐位相等。
- JAX/TRT最大单关节差**0.000691894rad**，夹爪最大差**0.002327973**（标称0～1范围的约0.233%）。最坏混合14D元素在case1/第48步/右夹爪，不是关节；混合14D P99=0.000415084。原服务数值门槛未放宽，前缀归一化显式quantile，两端前缀保持干净/flow time0，无guidance或普通回退。
- MAXN稳态纯推理P50/P95=**188.21/190.20ms**，Thor总处理192.20/194.18ms；首次包含CUDA Graph捕获的推理P95=383.02ms单列，不能冒充稳态。
- [临时九例协议](separation7000-candidate-smoke.json)和[正式九例协议](separation7000-online-smoke.json)均通过：有限50×14、前缀逐位保持、权重/norm/engine指纹匹配，dmax10/quantile/七步不变。正式服务推理P50/P95=**194.95/204.73ms**，**Thor本机WebSocket往返195.53/205.85ms**，不包含3588网络/采集/控制链路。
- 正式最大新后缀关节步进0.120382rad/tick（临时0.120539），低于保留的0.2拒绝值。协议随机采样两轮不用于后端误差比较；该粗大跳变保护不替代控制侧限位或任务精度评估。

输出仍是14D绝对物理目标：左6关节/左夹爪/右6关节/右夹爪；关节rad，夹爪连续值标称0闭/1开、不裁剪。30Hz policy tick，`action[0]`对应observation policy tick，已承诺前缀从同一tick起始；相机曝光到policy tick偏移仍未知，不把握手中的数据索引0偏移解释成设备0ms延迟。

## 部署产物与回退

冻结stage：`/home/wuyan-lyj/thor/pi/probes/rtc-separation-r1-7000-20261015-r2`；转换件：`checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/7000-pytorch-fp32-r1`；回放：`test-data/pi05-rtc-separation-r1-7000-replay-20261015-r2`。JAX/ONNX/TRT目录均独立命名为`artifacts/rtc-separation-r1-7000-…-20261015-r2`，未覆盖24万。

[引擎回执](engine_report.json)的engine SHA `d7d2c3baa192c9f37784c31430d402ccdd12a76b6ed40149aedc480385bd9cfa`。复用Pi镜像ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`，JAX参考镜像ID `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`，工具从已验收24万冻结stage复制，不同步整个工作树。20万/23万/24万均完整保留；下载与部署未删除任何模型。

[准备记录](prepare-receipt.txt)、[完整部署回执](deployment-receipt.txt)保留全部阶段；本地文本副本仅去掉行尾空白，Thor原始日志未改。`thor-rtc-separation7000-deploy-20261015.service`退出0/inactive，存在`DEPLOYED_PI05_240000_PLUS_SEPARATION_7000_SAME_SERVICE`，本次未触发24万回退。临时验证容器`pi05-rtc-validation-20261015`已自动移除，正式服务只保留同一Pi名称/地址；`--restart no`，不是开机自启，停止服务后MAXN会话恢复120W。当前可用479,603,331,072 bytes，约446.7GiB。

若用户要求回退24万，先停止当前Pi并等待按ID监控的MAXN退出/120W，再保留当前7000容器，把旧24万备份恢复同名服务、先启动MAXN并用旧24万指纹/九例重新smoke。不能同时占8000，也不自动恢复PLD推理。导出/构建/验证JSON中的阶段状态保持原样，不把`built_experiment_not_accuracy_validated`或`compared_not_robot_task_validated`改写成真机验收。

## 首次下载阶段（历史）

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

## 验收与任务边界

下载完整性单独只证明传输一致，本次已另行完成[训练时RTC手册](../../../reference/thor/13_trained_rtc_inference.md)的FP32保真转换、checkpoint自身quantile norm、独立七步JAX/Torch/TF32 TensorRT同输入对照及真实d=0/1/10协议。旧24万引擎或参考没有代替本次7000验收；0.2rad/tick拒绝、完整RTC语义和固定服务地址保留。转换/协议一致不证明专项模型任务成功率优于基线，后续由用户现场测试。

GPU推理/预热/测试均MAXN＋锁频；进入GPU阶段前再次核查实际服务和GPU状态。首次下载阶段的“不转换/不测试/不切服务”现已被本页当前状态中的新授权覆盖，不再等待重复授权；仍不读取/修改3588或自动发送客户端任务，不清理任何旧模型。
