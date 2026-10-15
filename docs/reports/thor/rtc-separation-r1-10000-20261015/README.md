# Pi0.5 DAgger RTC：240000＋专项10000

## 当前状态：10000已替换7000上线

**2026-10-15 16:49:36+08上线完成，16:50:06+08复核正常。** 用户明确授权拉取1万步DAgger模型、测试并替换7000，本次固定同run `lego_pi05_rtc_separation_r1_10k_20261014/10000`；源Orbax完成元数据确认15:32完整保存，合同与7000相同，parent原50h240000，仍是Pi0.5全量训练时RTC dmax10/H50/32D，不是新模型系列或原50h run的250000。

同名`pi05-rtc-infer`已运行新10000，固定`ws://192.168.250.1:8000`，healthz OK，`thor-pi-maxn-separation10000-20261015.service` active、MAXN/GPU1575MHz。七步FP32权重＋TF32、quantile前缀、缓存80/CUDA Graph和0.2rad/tick拒绝不变；20项源SHA、参数有限性/FP32转换、独立JAX/Torch/TRT及两轮各九例协议均通过。7000停止完整保留为`pi05-rtc-separation7000-preserved-20261015`，24万/其他旧权重和引擎未删；PLD推理仍停止、CPU数据审阅`pld-closing8-review-r06`继续运行。

本次不训练、不改服务器源、不读取/修改3588、不发送客户端任务；只完成Thor本机/离线验收，用户真机效果及跨IPC全链路待测试。以下是本轮下载和转换阶段记录。

16:29:48+08下载完成：[下载回执](download-receipt.json)确认20文件、12,440,554,566 bytes（约11.59GiB），[源SHA](source.sha256)20项全部通过，[完整传输记录](download-receipt.txt)退出0，未重复7000时的SSH超时。只取params/assets/完成元数据和[合同](training_contract.json)/[初始配置](initial_config.json)，不取train_state。16:30:59原始参数审计非有限0，CPU FP32转换进行中；7000仍在原Pi/原8000/MAXN运行，CPU数据审阅页面保留。GPU阶段才停止7000、等待旧MAXN按ID退出/120W，再独占MAXN测试。

16:32:24+08 CPU准备退出0：[原始审计](rtc-separation-r1-10000-audit-20261015-r2.json)非有限0、[转换审计](conversion_audit.json)811张量/3,353,433,872元素FP32映射逐位一致、[RTC manifest](rtc_manifest.json)及[真实九例](cases.json)完成。权重SHA `19f8db38b948d71b1decce69cb267f784a94249c9ceba5a8b5cbb14881468535`。随后暂停7000、其旧MAXN退出/120W，新独占MAXN/GPU1575MHz完成七步JAX/Torch/TRT对照。映射器基础配置打印`rtc_training_max_delay=0`不作为路线判据；实际合同、manifest、采样器均为训练时RTC dmax10。

下载阶段为`/home/wuyan-lyj/thor/pi/probes/rtc-separation-r1-10000-20261015-r1`，转换/部署为独立`rtc-separation-r1-10000-20261015-r2`。已冻结[下载](download-only.sh)、[CPU准备](prepare-20261015.sh)、[GPU/上线](deploy-20261015.sh)，不覆盖运行中的脚本。部署unit`thor-rtc-separation10000-deploy-20261015.service`退出0/inactive，存在`DEPLOYED_PI05_240000_PLUS_SEPARATION_10000_SAME_SERVICE`；本次没有触发GPU/上线失败回退，也没有重跑旧7000/24万脚本。

## 精度、性能与协议

[原JAX参考](reference_manifest.json)、[导出回执](export_report.json)、[引擎回执](engine_report.json)与[TRT验证](validation.json)均绑定本次权重/norm/真实九例（episode95/96/97早中晚，d=0/1/10），同输入/同固定噪声/同七步。FP32诊断关闭TF32、候选引擎明确TF32，无BF16/FP8量化。

- JAX/Torch最大物理数值差9.43815e-6、归一化最大1.88947e-5；九例1e-4门槛和缓存wrapper 1e-5门槛均通过。混合14D数值不统一冒充rad，缓存优化不宣称逐位相等。
- JAX/TRT最大单关节差**0.000774049rad**，夹爪最大差**0.002478397**（标称0～1范围约0.248%）。最坏混合14D元素为case1/第49步/右夹爪，不是关节；混合14D P99=0.000460549，保留原服务0.005 max/0.001 P99数值门槛，没有放宽。
- MAXN稳态纯推理P50/P95=**187.77/192.41ms**，Thor总处理191.50/196.95ms。首次包含Graph捕获的推理P95=378.38ms单列，不作稳态或现场全链路时延。
- [临时九例](separation10000-candidate-smoke.json)和[正式九例](separation10000-online-smoke.json)均有限50×14、前缀精确、握手权重/norm/engine匹配，trained RTC/quantile/dmax10/七步不变。正式服务推理P50/P95=**197.54/207.97ms**，**Thor本机WebSocket往返198.23/209.07ms**，不包含3588采集/网络/控制链路。
- 正式最大新后缀关节步进0.140482rad/tick（临时0.137687），低于保留的0.2拒绝值。两次随机协议采样不作后端误差比较；粗大跳变保护不替代控制层安全或任务精度评估。数值/协议通过不证明1万步任务效果优于7000步。

前缀仍按新observation state把12个关节转delta、两夹爪保持absolute，再显式quantile编码；模型内前缀干净/flow time0，只去噪后缀，无guidance/普通模式回退。输出逆变换为14D绝对目标，关节rad、夹爪连续值标称0闭/1开、不裁剪。30Hz policy tick，`action[0]`对应observation policy tick，已承诺前缀从同tick起始；相机曝光到policy tick偏移仍未知，不把数据索引0解释为设备0ms延迟。

## 产物、运行与回退

FP32产物为`checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/10000-pytorch-fp32-r1`，回放为`test-data/pi05-rtc-separation-r1-10000-replay-20261015-r2`；原JAX/ONNX/TRT独立位于`artifacts/rtc-separation-r1-10000-…-20261015-r2`。引擎SHA `ab51626268b33cba4a2fe8715dbc68245205016b94dc86f19173c1c95669906d`。合同SHA `f53267482d55d31d7df811253c04553677a9fb30d62a1e0f22a37511848cda1f`，norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`，与7000逐字节相同，仍不复用其参考或引擎。

只复制7000已验收冻结工具和热身JSON，镜像仍为Pi `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`、JAX参考 `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`；未同步整个工作树或新建DAgger镜像。临时验证容器`pi05-rtc-validation-10000-20261015`已自动移除；长期仍只有同一个Pi服务名称/地址，停止旧容器仅作回退，不是并行在线。

[准备日志](prepare-receipt.txt)、[部署完整回执](deployment-receipt.txt)保留原阶段记录，本地文本副本仅去掉行尾空白，Thor原始日志未改。阶段JSON中的`onnx_exported_engine_not_validated`、`built_experiment_not_accuracy_validated`、`compared_not_robot_task_validated`保留，不改写成实机已验收。当前可用427,630,284,800 bytes，约398.3GiB；没有删除任何模型。

在线MAXN按容器唯一ID监控，`--restart no`，不开机自启，停止服务后恢复120W。若用户要求回退，先停10000、等当前MAXN退出/120W，保留10000后将7000备份恢复同名服务，再先MAXN启动并按7000的`223b7dc8…`权重及旧九例重新smoke；不能同时占8000或自动启动PLD推理。一次性下载/转换/部署脚本只作本轮证据，不直接重跑。

## 路线与边界

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_separation_r1_10k_20261014/10000`。
- Thor原始目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-separation-r1-10k-20261014/10000`；FP32转换为同父目录`10000-pytorch-fp32-r1`，不覆盖7000。
- 只复用[7000已验收工具/镜像](../rtc-separation-r1-7000-20261015/README.md)，新权重独立审计/FP32保真映射、真实三相机14D state/prompt/d=0/1/10九例、原JAX七步参考、Torch/ONNX、TF32缓存80引擎及两轮协议。所有参考和大产物独立命名`rtc-separation-r1-10000-…-20261015-r2`。
- 正式仍为同一Pi系列镜像/同名服务/原8000，七步FP32＋TF32、quantile前缀、CUDA Graph与0.2rad/tick拒绝不变；不量化、不静默普通模式回退、不新建长期DAgger服务。临时工具容器`--rm`。
- GPU/推理/预热/测试MAXN＋锁频，停止服务后恢复120W。7000原始/转换/引擎和停止服务配置保留回退；GPU门槛或上线失败恢复并重新smoke7000，不删除24万或任何旧模型。
- 仅Thor侧，不训练、不改服务器源或3588、不自动发送客户端任务。源合同来自同run控制目录；训练配方及数据来源仍归[训练报告](../../training/lego-separation-20261014/replacement-10k/README.md)，本次不改变它。
- 数值一致和本机协议通过不等于新模型任务成功率更高，也不是跨IPC全链路时延；现场由用户测试。详细RTC语义归[手册13](../../../reference/thor/13_trained_rtc_inference.md)。
