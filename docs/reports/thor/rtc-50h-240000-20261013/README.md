# 乐高50h Pi0.5 RTC 240000：转换、验收与部署

## 当前状态

**2026-10-27 10:45 CST：用户要求开启24万推理，已恢复原50h240000基础版（非DAgger）。** 现场原Pi停止，但`yam-agent-perception`和`sun-yat-sen-ep02-audio`运行，宿主已是MAXN，无正在运行的Pi MAXN watcher或`maxn_session.py`进程。只启动原Pi唯一ID `d80c40497577a3028a4ca7bc16fd49c861c0c3893937df62f291e5887b5ef523`，不停止其他容器、不修改启动前已有的电源配置。通过sudo只读`jetson_clocks --show`核对MAXN/GPU1575MHz、CPU2601MHz、EMC4266MHz均锁频。不是重新进入120W再取MAXN快照，没有新建Pi MAXN unit；后续停服须核对并行任务和电源归属，不得宣称Pi停止会自动回120W，也不应擅自降频影响其他任务。

[本次九例协议回执](resume-20261027-smoke.json)于10:45:41+08通过，真实episode95/96/97早中晚、三相机/14D state/prompt、d=0/1/10全部有限50×14、前缀精确、指纹匹配。七步FP32＋TF32/quantile、缓存80/CUDA Graph、dmax10与0.2rad/tick保护不变；固定`ws://192.168.250.1:8000`、healthz OK。当前共享负载下推理P50/P95=200.26/223.62ms、本机往返201.04/224.88ms，最大新后缀关节步进0.122233rad/tick；不是独占GPU基准、3588全链路或任务成功率。复用既有权重/引擎，未重跑JAX精度对照，无下载、转换、构建、删除、训练或3588操作。

## 历史：2026-10-26关闭推理

2026-10-26 15:55:42+08：用户要求关闭Thor推理，`docker stop --timeout 20 pi05-rtc-infer`完成，实查running=false；`thor-pi-maxn-50h240000-resume-20261026.service` inactive，`nvpmodel -q`显示120W/模式1。容器、全部模型权重和引擎保留，不删除、不自动恢复、不恢复此前暂停的其他容器，不操作3588。以下为此前上线验收，不表示当前在线。

## 历史：2026-10-26恢复24万并停止其他容器

**2026-10-26 14:36 CST：按用户要求开启24万基础版（非DAgger），并按追加要求停止其他容器，已在线。** 现场原Pi停止/120W、无MAXN会话，复用原容器ID `d80c40497577a3028a4ca7bc16fd49c861c0c3893937df62f291e5887b5ef523`与已验收权重/引擎；新`thor-pi-maxn-50h240000-resume-20261026.service`先MAXN/锁频后启动，并按唯一ID监控，日志GPU1575MHz/CPU2601MHz/EMC4266MHz。原`yam-agent-perception`占GPU、CPU审阅`pld-closing8-review-r06`运行；起初保留它们，随后用户明确“把其他的容器先暂停”，才将这两个容器停止保留、不删除、不更改重启策略，复核只剩Pi容器运行且GPU进程仅Pi。

初次50秒就绪等待结束时服务仍未监听，随后协议尝试报ConnectionRefusedError，未产出通过回执；容器仍running、无OOM。没有重启或修改引擎，继续等待healthz真正OK后，于14:36:17+08完成[独占GPU九例协议复核](resume-20261026-isolated-smoke.json)。真实episode95/96/97早中晚、三相机/14D state/prompt、d=0/1/10全部有限50×14、前缀精确，握手weight/norm/engine与原24万一致。推理P50/P95=195.50/205.80ms，本机往返196.24/207.25ms，最大新后缀关节步进0.127803rad/tick。七步FP32＋TF32、quantile、缓存80/CUDA Graph/dmax10/0.2rad/tick不变；固定`ws://192.168.250.1:8000`、healthz OK/MAXN active。未重跑JAX精度对照，不冒称跨IPC/实机任务验收；没有下载、转换、构建、删除、训练或操作3588。停服恢复120W，不新增开机自启。

## 历史：2026-10-23停止推理

2026-10-23 11:07:46+08：用户要求“先取消推理”，`docker stop --timeout 20 pi05-rtc-infer`完成，实查running=false；`thor-pi-maxn-50h240000-resume-20261023.service` inactive，`nvpmodel -q`显示120W/模式1。容器及所有模型权重/引擎保留，不删除、不自动恢复、不操作3588；以下为此前上线和验收历史，不表示当前在线。

## 历史：2026-10-23恢复24万步

**2026-10-23 10:56 CST：用户要求从20万切换为本页50h240000基础版（非DAgger），已在线。** 先停止20万唯一容器，等待其MAXN watcher退出并确认120W，再保留为`pi05-rtc-50h200000-preserved-20261023`；将原24万停止容器恢复同名`pi05-rtc-infer`，ID `d80c40497577a3028a4ca7bc16fd49c861c0c3893937df62f291e5887b5ef523`。新`thor-pi-maxn-50h240000-resume-20261023.service`使用既有MAXN launcher和按该唯一ID监控的watcher，MAXN/锁频，healthz OK，固定`ws://192.168.250.1:8000`。直接复用原已验收权重/引擎，无下载、转换、构建或删除；耳机114000备份、DAgger资产、其余模型与CPU审阅页保留。

[本次九例真实协议回执](resume-20261023-smoke.json)于10:56:06+08通过：episode95/96/97早中晚、三路RGB/14D state/prompt、d=0/1/10均有限50×14、前缀精确、实际RTC启用，握手weight/norm/engine与原验收匹配。七步FP32权重＋TF32、quantile前缀、缓存80/CUDA Graph、dmax10和0.2rad/tick保护不变；推理P50/P95=195.80/206.78ms，本机往返196.35/208.07ms，最大新后缀关节步进0.115720rad/tick。仅恢复后的Thor本机协议测试，未重跑JAX对照，不代表3588全链路或实机成功率；不训练、不操作3588或发送客户端任务。停服恢复120W，不新增开机自启。以下为此前转换与上线历史。

2026-10-15 16:06 CST历史更新：用户另行授权的[240000＋专项7000](../rtc-separation-r1-7000-20261015/README.md)曾接管同名Pi服务/原8000，本页原24万当时停止保留为`pi05-rtc-50h240000-preserved-20261015`，资产未删；该停止状态已被上述10月23日恢复取代。

2026-10-14用户在恢复20万后，**另行明确授权“24w步开始部署上线”**，本次转换、GPU验收和验收后同地址上线均已获准；此前只下载限制是历史，不再作为本次阻断。新的冻结stage为`/home/wuyan-lyj/thor/pi/probes/rtc-50h-240000-20261014-r1`，不重跑10月13日的旧prepare/deploy脚本。

14:57+08完成CPU准备：18项源SHA重新通过；[原始参数审计](rtc-50h-240000-audit-20261014-r1.json)非有限0；[转换审计](conversion_audit.json)811张量/3,353,433,872元素FP32映射逐位一致；[RTC manifest](rtc_manifest.json)确认训练dmax10、checkpoint自身norm（仅末尾换行与训练norm不同）。FP32权重SHA256 `12b08533d61119bf9b7b98bc20a3f64553b42a7140fd99107f52610f19184288`。映射器打印的基础config中`rtc_training_max_delay=0`不代表本checkpoint普通推理，路线由训练合同、manifest及实际采样器共同确认。

**15:20:37+08完成上线，15:22:29+08再次复核正常。** [真实九例](cases.json)覆盖三路RGB/14D state/prompt、原始动作前缀及d=0/1/10；本次独立七步JAX→Torch/ONNX→TF32缓存80 TensorRT对照和候选/正式两轮协议均通过。固定`ws://192.168.250.1:8000`、`pi05-rtc-infer`，healthz OK，仅Pi运行，`thor-pi-maxn-50h240000-final-20261014.service` active、MAXN/GPU1575MHz。20万停止保留为`pi05-rtc-50h200000-preserved-20261014`，23万原备份不变，完整权重/引擎均保留；PLD继续停止。无清理，未操作训练、服务器源、3588或发送客户端任务。仅Thor本机/离线验收，真机及3588全链路待用户测试。

## 精度、延迟与协议结果

- [原JAX参考](reference_manifest.json)、[导出回执](export_report.json)：同输入、同固定噪声、同七步；九例JAX/Torch最大物理数值差4.58592e-6，门槛1e-4及缓存wrapper门槛1e-5通过。混合14D数值不冒充统一rad单位；ONNX保留动态`previous_actions`/`prefix_mask`。
- [引擎回执](engine_report.json)、[TRT验证](validation.json)：最大单关节转换差0.000824078rad，夹爪最大差0.000366097；输出有限、前缀精确。engine SHA `f37e0de738e108d718750222a82690fe74bef613219aadd24a7e530f557a321d`。FP32权重＋TF32计算，无BF16/FP8量化，不放宽门槛。
- MAXN稳态纯推理P50/P95=193.57/194.15ms，Thor总处理197.32/197.79ms。首次含CUDA Graph捕获的推理P95=375.25ms单列，不冒充稳态。
- [候选协议九例](50h240000-smoke.json)及[正式在线九例](50h240000-final-online-smoke.json)全部有限50×14、前缀精确；正式最大新关节步进0.147517rad/tick，小于保留的0.2拒绝值。服务推理P50/P95=199.39/209.71ms，**Thor本机WebSocket往返200.07/211.03ms**，不是IPC全链路。
- 握手绑定本页weight/norm/engine指纹、`trained`/`quantile`、7步、dmax10；14D布局为左6关节/左夹爪/右6关节/右夹爪，返回逆变换后的绝对目标。关节rad，夹爪连续值标称0闭/1开不裁剪；30Hz policy tick，第0步对应observation policy tick，不把相机曝光时间当目标tick。

`onnx_exported_engine_not_validated`和`compared_not_robot_task_validated`是原阶段状态，保持原样；数值一致和协议通过不证明实机成功率或泛化优于20万/23万。

## 正式交接故障与恢复

[原部署回执](deploy-receipt.txt)保留候选测试通过及后续交接失败，[诊断摘录](handoff-diagnostic.txt)保存systemd/journal观察。原`thor-rtc-50h240000-deploy-20261014.service`于15:14:54退出1，**不能宣称退出0**：`systemd-run`异步返回后，`ready()`立即把尚未Running的容器判失败；回退重新复用`pi05-rtc-infer`名称时，原MAXN watcher继续跟随该名称，新的回退MAXN unit在15:14:55报`BlockingIOError: Resource temporarily unavailable`（flock锁冲突）。这是启动/生命周期问题，未出现24万精度门槛失败。

未改写或重跑冻结脚本，单独使用[恢复脚本](rtc50h240000-promote-recovery-20261014.sh)：停止临时恢复的20万，等待旧watcher退出并恢复120W，保留20万容器，再恢复已验收24万为同名服务。新启动等待容器Running及healthz，MAXN watcher绑定容器唯一ID，不再跟随可复用名称；正式九例通过后才标为上线。[恢复回执](promote-recovery-receipt.txt)的`DEPLOYED_50H_240000_RECOVERY_VERIFIED_PRESERVED_200000_AND_230000`完整保留，`thor-rtc-50h240000-recovery-20261014.service`退出0/inactive。完整CPU记录见[prepare-receipt.txt](prepare-receipt.txt)；本地文本副本仅去掉行尾空白，Thor原始日志未改。

只保留一个在线Pi服务、`--restart no`，不开机自启；停止服务后MAXN会话恢复120W。恢复20万/23万须先停当前容器并等待MAXN退出/120W，再保留当前24万、切回备份并按对应权重指纹重新协议验收；不得并行占用8000或自动启动PLD。当前可用544,594,378,752 bytes，约507.2GiB，未删除任何模型。

## 当前产物定位

Thor根`/home/wuyan-lyj/thor/pi`；原始/转换分别为`checkpoints/lego-pi05-rtc-base-50h-20260921/240000`和`240000-pytorch-fp32-r1`；回放`test-data/pi05-rtc-50h-240000-replay-20261014-r1`；独立JAX/ONNX/TRT产物均以`artifacts/rtc-50h-240000-…-20261014-r1`命名。冻结工具与日志在`probes/rtc-50h-240000-20261014-r1`。Pi镜像ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`，JAX参考镜像ID `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`，与此前验证环境一致。

## 10月13日下载与授权边界（历史）

2026-10-13，用户先要求暂停`pld-rl`并开启240000；随后明确收窄为：**先下载，下载完成通知，取得新的明确授权后才能继续。** 当时严格按最后指令，仅下载推理必要文件及做传输完整性校验，没有转换、参考、构建、测试或上线。10月14日早先恢复20万也不构成24万授权；本页当前状态中的后续独立授权才允许推进。

**2026-10-13 15:34:48+08下载结束，专用任务退出0，18项逐文件SHA全部通过。** 含合同共18个文件、12,440,565,864 bytes（约11.59GiB）；原始校验输出见[download-only.log](download-only.log)。15:35:48+08再次复核：Thor的`pld-rl`已停止、其MAXN watcher退出0/inactive，宿主120W；原Pi200000、230000备份也停止，8000无监听，没有运行中的容器。

下载结束时24万FP32转换目录、有限性审计文件和TensorRT目录均不存在，证明当时未进入后续模型阶段。下载完成不等于权重数值有限、转换准确或推理就绪；当时等待新的授权，现已另获10月14日本次授权。所有既有权重、引擎、容器和服务器源保留；未读取或修改3588，未发送客户端任务。

## 源与目标

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/240000`，只读核对正式Orbax完成元数据、params元数据和checkpoint内norm。
- Thor目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-50h-20260921/240000`；独立于200000/230000，不覆盖旧版本。
- 只取`params/`、`assets/`、`_CHECKPOINT_METADATA`，共17个源文件、12,440,542,241 bytes；另取同run控制目录的`training_contract.json`。**不取train_state，不是完整训练续跑备份。**
- 完成元数据SHA256 `49c4288ca7fc4f260ce2025d7e39e4ea736e2604a914ea7fd7af4fc91ac0edfe`；norm SHA256 `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`；训练合同SHA256 `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`。18项完整清单[source.sha256](source.sha256)已全部核验。norm/合同与此前同run20万、23万相同，但它们的引擎/参考不能代替24万自身的后续验收。

## 下载边界与原自动流程的截断

初始完整接入脚本只进入服务器源文件SHA扫描，还没有实际下载或转换时，收到用户的新边界。立即SIGSTOP精确暂停本任务编排shell PID2709011，使其不可能自动进入CPU转换；源SHA扫描子进程2709014继续完成。随后独立下载专用任务复用这次扫描结果，核对源清单含所有必需元数据后，只终止该冻结编排shell，不终止数据传输；原SSH会话因此退出137。这是主动遵守新指令，不是权重损坏或下载失败。

唯一下载任务为[下载专用脚本](rtc50h240000-download-only.sh)，六路rsync由Thor直接读取服务器，通过当前SSH会话临时转发已有agent认证，不复制私钥、不在仓库保存凭据。任务末尾只有逐文件SHA校验和`DOWNLOADED_ONLY_50H_240000_WAIT_FOR_USER_AUTHORIZATION`，没有任何Docker模型执行命令；CPU转换及GPU部署单元没有启动。

Thor stage为`/home/wuyan-lyj/thor/pi/probes/rtc-50h-240000-20261013-r1`。其中此前准备的自动prepare/deploy脚本**未获新的继续授权，不得执行或恢复**。下载专用脚本也绑定本次源扫描PID，仅作操作证据，不应当成通用重跑入口。200000/230000备份和PLD容器保持原样，未经用户指令不清理或重新启动。

## 10月14日已获授权的部署路线

本轮授权后按[RTC接入手册](../../../reference/thor/13_trained_rtc_inference.md)执行原始参数有限性检查、FP32保真转换、checkpoint自身norm的真实d=0/1/10回放、MAXN七步JAX/ONNX/TF32 TensorRT对照、同地址协议测试与上线。继续保持分位数前缀、H50/14D绝对目标、0.2rad/tick跳变拒绝，不量化或放宽阈值。入口为[CPU准备](rtc50h240000-prepare-20261014.sh)及[GPU/上线/回退](rtc50h240000-deploy-20261014.sh)，执行前冻结，不覆盖正在执行的脚本。PLD/20万/23万和源数据均保留；只有全部验收通过才能标为上线。
