# 耳机clean-v2 retry4：62000步下载与离线转换（未上线）

## 当前：转换、构建与离线校验完成，Pi保持停止

2026-10-30下载完成后，用户追加授权转换、暂不启动推理；本轮执行CPU审计/FP32保真映射、七步JAX/Torch/ONNX/TRT离线构建与对照，不创建或启动任何Pi WebSocket服务，不改原8000容器。GPU资源授权明确为“只暂停H3，agent不暂停”：`comfyui-h3`在GPU阶段前停止，退出时恢复；`yam-agent-perception`始终保持运行。沿用宿主启动前已有MAXN/锁频，不新建专属MAXN watcher、不擅自恢复120W影响agent；本轮时延属于共享负载。

16:30+08 CPU准备完成：原始参数非有限0、811张量/3,353,433,872元素FP32逐位映射一致；权重SHA `80b97378cfa44a9a39dda91306a20f0940d1c33c77f5db8f18d7614b51c3d8c7`。约16:47，七步JAX/Torch/ONNX/TF32 TensorRT离线流程退出0，九例误差门槛与0.2rad/tick步进检查均通过。完整[GPU回执](gpu-offline-receipt.txt)包含`CONVERTED_VALIDATED_EARBUDS_62000_NO_PI_SERVICE`和`H3_RESTORED_AFTER_GPU_EXIT_0`。复核H3 running/healthy，agent运行且启动时间仍为10月26日，Pi running=false、8000无监听、原容器仍指向乐高24万，未创建任何新Pi服务。宿主沿用已有MAXN，未改变其他任务功耗，可用约267GiB。禁止重跑已有产物的冻结脚本；后续上线须另获授权并按新权重/norm/引擎重新配置同一个Pi服务，再做协议验收。

入口为[CPU准备](prepare.sh)、[回放重绑定](rebind_cases.py)、[GPU离线流程](gpu_offline.sh)、[离线步进检查](offline_guard_check.py)。原始九例仍来自同一clean-v2验证集episode0/1/2早中晚、d=0/1/10；源root与新合同一致，服务器当前三份选中data Parquet、episodes Parquet和info.json五项SHA均与原suite相同，观测/绝对前缀数组及其指纹不变。没有重新传输视频；本轮没有重新逐帧解码或全量视频复审，不外推任务成功率。

## 离线结果与产物

固定七步FP32权重＋TF32计算、text bucket80/时间缓存＋CUDA Graph，分位数前缀、干净token/flow time0、只去噪后缀，训练dmax10/H50不变；没有BF16转存、FP8量化、guidance或普通推理降级。原JAX/FP32参考关闭TF32，候选TRT开启TF32；同真实观测、同绝对前缀、同固定噪声比较完整逆变换后的50×14绝对动作。

| 指标 | 结果 | 证据/口径 |
|---|---:|---|
| 原始参数非有限数 | 0 | [原始审计](rtc-earbuds-clean-v2-62000-audit-20261030-r1.json) |
| FP32保真映射 | 811张量、33.53亿元素逐位一致 | [转换审计](conversion_audit.json)，不等同推理精度验收 |
| JAX/Torch混合14D最大物理数值差 | 2.90227e-6 | [导出报告](export_report.json)，九例1e-4门槛和缓存wrapper1e-5门槛均通过，不统一标rad |
| JAX/TRT最大单关节差 | 0.000625979rad，约0.0359° | [TRT验证](validation.json)，最坏case2/第33步/dim3，不是每关节都有该误差 |
| JAX/TRT最大夹爪差 | 0.000570210 | 连续夹爪值，标称0～1范围约0.057% |
| 混合14D P99差 | 0.000302881 | 原max≤0.005/P99≤0.001门槛未放宽 |
| 27次预热后纯推理P50/P95 | 188.20/188.58ms | H3暂停、agent仍运行的共享GPU离线测量 |
| 预热后本机总处理P50/P95 | 191.78/192.46ms | 非WebSocket往返、非3588全链路 |
| 首轮含Graph捕获推理P95 | 367.29ms | 单列冷路径，不冒充稳态 |
| 最大新后缀关节步进 | TRT 0.079821rad/tick | [独立离线步进回执](offline-guard-receipt.json)，JAX也通过0.2rad/tick，九例前缀精确 |

[原JAX参考](reference_manifest.json)、[转换身份](rtc_manifest.json)、[引擎回执](engine_report.json)、[九例绑定](cases.json)和[回放suite](suite.json)均绑定新checkpoint与自身norm。engine SHA为`fb79a8b13a19e763d93439e40a71f3881a263398ec4416fe52b57faf60c927f2`，TRT构建约224.78秒。阶段JSON保持原`built_experiment_not_accuracy_validated`/`compared_not_robot_task_validated`语义，不改写为实机或线上已验收。原驱动版本格式、Orbax params子目录缺独立metadata和TRT默认stream警告保留于完整日志，本轮未因此失败，也未修改驱动或放宽精度门槛。

Thor原始和FP32目录分别为`checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000`和`62000-pytorch-fp32-r1`；引擎`artifacts/rtc-earbuds-clean-v2-62000-trt-tf32-cache80-7step-20261030-r1`，对应JAX/ONNX产物同前缀，真实回放`test-data/pi05-rtc-earbuds-clean-v2-62000-replay-20261030-r1`。复用Pi镜像`a6227665…`和原JAX参考镜像`e3bb8e48…`，20份基础工具来源/指纹见[frozen_tools](frozen_tools.json)，本轮任务脚本独立冻结，没有同步整个工作树或更改镜像依赖。未做WebSocket、跨IPC或机器人任务测试，没有操作3588/训练或发送客户端任务。

## 下载阶段（历史授权与结果）

2026-10-30用户最初要求拉取最新耳机训练检查点，当时只下载与核对完整性；之后另行授权转换和暂不上线，见上节。查询服务器各耳机run后，锁定新分支最新完整62000，不能按跨run的数值大小误选旧114000；后续训练产生新保存点也不静默追更本轮目标。

**2026-10-30 16:20:15+08下载与校验完成。** [下载回执](download-receipt.json)为`downloaded_sha_verified_not_converted`，19文件/12,440,562,368 bytes（约11.59GiB），[源SHA清单](source.sha256)逐项通过，冻结脚本退出0。16:20后复核Pi仍running=false、目标无train_state，Thor可用约303GiB。没有转换、参数有限性审计、精度/时延测试、启动或切换推理；SHA通过不当作模型质量或部署验收。

已核对[训练合同](training_contract.json)、[初始配置](initial_config.json)与正数Orbax commit时间。checkpoint实际norm SHA `b7c63db5b78959d5da3ce720fe8f45c075eac57ed97bf949dacc9c86108f1878`，相对训练norm仅缺末尾一个换行，身份检查通过；这是新分支自己的norm，不是旧114000的norm。后续转换必须绑定本目录资产与独立新报告/引擎，不复用旧模型产物。

## 身份与路线

- 源run：`/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/62000`。已确认正式`_CHECKPOINT_METADATA`、正数commit时间、params/_METADATA和checkpoint自身norm存在，不取临时保存目录。
- 源control：`/home/wuyan/lyj/YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022`，配套`training_contract.json`和`initial_config.json`独立取回。
- 本run从旧耳机114000/params初始化、步数重新计数，使用clean-v2排除pantheon后的数据与新norm；62000不代表原run训练倒退。合同Pi0.5/H50/32D、训练时RTC dmax10，训练norm SHA `05f65ada286965f1c5c5b32e62bf3fcdafb7d41d2e22d6683e0c05efc443c5b2`，不能复用此前旧114000 norm或引擎。
- Thor目标：`/home/wuyan-lyj/thor/pi/checkpoints/earbuds-pi05-rtc-clean-v2-excl-pantheon-retry4-20261022/62000`；只取params/assets/完成元数据和合同配置，不取train_state/优化器。源params/assets共12,440,526,965 bytes，约11.59GiB。
- Thor stage：`/home/wuyan-lyj/thor/pi/probes/rtc-earbuds-clean-v2-62000-20261030-r1`。冻结[下载脚本](download.sh)通过工作站转发既有SSH agent，Thor直接从服务器读取六路rsync，不复制私钥或先落工作站大权重；flock防同stage重复下载，保留partial以供明确恢复。
- 下载后逐文件SHA核对，校验本run配置/commit/RTC合同及checkpoint自身norm身份；只容许既有Orbax少末尾一个换行的norm差异。SHA一致不等于参数有限性、精度或部署验收；随后另获授权完成的离线结果见上节，线上/实机尚未验收。

此前清理的旧114000模型不恢复；乐高20万/23万/24万与官方基础权重不动。其他容器仅按本次授权暂停并恢复H3，agent未中断。
