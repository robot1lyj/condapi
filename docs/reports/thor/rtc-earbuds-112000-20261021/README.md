# 耳机 Pi0.5 RTC 112000：Thor 转换与上线

## 2026-10-22已退役：114000替换并删除本地大产物

用户要求改为114000并删除112000；2026-10-22 11:05新服务真实协议通过后，本页112000旧容器、原始JAX/FP32权重、ONNX和TensorRT引擎已按明确路径永久清理。小体积回放、JAX参考、SHA/JSON和本页历史证据保留，但不构成可运行模型，**不能按下文旧步骤本地回退112000或重启其旧容器**。服务器文件未由本次操作删除；当前服务、清理分项及新验收归[114000报告](../rtc-earbuds-114000-20261022/README.md)。以下记录仅描述10月21日上线时的状态。

## 历史上线状态

**2026-10-21 18:20 CST上线通过，18:20:43复核正常。** 用户授权拉取最新耳机模型、转换、测试并上线。本次锁定接单时最新完整保存点 **112000**，不是继续追逐后台训练随后产生的新保存点。下载、原始有限性、FP32映射、独立JAX/Torch/TRT及两轮九例协议均通过；同名`pi05-rtc-infer`已运行耳机112000，固定`ws://192.168.250.1:8000`，healthz OK，MAXN watcher active、EMC4266MHz锁频。旧乐高DAgger10000停止保留，没有删除模型或改动3588。

在线容器ID为`dc2e00ad1e2581769e2b8aad832f9b1fc75aad122f32a1c60a665fecaa8d5221`；在线电源unit为`thor-pi-maxn-earbuds112000-20261021.service`，按该唯一ID跟踪。一次性部署unit`thor-rtc-earbuds112000-deploy-20261021.service`已退出0/inactive，[流水线回执](pipeline-receipt.txt)含`PIPELINE_COMPLETE_EARBUDS_112000`，未触发失败回退。用户真机任务效果和3588跨IPC全链路仍待测试。

## 模型、数据与精度合同

- 服务器原件：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012/112000`；`_CHECKPOINT_METADATA`有完成时间，未采用任何`*.orbax-checkpoint-tmp-*`目录。
- 实际训练合同：[training_contract.json](training_contract.json)，初始配置：[initial_config.json](initial_config.json)。这是官方base初始化、耳机50h全量训练、训练时RTC dmax10/H50/32D，不是乐高或DAgger权重；没有启动或修改服务器训练。
- 只拉取params/assets/完成元数据/合同/初始配置，不取train_state。[下载回执](download-receipt.json)记录20文件、12,440,586,815 bytes（约11.59GiB），[源SHA](source.sha256)全部通过。
- 耳机checkpoint norm为`ee56da7d2361623500ba4c4465a0836b88613c9b389460620d8fcd33a5ae7058`。与训练norm的`109af693…`只差Orbax省略的一个最终换行，按既有`checkpoint_norm_identity`验证；不修改norm、不重新统计、不复用乐高norm。最初下载末尾的严格字节断言因此退出1，后续[独立复核](verify_download.py)重验全部20项SHA及已知序列化差异后通过；[下载脚本](download.sh)已修正该已知情况，远端首次冻结副本/日志仍保留。
- [原始参数审计](rtc-earbuds-112000-audit-20261021-r1.json)非有限0；[转换审计](conversion_audit.json)811张量、3,353,433,872元素，FP32映射逐位一致。转换权重SHA `1751a7252fcd8d45027c791f36576f1c43b1783c4b752a0354b738eadec2b979`，[RTC manifest](rtc_manifest.json)绑定dmax10/norm/原始参数身份。通用映射器打印的基础配置`rtc_training_max_delay=0`不是此模型路线，实际合同/manifest/采样均为训练时RTC。
- 路线不变：七步、FP32权重＋TF32 TensorRT、80-token/时间缓存/CUDA Graph；JAX/FP32诊断关闭TF32。前缀按新state做12关节delta、夹爪absolute，再显式quantile归一化；模型内前缀干净且flow time0，只去噪后缀。输出逆变换为有限50×14绝对目标，关节rad、夹爪连续值，不裁剪、不guidance/普通采样回退；保留0.2rad/tick粗大跳变拒绝。

## 真实耳机回放与验证

使用已发布clean-v2耳机数据的`val`目录，选择episode0/1/2早中晚九例，d=0/1/10各三例；不是沿用乐高图像或伪造前缀。新增CPU工具`scripts/thor/prepare_lerobot_v3_rtc_cases.py`按LeRobot v3 episode metadata解析Parquet和三相机视频的路径/时间偏移，核对真实PTS与30Hz数据行，前缀直接取同集action；不更改原始数据。服务器只做只读CPU数据提取，随后将约1.9MiB固定观测/前缀传到Thor，推理不实时向服务器取数据。

[cases.json](cases.json)、[suite.json](suite.json)及[原始JAX参考清单](reference_manifest.json)保留输入身份。NPZ/视频来源的完整provenance与原始输出在Thor下述回放/产物目录，未把大二进制提交Git。JAX与Torch使用同一噪声seed0和七步，九例物理最大数值差`3.8448084636044655e-6`、归一化最大差`5.9604644775390625e-6`，全部通过既定1e-4门槛；该混合14D数值不统一标rad。缓存wrapper的1e-5门槛同样全部通过，不宣称缓存重排逐位一致。[导出回执](export_report.json)保留动态`previous_actions`和`prefix_mask`，无量化；[引擎回执](engine_report.json)记录strongly typed、TF32和约217秒构建。[验证回执](validation.json)绑定本次原始JAX/输入/norm/引擎，结果如下：

| 指标 | 本轮结果 | 解释 |
|---|---:|---|
| JAX/TRT最大单关节差 | 0.000810428rad，约0.0464° | 同输入同噪声；最坏case2/第47步/dim10，不是每关节都有这么大误差 |
| 左/右夹爪最大差 | 0.000558419 / 0.000278553 | 标称0～1连续值，不是rad |
| 混合14D P99数值差 | 0.000369341 | 小于原服务0.001门槛；最大元素小于原0.005门槛，未放宽 |
| MAXN稳态纯推理P50/P95 | 184.55 / 185.81ms | 九例再重复三轮；不含网络与控制端 |
| MAXN稳态Thor总处理P50/P95 | 188.02 / 189.38ms | 含本机输入/逆变换，不是跨IPC全链路 |
| 正式本机WebSocket往返P50/P95 | 192.83 / 203.95ms | 九例短快照，不能外推最坏时延 |

首轮含Graph捕获的推理P95为374.26ms，单列，不冒充稳态。[临时九例](earbuds112000-candidate-smoke.json)与[正式九例](earbuds112000-online-smoke.json)均通过：有限50×14、d=0/1/10前缀精确、权重/norm/engine握手匹配，最大新后缀关节步进分别0.075468/0.080041rad/tick，低于保留的0.2保护。正式服务推理P50/P95=192.19/202.74ms；两轮随机协议采样不是同噪声精度对照，也不是任务成功率。

引擎SHA `26839aac33baa2b64adb9b3ba9d2e5fcd5d4c538ddfd99e103560633180cd6e5`，配套norm和权重见上文。TensorRT日志中的`state`未直接使用符合Pi0.5离散state已进入语言token、物理逆变换仍使用state的现有路径；不代表丢弃观测state。CUDA/Orbax版本格式或子目录元数据的启动警告未阻断实际GPU输出，完整日志保留在Thor；未通过改驱动、升级依赖、降低精度或取消RTC来绕过。

新增工具的3项CPU合同测试通过，覆盖路径边界、错误PTS拒绝、v3视频时间偏移、原始前缀/目标tick和norm末尾换行；只使用数组/文件fixture，不是训练smoke。真实提取还完成了九路视频的实际解码。没有读取/修改3588、发送客户端任务或让机械臂执行动作；离线一致性不能替代任务成功率和跨IPC端到端验证。

## 产物、服务与回退

Thor根目录`/home/wuyan-lyj/thor/pi`：

- 原始：`checkpoints/earbuds-pi05-rtc-base-50h-20261012/112000`。
- FP32：同父目录`112000-pytorch-fp32-r1`。
- 回放：`test-data/pi05-rtc-earbuds-112000-replay-20261021-r1`。
- 参考/ONNX/引擎：`artifacts/rtc-earbuds-112000-{jax-quantile,onnx-fp32-cache80,trt-tf32-cache80}-7step-20261021-r1`。
- 冻结工具、全部阶段日志/回执：`probes/rtc-earbuds-112000-20261021-r1`；[CPU准备](prepare.sh)、[GPU/切换](deploy.sh)、[流水线](after_download.sh)只作本轮执行证据，不直接重跑。

Pi与JAX参考镜像沿用10000基线，分别为`sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`和`sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`。仅复制既有冻结工具与新增数据提取器，不同步整个工作树、不升级依赖、不新建耳机模型系列。

正式服务为`pi05-rtc-infer`、`ws://192.168.250.1:8000`；原10000容器已停止保留为`pi05-rtc-separation10000-preserved-20261021`，原始权重/引擎不删，CPU审阅页不动。推理预热/测试/服务运行均MAXN＋锁频，容器停止后按唯一ID watcher恢复120W，`--restart no`、不开机自启。临时验证容器测试后移除，原始模型及全部绑定产物保留；当前可用375,067,197,440 bytes，约349.3GiB。

如后续获准回退，先停止当前耳机容器并等其MAXN watcher退出/恢复120W；将耳机容器改为唯一备份名，恢复旧10000备份为`pi05-rtc-infer`。使用旧10000冻结工具、按恢复后容器唯一ID重新启动MAXN会话，并用旧九例及`19f8db38…`权重指纹复核同一8000；不重跑本轮一次性部署脚本、不同时启动两套GPU服务。

客户端仍发送三路RGB、14D state与RTC committed_actions，动作tick合同不变，任务指令改为训练原文：

```text
insert the wireless bluetooth earbuds into the charging case
```

此处仅展示对接信息，没有向客户端或其他agent自动发送指令。实际相机曝光到policy tick的偏移仍归控制侧，数据行`action[0]`与observation同tick不代表设备零延迟。
