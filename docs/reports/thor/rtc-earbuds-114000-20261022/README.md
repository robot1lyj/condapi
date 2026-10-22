# 耳机 Pi0.5 RTC 114000：替换112000并清理旧推理

## 当前状态

**2026-10-22 11:05 CST上线及旧版清理完成，11:06:24复核正常。** 用户明确要求删除Thor上的112000步耳机推理、替换为最新完整的114000。下载20文件、12,440,582,734 bytes全部SHA通过，原始参数非有限0、811张量FP32保真映射、九例真实回放独立JAX/Torch/TRT及临时/正式两轮协议均通过；新114000已在原`pi05-rtc-infer`/`ws://192.168.250.1:8000`运行，healthz OK，MAXN watcher active。旧112000在新模型上线通过后才永久清理，释放约48.40GiB；目前约349.39GiB可用。实际真机效果与3588全链路由用户测试，不冒称已验收。

当前容器ID为`6de9d18bb2385d3e5f15aeadb285aab4759967b1ec9b1fe37141b83486363528`，在线unit为`thor-pi-maxn-earbuds114000-20261022.service`，按该唯一ID监控。一次性部署及清理unit均退出0/inactive，[流水线回执](pipeline-receipt.txt)含`PIPELINE_COMPLETE_EARBUDS_114000`，未触发失败回退。CPU审阅页、共享镜像和乐高DAgger均未改动；本轮不训练、不操作3588或发送客户端任务。

## 模型与固定路线

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012/114000`；只取params/assets/完成元数据/训练合同/初始配置，不取优化器或train_state，不修改服务器训练。
- 与112000是同一耳机50h全量训练run，RTC dmax10/H50/32D；本次明确锁定114000，不自行换其他保存点。
- 配套norm SHA为`ee56da7d2361623500ba4c4465a0836b88613c9b389460620d8fcd33a5ae7058`，与112000相同；相对训练norm仅缺一个最终换行，按既有`checkpoint_norm_identity`确认，不重新统计或修改norm。
- 原JAX FP32 → 保真PyTorch FP32 → 七步、80-token/时间缓存、TF32 TensorRT＋CUDA Graph。JAX/Torch诊断关闭TF32，不用BF16或FP8；模型内仍是干净前缀/flow time0、只去噪后缀，前缀按新state做关节delta/夹爪absolute及quantile归一化，无guidance/普通模式降级。
- 原Pi镜像与JAX参考镜像继续复用，分别为`sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`、`sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`。同一个Pi系列服务、同一个`ws://192.168.250.1:8000`，不升级依赖或另建耳机系列镜像。

## 真实回放与验收

复用Thor已有的真实耳机clean-v2验证集episode0/1/2、早中晚九例，d=0/1/10各三例；原RGB/state/prompt及真实action前缀不变，不再次拉视频或实时从服务器读取。用[重绑定程序](rebind_cases.py)核对训练数据源及原始样本/前缀SHA，将provenance、cases重新绑定到114000的参数元数据和norm，再生成本检查点独立的JAX参考、ONNX、TensorRT及协议回执，不复用112000引擎或参考输出。

固定验收门槛不变：FP32 JAX/Torch各1e-4、缓存wrapper1e-5、TF32服务混合14D max≤0.005/P99≤0.001，分别报告关节rad与夹爪连续值；保留0.2rad/tick粗大新后缀跳变拒绝。推理/预热/测试/在线服务均MAXN＋锁频，停服恢复120W。以下均为离线数值或本机协议测量，不代表任务成功率或3588全链路。

| 指标 | 本轮结果 | 口径 |
|---|---:|---|
| JAX/Torch物理／归一化最大数值差 | 5.07567e-6／7.27177e-6 | 九例同输入同噪声，全部通过；混合14D不统一标rad |
| JAX/TRT最大单关节差 | 0.000626153rad，约0.0359° | 最坏case2/第47步/dim10，不是每个关节都有此误差 |
| 左／右夹爪最大差 | 0.000515767／0.000276637 | 标称0～1连续值，不是rad |
| 混合14D P99数值差 | 0.000282291 | 沿用原max0.005/P99 0.001门槛，未放宽 |
| MAXN稳态纯推理P50/P95 | 184.45／188.92ms | 九例再重复三轮，首轮Graph捕获单列 |
| MAXN稳态Thor总处理P50/P95 | 188.18／192.84ms | 含本机输入与逆变换，不含网络/控制端 |
| 正式本机WebSocket往返P50/P95 | 194.07／204.57ms | 九例短快照，不是最坏时延保证 |

[JAX参考清单](reference_manifest.json)、[导出](export_report.json)、[引擎](engine_report.json)与[验证](validation.json)均独立绑定114000；八个ONNX输入保留动态`previous_actions`/`prefix_mask`，缓存wrapper的1e-5门槛全部通过。首轮包含Graph捕获的推理P95=366.30ms，不冒充稳态。TensorRT约217.88秒构建，strongly typed/TF32，无量化。

[临时九例](earbuds114000-candidate-smoke.json)及[正式九例](earbuds114000-online-smoke.json)全部有限50×14、前缀精确、握手身份一致，最大新后缀关节步进分别0.085253／0.084225rad/tick，低于保留的0.2保护；正式服务推理P50/P95=193.50／203.52ms。随机协议输出不作为后端精度比较。新权重SHA为`b70344c1e09b4a0ce3436b50e9c00280210ba754ff4ee17743d7a4286ccdb0c5`，引擎SHA为`ee6744c2e8ec4d83162b413c67916851b49a7c3677acca78c145464c753d6aa3`，配套norm见上文；转换811张量、3,353,433,872元素FP32映射逐位一致。完整身份见[RTC manifest](rtc_manifest.json)、[转换审计](conversion_audit.json)及[源SHA](source.sha256)。

## 产物与执行

Thor根目录`/home/wuyan-lyj/thor/pi`：

- 原始与FP32：`checkpoints/earbuds-pi05-rtc-base-50h-20261012/114000`、`114000-pytorch-fp32-r1`。
- 回放：`test-data/pi05-rtc-earbuds-114000-replay-20261022-r1`。
- 参考/ONNX/引擎：`artifacts/rtc-earbuds-114000-{jax-quantile,onnx-fp32-cache80,trt-tf32-cache80}-7step-20261022-r1`。
- 冻结工具与完整日志：`probes/rtc-earbuds-114000-20261022-r1`；[下载](download.sh)、[复核](verify_download.py)、[CPU准备](prepare.sh)、[GPU/切换](deploy.sh)、[流水线](after_download.sh)仅为本轮执行证据，不能直接重跑。
- 部署unit：`thor-rtc-earbuds114000-deploy-20261022.service`；上线后电源unit按新容器唯一ID跟踪，`--restart no`、不开机自启。

## 112000清理边界

用户授权针对Thor的旧112000推理，不删除服务器原件、共享镜像、乐高DAgger或CPU审阅页。新服务通过后，[清理程序](cleanup_112000.py)核对114000真实协议回执、当前模型挂载、旧112000唯一ID及停止状态，于11:05:36完成永久删除旧容器和以下四个明确目录：

1. `checkpoints/earbuds-pi05-rtc-base-50h-20261012/112000`
2. `checkpoints/earbuds-pi05-rtc-base-50h-20261012/112000-pytorch-fp32-r1`
3. `artifacts/rtc-earbuds-112000-onnx-fp32-cache80-7step-20261021-r1`
4. `artifacts/rtc-earbuds-112000-trt-tf32-cache80-7step-20261021-r1`

小体积历史日志、SHA/JSON、真实回放和JAX参考输出保留，供追溯/后续数值对照；这不构成可运行的112000推理。**112000本地不能回退，已删除内容不在回收站。** [清理回执](cleanup-112000-receipt.json)记录四目录分项及旧容器ID`dc2e00ad…`，可用空间从323,182,272,512增至375,151,640,576 bytes，净释放48.400GiB；没有删共享镜像或服务器文件。旧112000报告已标记退役，旧回退步骤不应再执行。

检查包括本轮真实GPU/协议验收、Python AST、shell语法、清理非Thor拒绝及四项目标白名单、`git diff --check`；没有本地或服务器训练smoke。本轮冻结脚本只针对本次产物和既有容器ID，不作为后续模型可直接重跑的入口。

任务指令不变：`insert the wireless bluetooth earbuds into the charging case`。输入仍为三路RGB/14D state和严格RTC前缀，输出为50×14绝对物理目标；30Hz目标tick、关节rad、夹爪连续值及相机曝光偏移未知的边界均不变。没有读取或修改3588、发送客户端任务、改变训练配方或进行机器人动作测试。
