# 50h147000 RTC：Thor测试部署

2026-09-30 18:03 CST复核：固定 `ws://192.168.250.1:8000` 已运行50h147000，`pi05-rtc-infer`、healthz、`thor-pi-maxn-50h147000.service`正常，宿主MAXN。用户要求拉取147000并继续测试；完成转换及离线验收后接入同地址供用户现场测试。此前20h136000已停止，权重/引擎和容器 `pi05-rtc-20h136000-preserved-20260930`保留。

## 来源与配置

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/147000`。下载params/assets/完成元数据与同run控制目录合同，不取train_state；17项文件源/Thor SHA全部一致。
- 完成元数据 `_CHECKPOINT_METADATA` SHA：`6424968e606b8a0318ad8c901d8d404da000302685ccc2beb1d71552b5767688`。manifest内`checkpoint_metadata_sha256`指params/_METADATA，两者不是同一个文件。
- 合同SHA：`924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`。Pi0.5全量微调，H50/内部32D，训练RTC dmax10。
- checkpoint norm SHA：`fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`。训练合同norm SHA为`4e95507966eea44a58c2ecf32874e0e066adf1467f390c69395c4a2861dc30dd`；严格检查证明checkpoint只省略一个末尾换行，数值一致。不能沿用20h norm。
- FP32转换权重SHA：`2c3e2ea15b16178ebd3faed67defdd8b3d823b12415b7cfcc5289fcabca5475d`；engine SHA：`7caacdf30afab8e5217314412674615928e5fcfe16e5231f7c58eb2bbc0cfbea`。
- 七步FP32权重＋TF32 TensorRT，非量化，文本桶80/时间缓存/CUDA Graph。quantile前缀、d=0～10、0.2rad/tick关节跳变拒绝保留。返回50×14绝对目标，关节rad、夹爪连续值。
- 镜像仍为`openpi-pi:thor-trained-rtc-candidate-20260916`，ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`；JAX参考镜像r3 ID `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`。冻结复用此前已验证RTC脚本，未把当前工作树PARTS/RLT开发改动部署到Thor。

## 验收

- [原始有限性审计](rtc-50h-147000-audit-20260930-r1.json)：NaN/Inf总数0。
- [FP32转换审计](conversion_audit.json)：811张量映射到加载逐位一致。[RTC manifest](rtc_manifest.json)绑定源参数SHA、norm、合同和权重。
- 从既有真实三相机回放及源Parquet生成新norm绑定的九例，覆盖episode95/96/97早中晚、d=0/1/10。真实state/图像/前缀未改，离线tick来自记录数据；不证明现场相机曝光与控制时序。
- [JAX/Torch及wrapper对照](export_report.json)九例均通过原门槛；JAX→FP32最大物理输出差约1.4948e-5。
- [JAX/TRT对照](validation.json)：最大关节差0.001309rad，夹爪差0.0009483；混合14D数值P99差0.0004188，仅作既有数值门槛，不视为统一物理单位。全部输出有限、前缀保持，服务精度门槛通过。
- MAXN稳态推理P50/P95=188.58/189.68ms；总处理192.05/193.12ms。首次含图捕获的九例推理P95=370.25ms，和预热后稳态分别记录。
- [候选协议九例](50h147000-smoke.json)与[上线后九例](50h147000-online-smoke.json)均通过，握手权重/norm身份正确、输出有限50×14、前缀保持。上线后服务推理195.51/205.19ms，本机往返196.10/206.19ms（P50/P95），九例最大新后缀步进0.11829rad/tick。

任务成功率与3588全链路时延待用户现场测试；数值验收不代替任务评价。未操作3588、未向客户端任务发送指令。

## 恢复入口

Thor根 `/home/wuyan-lyj/thor/pi`：checkpoint父目录`checkpoints/lego-pi05-rtc-base-50h-20260921`下147000及147000-pytorch-fp32-r1；产物`artifacts/rtc-50h-147000-{jax-quantile-7step,onnx-fp32-cache80-7step,trt-tf32-cache80-7step}-20260930-r1`；回放`test-data/pi05-rtc-50h-147000-replay-20260930-r1`；脚本与回执`probes/rtc-50h-147000-20260930-r1`。

[本次部署脚本](rtc50h147000-deploy.sh)为固定路径的一次性执行证据，已执行完成，不重复运行。`thor-rtc-50h147000-deploy.service`18:01:52退出0，标记`DEPLOYED_50H_147000_136000_RETAINED`。在线MAXN由跟随容器的临时会话维持，停止后恢复120W，不设开机常驻。当前可用约606GiB；本轮没有删除旧模型。
