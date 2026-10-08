# 50h169000 RTC：部署与旧模型清理

2026-10-08 11:54 CST复核：`pi05-rtc-infer`已运行169000，固定 `ws://192.168.250.1:8000`、healthz OK、`thor-pi-maxn-50h169000.service` active/MAXN。七步FP32权重＋TF32 TensorRT、quantile前缀、缓存80/CUDA Graph，未量化。返回有限50×14绝对动作，关节rad、夹爪连续值，RTC d≤10；0.2rad/tick粗大关节跳变拒绝保持。

## 来源与精度

- 来源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/169000`。只取params/assets/完成元数据及同run控制目录合同，不取train_state；[19项源SHA](source.sha256)全部通过。
- 根目录完成元数据SHA：`15c2352d190294282da9aab6f38534cad004ed2198ea696b13079f84783e6849`。manifest中的metadata字段指params/_METADATA，不是根完成标记。
- 合同SHA `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`；norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`，与147000相同。严格检查证明与训练合同只差一个末尾换行，JSON数值一致。
- 转换权重SHA `287098d290e18bd57e7d5a43fdece1818c833a71a52318fb49f3f21f97e20dbf`；engine、参考输出及cases身份见[验证回执](validation.json)。
- [原始参数审计](rtc-50h-169000-audit-20261008-r1.json)NaN/Inf为0；[转换审计](conversion_audit.json)811张量映射/加载逐位一致；[manifest](rtc_manifest.json)保存源参数文件身份。
- 复用已验证Pi系列镜像`openpi-pi:thor-trained-rtc-candidate-20260916`（ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`），JAX参考镜像r3；冻结复用147000阶段RTC脚本，新目录保存本次产物，没有部署工作树其他功能改动。

## 测试

从既有真实三相机suite及源Parquet重新生成本次norm绑定的九例，episode95/96/97早中晚，覆盖d=0/1/10。同输入/同噪声/七步，完整YAM逆变换后比较。离线数据tick不证明现场曝光与控制时序。

- [JAX→FP32及缓存wrapper](export_report.json)九例均通过，前者最大物理输出差约9.8885e-6。
- [JAX/TRT验证](validation.json)最大关节差0.00075175rad，夹爪最大差0.00080962；混合14D数值P99差0.0004390，用于既有部署数值门槛，不当成统一物理单位。输出有限、前缀保持，服务门槛通过。
- MAXN稳态推理P50/P95=187.62/188.57ms，总处理191.14/192.51ms。首次含CUDA Graph捕获的九例推理P95=365.85ms，不能混作稳态延迟。
- [候选协议九例](50h169000-smoke.json)和[上线后九例](50h169000-online-smoke.json)均通过，握手权重/norm匹配、输出有限50×14、前缀保持。上线后服务推理193.46/203.85ms，本机往返194.17/204.90ms（P50/P95），最大新后缀步进0.08538rad/tick。

这是Thor本机协议及离线数值验收，真机任务效果与3588全链路仍待用户测试。未操作3588或发送客户端任务指令。

## 清理与恢复

用户授权删除147000/136000。在169000上线协议通过后永久删除两者原始params、转换model.safetensors、各自七步ONNX外部权重与TRT engine，以及两个已停止旧服务容器。配置、norm、小报告与服务器原件保留；旧模型不能本地直接回退，恢复须重新下载/转换。清理前后磁盘Used为355682676736和251810775040字节，释放103871901696字节，约96.74GiB；当前可用约654GiB。

Thor根 `/home/wuyan-lyj/thor/pi`：新权重在`checkpoints/lego-pi05-rtc-base-50h-20260921/169000-pytorch-fp32-r1`；engine在`artifacts/rtc-50h-169000-trt-tf32-cache80-7step-20261008-r1`；stage在`probes/rtc-50h-169000-20261008-r1`；cases在`test-data/pi05-rtc-50h-169000-replay-20261008-r1`。原JAX、JAX参考和本次ONNX保留。

[一次性部署脚本](rtc50h169000-deploy.sh)已完成，不重复执行；unit `thor-rtc-50h169000-deploy.service`11:54:00退出0，标记`DEPLOYED_50H_169000_CLEANED_147000_136000`。在线MAXN临时会话跟随容器，停止后恢复120W，不设开机常驻。

本轮旧SSH地址10.18.10.89失效，通过设备广播发现并用原主机密钥验证Thor当前Wi-Fi为10.18.10.8/23，已更新本机thor别名。管理SSH中断一次，Thor已有传输继续；重连后收尾rsync与19项SHA通过，没有把部分下载当成完整检查点。地址仍是DHCP观察值；生产直连192.168.250.1不变。
