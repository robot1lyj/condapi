# 50h193000 RTC：替换169000

2026-10-09 10:06 CST复核：193000已替换169000，`pi05-rtc-infer`运行、healthz OK、`thor-pi-maxn-50h193000.service` active/MAXN，固定`ws://192.168.250.1:8000`。七步FP32权重＋TF32、quantile前缀、缓存80/CUDA Graph；18项源SHA、有限性/保真转换、九例JAX/TRT及候选/上线两轮协议均通过。169000本地大权重/引擎和旧容器已永久删除；真机任务效果及3588全链路未验收。

## 来源与身份

- 完整源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/193000`，完成元数据SHA `95b11aecaad752e0e6da94678f2536f4245d6031b53fc2b4df25d983dbe0f449`。
- 只取params/assets/完成元数据及同run合同，不取train_state。六路Thor直接下载，[18项源SHA](source.sha256)含训练合同，全部通过。
- 合同SHA `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`；norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`，与169000相同。训练合同H50/32D/dmax10；严格检查确认norm只比训练源少一个末尾换行，不是数值改变。
- [原始参数审计](rtc-50h-193000-audit-20261009-r1.json)非有限0；[转换审计](conversion_audit.json)811张量、3,353,433,872元素映射加载逐位一致，FP32、无LoRA。转换权重SHA `4c308c17f4c65bbc06c29a33823ab48a90238750c73100472660ca74b88ee126`；原参数文件身份见[manifest](rtc_manifest.json)。
- 复用169000已验收的冻结RTC脚本及Pi系列镜像`openpi-pi:thor-trained-rtc-candidate-20260916`（ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`），JAX参考镜像`openpi-pi:thor-trained-rtc-jax-ref-20260917-r3`（ID `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`）。不是同步当前整个工作树；[部署脚本SHA](deployed_scripts.sha256)绑定实际执行源码。

## 测试

从真实三相机suite及源Parquet重新生成本次norm绑定[九例](cases.json)，episode95/96/97早中晚、d=0/1/10。同输入/噪声、同七步，对比完整YAM逆变换后物理动作。七步FP32权重＋TF32 TensorRT、quantile前缀、80-token时间缓存/CUDA Graph，不量化；0.2rad/tick粗大跳变拒绝未放宽。

- [原JAX参考](reference_manifest.json)和[导出回执](export_report.json)九例均通过，JAX→Torch最大物理差4.6981e-6，缓存wrapper门槛1e-5通过。混合14D差异没有统一物理单位，不当成关节误差。
- [TensorRT构建](engine_report.json)与[验证](validation.json)通过；engine SHA `7e245cc052f8f5181105f61062e92127501b4a4f60b78d904974d546aba22159`。最大单关节转换差0.00083352rad、双夹爪最大差0.00092324；混合14D P99=0.00042757，仅作既有部署数值门槛。输出有限，已承诺前缀保持。
- MAXN稳态推理P50/P95=187.46/189.03ms，总处理191.06/192.88ms。首轮包含Graph捕获的推理P95=369.15ms，单列，不冒充稳态。
- [候选九例协议](50h193000-smoke.json)与[上线后九例](50h193000-online-smoke.json)均通过，握手权重/norm/engine匹配、quantile、七步、trained RTC。返回有限50×14绝对目标，关节rad、夹爪连续标称0闭/1开（不裁剪），d≤10；30Hz policy tick语义未变。
- 上线本机服务推理P50/P95=195.00/204.58ms，WebSocket往返195.56/205.82ms；最大新后缀关节步进0.13600rad/tick，小于0.2粗大跳变拒绝值。这个阈值和数值一致性不证明所有机器人任务安全/有效。

阶段回执的`engine_not_validated`/`compared_not_robot_task_validated`描述各自阶段，后续协议通过不修改原回执。没有对专家真值、闭环成功率或相机曝光到控制tick做新结论。

## 清理与恢复

上线复测通过后，按本轮授权永久删除169000原始params、转换model.safetensors、七步ONNX外部权重、TRT引擎和旧停止容器`pi05-rtc-50h169000-preserved-20261009`。保留norm、配置、小报告与服务器原件；旧版不能本地直接回退，恢复须重新下载转换。其他检查点、基础模型和共享镜像不动。删除不是回收站。

[部署与删除回执](deployment_receipt.txt)记录清理前后Used为303941111808/252005105664字节，释放51936006144字节，约48.37GiB；可用701072896000字节，约653GiB。复核旧四目录无大于100MB的遗留权重，旧容器已移除。

Thor根`/home/wuyan-lyj/thor/pi`：检查点`checkpoints/lego-pi05-rtc-base-50h-20260921/193000`，转换件`193000-pytorch-fp32-r1`；engine为`artifacts/rtc-50h-193000-trt-tf32-cache80-7step-20261009-r1`，stage为`probes/rtc-50h-193000-20261009-r1`，cases为`test-data/pi05-rtc-50h-193000-replay-20261009-r1`。新原始JAX、转换件、参考、ONNX及TRT保留。

一次性unit `thor-rtc-50h193000-deploy.service`已于10:05:17 CST退出0，最终标记`DEPLOYED_50H_193000_CLEANED_169000`。本次[脚本](rtc50h193000-deploy.sh)已执行，不重复运行；恢复先核对实际服务与回执。在线MAXN跟随容器，停止后恢复120W，非开机常驻。管理SSH仍为`thor`别名/10.18.10.8（DHCP实测，不是地址保留）。

本次只操作Thor，不访问3588，不发送客户端指令；离线及Thor本机smoke不代替全链路或真机任务效果。
