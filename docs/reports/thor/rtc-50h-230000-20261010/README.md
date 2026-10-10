# 50h230000 RTC：替换193000

2026-10-10 10:26 CST复核：本轮选择当时最新完整230000，已替换193000。`pi05-rtc-infer`运行、healthz OK、`thor-pi-maxn-50h230000.service` active/MAXN，固定`ws://192.168.250.1:8000`。七步FP32权重＋TF32、quantile前缀、缓存80/CUDA Graph；18项源SHA、有限性/保真转换、九例JAX/TRT及候选/上线两轮协议均通过。旧193000本地大权重/引擎和旧容器已永久删除；真机任务效果及3588全链路未验收。

## 来源与身份

- 完整源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/230000`，完成元数据SHA `33554428d9198932ad40965ff72c8f94bebc6fce70e670b83a85a081388de860`。
- 六路Thor直拉，仅params/assets/完成元数据及同run控制目录合同，不取train_state；约09:57开始、10:06下载完成。[18项源SHA](source.sha256)含训练合同，全部通过。
- 合同SHA `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`，H50/32D/dmax10；norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`，与193000相同。严格检查确认norm只比训练源少一个末尾换行，不是数值改变。
- [原始参数审计](rtc-50h-230000-audit-20261010-r1.json)非有限0；[转换审计](conversion_audit.json)811张量、3,353,433,872元素映射加载逐位一致，FP32、无LoRA。转换权重SHA `1d6a30a0dcbd15cc491985f46b73e46fd5278acf5fe8f635a3e85f06173d77a6`；原参数文件身份见[manifest](rtc_manifest.json)。
- 冻结复用193000脚本及Pi镜像`openpi-pi:thor-trained-rtc-candidate-20260916`（ID `sha256:a6227665c32d1c7b2b7ca01a2d1a29e18d64bf764e5914db6eb06db5419a8e56`），JAX参考镜像`openpi-pi:thor-trained-rtc-jax-ref-20260917-r3`（ID `sha256:e3bb8e489e896b2ba88a7d811f845dd09fa73150d6a510d1d6cb5709fd22644d`），本轮实际镜像身份已复核。没有同步整个当前工作树；[脚本SHA](deployed_scripts.sha256)绑定实际执行源码。

## 测试

从真实三相机suite及源Parquet重新生成本次norm绑定[九例](cases.json)，episode95/96/97早中晚、d=0/1/10。同输入/噪声、同七步，对比完整YAM逆变换后物理动作。保持FP32权重＋TF32 TensorRT、quantile前缀、80-token时间缓存/CUDA Graph，不量化；0.2rad/tick粗大关节跳变拒绝未放宽。

- [原JAX参考](reference_manifest.json)和[导出回执](export_report.json)九例均通过，JAX→Torch最大物理数值差9.6295e-6，缓存wrapper门槛1e-5通过。混合14D数值没有统一物理单位，不当成关节误差。
- [TensorRT构建](engine_report.json)与[验证](validation.json)通过；engine SHA `1c01c63d0381280b9c33a5471f208ec0aa8dce0a161789cc56cddf4551f7466a`。最大单关节转换差0.00106404rad、双夹爪最大差0.00075294；混合14D P99=0.00043395，仅作既有部署数值门槛。输出有限、已承诺前缀保持。
- MAXN稳态推理P50/P95=189.73/193.95ms，总处理193.25/198.46ms。首轮含Graph捕获的推理P95=377.26ms，单列，不冒充稳态。
- [候选九例](50h230000-smoke.json)与[上线后九例](50h230000-online-smoke.json)协议均通过，握手权重/norm/engine匹配、quantile、七步、trained RTC。返回有限50×14绝对目标，关节rad、夹爪连续标称0闭/1开（不裁剪），d≤10；30Hz policy tick语义未变。
- 上线本机服务推理P50/P95=198.70/208.97ms，WebSocket往返199.45/210.09ms；最大新后缀关节步进0.09030rad/tick，小于0.2拒绝值。候选阶段随机采样最大步进0.14903rad/tick也通过；两次非固定噪声协议测试不作为后端误差比较。

阶段原始回执的`engine_not_validated`/`compared_not_robot_task_validated`分别描述导出/离线阶段，后续协议通过不修改它们。未评估专家真值、闭环成功率或相机曝光到控制tick物理偏移。

## 清理与恢复

上线复测通过后，按本轮授权永久删除Thor193000原始params、转换model.safetensors、七步ONNX外部权重/TRT引擎及旧停止容器`pi05-rtc-50h193000-preserved-20261010`。保留norm、配置、小报告、共享镜像和其他模型。删除不是回收站。

[部署与删除回执](deployment_receipt.txt)记录清理前后Used为303989633024/252053745664字节，释放51935887360字节，约48.37GiB；可用701024256000字节，约653GiB。旧四目录已无大于100MB的遗留权重，旧容器已移除。**操作前09:56及完成后复核，服务器193000源目录均已不存在**；旧版不能本地直接回退，也不能承诺从该服务器重新下载，恢复须另有可用备份。本轮未删除服务器任何权重；230000服务器源仍存在。

Thor根`/home/wuyan-lyj/thor/pi`：检查点`checkpoints/lego-pi05-rtc-base-50h-20260921/230000`，转换件`230000-pytorch-fp32-r1`；engine `artifacts/rtc-50h-230000-trt-tf32-cache80-7step-20261010-r1`，stage `probes/rtc-50h-230000-20261010-r1`，cases `test-data/pi05-rtc-50h-230000-replay-20261010-r1`。新原始JAX、转换件、JAX参考、ONNX及TRT保留。

一次性unit `thor-rtc-50h230000-deploy.service`已于10:26:02 CST退出0，最终标记`DEPLOYED_50H_230000_CLEANED_193000`。[脚本](rtc50h230000-deploy.sh)已执行，不重复运行；恢复先核对实际服务及回执。在线MAXN跟随容器，停止后恢复120W，非开机常驻。管理SSH仍为`thor`别名/10.18.10.8（DHCP实测，不是地址保留）。

本次只操作Thor，不访问3588、不发送客户端任务；本机/离线测试不代替用户真机任务效果或全链路验收。
