# RTC 80000 · Thor接入

## 输入和转换

源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/80000`。目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-10h-20260916/80000`。只取params/assets/完成元数据与合同，不取train_state。[源文件清单](source.sha256)在流水线中逐项校验通过；CPU原始参数[审计](rtc-80000-audit-20260920-r1.json)非有限0。

转换目录为同父目录`80000-pytorch-fp32-r1`，[转换审计](conversion_audit.json)811张量、3,353,433,872元素映射与加载逐位一致。[RTC manifest](rtc_manifest.json)记录源文件/合同/norm身份；FP32权重SHA-256 `f622b7e37883798df9676f509712a00403a957f82097627d49e118db873ae7ee`。训练合同和分位数norm与60000一致；保持H50、内部32D、物理14D、dmax10。

## 固定配置和产物

按既有七步、FP32权重＋TF32 TensorRT、文本桶80、时间条件缓存、CUDA Graph路线，训练时RTC分位数前缀不变。JAX参考FP32最高矩阵精度、Torch导出关闭TF32。复用三相机真实九例d=0/1/10和历史事故四例d=9；同输入同噪声比较，事故前缀由旧回包重建、现场原噪声未知，不能声称逐位重现现场。测试在MAXN+锁频，结束恢复120W。

Thor `/home/wuyan-lyj/thor/pi/artifacts/` 下本轮目录：

- `rtc-80000-jax-quantile-7step-20260920-r1`
- `rtc-80000-onnx-fp32-cache80-7step-20260920-r1`
- `rtc-80000-trt-tf32-cache80-7step-20260920-r1`
- `rtc-80000-incident-jax-7step-20260920-r1`
- `rtc-80000-incident-trt-7step-20260920-r1`

暂存脚本`probes/rtc-80000-20260920-r1`，使用接入时仓库版本`21cec50`，之后本任务只改记录。一次性任务日志`journalctl -u thor-rtc-80000-test-20260920.service`。候选镜像`openpi-pi:thor-trained-rtc-candidate-20260916`，原版参考镜像`openpi-pi:thor-trained-rtc-jax-ref-20260917-r3`。

## 当前结果

2026-09-20 10:18 CST：转换、构建、离线回放和本机协议测试完成，`pi05-rtc-infer`已替换为80000，**最终仍保持停止、120W、8000无监听**。原60000停止服务容器在80000协议测试成功后删除，60000权重/引擎保留；只维护一个Pi推理服务。没有操作3588或执行真机任务。下次恢复时沿用`ws://192.168.250.1:8000`，须使用MAXN会话包装，不能直接常驻120W测试。

JAX↔Torch九例全部通过既定1e-4数值门槛，最大物理输出差6.27559e-6。TensorRT引擎SHA-256 `bc07331e11faad0e9580d12755f43926780ef4b6c763eecb3a5b3cff7724cf4e`。完整证据见[导出](export_report.json)、[构建](engine_report.json)、[JAX参考](reference_manifest.json)、[离线验收](validation.json)。

| 指标 | 结果 |
| --- | --- |
| 九例d=0/1/10 | 全部有限50×14、前缀逐位保持 |
| 对JAX最大关节差 | 0.00111453 rad，约0.064°，case7/action49/dim1 |
| 物理14D P99差（混合单位） | 0.000453650 |
| 预热后推理P50/P95，27次 | 186.86/187.33 ms |
| 预热后总处理P50/P95 | 190.42/191.14 ms，不含网络 |
| 本机WebSocket九例P50/P95 | 194.12/205.03 ms，短样本含首请求 |

满足既有九例数值门槛max≤0.005、P99≤0.001，未放宽门槛。首轮推理P95=367.33ms，包含首次Graph捕获；不能以稳态值保证首次请求或网络端到端时延。与60000历史测试不是同时A/B，不推断权重导致速度变化。

历史事故四例[回放](incident_receipt.json)的TRT前后缀交界分别0.006136/0.007984/0.097116/0.090930rad；含交界与全新后缀的最大关节步进为0.068497/0.056570/0.097116/0.090930rad/tick，全部有限且低于0.2保护阈值。与JAX的最大关节差分别0.000639/0.000765/0.000907/0.001703rad。回放未重现旧归一化事故大跳变，不证明所有场景安全或任务成功率。

最后在MAXN下短暂启动新服务做[真实协议smoke](rtc80000-smoke.json)，9/9通过，确认80000权重/引擎指纹、quantile前缀、七步和0.2rad保护。随后停止新容器，替换同名`pi05-rtc-infer`，恢复120W；一次性日志`journalctl -u thor-rtc-80000-promote-paused.service`退出0。

## 已执行清理

永久删除30000的转换权重、七步ONNX外部权重、七步TensorRT引擎和已停止回退容器，保留原始JAX与报告，约释放37GiB。100000非RTC大权重和引擎此前已删除，本次确认没有残留大模型文件，只保留约71MB追溯文件。60000原件和转换产物保留。
