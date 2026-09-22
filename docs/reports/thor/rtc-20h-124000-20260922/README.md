# 20h124000 RTC部署验收

2026-09-22 11:09 CST复核：`pi05-rtc-infer`运行、healthz OK，`thor-pi-maxn-20h124000.service` active/MAXN。固定地址 `ws://192.168.250.1:8000`，Pi共用容器镜像不变，七步FP32权重＋TF32 TensorRT、缓存80/CUDA Graph、quantile前缀，训练dmax10。输出为50×14绝对动作；关节rad，夹爪连续值。0.2rad/tick粗大跳变拒绝保留。

## 身份与证据

- 来源：服务器 `lego_pi05_rtc_base_20h_20260918/124000`，17项源文件SHA校验通过，原始参数NaN/Inf为0。
- 转换权重SHA：`088b1f4bb70addf8e852dc2d7500a530cae9c2997cda61166bc814feb260852b`。
- norm SHA：`d5493ce7b71790526312779d8fbccc374b9569f14d88722682b0e7c8bfcf1d52`。
- [转换审计](conversion_audit.json)：811张量映射与加载逐位一致；[manifest](rtc_manifest.json)记录合同及产物身份。
- [导出对照](export_report.json)：九例d=0/1/10，JAX→Torch FP32以及缓存wrapper的既定数值门槛全部通过。
- [TRT验证](validation.json)：九例有限输出、前缀保持，最大关节差0.00061560rad；夹爪最大差0.00035878。混合物理维度P99差0.00035880仅用于既有数值门槛，不当成统一物理单位。
- 稳态服务推理P50/P95为191.47/192.40ms；[上线后本机WebSocket九例](20h124000-online-smoke.json)全部通过，往返P50/P95为197.42/208.16ms。不是3588全链路延迟，也不是机器人任务成功率。

## 产物与清理

Thor根目录 `/home/wuyan-lyj/thor/pi`。权重在 `checkpoints/lego-pi05-rtc-base-20h-20260918/124000-pytorch-fp32-r1`，engine在 `artifacts/rtc-20h-124000-trt-tf32-cache80-7step-20260922-r1`。原始JAX、参考输出、九例回放及本次部署脚本均保留。作业 `thor-rtc-20h124000-deploy.service` 11:05:06正常结束，日志最终标记 `DEPLOYED_124000_AND_CLEANED_90000_30000`。

上线后再次协议验收通过才永久删除20h90000原始params、转换model.safetensors、七步ONNX外部权重和TRT engine，以及10h30000剩余原始params；配置、norm、报告和源服务器原件保留。删除前后磁盘Used分别441520898048、377144754176字节，释放64376143872字节（约59.96GiB）。当前可用约537GiB。旧模型不能就地回退，须从服务器重新下载/转换；其他模型未动。

本次只操作Thor，未做3588端到端测试或机器人任务验收。服务停止后MAXN会话恢复120W，未设置开机MAXN常驻。
