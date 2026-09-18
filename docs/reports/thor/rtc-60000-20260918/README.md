# RTC 60000 · Thor转换与离线验收

## 输入与边界

2026-09-18，服务器 `yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916/60000` 完整保存点。只下载推理所需params/assets、完成元数据及该run的训练合同，没有下载train_state。服务器与Thor的17个参数/资产文件SHA-256逐项一致，完成元数据SHA-256 `0dd2838cd99a872fd4f3e0b44880fb7a8c6aa6730e26c5ce618390fc2074dcb5`。

Thor根目录 `/home/wuyan-lyj/thor/pi`：原件 `checkpoints/lego-pi05-rtc-base-10h-20260916/60000`，转换件为同父目录`60000-pytorch-fp32-r1`。代码提交`0e161c5`，独立脚本暂存`probes/rtc-60000-20260918-r1`，避免修改线上挂载代码。用户授权暂停30000进行独占GPU离线测试，结束恢复原30000，不自动将60000上线。

## 已完成的权重验收

- 原始JAX参数非有限元素0，见[原件审计](rtc-60000-audit-20260918-r1.json)。
- FP32映射811张量、3,353,433,872元素，映射→加载逐位一致，见[转换审计](conversion_audit.json)。此结论不是动作精度结论。
- 权重SHA-256 `b519a67b3c4717f1d0b3a7e2a27acbfb8161e2bb5ea99f4d6bbfc14c732b04b4`。
- checkpoint norm SHA-256 `b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`；合同SHA-256 `ef773631bb5dac8d4de055dc1d7161cf57c691b68ae50ec983e3a6297cb5ed5f`。训练norm与checkpoint norm仍只差一个末尾换行，不是数值更改；完整参数身份见[RTC manifest](rtc_manifest.json)。

## 测试配置与进展

复用正确的分位数前缀，H50/14D物理输出、内部32D、dmax10；七步Euler、80-token桶和时间条件缓存。JAX原版FP32/最高矩阵精度为参考，Torch导出FP32关闭TF32，TensorRT候选显式TF32、不量化、CUDA Graph。模型系列镜像分别为`openpi-pi:thor-trained-rtc-jax-ref-20260917-r3`、`openpi-pi:thor-trained-rtc-candidate-20260916`。

常规9例使用`test-data/pi05-rtc-20000-replay-20260917-r2/cases.json`，每个d=0/1/10各3例，norm与60000一致。事故4例使用`test-data/lego3-rtc-incident-20260917-r2/cases.json`，d=9；为历史观测及重建前缀、固定噪声，不声称逐位复现现场随机轨迹。MAXN/锁频会话记录在`journalctl -u thor-rtc-60000-test-20260918.service`；退出自动恢复30000服务及它自己的MAXN会话。

## 完整离线结果（11:07 CST）

流水线正常退出0。九例JAX↔Torch FP32全过`1e-4`门槛，最大物理差`1.06556e-5`；缓存wrapper最大物理差`3.01737e-6`，全过`1e-5`优化门槛。见[ONNX报告](export_report.json)与[JAX参考manifest](reference_manifest.json)。

七步TensorRT引擎位于`artifacts/rtc-60000-trt-tf32-cache80-7step-20260918-r1`，SHA-256 `8f8e642bb33b4e5074cb901b742d78b0609b5ccf4b0de9e61090a7cc5f82aa0a`；相应ONNX目录为`artifacts/rtc-60000-onnx-fp32-cache80-7step-20260918-r1`，原始JAX参考为`artifacts/rtc-60000-jax-quantile-7step-20260918-r1`。见[构建报告](engine_report.json)、[数值/时延回执](validation.json)。

| 指标 | 实测 |
| --- | ---: |
| 真实常规案例 | 9/9输出有限50×14、前缀逐位保持 |
| JAX→TRT最大单关节误差 | 0.00171105 rad，约0.098° |
| 物理14D P99绝对差（混合单位，不称统一rad） | 0.000513865 |
| 稳态推理P50 / P95 | 190.73 / 193.40 ms |
| 稳态Thor总处理P50 / P95 | 194.52 / 197.54 ms |

稳态为预热后9例×3轮；不含网络和3588队列。首轮推理P95为377.18ms，含惰性CUDA Graph捕获，不能拿它或稳态值代替全链路承诺。常规九例满足既有服务数值门槛max≤0.005、P99≤0.001；最大项case1/action49/dim1。相较30000修正后离线采样P50约186.98ms，本次约慢3.75ms，但不是同时同负载A/B，不能据此归因于权重。

四个事故观测另作诊断：见[回执](incident_receipt.json)，原始输出保存在`artifacts/rtc-60000-incident-trt-7step-20260918-r1`。TRT前后缀交界分别`0.007603/0.011184/0.103139/0.081593 rad`；包含交界与整个新后缀的最大关节步进分别`0.066723/0.055364/0.103139/0.081593 rad/tick`，均有限且低于现有0.2拒绝阈值。与同输入JAX最大关节误差分别`0.001059/0.000840/0.000894/0.002059 rad`。第四例关节P99为0.001745，不能声称每个诊断样本均满足九例集合的P99≤0.001门槛；没有据此调宽上线门槛。回放未重现旧归一化错误的大跳变，但不能证明真机任务成功率或所有新观测安全。

结束后自动恢复原`pi05-rtc-infer`30000七步服务与MAXN会话，11:08 CST `http://192.168.250.1:8000/healthz`返回OK。11:09恢复后[真实协议smoke](restored-30000-smoke.json)九例全过，核对30000权重指纹、quantile前缀、有限50×14及前缀保持；本机往返P50/P95=194.24/204.95ms。**60000未上线**；没有操作3588或下发机械臂命令。
