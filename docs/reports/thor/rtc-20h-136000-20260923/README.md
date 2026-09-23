# 20h 136000 RTC：Thor 转换、验证与上线（2026-09-23）

15:22 CST 现场复核：`pi05-rtc-infer` 运行 136000，固定 `ws://192.168.250.1:8000`、`healthz=OK`，`thor-pi-maxn-20h136000.service` active，宿主 `nvpmodel -q` 为 MAXN。镜像仍为 Pi 系列共用的 `openpi-pi:thor-trained-rtc-candidate-20260916`，没有每个 checkpoint 新建镜像。后续运行状态须重新复核。

## 身份和转换

- 来源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_20h_20260918/136000`。只取推理所需 `params/assets/_CHECKPOINT_METADATA`，不取 `train_state`。完成标记 SHA256 `d084f4de57398e4b42770bf1f19579e202b0f7aa9938ae15238cf1611811950e`；源端生成的 [18 项 SHA 清单](source.sha256)在 Thor 全部通过。
- 136000 源目录未附 `training_contract.json`；从同一 20h run 的 Thor 124000 副本复制合同，SHA256 `42c635bd92535797a00bdbc63ac8b7c09d8ee6754d6cd24e7e93800319192ac2`，并确认两个 checkpoint 的 norm 逐字节相同。合同中 Pi0.5、H50/32D、训练 RTC 最大延迟 10；这份合同的来源是同 run 的 124000，**不是 136000 独立提供的训练合同**。
- norm SHA256 `d5493ce7b71790526312779d8fbccc374b9569f14d88722682b0e7c8bfcf1d52`；[原始参数有限性审计](checkpoint_audit.json)无 NaN/Inf。[转换审计](conversion_audit.json)确认 811 张量 FP32 映射与加载逐位一致；[manifest](rtc_manifest.json)权重 SHA256 `9778949c85824a563eb997764b91dcde12d14765a400b6727581586d84dd8bb4`，无 LoRA。
- 复用相同 norm 的 124000 九个真实观测回放作为固定输入；这是**离线同输入比较**，并非新采集的 136000 真机数据。七步训练型 RTC、quantile 前缀、FP32 权重/TF32 计算、80-token 时间调制缓存与 CUDA Graph；未用 BF16/FP8 量化。[导出报告](export_report.json)九例 JAX→FP32 PyTorch 最大物理差约 `4.16e-6`，wrapper 自身最大约 `1.25e-6`。

## 精度与时延

- [TensorRT 九例验证](validation.json)：九例 `d=0/1/10`，输出有限、已承诺前缀保持；相对**同 checkpoint、同输入、同七步**原生 JAX，最大关节差 `0.00074687 rad`，夹爪最大差 `0.00031977`（夹爪连续值，不与 rad 混为同一单位）。引擎 SHA256 `788e70090045551530fcb1dbfb44d900bbdff673aefc25262100253d1ef005bd`。稳态服务推理 P50/P95 `190.95/192.98 ms`；全九例含预热阶段的 P95 较高，不作为稳态值。
- 候选阶段 [WebSocket 九例](20h136000-candidate-smoke.json)和同名上线后 [WebSocket 九例](20h136000-online-smoke.json)均通过，返回有限 `(50,14)` 绝对目标动作、quantile RTC 前缀逐步不变，并检查了握手权重指纹。上线后服务推理 P50/P95 `197.87/207.56 ms`，Thor 本机往返 P50/P95 `198.73/208.77 ms`。不是 Thor↔3588 端到端时延，也不是机器人任务成功率。
- 推理阶段 MAXN 生效；服务停止后由 `maxn_session.py` 恢复日常 120W。当前 `--restart no`，不代表开机自启。RTC 关节粗大跳变保护仍为 `0.2 rad/tick`，不等于真机安全验收。

### 15:28 CST 用户现场反馈：左臂行为异常（待定位）

用户认为 136000 比上一版更容易出现左臂“不知道干什么”。这是真机场景反馈，**不能被上述九例离线数值通过所否定**；在定位前不应继续下发其动作驱动真机。为区分转换误差与 checkpoint 行为变化，对同九例、同 norm/七步/噪声的原生 JAX 124000 与 136000 物理动作做了只读对照：左六关节平均绝对变化 `0.00591 rad`、P95 `0.01957 rad`、最大 `0.04418 rad`，右六关节分别约 `0.00625/0.02002/0.03737 rad`。13.6 万步 TensorRT 相对其自身 JAX 的左六关节最大差仅 `0.00074687 rad`，较124000的转换最大关节差约多 `0.000131 rad`；离线证据**不支持单靠转换误差解释该现场左臂异常**，也不证明新模型任务行为正确。当前 Thor 服务日志没有保存导致异常的三相机观测、14D状态、prompt和 RTC 前缀，无法判定该特定请求的 JAX 与 TRT 是否同样异常。需要先取得一次故障输入，在 Thor 离线固定噪声重放两后端，并检查 observation/前缀时间对齐；不动3588控制实现。

## 清理与恢复边界

新服务上线后再次九例 smoke 通过，才删除 Thor 本地 124000 的原始 `params`、转换 `model.safetensors`、ONNX 外部权重、TRT 引擎及停止的旧容器；小型配置/norm/报告仍留 Thor，服务器未被本次清理修改。删除前后 Thor `df -B1` Used 为 `325787705344`→`273851723776` 字节，释放 `51935981568` 字节（约 48.36 GiB）。**旧 124000 无法就地回退**；其服务器源目录在本次查询时也已不在原路径，不能笼统宣称可以重新下载旧版。

按用户此前暂停 OpenWAM 的要求，另删除 Thor 上 21 个已停止的 OpenWAM 试验/下载容器及其可写层；保留两个 OpenWAM 环境镜像、官方原始权重、源码和日志，未删除通用 Docker 构建缓存。未操作 3588。工作站中转曾下载约 8.2 GiB 临时片段，Thor 完整源 SHA 通过后删除该本机临时目录。

Thor 路径：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-20h-20260918/136000{-pytorch-fp32-r1,}`，`/home/wuyan-lyj/thor/pi/artifacts/rtc-20h-136000-{jax-quantile-7step,onnx-fp32-cache80-7step,trt-tf32-cache80-7step}-20260923-r1`，`/home/wuyan-lyj/thor/pi/probes/rtc-20h-136000-20260923-r1/`。GPU 流水线脚本见 [deploy_20h136000_gpu.sh](../../../../scripts/thor/deploy_20h136000_gpu.sh)。
