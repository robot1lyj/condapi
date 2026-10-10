# 50h200000 RTC：候选回退对照，保留230000

2026-10-10：用户怀疑230000现场表现有过拟合，指定同50h run的200000用于对照；随后明确要求验收后直接切换200000。怀疑不等于已经确认过拟合，本轮转换一致性/离线回放不评估泛化成功率。230000原始权重、FP32转换件、ONNX、TensorRT引擎和服务配置全部保留，不执行清理。服务器源不改动，不操作3588或发送客户端任务。

## 当前：200000已上线，230000完整保留

11:38 CST复核：`pi05-rtc-infer`已运行200000，固定`ws://192.168.250.1:8000`、healthz OK、`thor-pi-maxn-50h200000.service` active/MAXN。19项源SHA、原始非有限0、811张量/3,353,433,872元素FP32逐位映射、九例JAX/Torch/TRT及两轮各九例真实协议测试全部通过。230000完整保留为停止容器`pi05-rtc-50h230000-preserved-20261010`及原权重/引擎，无删除。本轮仅Thor本机/离线验收，3588全链路和真机任务效果待用户测试。

## 路线与产物

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/200000`。只取params/assets/完成元数据及同run控制目录训练合同，不取train_state。
- 完成元数据SHA `cdfaafc3a47983780e4e12d27ffa45e708ad9eda310667c6df3346f73bfc3e80`。源文件共19项，OCDBT分片数量与230000不同是正常存储差异，不用旧18项固定数量判断损坏。
- 合同SHA `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`，H50/32D/dmax10；norm SHA `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`，与230000相同。仍由现有严格norm身份检查确认训练合同关系。
- 冻结复用230000已验收脚本和Pi候选镜像`openpi-pi:thor-trained-rtc-candidate-20260916`、原JAX参考镜像`openpi-pi:thor-trained-rtc-jax-ref-20260917-r3`；实际镜像ID已核对与230000报告相同。没有同步整个工作树。
- [源SHA](source.sha256)共19项通过；[原始有限性](rtc-50h-200000-audit-20261010-r1.json)非有限0；[转换审计](conversion_audit.json)FP32映射811张量逐位一致。[RTC manifest](rtc_manifest.json)确认dmax10和训练norm仅末尾换行不同；权重SHA `298714be810f5e7e822bcac55c5efe55789f1c96c0ac4ed0a1caa86f2670e42a`。映射器基础配置日志中的`rtc_training_max_delay=0`不用于判定路线，原JAX/导出/服务均须使用训练合同和manifest的dmax10。
- [准备脚本](rtc50h200000-prepare.sh)负责直拉、源SHA、原始有限性审计、FP32保真转换及重新生成真实三相机九例（d=0/1/10）。CPU转换不使用CUDA、不停在线服务。
- [部署脚本](rtc50h200000-deploy.sh)在准备完成后执行停止230000、MAXN独占七步JAX参考、FP32/TF32缓存80/CUDA Graph构建、精度对照及候选协议测试。本次自动unit在最后交接前退出1，已显式接续同地址上线并再次协议验收，详见下节；不得重新运行整个脚本。没有放宽精度或0.2rad/tick关节跳变拒绝。

## 精度与延迟

从真实三相机suite和源Parquet按本checkpoint norm重新生成[九例](cases.json)，episode95/96/97早中晚，d=0/1/10。对比同输入/固定噪声、同七步、完整逆变换后的50×14绝对目标；训练前缀为quantile、干净token/flow time 0，无guidance或普通模式回退。

- [原JAX参考](reference_manifest.json)和[导出回执](export_report.json)通过。JAX→Torch最大物理数值差`1.9727269589930874e-5`，JAX门槛1e-4及缓存wrapper门槛1e-5均通过。这一混合14D最大数值差不冒充统一单位的关节误差。
- [引擎回执](engine_report.json)和[TensorRT验证](validation.json)通过，engine SHA `15ae4a5ceb57b6b6d8743c721cbe43178ab29c714be95cfbf1b9b893574ba71c`。最大单关节转换差0.000730822rad，夹爪最大差0.000594768；混合14D P99=0.000373148。输出有限、前缀逐位保持，不量化。
- MAXN稳态推理P50/P95=193.59/194.67ms，Thor总处理197.42/198.20ms。首轮包含Graph捕获的推理P95=372.48ms单列，不作为稳态。
- [候选九例](50h200000-smoke.json)和[正式上线九例](50h200000-online-smoke.json)均通过，握手权重/norm/engine匹配、七步/quantile/trained RTC、dmax10。上线服务推理P50/P95=199.10/209.13ms，**Thor本机WebSocket往返199.72/210.34ms**，不是3588全链路时延。
- 上线九例最大新后缀关节步进0.129985rad/tick，候选0.101978，均小于0.2拒绝值；两次随机采样的协议回包不用于后端误差比较。12个关节单位rad、两个夹爪标称0闭/1开连续值，不裁剪，30Hz policy tick合同不变。

这些检查证明转换/部署数值一致性，不证明20万步比23万步任务成功率更高，也没有确认过拟合。阶段回执的`engine_not_validated`/`compared_not_robot_task_validated`保留其原阶段语义，不重写历史报告。

## 服务交接与恢复

[完整运行回执](deployment_receipt.txt)保留失败与恢复证据：一次性`thor-rtc-50h200000-deploy.service`在11:36:31候选smoke通过、GPU会话恢复120W后退出1，没有执行容器重命名/正式启动。运行期间曾补写启动失败检查到该Bash脚本，日志没有额外错误，**准确退出根因未独立复现**；后续必须先冻结脚本再执行，不覆盖正在运行的脚本。

已检查候选九例通过及200000权重身份，显式把原`pi05-rtc-infer`改名为`pi05-rtc-50h230000-preserved-20261010`，把候选改回同名Pi服务，再启动新的`thor-pi-maxn-50h200000.service`。11:37:55正式九例smoke通过，11:38:30再次复核在线/healthz/MAXN。这不是重跑下载/转换；不能宣称原部署unit退出0。

230000原始params、FP32 model.safetensors、ONNX外部权重和TensorRT引擎均核对仍在，原服务配置/挂载也保留；无任何模型删除，可用649043025920字节，约604GiB。若用户要求回退，先停200000并等待MAXN退出/120W，再保留200000容器、将230000恢复同名服务，重新启动MAXN会话并用230000的旧回放/指纹做协议验收；两个服务不能同时占8000。保留的230000不自动启动，未更改开机自启策略。

Thor根`/home/wuyan-lyj/thor/pi`，本轮stage `probes/rtc-50h-200000-20261010-r1`，检查点`checkpoints/lego-pi05-rtc-base-50h-20260921/200000`，转换件`200000-pytorch-fp32-r1`，回放`test-data/pi05-rtc-50h-200000-replay-20261010-r1`；独立JAX/ONNX/TRT产物均以`rtc-50h-200000-…-20261010-r1`命名。服务地址保持`ws://192.168.250.1:8000`，只运行一个在线Pi服务。230000的停服保留容器用于快速恢复，不是另起一个模型系列或并行在线服务。

权重收齐后直拉SSH会话退出255，已使用已有文件断点接续补合同，没有重下大权重；随后全部源SHA一致。准备脚本也已去掉复用230000固定18项数量的假设，按本checkpoint实际19项及必需文件校验；这两项是传输/流程问题，不是模型权重损坏证据。
