# Thor RTC 前缀归一化事故与离线复测（2026-09-17）

## 结论与运行边界

30000 RTC 的 `pi05_yam` 训练数据采用**分位数归一化**，旧 Thor 服务却把控制侧已承诺绝对动作转成**均值/标准差归一化**的模型前缀。旧 JAX 对照脚本使用同一个错误默认值，因此旧版 JAX↔TensorRT 数值接近、9/9 WebSocket smoke 及延迟记录**不能证明 RTC 条件化正确**。TensorRT 引擎本身接收归一化前缀作为输入；本次修正主机侧前缀编码，未改权重/引擎文件。根据同输入同噪声回放，错误前缀是现场大跳变的强支持原因；原在线噪声及独立请求前缀未记录，不能声称逐位复现了当时请求，也不能据此判定训练集或模型能力已全面合格。

Thor `pi05-rtc-infer` 在发现事故后停止，`--restart no`、8000无监听，日常电源120W。**未经受控真机复核不得把旧容器或旧10步容器恢复为在线 RTC。** 新代码强制显式选择归一化模式，新7步服务必须读取带 `prefix_use_quantiles=true` 的新版验证回执；旧容器原命令缺少 `--validation` 会拒绝启动。服务还设关节相邻目标 `0.2 rad/30Hz tick` 的临时粗大跳变闸门：超过即返回错误、不返回动作，不做隐式普通推理回退；这不是机械臂安全认证，也不能替代控制侧保护。

## 事故证据

- 用户核对“乐高分拣3”最新集的 manifest `rtc=false / policy_fusion=tda_smooth` 是会话启动默认值未随页面热切换更新。只读 HDF5 复核：42帧 `details.policy_fusion=rtc`，4个实际回复 `server_timing.rtc_used=true`；用户另外核对其中33帧实际执行RTC动作。因此以逐帧/回复为准，不把事故归给TDA。
- 原始集：IPC `/data/YAM/data/episodes/5dbcd16f-6d45-4e60-9791-10cdbe6f1191/session_20260917_193745_eb8952/episode_000001`，`samples.h5` SHA256 `ffa6b3cb32595d14e2b8d3f11835ac0b72719abc202b58b9196a17390f0d6a4f`。仅只读复制提取，未改3588代码/设备。Thor 重建夹具位于 `/home/wuyan-lyj/thor/pi/test-data/lego3-rtc-incident-20260917-r2/`，包含三路视频精确帧、14D状态、prompt、4个回复及从回复前9步重建的前缀。`cases.json`/每例 provenance 保留源哈希。
- 训练配置 `DataConfigFactory.create_base_config` 对 Pi0.5 设置 `use_quantile_norm=true`；`src/openpi/training/data_loader.py` 和 policy 输入/输出 transform 依此处理。旧 `scripts/thor/rtc_policy.py` 却写 `self.use_quantiles=False`，旧 `rtc_jax_reference.py` 调用 `encode_committed_actions` 时落入默认 `False`。尽管返回给客户端的前9步被原样覆盖，模型内部用于前缀条件的32D张量仍是错误的。

## 同一观测/前缀/固定噪声0的对照

下表是交界处 `action[d] - action[d-1]` 的**最大单关节绝对差，单位 rad**；4例均 `d=9`。原在线请求的随机去噪噪声未记录，所以“记录值”只作现场事实，离线 JAX/TensorRT 两列才是严格同噪声的后端对照。

| 回复行 / request | 现场记录 | 错误均值/标准差 JAX | 正确分位数 JAX | 正确分位数 TensorRT |
| --- | ---: | ---: | ---: | ---: |
| 9 / 1 | 0.109656 | 0.110053 | 0.009453 | 0.009472 |
| 18 / 2 | 0.293689 | 0.279978 | 0.013476 | 0.013484 |
| 30 / 3 | 0.931786 | 0.947572 | 0.104250 | 0.104349 |
| 40 / 4 | 2.505012 | 2.675350 | 0.067022 | 0.067181 |

同输入同噪声、完整H50/12关节的正确 JAX↔TensorRT 最大单元素差按4例分别为 `0.000710 / 0.000715 / 0.000597 / 0.002116 rad`。Thor 原始参考在 `.../jax-reference-quantile-seed0-7step/`，部署回执 `.../trt-quantile-seed0-7step-r1/receipt.json`（两者均相对于上面的 Thor 夹具目录）。旧错误参考保留在 `.../jax-reference-seed0-7step/`，只作事故对照。

4例正确 TensorRT 的整个新后缀相邻关节最大变化分别为 `0.0671 / 0.0608 / 0.1044 / 0.0672 rad/tick`，在临时 `0.2` 闸门内；将同一函数直接作用于4个现场记录/4个正确回放，结果分别是`通过、拒绝、拒绝、拒绝`与`全部通过`。闸门是对**未来服务输出**的拒绝条件，不会追回现场已经执行的动作。

另外将原有9个真实 YAM 离线样本的7步 JAX 前缀按正确分位数重跑：`d=10` 的3例交界最大单关节差从错误参考约 `0.712 / 0.723 / 1.112 rad` 降为 `0.040 / 0.047 / 0.050 rad`。修正后的同9例 TensorRT 对原JAX，物理H50×14混合维度 MAE `8.93e-5`、P99 `5.07e-4`，最大为动作49/关节1的 `0.004520 rad`；关节/夹爪分维数据及MAXN离线稳态推理P50 `186.98 ms` 位于 Thor `.../pi05-rtc-20000-replay-20260917-r2/validation-30000-quantile-7step-20260917-r1.json`。`0.004520` 是最坏一个关节标量，不是每关节/每步误差，也不代表相对任务真值；离线耗时不是跨IPC往返。

## 后续闸门

`rtc_candidate_test.py` 的12项CPU合同测试通过，含分位数/均值编码分离、缺省模式拒绝、前缀边界、`0.2 rad/tick`拒绝及训练时RTC不会退回普通推理测试。旧7步验证文件缺少正确前缀标记，不能再作为上线依据；5/6/8/10步的旧 RTC 数值报告亦受同一共同错误影响，需要逐步重算，不作为可回退生产版本。当前只完成离线复测，**尚未重启8000、未做新版本WebSocket smoke或3588闭环**。恢复前还需在受控条件下启用新版服务/临时MAXN、核对握手 `rtc_prefix_norm=quantile` 和守卫、从Thor本机及直连各做真实请求，再由用户协调真机观察；尤其实际延迟和任务成功率仍未知。
