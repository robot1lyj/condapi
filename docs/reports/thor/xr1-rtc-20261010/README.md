# XR-1 原生异步 RTC：Thor 延迟实测

2026-10-10，用户授权暂停在线 Pi200000，独占 Thor 测试 XR-1 原生动作前缀异步推理，用于确定未来微调的 RTC 时序预算。不启动训练，不操作3588，不向机器人发动作。Pi230000备份没有更改。

## 测试身份与边界

- 官方通用后训练起点 [Xiaomi-Robotics-1-5B](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-5B/tree/ee21d524b5c52ac961d941e1bc7d6d92836c3d5e)，`model_states.pt`，10,226,684,862 bytes，SHA256 `94d55a79122050a654b379664b644e874ff90d64ccd30a6a633f816555bcecf7`；本地传入Thor后摘要一致。不是9月23日测试的RoboCasa HF专用权重。
- 使用固定上游 [`0dd7aef8`](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/tree/0dd7aef8dc87296246aae812a1f59ccb708e5546/xr1) 的原生 `mibot.models.VLA.XR1.xr1`，不是缺少前缀逻辑的旧HF包装。模型文件SHA256 `eeb4316e7125a0c414875e178da677c0c254af7c6a0536f8da0aace13a3928c0`。本仓库扩展1～10步的补丁仅改变训练前缀采样/配置，eval分支与本次原生实现相同。
- 原始1135个张量均为BF16，`strict=True`全部匹配；不转换权重，不量化，不缩减去噪步数。关闭训练专用FFN checkpoint wrapper，模型始终`eval()`、`inference_mode()`，没有训练循环。输出为 **1×30×60原生模型空间动作**，不是Pi的50×14关节目标。
- Thor T5000，测试单元日志确认MAXN0、`jetson_clocks`锁频，GPU GPC 1575MHz；自动风扇策略保留。测试结束由`maxn_session.py`恢复120W，恢复Pi服务时另开MAXN会话。没有设置开机永久MAXN。
- XR-1系列镜像 `xiaomi-xr1:thor-native-rtc-20261010`，ID `sha256:9d0be210f8ce56010790e5ffef9b65bd5760cfb1cae6e8cf206cddc887496f7e`，基于已有26.05 NVIDIA Torch/FlashAttention镜像增加原生import依赖；不污染Pi容器或宿主Python。[实际包清单](requirements.observed.txt)包括Torch2.12.0a0/nv26.5、Transformers4.57.1、FlashAttention2.7.4.post1、Liger0.6.5、mmengine0.10.7、Lightning2.5.3。
- 三个真实YAM录制样本来自Thor `pi/test-data/pi05-replay-v1/episode-{000095,000096,000097}-early.npz`，读取三路RGB和prompt；**不借用Pi norm或delta动作变换**。state和前缀是显式构造的归一化模型空间测试值，预留槽补零，action mask调用原生函数；不能解读成已经通过YAM末端坐标、FK/IK、动作质量或训练效果验收。
- 每路原始fixture为224×224；384×384档由其放大，测视觉token负载，不声称恢复原图细节。调用官方客户端消息模板/resize/processor；三路224为216总token、588个pixel patch，三路384为501总token、1728个pixel patch。实际训练/部署长宽、视图数、prompt长度变化后必须重新测量。

## RTC与图加速做法

保留原生30步、5步Euler积分、因果mask、后缀位置`+10`、固定干净前缀和前缀速度归零逻辑。测`N=0/1/3/6/8/10`；0是无前缀对照，不把它当RTC性能。前缀长度变化不减小30步计算张量，也不调用Pi特有的逐token flow-time/guidance算法。

只给原生`dit_forward`加固定形状、固定N的CUDA Graph，VLM每次重新计算。所有当前KV、state embed、position embed、mask、噪声和timestep均重新复制；不能复用上一帧视觉KV。每个N单独捕获，切换形状必须重新捕获。三个不同观测/状态/前缀/seed均与eager同输入对照，**36组全部逐位相等、max_abs=0**。所有测试输出有限、前缀逐位不变；改变非零前缀时新后缀也改变。零新增误差仅指相对**原生BF16实现**，不是与FP32/JAX对照，也不等于实机任务精度已验证。

## 主测试结果

固定输入已在GPU，CUDA同步后计model forward；每档每N预热3次、正式30次，轮换三个观测。完整本地路径另测12次，包含Qwen processor、H2D、forward、D2H；fixture磁盘读取、图像解码/尺寸调整在计时外，也不包含相机采集、网络、物理norm逆变换、末端恢复或IK。不同批次独立计时，不通过相加两个P95伪造总P95。

| 每路图像 | 前缀N | 原生eager P50/P95 ms | DiT Graph P50/P95 ms | 本地完整路径 P50/P95 ms |
|---|---:|---:|---:|---:|
|224×224|0|181.95 / 185.86|135.17 / 135.84|134.58 / 138.71|
|224×224|1|171.66 / 172.43|131.58 / 131.99|135.15 / 135.88|
|224×224|3|172.10 / 173.51|130.21 / 130.85|134.18 / 135.66|
|224×224|6|172.32 / 181.91|131.97 / 133.34|135.27 / 136.07|
|224×224|8|171.82 / 172.62|130.25 / 130.91|137.75 / 138.42|
|224×224|10|171.82 / 172.58|130.26 / 137.07|134.24 / 138.73|
|384×384|0|209.12 / 217.51|173.98 / 175.25|179.72 / 183.06|
|384×384|1|209.19 / 220.06|181.23 / 182.11|206.51 / 284.60|
|384×384|3|211.72 / 223.42|179.31 / 179.97|179.87 / 183.63|
|384×384|6|208.23 / 210.68|174.37 / 174.71|180.71 / 181.69|
|384×384|8|204.76 / 205.71|173.20 / 173.71|179.06 / 180.78|
|384×384|10|205.13 / 205.76|173.24 / 173.64|179.44 / 183.71|

P50是典型耗时，P95是约95%样本不超过的耗时，不是硬实时最坏值。本轮峰值GPU分配9.892GiB。前缀N=1的大图完整路径曾出现 **344.77ms** 单次峰值；没有删除离群值或把它记成网络问题。原始样本见[results.json](results.json)，完整MAXN/运行过程见[benchmark.log](benchmark.log)。补测结果在本报告后续小节单列，不能覆盖这次观察。

### 大图尾延迟补测

保持同一镜像、权重、原生算法和默认Python GC，重新加载模型，独立复测384档N=1/10。各测60次eager、60次Graph和60次本地完整路径，前置3次预热；只增加GC耗时记录，没有关闭GC或丢弃样本。

| 前缀N | DiT Graph P50/P95 ms | 本地完整路径 P50/P95 ms | 完整路径最大 ms |
|---:|---:|---:|---:|
|1|179.79 / 180.40|179.42 / 183.42|185.30|
|10|174.33 / 175.48|179.57 / 183.37|183.93|

六组同输入Graph/eager对照仍逐位一致，前缀不变、后缀条件化有效。本轮没有复现344.77ms；GC仅各发生一次gen0，耗时0.128/0.158ms，**不足以认定上次峰值是GC造成，也不足以确认峰值已消除**。需要常驻服务下更长的全链路监测才能给最坏时延界限，本次不继续扩大测试。[补测原始样本](tail-results.json)、[补测日志](tail.log)保留全部数据。

## 微调时需要固定的合同（建议，不生成新配方）

当前候选50h recipe已有`rtc_mode=native_async`、`async_prefix_min=1`、`async_prefix_max=10`，保持原生50%无前缀、50%均匀采样1～10步；本轮不改配置、不启动训练。建议保持这个范围、30Hz标签和H30：最大前缀10覆盖约333.3ms，N=10时仍有20个新后缀目标。旧配方缺字段时仍是官方1～6步默认，不能把两种范围混用。

部署先保留原生BF16、5步去噪和DiT Graph。5是推理数值积分步数，不是训练优化步数，微调仍按原生随机flow时间训练。图像尺寸/视觉token预算必须与训练预处理对齐；不能仅为速度把已在高分辨率训练的模型偷偷缩成224并宣称精度不变。

运行时的N由**整个请求从对应观测到新动作可用之间已承诺执行的控制tick**决定，而非永久写死10，也不能用相机曝光时间替代目标tick。预算草案为`ceil((P95_local + transport_budget + margin) × 30)`（秒单位）；暂借用户此前20ms额外链路预算，并留1 tick约33.3ms余量，主测试224档约需6步、384档约需8步。这是规划估计，不是3588全链路实测；罕见超333ms不能依赖训练1～10范围硬接，应拒绝过期结果/按安全控制策略处理。

XR-1与Pi的norm/action语义不同：XR-1原生动作/前缀走每步mean/std，state走q01/q99；原生末端相对目标和夹爪增量需相对新观测参考系正确重表达，不能直接把Pi的绝对14D关节前缀复制进去。训练labels第0步对应哪一个控制tick必须绑定派生数据的时序审计；本次模型空间延迟probe不证明物理时间对齐或IK可用。本轮没有提交客户端任务或客户端指令。

## 可复现入口

源码：[主probe](../../../../scripts/thor/benchmark_xr1_rtc.py)、[主MAXN launcher](../../../../scripts/thor/run_xr1_rtc_benchmark.sh)、[容器Dockerfile](../../../../scripts/thor/Dockerfile.xr1-rtc)、[尾延迟probe](../../../../scripts/thor/benchmark_xr1_rtc_tail.py)、[尾延迟launcher](../../../../scripts/thor/run_xr1_rtc_tail.sh)。Thor stage为`/home/wuyan-lyj/thor/xr1/rtc-20261010-r1`，checkpoint为`/home/wuyan-lyj/thor/xr1/checkpoints/base-5b-ee21d524/model_states.pt`，Qwen config/tokenizer/processor缓存为stage下`hf-home`。

先停Pi并等待旧MAXN会话恢复120W，再在宿主sudo运行冻结launcher；不要覆盖正在运行的脚本。主测试unit为`thor-xr1-rtc-benchmark-20261010-r1.service`，补测unit为`thor-xr1-rtc-tail-20261010-r1.service`；它们不是常驻XR-1推理服务。旧9月23日HF RoboCasa H10/普通推理约151ms的Graph结果不具有原生RTC前缀语义，不能替换本报告。

## 完成状态与Pi恢复

主测、补测unit均`ExecMainStatus=0`，结束恢复120W，两个临时XR-1容器自动移除；没有开启XR-1线上控制服务。17:48以新`thor-pi-maxn-50h200000-xr1-restored-20261010.service`先进入MAXN再启动原同名Pi容器。17:49:30九例[恢复后WebSocket smoke](pi-restored-smoke.json)通过，50×14输出有限、前缀精确、最大新后缀关节步进0.12451rad/tick，小于原0.2阈值；权重SHA `298714be810f5e7e822bcac55c5efe55789f1c96c0ac4ed0a1caa86f2670e42a`、norm与engine身份、七步FP32/TF32和quantile前缀未变。

恢复后Pi本机推理P50/P95=197.78/207.86ms，本机WebSocket往返198.30/209.03ms。`ws://192.168.250.1:8000`不变、healthz OK，只有`pi05-rtc-infer`运行；新MAXN watcher随容器停止退出并恢复120W，不开机自启。230000停止备份及所有产物保留。新增XR-1权重与镜像后可用约594GiB；没有删除任何原模型、数据或容器。实际单元、Pi启动参数和功耗状态见[运行回执](runtime_receipt.txt)。本轮没有读取3588或做跨IPC/实机验收。
