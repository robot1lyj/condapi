# 05 · 训练后 policy 端侧与 IPC smoke

本页保留 YAM 训练后 policy 的输入输出协议、Thor 本地推理验收和 Thor↔3588 的直连以太网推理通道。模型运行在 NVIDIA Jetson AGX Thor；3588 负责相机采集和机械臂控制，YAM 机械臂控制、CAN、GUI、home pose、控制频率和真机安全不在本仓库适配范围，也不要从独立 YAM-ABC-Reproduce 代码推断本项目合同。

Thor 系统、容器、Pi0.5 转换和 TensorRT 方案见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)。

## 1. 部署前 gate

必须同时满足：

1. checkpoint 参数元数据完整，且有与训练数据绑定的 `assets/yam/norm_stats.json`。
2. 当前全量微调配置是 `pi05_yam`，数据和 checkpoint 属于同一 14D 合同；旧 LoRA 配置不是默认入口。
3. Thor 已核验 JetPack/L4T、GPU、Docker runtime 和 Pi 系列容器内的 CUDA/JAX；采用其他候选后端时额外核验其依赖。镜像、只读模型挂载与端口发布按 [08](08_thor_edge_deployment.md) 记录。
4. 原 JAX checkpoint、golden 输入/噪声/输出已保存；任何候选后端均需按 [08](08_thor_edge_deployment.md) 做分阶段精度验收，不能只与转换后的 Torch 比较。
5. Thor 本地推理 smoke 确认输入键、输出 shape、有限值和 checkpoint/norm 绑定；随后必须做 3588↔Thor 的真实直连以太网 smoke，验证 observation/action 往返。

## 2. Thor 本地推理边界

默认数据流为：

```text
3588 camera/state/prompt
  == direct Ethernet / WebSocket or agreed transport ==>
Thor Pi 系列容器：YamInputs + norm
  -> 已完成离线原 JAX 对照的 TensorRT policy（当前 RTC 30000）
  -> YamOutputs + absolute action
  == direct Ethernet / action response ==>
3588 controller: 有限的 (50,14) YAM action chunk
```

端侧 bundle 必须来自当前指定的 `pi05_yam` 全量微调 checkpoint。成熟案例的 `pi05_libero` 是 7D、horizon 10；不能直接复用其权重资产、TensorRT engine 或 49/54 ms benchmark 来代表 YAM。YAM 需要三路图像、模型内部 horizon 50 和真实 14D 输出，详见 [数据合同](04_data_contracts.md)。

首个端侧运行顺序固定为：

1. 原 JAX policy 以实际 YAM 样本和同一份噪声数组生成 golden；保留原始全量 checkpoint。
2. 在 Pi 系列容器内核验 Thor 原生 JAX 可行性；若转换到 Torch，先审计 FP32 构造/存储、norm 绑定和未量化计算对齐。不同 checkpoint 通过配置选择，不各自建立常驻容器。
3. 精度通过后再按延迟需求决定是否导出 engine；FP8/NVFP4 和定制 FP16 是独立候选，必须与原 JAX 比较。
4. 在 Thor 本地用回放样本直接调用 policy，验证三路图像、14D state、prompt 和 `(50,14)` 输出；再用 3588 的真实 observation 做跨 IPC 直连 smoke。

原生 JAX、经审计的 Torch 和加速 engine 均须有 Thor 实测证据；具体精度、环境限制和验收层次由 [08](08_thor_edge_deployment.md) 持有。保持 `action_horizon=50` 与默认去噪 `num_steps=10` 分别记录。跨 IPC smoke 只证明传输和模型可运行，不能替代任务成功率验收。

## 3. YAM 输入输出合同

YAM policy 输入使用和 LeRobot 导出一致的键：

```text
observation.state                  float array, shape (14,)
observation.images.top_rgb         RGB image
observation.images.left_rgb        RGB image
observation.images.right_rgb       RGB image
prompt                              scalar string（或由 task 注入）
```

图像可由 `YamInputs` 兼容 CHW/HWC，正式 smoke 建议使用 HWC `uint8`。推理返回：

```text
actions: float array, shape (50, 14), all finite
```

模型内部 padding 到 32D 只属于 OpenPI 模型边界，不应把 32D 直接当作 YAM 真实动作发送给机器人侧。单位、限幅、执行频率和安全检查由已核实的机器人侧系统负责。

## 4. 本地 smoke 验收

不得以容器启动、端口监听或只返回 metadata 作为成功。按顺序检查：

- 同一批三路实际图像、14D state 和 prompt 在 JAX reference、PyTorch、TensorRT（或 FlashRT）中都能完成推理；
- 输出严格为 `(50,14)`，所有值有限，action 顺序和 norm asset 绑定到 YAM；
- 记录 engine/backend、config、checkpoint、训练 commit、norm 路径、JetPack/L4T、CUDA/TensorRT、warmup、时延和功耗模式；
- 端侧本地路径通过后，才允许进入低速、限位和人工急停可用的机器人侧测试。

旧 100000 服务使用 [serve_pi05_trt.py](../scripts/thor/serve_pi05_trt.py) 与 [smoke_pi05_ws.py](../scripts/thor/smoke_pi05_ws.py)；其独立真实回放和原 JAX 对照见 [100000 报告](reports/thor/100000.html)。当前RTC30000见[训练时RTC冷手册](reference/thor/13_trained_rtc_inference.md)。不要把旧 OpenArm 16D 工具重新作为默认入口。

## 5. Thor↔3588 网络推理通道

Thor 服务端在 Pi 系列容器内加载已验收的 TensorRT engine 与训练 norm，3588 通过 Thor 直连网卡上发布的 policy 端口发送 observation 并接收 action。Pi 系列固定 `ws://192.168.250.1:8000`，切 checkpoint/普通或RTC模式时更改服务配置并重启、重新预热和验收，**不因模型切换另设客户端 URL**，也不假定支持热切换。旧100000 Docker `pi05-infer` 已停；当前RTC30000为 `pi05-rtc-infer`、`--restart no`，**尚未实施 Compose**。GPU 接入与端口规则见 [08](08_thor_edge_deployment.md)。

### 2026-09-16 · 100000 服务历史验收（2026-09-17 已停）

- Thor 直连 URL：`ws://192.168.250.1:8000`；只绑定该网口，不绑定 Wi-Fi 管理地址。宿主 `curl http://192.168.250.1:8000/healthz` 返回 `OK`；`docker inspect pi05-infer` 为 running、restart=unless-stopped、当次重启0。运行状态在使用前复核。
- 入口为 `python /service/serve_pi05_trt.py`，Pi v6 镜像，独立服务目录 `/home/wuyan-lyj/thor/pi/services/pi05-100000-20260916`；容器只读挂载对应 100000 FP32 checkpoint、训练 norm、100000 W/80 engine、真实回放，tokenizer 缓存本地挂载。新请求由 Thor 生成 `(50,32)` FP32 噪声，客户端只发标准 observation；服务关闭 RTC。
- 握手 metadata 声明 `checkpoint_step=100000`、`backend=tensorrt`、`norm_stats_sha256=044aad51…4dcc`、`engine_sha256=70b366c5…9d54`、`precision=BF16 main/FP32 sensitive+time cache, no quantization, TF32 off`、`action_horizon=50`、`robot_action_dim=14`、`denoising_steps=10`。完整字段及哈希在 [Thor 本机 WebSocket smoke 回执](reports/thor/evidence/20260916/100000-service/local-ws-smoke-20260916.json)。未核实第 0 步相对 observation 的物理时间偏移，不在握手中臆造 `action_dt_s`/时间偏移。
- 返回 `actions` 为训练逆变换后的 50×14 **绝对目标**，顺序 `[左 6 关节, 左夹爪, 右 6 关节, 右夹爪]`。根据当前数据发布合同，关节数值按弧度、夹爪名义 0 闭 1 开；这是数据语义，不宣称硬件标定/限位。模型输出未裁夹爪；机器人侧不能把本服务输出视为已做安全约束。
- Thor 本机经直连地址发送一条真实记录的三路 224×224 RGB、14D 状态和 prompt，普通 WebSocket 握手及推理成功，返回有限 50×14。初次 `120W` 的服务端推理 `174.71 ms`、请求往返 `177.07 ms`；按用户 2026-09-16 明确要求切入 **MAXN 推理/测试阶段**，同一路 WebSocket 复测服务端 `106.10 ms`、请求往返 `107.16 ms`，GPU 1575 MHz、EMC 4266 MHz 锁到该模式上限。MAXN 原始回执见 [本轮 MAXN smoke](reports/thor/evidence/20260916/100000-service/local-ws-smoke-maxn-20260916.json)；这仍只是单次在线协议 smoke，不等于离线 180 次 P50 `104.34 ms`，也不是 3588↔Thor 跨 IPC、真机闭环或任务效果验收。
- **电源阶段规则**：以后推理和测试阶段启用 MAXN＋`jetson_clocks`；准备、下载、安装和日常空闲为 120W/动态调频/自动风扇。当时 Thor 宿主用 `maxn_session.py -- docker wait pi05-infer` 保持模式；容器后续重启暴露该等待方式会卡住，已停用并恢复120W。若主机断电/SIGKILL，Python 清理无法执行，重启后先检查 `nvpmodel -q`。旧100000 Docker容器虽有自动重启策略，但目前手动停止；不能只看旧容器配置推断当前服务。
- 当时 W/80 engine 仅接受有效 token ≤80 的 prompt；更长文本需为**本次 100000**另建 200 桶引擎，当时没有自动路由。历史日志可用 `ssh thor 'docker logs --tail 100 pi05-infer'` 查看；这不是当前RTC服务的日志入口。

### 2026-09-17 · 固定8000的 RTC 30000 服务

- 旧100000容器已停、权重与引擎保留。当前 `pi05-rtc-infer` 只监听 `ws://192.168.250.1:8000`；曾短暂试验的8001已停且无监听。后续Pi系列模型共用此固定URL，通过握手metadata区分实际 checkpoint/backend/模式。
- 握手 `rtc_mode=trained`、`backend=tensorrt_cuda_graph`、`rtc_max_delay_steps=10`，`checkpoint_weights_sha256=fc60f64cfd05906edda9f446e113c159e1df6ece4ea479e27ba22b01791d5f60`、`norm_stats_sha256=b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`，`action_dt_s=1/30`、第0步相对**observation数据行的30Hz policy tick**为0；相机曝光到物理控制tick的偏移未知。输出为 `(50,14)` 绝对目标，关节rad、夹爪连续值名义0闭1开，未裁剪。
- Thor本机9组真实三相机/状态/prompt的d=0/1/10协议smoke均返回有限50×14，已承诺前缀逐位不变，MAXN服务/往返P50约1033/1034ms。原始JAX↔TensorRT物理最大绝对差约8.55e-6，但**约1秒延迟远未满足低延迟目标**；3588→Thor跨IPC和真机闭环尚未验收。准确性、计时、引擎、操作细节与收据见[RTC冷手册](reference/thor/13_trained_rtc_inference.md)。
- 当前MAXN由 `thor-pi-maxn-30000.service` 监视容器；停止 `pi05-rtc-infer` 后会恢复120W。容器 `--restart no`，重启机器后不能仅凭历史状态认为它会自动服务；按固定地址部署流程重新启动并核对模式。

### 2026-09-14 直连网络基线

用户授权本轮仅对 3588 的独立网口进行联调，不读取或修改其机械臂、相机和控制实现。两端 NetworkManager 均已保存 `yam-thor-direct`，自动连接并绑定物理接口和 MAC：

- Thor `enP2p1s0`：`192.168.250.1/24`；
- 3588 `lan1`：`192.168.250.2/24`。

直连配置不设 gateway、DNS 或附加 route，`ipv4.never-default=yes`、IPv6 disabled、MTU 1500、自动协商。实机插线后两端均为 `UP/LOWER_UP`，协商 2500 Mb/s、full duplex；双向各 10 次 ICMP 为 0% 丢包，Thor→3588 平均 0.245 ms、3588→Thor 平均 0.235 ms，双方 TCP/22 均可达。两台机器到公网的路由仍分别使用 Wi-Fi，绑定 Wi-Fi 接口的 HTTPS 请求均返回 HTTP 200。Thor 当次 Wi-Fi DHCP 地址为 `192.168.110.250/23`，3588 为 `192.168.110.140/23`；DHCP 地址是临时观察值，直连服务应只使用 `192.168.250.0/24`。

这是 2026-09-14 的网络层基线；当时尚未启动 policy 端口，也没有完成 observation/action 或机器人闭环 smoke。上面的 100000 服务是 2026-09-16 的新事实，不能回写为网络基线当天的结果。旧 `scripts/serve_policy.py` 载入的是普通 policy，不是本次 W TensorRT engine，不作为 100000 默认启动入口。

使用真实 `openpi-client` WebSocket 协议（或后续核定的等价直连协议）从 3588 发送三路图像、14D state 和 prompt，检查：

- 输入无缺少 image/state/prompt 错误；
- 输出严格为 `(50,14)`，无 NaN/Inf；
- metadata 的 `robot_action_dim/output_action_dim=14`、`model_action_dim/action_dim=32`、`action_horizon=50` 与 checkpoint config 一致；
- 日志记录 commit、checkpoint、config、端口、prompt 和 smoke 结果。

跨 IPC 直连 smoke 由 3588 侧测试者执行并保存报告；本仓库只提供 Thor 推理服务和上述 Thor 本机回执。不要触碰 3588 的系统、控制进程、相机进程或其他用户任务。

## 6. 停止条件

Thor 本地或跨 IPC 直连路径出现以下任一情况，停止 rollout 并回到 checkpoint/data gate：

- shape 不是 14D 输入或 50x14 输出；
- 动作非有限、checkpoint/norm 不匹配、转换产物未通过 JAX reference 比较；
- 三路图像缺失、通道/时间同步不明；
- 把 OpenArm/Piper 单位、顺序或 transform 混入 YAM；
- 只看到端口监听或服务进程存在，尚无 Thor 本地推理和 3588↔Thor 网络结果。
