# OpenWAM-α Thor BF16 加速跟进（r6–r9）

## 范围

官方 `OpenWAM-Alpha-Sim-RoboTwin-Full`，固定权重和镜像见[首轮报告](gpu-r4.md)。Thor 独占 MAXN，BF16、三路随机 RGB 经官方预处理、20D 合成 EEF 状态、固定 prompt/seed 42；输出是有限 `(32,20)`。每组一次首调用和十次稳态完整 `engine.generate`，计时包括必要 CUDA 同步和输出落 CPU。**这些实验不能说明 RoboTwin 或 YAM 任务成功率。** 物理位置单位未独立验明，rot6d 是表示分量，最大差不是关节角误差。

| 运行 | 10步官方缓存＋编译 P50/P95 | 主要变更 | 结果 |
|---|---:|---|---|
| r6 | 407.5/408.8 ms | 首次组合 | 10步跳过6次联合前向；编译图调用可见，但轨迹没有 CUDA Graph 重放证据 |
| r7 | 405.2/408.0 ms | `reduce-overhead` | 轨迹明确 `cudaGraphLaunch` 4次；速度几乎不变 |
| r8 | 404.3/406.5 ms | 同输入对照7步 | 7步组合 406.1/416.1 ms；10步跳6次、7步跳3次，均实际前向4次 |
| r9 | 404.8/405.8 ms | `max-autotune` | 自动调优运行并确认 CUDA Graph 4次；稳态无显著收益，编译首调用约125秒 |

r8 不开缓存的编译路线，10步 963.9ms、7步 682.3ms；步数缩减本身提速约29%，但官方缓存使两者的完整前向次数相同。r8 7步对10步的同配置动作差：不开缓存 MAE 0.00319、位置最大 0.01089、夹爪最大 0.05469；开缓存 MAE 0.00341、位置最大 0.01261、夹爪最大 0.07031。它们只诊断数值变化，没有任务级阈值，7步不晋级。

r7 的完整调用约405ms，其中上游 `[WAM_PROFILE]` 报去噪循环约379–392ms；Torch profiler 中 GPU kernel 累计约388ms。按 kernel 名汇总，矩阵乘法约249ms，masked SDPA attention 约60ms，其余为 RoPE、归一化、VAE 编码等。该加总来自一个 profile 调用，不是可直接相加到墙钟的分项基准；但与墙钟接近，足以说明主瓶颈在 GPU 计算。图重放改善的是主机调度，无法单独达到300ms。

[官方RoboTwin复现实录](https://github.com/OpenWAM-Official/OpenWAM/blob/main/benchmarks/robotwin/README.md)确认 Alpha 使用10步、同步去噪、缓存阈值0.99/最多连续跳3步及编译，与本轮核心设置一致。论文所述约170ms是RTX 5090的去噪循环；Thor本轮对应循环约380ms，完整 `engine.generate` 约405ms。硬件和计时范围不同，不能把170ms直接作为Thor整次调用承诺。目前没有发现“官方实际用了7步”的证据。

Alpha `dual_system/joint_self_attn.py` 的 MoT 驱动对视频与动作拼接 token 调用 `F.scaled_dot_product_attention(q,k,v,attn_mask=bool_mask)`；掩码同时约束视频帧因果关系与双模态可见性。之前容器的 `WAM_ATTENTION_IMPL=sdpa` 控制 ActionDiT 组件，`DIFFSYNTH_ATTENTION_IMPLEMENTATION=torch` 控制 Wan 独立注意力，均不是这条联合注意力的直接切换器。把普通 Wan 的 FlashAttention 开关当成 Alpha 联合注意力加速开关会误导实验。

BF16 同形状布尔掩码探针 r10：自动模式已走 `efficient`/CUTLASS，单算子 P50 0.354ms；强制 `efficient` 0.385ms，强制 cuDNN 0.377ms（相对自动最大差0.001953125、MAE 1.56e-7），强制 Flash 报 `No available kernel`，宿主日志说明它不支持非空自定义掩码。这只是合成掩码/QKV算子对照，**不是完整模型精度或时延结论**。单纯切换后端没有观察到值得做完整模型实验的收益；即使 attention 的约60ms全被省掉，仍不足以单独把405ms压到300ms。入口 [`probe_attention.py`](../../../../scripts/thor/openwam/probe_attention.py)，[原始回执](attention-probe-r10.json)。

另一个**精确复用候选**：Alpha TI2V 把第一帧潜变量在每个去噪步固定，[固定版本的时间条件代码](https://github.com/OpenWAM-Official/OpenWAM/blob/90e94ae31efddd64b59e00365cfc501d9a972eb1/openwam/model/video_backbone/wan/dit_forward.py)把该帧时间条件钉在0；[视频掩码](https://github.com/OpenWAM-Official/OpenWAM/blob/90e94ae31efddd64b59e00365cfc501d9a972eb1/openwam/model/video_backbone/wan_backbone.py)与[跨模态掩码](https://github.com/OpenWAM-Official/OpenWAM/blob/90e94ae31efddd64b59e00365cfc501d9a972eb1/openwam/model/architectures/utils/mask_modes.py)均禁止首帧查询看到未来视频或动作。这里查的是Thor实际固定的 `90e94ae` 版本，不以本仓库另一份 `7c5861e` 厂商快照替代。r11 在固定官方检查点、合成观测、10步 BF16 eager 下截取前两个去噪前向：首帧输入120 tokens×3072及后续30层输出的最大差、MAE均为**0**，完整输出有限 `(32,20)`；[原始回执](prefix-invariance-r11.json)。这验证了该输入下逐层不变性，**不是缓存实现、完整精度或加速验收**。下一步应实现每层首帧 K/V 与残差的分块前向，和原完整模型逐维比较，再测 MAXN 稳态时延；不能只缓存第一层或忽略动作查询对首帧 K/V 的读取。

r12 用同一模型和合成观测扩展到全部10次去噪前向，以第一次为参考比较其余9次的首帧输入及30层输出，共279项，最大差均为**0**；[原始回执](prefix-invariance-r12.json)。这排除了仅前两次恰好相同的解释，但仍未覆盖其他输入/编译路线，也不等于缓存已实现。

原始回执：[r6](benchmark-r6.json)、[r7](benchmark-r7.json)、[r8](benchmark-r8.json)、[r9](benchmark-r9.json)、[r10](attention-probe-r10.json)、[r11](prefix-invariance-r11.json)、[r12](prefix-invariance-r12.json)。Thor宿主各有 `logs/benchmark-rN-host.log` 和原始 profile；当前未下载整份 trace。所有容器实验后退出并恢复宿主120W，Pi按用户最新指示保持停止。2026-09-23 USB恢复管理后，将Thor连接至工作站当前5 GHz AP，Wi-Fi DHCP `10.18.10.89`；见[网络状态](../../../08_thor_edge_deployment.md#7-当前状态)。
