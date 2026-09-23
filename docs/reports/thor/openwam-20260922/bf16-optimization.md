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

下一候选是先测 BF16 同形状自定义掩码的 SDPA `efficient`、cuDNN、Flash 后端是否可用，以及单算子速度/数值差。入口 [`probe_attention.py`](../../../../scripts/thor/openwam/probe_attention.py) 只用合成掩码和 QKV，**不能作为完整模型精度结论**；如后端支持，再在固定 checkpoint 上做同输入完整动作比较。即使 attention 的约60ms全被省掉，仍不足以单独把405ms压到300ms；矩阵乘法需要另找方法。

原始回执：[r6](benchmark-r6.json)、[r7](benchmark-r7.json)、[r8](benchmark-r8.json)、[r9](benchmark-r9.json)。Thor宿主各有 `logs/benchmark-rN-host.log` 和原始 profile；当前未下载整份 trace。所有容器实验后退出并恢复宿主120W，Pi按用户最新指示保持停止。2026-09-23 工作站获同一 SSID 的 `10.18.10.43/23`，旧 Thor 管理地址 `192.168.110.250` 无法直达，USB设备亦未连；新探针已准备但**尚未在 Thor 执行**。需恢复 Thor 管理连接后再验证当前运行状态和继续实验。
