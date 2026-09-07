# 10 · 100 ms 加速实验执行记录

部署边界与最终验收仍由 [Thor 主文档](../../08_thor_edge_deployment.md) 持有；本页只保存本轮加速适配的执行细节，不表示已经选定生产后端。

## 固定比较条件

目标为完整本地 policy 调用约 100 ms 或以下，同时报告 P50/P95；包含输入变换、CPU/GPU 传输、模型与输出变换，不只计异步 CUDA 提交时间。保持三相机、horizon 50、10 次去噪、内部 32D / YAM 14D。同一真实回放输入、归一化和初始噪声用于跨后端对照。仅推理期间 MAXN + 锁频，结束恢复 120W 与动态调频。

优先顺序：FP32 跨框架参考 → BF16 混合精度 / torch.compile → TensorRT 强类型混合精度。FP8/NVFP4 是独立候选，不能在数值与任务验收前默认接受。社区 H10 / LIBERO 7D 的延迟不能替代本项目 H50 / YAM 14D 实测。

## 2026-09-07 · Git 接入完成

用户添加部署公钥后，Thor 已成功读取 Gitea 的 main。原 rsync 代码完整备份至 Thor `/home/wuyan-lyj/thor/code-backups/rsync-before-gitea-20260907/`；新建 Git 索引后检查差异，再从已验证的 origin/main 恢复完整受跟踪代码。没有删除权重、测试数据或旧代码备份。

`python3 scripts/thor/sync_code.py --host thor-usb` 实机通过，首次对齐提交 `6339d1d4a653a140ceb063c1739046fc1d2ae783`。这是对 [09](09_gitea_code_sync.md) 中“授权待完成 / 尚无 HEAD”历史状态的更新。后续仍由本机先提交并推送 Gitea，然后 Thor 仅做快进同步；不会自动重建镜像或重启推理。

## 2026-09-07 · FP32 转换首次完成

- 原始基础权重只读挂载，没有 YAM 微调，也没有 LoRA 张量。
- 原转换器存在两项风险：初始化 dtype 默认 BF16、`strict=False` 未审核缺失与多余参数。现明确先构造 FP32 模型，拒绝未映射参数、非 FP32 源权重和 LoRA，检查载入值一致性，之后才允许显式另存 BF16。
- Pi0.5 AdaRMSNorm 根据 `model_config.pi05` 选择，不再根据文件夹名称猜测模型结构。
- 新输出目录已存在即拒绝；不覆盖历史产物。转换器不是 LoRA 合并实现；未来 LoRA 需另行完成保留/合并与原生等价验证。
- 推理工厂新增显式 `pytorch_precision`、`pytorch_compile`，防止 FP32 对照被默认 BF16 转换覆盖；旧调用仍默认 BF16 + 编译。

首次转换镜像：`openpi-pi:thor-convert-fp32-20260907`，实测 image ID `sha256:5e9202b3f4dd1caee7e86e0c1f45f3008ab27352fa05ff4131d8360ec0308955`。基于已验证的 Pi JAX 容器增加 CPU PyTorch 转换依赖；没有 GPU 挂载，在 120W 模式完成。

Thor 产物：

- `/home/wuyan-lyj/thor/pi/checkpoints/pi05_base_pytorch_fp32_v1/model.safetensors`
- 同目录 `config.json`、`conversion_audit.json`
- 日志 `/home/wuyan-lyj/thor/pi/logs/convert-pi05-fp32-20260907-r1.log`

首次实测：811 个映射张量、3,353,433,872 个参数，源 metadata SHA-256 为 `250d0ad0f6540a5e4bdd70442d6864cef729724f2d23e2dd2cc669563a668d70`。映射值与加载后 FP32 张量逐项相等。两个未映射语言输出头明确列为例外：PaliGemma 头绑定词嵌入；expert 语言头不参与 OpenPI 动作前向。它们不是丢失的动作层。

注意首次镜像中的一致性检查使用 `torch.equal`；审计字段 `mapped_to_loaded_bit_exact` 在这里严格只能解释为数值完全相等（该操作不区分正负零）。工作树后续已加强为 int32 位视图比较，并增加源/目标总参数量检查；这些加强尚不能追认为首次镜像已执行。转换通过不是跨框架推理通过，也不是机器人任务精度验收。

## 候选 GPU 容器与下一次测试

官方底座 `nvcr.io/nvidia/pytorch:26.05-py3`，ARM64；[NVIDIA 26.05 说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-05.html) 对应 PyTorch 2.12.0a0、CUDA 13.2.1、TensorRT 10.16.1.11。首次查询 registry digest `sha256:222d8b18e671be5c3ef91cb41727a2572a0b23f59ded6c39f373a96946f6f2ba`；实际落地后另存 image ID 与完整包版本。

最初启动了 Thor 直拉与本机下载备选，后者保存在 `/home/wuyan-lyj/thor-system/images/pytorch-26.05-arm64/`；最终改用下方记录的内容缓存复用与 USB 传输。`scripts/docker/thor/pytorch.Dockerfile` 已完成候选构建与下方 D/E 实测；保留 NVIDIA 的 torch/CUDA/TensorRT 约束，不安装项目服务器训练整套依赖，不执行教程脚本来覆盖本项目代码。

首次转换阶段的安排是先跑既有 9 输入，检查 FP32 / BF16 输出差异和实际同步延迟；候选速度有价值才扩大样本。FP32 禁用 TF32，记录实际参数 dtype、编译开关、输入与权重摘要、首次编译时间。随后完成的实测见下方更新；原 JAX 约 177 ms 最快结果仍未满足用户目标。

### 统一回放入口

`scripts/thor/run_suite_host.py` 保留 A/B/C 原生 JAX 路径，新增 D（PyTorch FP32 eager）、E（混合 BF16 eager）、F（混合 BF16 编译）、G（混合 BF16 编译 + SDPA）。使用新建的 GPU 候选镜像时显式传 `--image`，并选择 `--modes D E F G`；不能用 CPU 转换镜像冒充 GPU 后端。

`benchmark_suite.py` 的 PyTorch 路径检查 CUDA 可用及实际参数设备；在旧模型构造器之后重新禁用 TF32，记录真实参数 dtype、完整 safetensors SHA-256、转换审计、编译/注意力设置和 CUDA 内存峰值。计时包含最终设备同步；没有 CPU 推理降级。默认仍为每组 9 输入、每输入 20 次正式调用。

跨框架对照必须显式使用 `compare_suites.py --cross-backend`：只放开后端、运行库版本和 matmul 实现的差异，并将它们列在结果中；输入、原 checkpoint metadata、noise、norm、steps、horizon、重复次数和样本顺序仍须匹配。普通 A/B/C 比较仍拒绝运行时差异。

### 镜像缓存复用

官方 ARM64 manifest 固定为 `sha256:aa400d4373fa71f30e1714664beabcc64c2d198e72d65e9a3641440b07e7cc83`，压缩层总量约 10.140 GiB。现有 JAX 层中命中 37 个 manifest 条目、约 6.454 GiB。`skopeo copy` 的 dir 目标不能作为本次缓存复用的可靠入口：实机观察其重建了目标目录，因此改用 `download_ngc_image.py` 按 SHA-256 检查、复制缓存，只下载缺失层，最后才发布完整 manifest。

原 JAX 缓存保持不变；停止的是本轮自己启动的两条重复 workstation skopeo 下载，已完成文件仍保留。新下载目录为本机 `/home/wuyan-lyj/thor-system/images/pytorch-26.05-arm64-cached/`，经 USB 同步到 Thor `/home/wuyan-lyj/thor/images/pytorch-26.05-arm64-cached/`。传输不包含 `.partial` 文件，最终镜像仍需完整指纹校验与导入验证。

### 社区导出器的额外语义差异

本机只读参考仓库 `/home/wuyan-lyj/thor-system/references/openpi-thor/` 固定提交 `40c88146e5e6b82526db6583c8a75520e02e551e`。其 [export.py](https://github.com/xuweiwu/openpi-thor/blob/40c88146e5e6b82526db6583c8a75520e02e551e/src/openpi_thor/export.py) 支持从 config 读取 action horizon，但公开导出入口将 compute dtype 固定为 FP16，并对 Gemma MLP 中间结果使用 `nan_to_num`。因此“最后输出有限”不能单独证明该路径没有溢出或精度变化。

后续若借用该导出器，应单独核对 FP32 稳定层、溢出修正是否触发和原 JAX 动作偏差；不能直接覆盖项目源码、沿用它的环境降级或照搬社区的精度阈值。当前优先测试不依赖这种 FP16 修正的 PyTorch BF16/SDPA 路径，再决定 TensorRT 适配细节。

### 镜像文件校验已完成

2026-09-07 20:10（北京时间），Thor 上 62 个唯一 config/blob 文件全部通过 SHA-256 验证，见 [原始校验日志](../../reports/thor/evidence/20260907/acceleration/pytorch-image-verify.log)。最后 9 个本机未完成的层从 Thor 本轮 Docker 拉取已完成的 containerd 内容缓存取出，逐个核对哈希；没有删除 Docker 缓存。

随后停止本轮自己的剩余重复网络拉取，保留部分下载文件。Thor 安装了镜像工具 skopeo 1.13.3；直接写 docker-daemon 返回 `io: read/write on closed pipe`，因此使用已经验证过的 docker-archive → `docker load` 路径。该错误的底层原因没有进一步认定。

下载器在文件齐备时现可完全离线校验/发布 manifest，不再为了检查本地缓存请求 NGC token。GPU 候选 Dockerfile 在装依赖前生成已装 torch / NVIDIA / TensorRT / NumPy / SciPy 版本约束；不依赖底座里内容为空的 `/etc/pip/constraint.txt` 来保证底层库不变。

## PyTorch 第一轮实测（D/E 完成，F 首试失败）

官方底座导入后的 Docker image ID 为 `sha256:9024018b27e9ad043d1b88984b4fb7b705df9778f3688ca7cbaf399031421cde`；格式转换后的 ID 不与 registry manifest ID 混用。Pi 候选镜像 `openpi-pi:thor-pytorch-candidate-20260907` 的实测 image ID 为 `sha256:4a878f9b56d4e35876a2db5fa3ca7c245143e0a6fd6ed5179b9e2948b896f2c2`。GPU 检查实测 PyTorch `2.12.0a0+5aff3928d8.nv26.05` / CUDA `13.2` / TensorRT `10.16.1.11`，设备 NVIDIA Thor `(11,0)`。

全部维持原回放合同与噪声，每个完成配置含 9 输入 × 20 正式调用；MAXN / 锁频仅在测试期间启用。对照参考为原 JAX FP32（A）：

| 配置 | P50 / P95（ms） | 动作 MAE | 动作最大绝对差 | 当前判断 |
|---|---:|---:|---:|---|
| D · PyTorch FP32 eager | 1249.229 / 1251.134 | 0.000001086 | 0.000015259 | 跨框架 FP32 数值接近；速度不合格 |
| E · 混合 BF16 eager | 221.342 / 227.245 | 0.002794213 | 0.024288177 | 比原 JAX C 更慢，最大偏差也更大，不选作最终路线 |
| F · 混合 BF16 compile，首次 | 无有效结果 | — | — | 融合注意力 bias/query dtype 不匹配；已退出并恢复 120W |

D 的 812 个实际参数张量均为 FP32；E 为 122 个 FP32 / 690 个 BF16，保留既定稳定层。D/E 每个输入重复输出的最大变化均为 0。误差仍使用数据集原单位，不宣称机器人任务精度合格。完整结果及运行库差异进入 [中文测试页](../../reports/thor/index.html)，原始动作保存在 Thor 与本机 `test-data/results/pi05-{D,E}-20260907-r1/`。

F 首次失败发生于编译器生成的 `aten._scaled_dot_product_efficient_attention`，错误为 `invalid dtype for bias - should match query's dtype`。原始失败日志保留在两端，Git 另存无损 [gzip 副本](../../reports/thor/evidence/20260907/acceleration/pi05-F-20260907-r1.log.gz)。不是 GPU 不支持 BF16，也不是得到了一组可用延迟。G 在该批次因 F 失败尚未启动。

下一批增加显式 `--native-attention-mask`，让融合算子的 bias 与 query dtype 对齐；旧 eager 路径仍保留 FP32 掩码。另加 H 配置（SDPA + 编译 + 三相机视觉编码合批），合批不改变图像数量、视角顺序或 action horizon。掩码与合批均有小型 CPU 语义测试，但是否真正加速、是否影响完整动作精度，必须以重跑结果为准。

## 2026-09-07 晚间追加：编译、相机合批与整图重放

本节追加实机观测，不修改前面的历史实验记录。固定合同仍为三路 RGB、H50、10 次去噪、模型 32D / YAM 输出 14D、9 个真实录像输入、每输入 20 次正式调用，全部在 Thor 断网容器内、MAXN 锁频下运行；每组退出恢复 120W。

| 运行 | 实现 | 完整调用 P50 / P95 ms | 对 JAX FP32 的动作 MAE / 最大绝对差 |
| --- | --- | --- | --- |
| F-r2 | BF16 + FP32 稳定层，编译，原 attention，原生 dtype 掩码 | 135.227 / 136.098 | 0.00234401 / 0.0192254 |
| G-r2 | F 改为显式 SDPA | 151.880 / 153.348 | 0.00223713 / 0.0191467 |
| H-r2 | G 加三相机合批 | 142.293 / 143.762 | 0.00207001 / 0.0135294 |
| I-r1 | F 加三相机合批 | 124.151 / 125.365 | 0.00214831 / 0.0160803 |

上述运行的日期前缀均为 `20260907`，例如 `pi05-I-20260907-r1`。I 是目前已完成的最快配置，仍未达到约 100 ms 目标。误差使用数据集原单位，不能解释成度、弧度或毫米；I 相比原生 C 的最大差较小，不等于已保证机械臂任务效果。基础模型没有 YAM 微调，也没有 LoRA。不同真实输入的中间张量都重新计算，没有缓存上一帧特征或动作。

所有正式重复输出差异为 0。D 组 FP32 跨框架参考最大差 `1.52587890625e-5`，报告必须保留这个小非零值，不能显示成零。F/G/H/I 均未启用 TF32，也未做 FP8/NVFP4 量化。原始 JAX checkpoint 和 FP32 转换文件保持只读。

整图重放 J-r1 首次尝试在 `embed_prefix` 的 CPU→CUDA 常量拷贝处失败，未产生有效延迟；退出码 1，120W 恢复记录已确认。新增的固定步数分支保留 FP32 时间递推和完整 Euler 次数，默认关闭，旧 while 路径保留。后续修复只把固定掩码和时间常量直接创建在 GPU，并逐输入对照旧 while 路径的完整输出；不能在修复后重标旧失败运行成功。

证据：仓库 `docs/reports/thor/evidence/20260907/acceleration/pi05-{F,G,H}-20260907-r2.*`、`pi05-{I,J}-20260907-r1.*`；原始日志在 Thor `/home/wuyan-lyj/thor/pi/logs/`，完整动作数组在 `/home/wuyan-lyj/thor/pi/results/`，工作站镜像在 `/home/wuyan-lyj/thor-system/test-data/results/`。Git 中 `.log.gz` 是原始日志的无损压缩，不修改原始文件。中文 HTML / JSON 含实际命令、镜像 ID、源码哈希、逐输入时延、动作差异和遥测。代码同步状态由 [09](09_gitea_code_sync.md) 持有。

### 2026-09-07 21:42 后续观测：整图重放跑通但未胜出

固定常量创建修复后的 `pi05-J-20260907-r2`（原 attention）P50/P95 为 168.994/170.154 ms，动作 MAE/最大差为 0.00249998/0.0235503；`pi05-K-20260907-r2`（SDPA）为 176.467/177.259 ms，动作 MAE/最大差为 0.00203562/0.0168608。二者都使用三相机合批、混合 BF16/FP32、完整 10 步及 H50，没有权重再转存和量化。参考依旧是 A 的 JAX FP32。

J/K 的九个输入都额外执行一次未捕获的旧 while 循环对照，模型 32D 输出最大差均为 0。这证明本批数据上的重放与旧循环等价，不证明 BF16 与 JAX FP32 等价，也不证明机械臂任务成功率。额外对照不计入正式时延样本。两组退出均恢复 120W。

整图重放自身并未超过 I 的 124.15 ms，不能因为用了 CUDA Graph 就默认推荐。下一步候选 L/M 使用 `max-autotune-no-cudagraphs` 分别编译图像前缀、模型 forward 和去噪段，再由一个外层 CUDA Graph 重放全部 10 步。这样保留算子融合，不把十个 action expert 展成一个巨大编译图；需要独立实测，不能借用 I/J 的数字。L/M 的非图对照使用相同分段编译后端的旧 while 循环，字段 `cuda_graph_vs_eager_max_abs` 为历史命名，报告按“非图旧循环”解释。

J/K-r2 的 manifest、exit、power-after、tegrastats 和无损压缩日志与前述证据放在同一目录；完整结果在各自独立运行目录。此时共 11 组完整结果、1980 次正式调用，另保留两个失败运行。100 ms 是 Thor 完整 policy 调用目标；尚未包含未操作的 3588 采集/控制和生产网线传输延迟。

### 2026-09-07 后续观测：分段编译的收益与数值分歧

`pi05-L-20260907-r1` 的完整调用 P50/P95 为 **118.8537/119.9279 ms**，对 A 的原单位动作 MAE/最大差为 **0.00209398/0.0197093**。`pi05-M-20260907-r1` 为 **135.7602/136.9710 ms**，MAE/最大差为 **0.00202810/0.0160533**。两组均完成 9 输入 × 20 次正式调用，重复差异为 0，退出恢复 120W。

L 是目前纯延迟领先者，但不能称作无差异替换：额外旧循环对照的模型 32D 输出最大差，L 为 **0.0192124**，M 为 **0.0211713**。这是图捕获、固定常量创建分支和分段编译组合相对旧循环分支的差异，尚未进一步分离原因；不是原单位动作误差。不能把 J/K 的零差异结论延伸到 L/M。I 仍作为约 124 ms、原单位最大差约 0.0161 的保守候选保留，而不是为了快约 5 ms 就宣告 L 精度通过。

N 是后续独立候选，不借用已有延迟：保持 I 的编译、三相机合批、原 attention 和混合 BF16/FP32；仅在单 SM110 Thor 测试进程内放开 Inductor 的 `is_big_gpu` 68-SM 门槛，让 ATen 与 Triton 矩阵内核通过实测选型。实际硬件属性不改写，记录原函数 SHA256、实际 SM 数和前后候选后端；使用 `/cache/torchinductor-thor-triton-v1` 独立缓存，避免复用未放开门槛时的编译结果。此私有接口实验不作为生产默认，也不启用 TF32 或量化。

L/M 原始动作、日志、manifest、恢复记录仍按独立 run ID 留存，中文报告包含全部逐输入数据；此阶段共 13 组完整配置、2340 次正式调用。约 100 ms 的目标仍未完成。下一轮除测试 N 外，应把“捕获图 vs 未捕获的同一固定循环”与“固定循环 vs 旧 while”分别记录，定位 L/M 的数值分歧。

### 2026-09-07 本轮收尾：N 约 120 ms，下一主线 TensorRT

`pi05-N-20260907-r1` 完成，P50/P95 为 **119.8713/120.9537 ms**，对 A 的原单位动作 MAE/最大差为 **0.00211876/0.0173282**，归一化有效 14D 最大差为 **0.0347049**。实际 20 SM；原 Inductor 门槛判定为 false，放开后的日志确认 Triton 候选参与实测，并非只改了配置名称。部分候选超过单块资源上限而被淘汰，不是整机内存耗尽；最终运行退出 0，全部正式重复输出差异 0，恢复 120W。

截至此记录共 **14 组完整配置、2520 次正式调用**，另有加载、预热、图对照和两个独立失败运行。没有一组达到约 100 ms 且完成任务精度资格。L 约 119 ms 仅是延迟领先者，N 约 120 ms 使用私有编译器门槛实验，I 约 124 ms 则保留为更简单的候选；不把这三者自动升格为生产推荐。

下一步先做 TensorRT **非量化 BF16 + FP32 敏感层** 的导出与 engine 验证，分别对照 JAX A 和 PyTorch，核对 FP32 路径是否意外启用 TF32。不能直接采用社区 FP16 入口的 `nan_to_num` 溢出补救作为“保精度”；仅借鉴工程结构。再独立评估选择性 FP8，NVFP4 不默认接受。[NVIDIA 数值原则](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) 支持保留敏感运算精度，但不能证明本模型的性能或任务效果；具体 API 仍按容器内 TensorRT 10.16.1.11 核验。

已补充图诊断代码，分别记录捕获与未捕获同固定循环、固定与旧 while 分支的差异；65 项相关 CPU 单元测试通过，但拆分诊断尚未重跑 L/M，不能反写旧运行数据。后续微调产物仍为 JAX/LoRA：当前转换器拒绝含未处理 LoRA 的输入，未来必须先验证 FP32 合并或等价 adapter 路径，不得漏掉 LoRA。

本轮记忆读取沿用预算账本 `thor-accel-20260907-c3`，准入用量 11678 UTF-8 字节；为真实压缩后保留的摘要另预留预算，未重置账本绕过限制。完整请求 tokenizer/封装对宿主不可见，不能宣称这是完整请求 token 限制。详细日志与动作数组不因记忆预算而删除。

## TensorRT 非量化导出准备（2026-09-07 追加）

已接入 `scripts/thor/onnx_sampler.py`、`export_pi05_onnx.py` 和 `run_export_host.py`。使用现有 NVIDIA PyTorch 26.05 系列镜像；实查 TensorRT 10.16.1.11、ONNX 1.21.0、ONNXScript 0.7.0，无需升级宿主机或重新下载基础环境。

保持三相机、H50、10 步和 BF16/FP32 敏感层。固定时间步仍按原 FP32 递推产生，各步正弦嵌入在 Thor 按原 GPU FP64 实现预计算后转为 FP32；导出图直接引用这些常量，不把运行中的 FP64 运算粗略改成 FP32。未做 FP16、量化或非有限值截断，原始权重保持只读，旧采样器继续可用。

`pi05-onnx-bf16-20260907-r1` 与 `r2` 的九个真实输入均通过导出准备对照：模型 32D 和反变换后 14D 输出与原 PyTorch eager 采样器完全一致。这不是 JAX 等价、ONNX 等价或 engine 精度通过。两次实际导出均失败并恢复 120W：

- r1：legacy 导出器报 `ScalarType ComplexDouble`。在同一容器用“FP32 标量 buffer × BF16 张量”的 CPU 最小例复现；普通 Python 浮点标量例可通过，不能把问题泛化成所有 BF16 乘法。新版 dynamo 导出器通过了相同最小例。[PyTorch 上游相关问题](https://github.com/pytorch/pytorch/issues/158658)提供了关联证据，实际版本仍以本机复现为准。
- r2：新版导出器在输出模型信息时触发自适应 `GemmaRMSNorm.extra_repr()` 访问不存在的 `weight`。最小修复改用已有 `dim`，不改变前向计算。相关 CPU 测试目前 70 项通过，下一运行仍须重新进行真实输入准备对照及完整导出。

`build_trt_engine.py` 已准备 strongly typed、显式关闭 TF32、固定输入形状的构建入口，校验 ONNX 和外部权重指纹后再调用现有 trtexec；尚未有成功 engine，不得报 TensorRT 推理成绩。GPU tactic 选型和模型验证才使用 MAXN 会话，退出恢复日常模式。

原始导出产物位于 Thor `/home/wuyan-lyj/thor/pi/artifacts/<run_id>/`，工作站副本为 `/home/wuyan-lyj/thor-system/test-data/artifacts/<run_id>/`；报告、host manifest、exit、power-after、tegrastats 和压缩日志归档到 `docs/reports/thor/evidence/20260907/acceleration/`。中文 HTML 单列导出准备，不把失败导出计入原 14 组/2520 次正式推理调用。
> 2026-09-07 22:44 更新：`pi05-onnx-bf16-20260907-r3` 已成功导出并通过 ONNX checker，退出码 0，结束恢复 120W。现代导出器保留 BF16/FP32，未量化、未开启 TF32、未做非有限值截断；三路 224RGB、H50、去噪 10。9 个输入的导出准备包装器与原 PyTorch eager 路径原始/反归一化输出均逐值相等，但这不等于 ONNX/TensorRT 执行精度已验收。导出本身耗时 732.06 秒、18,305 节点，外置权重约 6.4GB；耗时主要停留于图优化，最终正常完成，未因等待而重启。镜像 `openpi-pi:thor-pytorch-onnx-v6-20260907`，代码 `fc41fe2f9ae8be52947005f7553cc0235f66d9f8`。ONNX SHA256 `57d565c39c573137205152a274adb65b82e668aff98c3589d79993a0539d3f8e`，外置权重 SHA256 `7be25298ffe2bf6fdd0d751c6f753649ac0b155037abaa1905dfa0e50ef98bc9`。详见报告中的独立导出表及 `docs/reports/thor/evidence/20260907/acceleration/pi05-onnx-bf16-20260907-r3.*`。下一阶段使用 `build_trt_engine.py` 的 strongly typed / noTF32 构建，再以 `run_suite_host.py --modes T --engine-id ...` 回放；在真实完成前不把导出耗时当推理延迟、不增加已完成的 14 组推理计数。

TensorRT 回放接口保留现有 YAM 输入/归一化/输出变换，不加载第二份 PyTorch 权重；引擎 IO 绑定要求原 dtype/shape、CUDA 执行，不允许静默转换。每组同时记录对 JAX FP32 的误差和对同一导出准备 BF16 eager 的误差，后者固定相同 noise；二者用于区分跨精度差异与引擎转换差异。CPU 单元测试只是接口合同测试，不冒充真实 TensorRT 推理验证。引擎显存不纳入 PyTorch allocator 统计，因此同时保留整机 tegrastats，不能把 Torch allocator 值当作引擎总显存。

2026-09-07 22:57：`pi05-trt-bf16-20260907-r1` 构建成功，TensorRT 10.16.1.11、537.30 秒（宿主调用计时），引擎 SHA256 `bf1fd25e7e72b06f27eecc834ff914bda54d57190339e75c0cf289a8bc04528c`，约 6461.65 MiB。strongly typed / noTF32，层报告输出为 BF16、FP32、Bool、Int64，无 FP16/FP8 输出；引擎报告及完整压缩层报告已归档。构建退出恢复 120W。

首轮回放 `pi05-T-20260907-r1` 在首个正式模型调用前因 `state` dtype 合同不符退出，未生成有效速度或动作比较。原因：导出接口 state 是 FP64，但 Pi0.5 独立 state 输入没有 ONNX 消费者，TensorRT 保留这一死输入时绑定为 FP32。修复只给经 ONNX 指纹与无消费者检查确认的死输入配置 dummy buffer；不改变实际状态值、提示词编码或 YAM 动作还原，也不重建或修改引擎。该失败保留原始日志，不能计入已测配置。

## 23:08 · TensorRT 非量化 T/U 完成，未达到 100 ms

两组均使用同一引擎 `pi05-trt-bf16-20260907-r1`，三相机 224RGB / H50 / 去噪 10、固定种子 0、9 个真实录像输入各 20 次正式调用。T 为常规 execute，U 仅加静态 CUDA Graph 重放；MAXN 与最高 CPU/GPU/EMC 频率仅在测试会话启用，两个会话结束均恢复 120W、动态频率及自动风扇。

| 配置 | P50 / P95（ms） | 对 JAX A 的动作 MAE / 最大差 | 对同引擎普通执行 |
| --- | --- | --- | --- |
| T · `pi05-T-20260907-r2` | 129.1304 / 129.8775 | 0.002512555 / 0.017153978 | 参考 |
| U · `pi05-U-20260907-r1` | 127.3699 / 127.9938 | 0.002512555 / 0.017153978 | 9 输入、14D/32D 均精确零差异 |

T/U 所有输出有限、重复最大差 0，但对导出前 BF16 eager 的归一化 14D 最大差为 0.03393110；对 JAX A 的归一化 14D 最大差为 0.02510925。这些不是硬件单位容差或任务成功率。引擎编译带来的运算融合/顺序改变不能因为“非量化”就视为数学逐值相同。T 首次运行出现默认 CUDA stream 同步性能警告；U 的捕获采用独立 stream，且逐输入与未捕获引擎验证完全相等，但只节省约 1.76 ms，因此不继续把调度微调当作主要加速路线。

累计 **16 组有效配置、2880 次正式调用**；新增 T 首轮绑定失败不计入。最快仍为 L 约 118.85 ms（旧循环数值分歧未单独定位），更简单的 I 约 124.15 ms 继续保留，T/U 不能替代它成为默认。下一步先用同一真实输入分析引擎各算子耗时，定位注意力/矩阵计算热点，再决定 SDPA/融合注意力的 BF16/FP32 导出或局部精度方案；不直接跳 FP16/FP4，不减少相机、H50 或去噪步数。未来 JAX LoRA 合并与任务级准确率仍需微调产物验证。

证据：`docs/reports/thor/evidence/20260907/acceleration/pi05-{T-20260907-r2,U-20260907-r1}.*`、`compare-A-{T,U}-*.json`、`compare-T-U-20260907-r1.json`；原始动作数组分别保留于 Thor 和工作站各自 results 目录。中文 HTML 已纳入两组及新的后续计划，不把引擎构建时间写成推理延迟。

## 固定时间条件投影：层级测量改变下一步选择

`pi05-trt-profile-20260907-r1` 在同一非量化引擎、同一 9 个真实输入、同一 seed/noise/H50/10 步上完成 27 次带 Profiler 的调用。每个输入开启分析前后的原始 32D 和最终 14D 输出均相等。该诊断不增加正式 16 组/2880 次计数；TensorRT Profiler 会增加开销，各层平均时间之和 131.795 ms 不是新的端到端成绩。类型累计为 kgen 107.339 ms、gemm 24.100 ms、correlation 0.337 ms、custom_layer 0.019 ms；kgen 内含矩阵及融合计算，不能称为纯逐元素瓶颈。官方接口说明：https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/_static/python-api/infer/Core/Profiler.html 。

最大单个融合层为 18.6453 ms，包含 `node_linear_292` 到 `node_linear_1957` 等 370 个原始线性投影节点。检查原 ONNX 的 namespace / 输入依赖确认，前者对应 `gemma_expert.model.layers.0.input_layernorm.dense`，后者对应 `gemma_expert.model.norm.dense`；其输入来自 `time_mlp_in`、`time_mlp_out`，而不是相机或 noisy action。故本轮先测试固定时间条件缓存，不先假定瓶颈是注意力。

候选 `--cache-time-modulation` 对 37 个 AdaRMS 投影分别缓存 10 个时间点的 FP32 输出；每个时间点仍用原 batch=1 GEMV，避免合批选核改变舍入。原始线性权重保留，wrapper 外仍可执行原路径；缓存是 non-persistent buffer，不写回 checkpoint。任何权重/LoRA 或去噪时间计划变化都使缓存失效，必须按新权重重建。该选项仅存在于推理测试/导出 wrapper，不改训练默认路径。

`pi05-onnx-timecache-20260907-r1` 的 9 输入导出准备验证已通过：原始与最终动作均精确零差异。代码 `81356323839ab50a633432f80d4018b247c962a5`，仍使用 v6 镜像；后续 ONNX/引擎状态以其 exit/report 为准，准备通过不等于引擎或任务精度验收。下一配置 V 使用缓存引擎 + CUDA Graph，强制校验缓存来源，并继续验证同引擎图/非图完全一致及对 JAX A 的误差。

原始层级数据与动作保留在 Thor/工作站 `artifacts/pi05-trt-profile-20260907-r1/`，报告和宿主证据已归档至 `docs/reports/thor/evidence/20260907/acceleration/pi05-trt-profile-20260907-r1.*`。分析会话退出恢复 120W；新导出会话单独记录功耗恢复。

时间缓存 ONNX 导出已完成：`pi05-onnx-timecache-20260907-r1`，722.54 秒、18,225 节点，ONNX SHA256 `45e95be8da29bb1d8e7873375bcc5e378378329b5e4fbb15afc05db740f4fd96`，外置权重 SHA256 `83c38090d1882386b4ed8840c587eece6649c4222c5570da9f003f2ffb50a620`，缓存 NPZ SHA256 `ccc731e87c791ab1b29495aba5721ae674ddffc2804689d57e2164e6c6c1b8b2`。checker/宿主退出码 0、恢复 120W；不修改原始权重。引擎另存为 `pi05-trt-timecache-20260907-r1`，状态须查该 run 的报告。

备用优化观察（尚未实施）：用同一输入变换、norm 和本地 tokenizer 在工作站重算 9 个输入，实际文本 token 数依次为 `[69,64,67,68,69,69,69,69,70]`；当前 ONNX/引擎仍按 200 位置执行。本轮 V 不改变这一合同。若后续评估移除末尾 mask=false 填充，必须完整保留实际 prompt/状态 token，超出候选桶长时回退 200 路径，且重新做完整输出对照；不能把直接降低 tokenizer max_len、截断有效 token 当作同任务加速。

## V · 时间缓存引擎完成：108.85 ms

`pi05-trt-timecache-20260907-r1` 构建成功，477.41 秒，引擎约 6009 MiB，SHA256 `abc91e29f7b0d022fe187a6f82325c3e61402c6d8bcbd1f7981a2f204625d954`。层报告 2404 层（kgen 2310、gemm 91、custom 2、correlation 1）；旧引擎那个包含 370 个线性节点的巨大时间投影融合层已经消失，新层最多包含 3 个线性节点。

`pi05-V-20260907-r1` 使用 b5d6bc303276188effc0bbfd802308423c04b3da / v6 镜像，9 输入 × 20 次正式调用、每输入 2 次预热。P50 **108.8454 ms**、P95 **109.5225 ms**，比同样 CUDA Graph 的 U 组减少 **18.5244 ms**。全部输出有限，重复差 0，逐输入图/非图原始输出差 0；输入仍是三相机 224RGB、文本 200、H50、去噪 10，没有删减任务内容或启用 FP16/FP8/FP4。

精度分开记录：对 JAX A 的动作 MAE **0.002517905**、P95 绝对误差 **0.009023270**、最大差 **0.017522693**；归一化 14D 最大差 **0.026319869**。对导出前 BF16 eager 的归一化 14D 最大差 **0.034132626**。V 相对旧 U 引擎的动作 MAE **0.000261664**、最大差 **0.002023801**，因此只能称“缓存准备零差异、同一引擎图重放零差异”，不能称“新旧引擎或 JAX 全链路零差异”。重新编译的融合/舍入有变化，当前没有任务级精度放行。

累计 17 组有效配置 / 3060 次正式调用，27 次 Profiler 调用另列；本轮构建与推理均退出 0，并恢复 120W、动态调频与自动风扇。V 成为下一步性能候选，原 JAX 和 I 路径保留。仍未达到 ≤100 ms；下一项可独立验证 masked padding 消除（不能降低 tokenizer max_len 截断有效 token），需以 FP32 对照区分语义错误与 BF16 内核舍入，长输入保留原文本 200 路径。

证据归档：`pi05-trt-timecache-20260907-r1.*`、`pi05-V-20260907-r1.*`、`compare-A-V-20260907-r1.json`、`compare-U-V-20260907-r1.json`，仍位于 `docs/reports/thor/evidence/20260907/acceleration/`；动作原始数组在 Thor/工作站 results 中。候选复测与后续方案见 [Pi0.5 候选测试方案](11_pi05_candidate_test_plan.md)。
