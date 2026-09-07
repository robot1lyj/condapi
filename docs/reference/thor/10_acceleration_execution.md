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
