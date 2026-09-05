# 08 · Thor 端侧部署

本页是 NVIDIA Jetson AGX Thor IPC 的端侧系统、Pi0.5 转换/加速和推理验收的唯一 owner。训练仍在服务器 GPU 上进行；3588 IPC 负责相机采集、机械臂驱动、CAN、GUI、home pose、限位和急停，本仓库不读取或修改 3588 的实现。

## 1. 当前决策

- 默认推理目标是 Jetson AGX Thor Developer Kit（T5000 口径）；设备到手后仍需用 `jetson_release`/`cat /etc/nv_tegra_release` 核对实际 SKU。若实际是 T4000 或定制载板，不能直接套用开发套件 ISO。
- 官方系统基线选 JetPack 7.2.1 / Jetson Linux r39.2.1。系统盘制作介质是 Jetson ISO USB 安装盘，实际 BSP 安装到 Thor 的 NVMe，不把 ISO 当作 Live USB。
- 部署形式确定为 Docker + NVIDIA Container Toolkit，以 Docker Compose 管理每模型独立容器。首版在容器内验证原生 JAX；“原生 JAX”表示保留原模型实现与权重，不表示直接安装到宿主机。模型 Python 依赖、转换工具和开发环境均封装在各自镜像中。
- 用户确认训练产物必定为 JAX/Flax（Orbax）checkpoint；当前 `pi05_yam_lora` 的两条 Gemma 分支均有 LoRA。原始 checkpoint、配置、norm 和原 JAX policy 行为是部署验收依据，必须完整保留。
- 精度优先：先建立原 JAX golden，并验证 Thor 容器内原生 JAX 的可行性；需要转换时，先审计 LoRA 与权重映射，使用 FP32 中间产物，再验证与参考一致的未量化混合精度运行。容器部署形式已确定，模型后端的生产验收仍待实测，不预设 PyTorch BF16 → TensorRT FP8 为必经路线。
- TensorRT BF16/FP32 混合精度是未量化候选，须另行验证 exporter 支持；FP8、NVFP4、定制 FP16 均为可选实验。只有逐层、逐去噪步、完整 action 与任务验收通过后才能晋级；cosine、有限输出或时延不能单独证明精度保持。
- Thor 只在本地加载和执行模型；3588 通过直连以太网发送相机/状态/prompt observation，Thor 通过同一条直连链路返回 action chunk。WebSocket 或后续约定的直连协议是两 IPC 的生产数据通道，不等于远程模型推理；Thor 内部仍可用本地 direct API 做基准 smoke。

## 2. 官方系统基线

截至 2026-09-05，NVIDIA 官方下载页将 JetPack 7.2.1 列为最新版本，核心版本如下：

| 项目 | 版本 |
|---|---|
| JetPack | 7.2.1 |
| Jetson Linux / L4T | r39.2.1 |
| Ubuntu | L4T Ubuntu 24.04 |
| Kernel | 6.8 |
| CUDA | 13.2.1 |
| cuDNN | 9.20.0 |
| TensorRT | 10.16.2 |
| NVIDIA Container Toolkit | 1.19（ISO 安装介质包含） |

权威入口：

- [JetPack SDK Downloads and Notes](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson AGX Thor Quick Start](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html)
- [Jetson AGX Thor BSP Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_bsp.html)
- [Jetson AGX Thor JetPack SDK Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_jetpack.html)
- [Jetson Linux r39.2.1 Release Notes](https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf)

本次下载的官方 ISO：

```text
URL:  https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/iso/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso
PATH: /home/wuyan-lyj/thor-system/jetpack-7.2.1/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso
```

2026-09-05 下载正常退出；已核对官方 HTTP 文件长度 `5,035,601,920` bytes，识别为可启动 ISO 9660，并计算本地 SHA-256：`f31cf35023cd072707dfc46ef23ed71318449516333055fe3bb9455a615395d0`。这是本地文件摘要，尚未与 NVIDIA 发布的参考摘要或签名比对。介质约 4.69 GiB（5.04 GB），制作 USB 需要至少 16 GB U 盘和至少 25 GB 的下载/制作空间。

## 3. 从零安装顺序

1. 在任意 Windows、macOS 或 Linux 主机下载 ISO，用 Balena Etcher 或同类工具写入至少 16 GB U 盘；不要把 ISO 文件简单复制到 U 盘。
2. Thor 插入 U 盘并上电，按官方 Quick Start 进入安装器；若提示 QSPI capsule update，确认 `Y`，否则新 ISO 与出厂 UEFI 可能不兼容。
3. 安装目标选择 `Install on NVMe`。r39.2.1 release notes 说明 Thor 的默认 BSP 假定 NVMe 大于 234 GiB；刷写前核对实际盘容量，较小盘需要按官方 `EXT_NUM_SECTORS` 方案调整。安装结束后拔出 U 盘，再完成 `oem-config` 的用户、网络和时区设置。官方 USB 介质只用于安装 BSP，不能作为完整系统试运行盘。
4. 首次启动记录以下版本和硬件信息：

   ```bash
   cat /etc/nv_tegra_release
   uname -a
   jetson_release
   docker --version
   dpkg-query -W nvidia-container-toolkit
   nvidia-smi
   ```

5. ISO 安装方式通常已经包含 Docker 和 NVIDIA Container Toolkit；先做 GPU 容器 smoke，不要重复覆盖系统软件。如果不是 USB ISO，而是 SDK Manager 或 `Linux_for_Tegra` 手工刷写，按 [NVIDIA Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html) 安装并配置 Docker/Container Toolkit。
6. 核验 Docker Compose 与 NVIDIA GPU 容器接入，再构建第 3.1 节的 JAX 推理镜像。宿主机不安装模型专属 Python/conda 环境；系统级组件确有需要时从 JetPack APT 源安装，不安装 Ubuntu 的 `nvidia-cuda-toolkit`。基镜像必须匹配 ARM64、Thor GPU、宿主驱动和 CUDA 用户态库；具体镜像 digest 与 JAX/Flax/Orbax 版本组合通过实机验收后锁定，不直接采用 PyTorch 教程镜像作为 JAX 已验证环境。

### 3.1 每模型独立容器的部署约定

2026-09-05 用户指定容器化作为默认部署方式。本节为待实施的容器方案；尚未创建或验证 Dockerfile、Compose 文件或模型镜像。架构为：

```text
3588 IPC（相机、状态、机械臂控制）
  <== 直连以太网 / observation、action ==>
Thor 专用网卡 IP + policy 端口
  -> Docker 发布端口
  -> 当前激活的模型容器：OpenPI JAX + 原始 checkpoint/LoRA
  -> NVIDIA Container Toolkit -> Thor GPU
```

| 层次 | 管理内容 | 生命周期 |
|---|---|---|
| Thor 宿主机 | JetPack/L4T、GPU 驱动、Docker、Compose、NVIDIA Container Toolkit、网卡 | 统一维护，不随模型安装 Python 包 |
| 版本化镜像 | 固定代码 commit、Python、JAX/Flax/Orbax、CUDA 用户态依赖及启动入口 | 构建后以不可变 digest 记录；依赖或代码改变时构建新版本 |
| 每模型容器 | 独立服务名、模型配置、端口与运行参数 | 一个模型一个容器实例；依赖相同的模型可复用同一镜像，不共用可写运行目录 |
| 模型与运行数据 | 原始 checkpoint/norm/tokenizer、golden、日志和编译缓存 | 模型与 golden 只读挂载；日志和缓存独立持久化，重建容器不覆盖模型 |

具体实施规则：

1. 在镜像构建阶段通过项目约定的 conda/pip 安装固定依赖，记录基础镜像 digest、依赖清单和代码 commit；不使用 `latest` 作为生产版本，不在运行中容器临时升级依赖来形成不可复现的环境。需要尝试 Torch/TRT 时新增对应候选镜像和容器。
2. checkpoint 整目录以只读方式挂载，保持 `params/` 与 `assets/yam/norm_stats.json` 的绑定；tokenizer 和 golden 也只读。镜像内包含固定版本代码，生产不依赖可变的宿主源码挂载。编译/下载缓存按模型与运行环境版本隔离并显式指定可写位置；日志按模型与运行批次保存。
3. 用 Compose 显式声明 GPU 访问、挂载、环境变量、端口、启动命令和健康检查，便于查看状态、日志与重建。默认一个模型容器占用生产 GPU 服务；多个模型可以同时保留镜像和配置，但同时运行必须另测统一内存占用与延迟。容器不是 GPU 资源完全隔离的虚拟机。
4. 首版使用 bridge 网络，将 policy 端口明确发布到 Thor 的直连网卡 IP；容器内服务监听 `0.0.0.0`，3588 连接 Thor IP，而非容器内部地址。IP/实际端口联调时确定；多候选容器使用不同测试端口。仅通过网线接收 observation，不给模型容器配置相机或机械臂设备透传。
5. 首次启动必须完成 GPU 实际运算、checkpoint/norm 校验、JAX 编译预热和固定样本推理，之后才报告模型就绪；健康检查不能只看容器进程或 TCP 端口。后续轻量健康检查读取就绪状态，不反复触发大模型冷编译。
6. 发布单元同时绑定镜像 digest、模型摘要、配置、Compose 参数和精度报告。先在独立候选容器完成 golden/延迟/网络验收，切换时由控制侧负责人配合暂停请求，再把生产端口交给通过的容器。保留前一版本镜像、模型及配置以回退，不删除已有模型资产。
7. 容器仍共享宿主内核、GPU 驱动与硬件。更换镜像不能绕过 Thor/CUDA/JAX 兼容检查，也不自动保持数值精度；镜像、依赖、驱动或编译设置变动后重新跑第 6 节验收。

实施交付物计划为 Dockerfile、版本锁定依赖清单、每模型 Compose 配置及镜像/模型 manifest；待 Thor 环境核验后实现，不在本次方案更新中填入未经验证的基镜像标签。依据：[NVIDIA Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)、[Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)、[Docker 端口发布](https://docs.docker.com/engine/network/port-publishing/)（2026-09-05 核对）。

## 4. JAX 训练产物的精度深度调研

调研日期：2026-09-05。对象是本项目部署负责人，问题是 JAX 训练的 Pi0.5 LoRA 能否在 Thor 保持行为。结论依据官方文档、上游 issue/PR、作者实验记录和本仓库静态代码审计；本次没有运行 YAM checkpoint 的跨框架实验。以下区分已确认机制、社区测量和待验候选，不把社区阈值作为本项目验收标准。

### 4.1 先分清格式、存储精度与计算精度

JAX/PyTorch 是实现框架，Orbax/SafeTensors 是存储格式；BF16、FP32、FP8 等是数值类型。换容器格式本身不要求量化，但转换脚本可以同时改变权重数值、模型结构、预处理和算子行为。JAX checkpoint 也不能仅凭文件格式断定所有参数均为 FP32，必须读取各 tensor 的 dtype。

本仓库在调研基线 `09ce14f` 下的关键发现：

| 环节 | 静态代码证据 | 对本项目的影响 |
|---|---|---|
| 转换时提前降精度 | [转换脚本](../examples/convert_jax_model_to_pytorch.py) 先以 FP32 restore，再用原 `model_config` 构造模型、加载权重，最后才 `.to(float32/bfloat16)`；[配置](../src/openpi/models/pi0_config.py) 默认 BF16 | 仅加 `--precision float32` 不能保证无损：FP32 值可能已在加载到 BF16 参数时舍入，之后升回 FP32 无法恢复 |
| BF16 并非整网单一精度 | [gemma_pytorch.py](../src/openpi/models_pytorch/gemma_pytorch.py) 将部分视觉 embedding、LayerNorm/RMSNorm 参数保留为 FP32；转换脚本最终整网 `.to(bfloat16)` | 在磁盘上先把这些参数降为 BF16，再加载为 FP32，也无法恢复被舍去的信息 |
| LoRA 未合并且加载宽松 | 转换脚本没有 LoRA 合并步骤，`load_state_dict(..., strict=False)` 的返回值未检查 | adapter 更新可能被静默忽略；打印转换成功不证明保留了微调结果。不能用 `strict=False` 作为 dtype/shape 错误的修复办法 |
| 默认加载器会改变 dtype | [policy_config.py](../src/openpi/policies/policy_config.py) 的 JAX 路径显式 BF16 restore，Torch 路径再次选择性 BF16 cast | 应保存“原生产 JAX 路径”作为行为基线，另建受控 FP32 诊断对照。一个 FP32 文件经默认 loader 加载后不等于全 FP32 运行 |
| norm 资产位置 | 转换脚本从 `checkpoint_dir.parent/assets` 复制，policy 从 `checkpoint_dir/assets` 读取 | 必须验证转换 bundle 实际使用的是同一份 `assets/yam/norm_stats.json`；不能让缺失/错误 norm 被误判为精度问题 |

这些是代码审计发现，尚不是本项目实测误差。上游 [PR #978](https://github.com/Physical-Intelligence/openpi/pull/978) 专门提出在构造转换模型前把 `model_config.dtype` 设为 FP32；2026-09-05 API 核验仍为 open、未合并。该提案不修改最终输出 dtype，也不解决 LoRA 合并。

### 4.2 LoRA 比单纯换 dtype 更关键

[Issue #958](https://github.com/Physical-Intelligence/openpi/issues/958) 报告转换静默丢弃 LoRA；[PR #960](https://github.com/Physical-Intelligence/openpi/pull/960) 尝试合并 adapter；[PR #984](https://github.com/Physical-Intelligence/openpi/pull/984) 只增加未合并 LoRA 检查。两个 PR 在核验时均未合并。#984 不能当作完整转换方案。

本项目 `pi05_yam_lora` 同时启用 `gemma_2b_lora` 与 `gemma_300m_lora`。本地 [lora.py](../src/openpi/models/lora.py) 中，Einsum 分支使用 `scaling_value`，FeedForward 的 `_dot()` 不乘该缩放；`attn_vec_einsum` 的合并还须遵循实际 einsum 的跨 head 求和语义。应按本仓库执行方程逐项验证，不能直接套另一实现的 `W + alpha/rank * AB`。

PR #960 作者给出的局部 checkpoint 对照如下；样本范围、硬件及完整任务评估不足以作为 YAM 结论：

| 合并后的存储方案 | mean_abs | max_abs |
|---|---:|---:|
| FP32 | 0.0000929647 | 0.0016714334 |
| BF16 | 0.0011840940 | 0.0170528293 |

本项目由此采用的工程要求是：保留原始 LoRA checkpoint，转换中间主副本使用 FP32，并分别比较“JAX 在线 LoRA → JAX 合并模型 → Torch 合并模型”。FP32 合并也不保证与低精度在线 LoRA 完全相同，因为运算顺序发生变化；后续 BF16 计算仍可能再次损失小幅 adapter 更新。若不能过 gate，应保留在线 LoRA 或原 JAX 路径，不能放宽阈值来接受转换。

### 4.3 真实负面反馈与官方示例的证据范围

[Issue #840](https://github.com/Physical-Intelligence/openpi/issues/840) 的用户报告 JAX 模型表现良好，但转换后机械臂动作异常；该报告同时改变了 Torch 版本和编译设置，不能把原因全部归于量化。[Issue #810](https://github.com/Physical-Intelligence/openpi/issues/810) 的初始脚本只固定 observation，没有向两个 policy 注入同一噪声，因此原帖差值也不能直接作为转换误差。

与 Thor 直接相关的 [#826 评论](https://github.com/Physical-Intelligence/openpi/issues/826) 中，`openpi-thor` 作者报告按照当时教程转换后表现很差；自己的修订版 FP16 接近 JAX，FP8 动作较嘈杂。其最初 NVFP4 失败描述后来被更新为可运行。它是作者经验，未提供 YAM 任务的受控统计，不能解读为所有 FP8 或 NVFP4 都不可用。

[NVIDIA Jetson AI Lab 教程](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) 使用 `pi05_libero`，上游固定 commit 为 `15a9616a00943ada6c20a0f158e3adb39df2ccac`；JP 7.2、horizon 10 的延迟约为 BF16 Torch 132 ms、TRT FP8 54 ms、混合 NVFP4 49 ms。它提供可复现的性能流程，但不足以证明本项目 JAX LoRA 精度保持。

本次进一步读取 NVIDIA 发布的脚本，发现需要单独补齐的验收环节：

- [转换 overlay](https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/examples/convert_jax_model_to_pytorch.py) 仍有先构造低精度模型、后输出转换和未检查的 `strict=False`，没有完整 LoRA 合并。
- [比较脚本](https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/deployment_scripts/pi05_inference.py) 的 `compare` 比较 Torch 与 TRT，共用 noise；默认 `--use-dataset=False`，使用合成 observation。它不替代 JAX 原模型与真实 YAM 数据对照。
- [ONNX exporter](https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/deployment_scripts/pytorch_to_onnx.py) 的 `_prepare_model_for_export()` 先把模型转为 FP16，再进行 FP8 量化；因此“FP8 engine”不表示所有其他计算都保留 BF16。它还会跳过失败的 calibration batch，并保留 dummy calibration 分支；YAM 验收必须检查实际校准成功数与来源。
- [下载脚本](https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/download.sh) 会覆盖上游源文件。固定 OpenPI commit 还不够，必须记录 overlay 文件摘要，不能直接覆盖本仓库的 YAM/RTC 改动。

### 4.4 FP16、BF16 和低比特候选如何判断

[TensorRT 数值精度文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html) 说明 FP16 最大有限值为 65504，BF16 的动态范围更大，但有效尾数更少。BF16 并非在所有数值上都比 FP16 精细；这里关注的是源模型的范围与敏感计算。Softmax、归约和归一化等需检查中间值，不能只检查最终 NaN/Inf。

`openpi-thor` 的 [作者说明](https://github.com/xuweiwu/openpi-thor) 使用保留部分 FP32 计算的定制 FP16 和 strongly typed TRT；这与官方“不支持纯 FP16”的警告并不矛盾。作者还报告广泛 attention+MLP NVFP4 在 TRT lowering 后 MAE 约为 FP8 的 30 倍，后续缩小量化范围才改善。其验证阈值随精度变化且与任务相关，不能复制到单位未定的 YAM。

对 TRT 10.x，strong typing 约束图中类型，但不自动修复图中已有的错误 cast，也不承诺与 JAX 逐位一致。必须显式审计类型、Q/DQ、累加精度及 TF32；不要混用 strongly typed 与 `--fp16/--bf16/--fp8` builder 精度标志。依据 [TRT 10.x precision control](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)。

[FlashRT](https://github.com/flashrt-project/FlashRT) 的证据比单纯延迟表更完整：作者公开 Thor LIBERO Spatial 上 FP8 与某个 NVFP4 encoder 配置均为 491/500 次成功。但该对照不是原始 JAX，也不是所有 FP4 preset。其 Orbax/JAX frontend 会调用自定义 CUDA kernels；能直接读取 JAX 权重不等于执行原 OpenPI JAX 数学实现。

尤其要保留 [FlashRT 2026-08-05 详细报告](https://github.com/flashrt-project/FlashRT/blob/054bea4d02ebc63f6a0c45991c6061b1e1caa46c/docs/pi05_thor_decoder_fp4_e2e.md) 的限制：23.01 ms 单视角 NVFP4+FA4 配置未过逐样本 fidelity gate；2/3 视角的 27.17/31.74 ms 配置在该报告中通过，但比较基准仍是同轮 FP8。作者将部分单视角偏差解释为去噪轨迹分岔，这是作者的归因，不能据此豁免 YAM gate。

### 4.5 Thor 原生 JAX 必须保留为待验选项

[JAX 官方安装页](https://docs.jax.dev/en/latest/installation.html) 当前列出 Linux aarch64 与 CUDA 13 支持；它不能保证任意 wheel 含 Thor 所需全部 GPU kernels。[NVIDIA 论坛](https://forums.developer.nvidia.com/t/how-to-install-jax-0-5-3-on-jetson-thor-device/362519) 记录了旧 JAX 0.5.3/CUDA 13 构建不兼容，另有 0.9.1 能枚举 GPU 却在 warmup 报 `no kernel image`；NVIDIA 回复指向针对 Thor 的 0.10.0 构建配置，未给出本项目 Pi0.5 完整验收。

本仓库依赖为 `jax[cuda12]==0.5.3`、`flax==0.10.2`、`orbax-checkpoint==0.11.13`（[pyproject.toml](../pyproject.toml)）。因此后续在独立 Thor JAX 容器内验证兼容组合、Orbax restore、Flax/NNX API、真实 GPU 运算及完整 policy；不得只升级一个 JAX 包就宣称可部署，也不得通过跳过 CUDA 兼容检查来接受结果。若原生 JAX 满足精度与延迟要求，该容器可晋级生产，避免增加转换环节。

目前没有找到已公开验证的“本项目 JAX LoRA + YAM 三路图像 + 14D 输出 + horizon 50”完整部署案例。部署后端是待实测决定的事项。

## 5. YAM 端侧落地计划

以下是待实施方案，不代表已经完成转换或部署。bundle 绑定训练 commit、原 checkpoint、完整 config、norm、tokenizer 和 YAM 数据版本；训练用 JAX 这一事实不变。

1. 在已验证的训练环境保存原 JAX policy golden：原始 observation、实际模型输入、参数 dtype 清单、完整噪声数组和各步输出。确认使用训练 checkpoint，而非基础权重或错误的 EMA 分支；当前 LoRA 配置 `ema_decay=None`。
2. 按第 3.1 节构建独立 JAX 候选容器，优先做 Thor 原生 JAX 可行性核验；在同一 golden 上比较服务器与容器输出和实际延迟，镜像/环境变化造成的差异也必须定位。
3. 如需 Torch，先在独立输出目录进行 LoRA-aware FP32 转换。逐参数检查 key/shape/dtype/布局，所有 adapter 均须被映射或经验证合并；missing/unexpected keys 必须为空或仅有已核实的 tied-weight 白名单。不要直接运行现有通用脚本充当验收通过的转换器。
4. 建立两种互补对照：原 JAX 生产行为对照，以及双方对齐 FP32 设置的诊断对照。先逐层验证，再启用与源路径匹配的 BF16/FP32 混合精度；逐个开启 compile、attention 优化、CUDA Graph 等变量。FP32 文件经默认 loader 自动 cast 的情况必须被记录。
5. 只有精度合格但延迟仍不满足需求时，才评估 TRT 或 FlashRT。先确认未量化图/导出改写保持行为，再单独实验 FP8，最后按需实验 NVFP4；每个阶段都与上一阶段和原 JAX 比较。TensorRT BF16/FP32 导出需另行实现/验证，不能声称官方示例已直接支持。
6. 先做真实 YAM 留出集回放与 Thor 本地 smoke，再做直连以太网 shadow observation/action 往返。闭环任务验收由控制侧负责人配合完成；本项目只操作 Thor，不修改或操作 3588 的控制、相机和系统。任何候选失败均回退到上一个通过的产物。

## 6. 端侧验收闸门

- 系统为 JetPack 7.2.1 / L4T r39.2.1，GPU/容器可见，Docker runtime 正确。
- 镜像 digest、Compose 启动配置及只读/可写挂载可追溯；容器内完成 GPU 运算、预热和真实 policy 推理，重建后可恢复同一模型/配置，持久化日志与原始权重完整。
- checkpoint、config、norm stats、tokenizer、LoRA 合并清单、参数 dtype、转换/overlay 版本与哈希、engine/插件摘要可追溯。
- 固定同一份噪声数组：内部形状 `(1,50,32)`，policy API 可传 `(50,32)`；只设置相同 seed 不保证 JAX/Torch 生成相同噪声。固定输入图像、resize/padding、RGB/布局、token IDs、mask、量化 state、去噪时间表和 RTC 设置。
- `action_horizon=50` 是预测序列长度；`num_steps=10` 是当前默认 flow 去噪次数，二者必须分别记录，不可混淆。
- 逐层定位首个偏差：视觉特征 → token/prefix → KV cache → 每个去噪步的 velocity/action → 完整 32D 原始结果 → 14D normalized/delta action → unnormalize 后关节/夹爪 → 加当前 state 后的 absolute action。只比较最终 absolute 值可能掩盖 delta 偏差。
- 对固定真实留出集报告逐关节、逐时刻 MAE、P95/P99、最大绝对误差、最差样本、cosine、动作跳变和夹爪转换时刻；逐维物理单位从数据 audit 确认。零向量附近 cosine 不稳定；`a` 与 `2a` 的 cosine 为 1，仍可有不可接受的动作幅度差。
- calibration 与 validation 按 episode 分离，覆盖双臂、三相机、任务阶段和场景变化；记录成功/失败校准样本数，禁止将 dummy 或全失败 calibration 当作生产证据。阈值必须在评估候选前按 YAM 单位、原 JAX 重复性和控制容差确定；当前没有经过认可的数值阈值，不能从社区 cosine 门槛推出“通过”。
- 回放误差不能替代闭环任务成功率。实际验收需匹配任务、初始条件、执行 chunk 长度与延迟，记录试验数、成功率及置信区间；在固定执行节奏下比较行为，再独立比较低延迟收益。控制侧试验结果由其负责人提供。
- 输入保持 `observation.state`、三路 `observation.images.*` 和 prompt；输出为有限 `(50,14)`，不得把内部 32D padding 发送给机器人。
- 记录 Thor 温度、功耗模式、warmup、端到端时延和 action horizon；不得把 10 步 LIBERO benchmark 当作 YAM 闭环频率。
- 端口监听或容器启动不算通过；必须有真实 Thor 本地推理结果和 Thor↔3588 直连以太网 smoke。

## 7. 当前状态

- 官方 JetPack 7.2.1 ISO：2026-09-05 已下载，长度/类型已核对且本地 SHA-256 已记录，详见第 2 节；USB 制作与刷写未完成。
- 每模型独立 Docker 容器方案：已确定并记录；首版候选为容器内原生 JAX。Dockerfile、Compose、镜像构建与容器内 GPU/精度/网络验收仍待实施。
- 精度调研与静态代码审计：已完成；已确认训练产物为 JAX，并撤回无条件 FP8 默认路线。LoRA-aware 转换器、FP32 诊断工具、YAM 精度阈值尚未实现/确定，不能使用现有脚本直接宣称转换可靠。
- Thor 实机刷写、Docker GPU smoke、YAM checkpoint 转换、TensorRT engine 和真实三路输入 smoke：计划中，尚未宣称已验证。

## 8. 调研溯源与复核范围

本页同时作为本次仓库 Markdown 调研报告与部署事实 owner，避免建立平行报告缓存。来源访问日均为 2026-09-05；动态网页内容、PR 状态和依赖版本在实际部署时复核。

| 关键主张 | 来源与日期/版本 | 证据等级与缺口 |
|---|---|---|
| 转换后动作异常 | OpenPI #840（2026-01-07）；#826 作者评论（2026-03-30，后续编辑） | 用户/作者报告；原因混有环境、dtype、转换与量化，非完整因果证明 |
| LoRA 被忽略、合并特殊语义 | OpenPI #958；#960（2026-06-02，head `46fe49901bfee3b6d8c768a755e86699d5006b48`） | 作者局部数值结果 + 本仓库代码互证；PR 未合并，无 YAM 闭环证据 |
| 防止低精度中间加载 | #978（2026-06-16，head `992b4e14260d045cb06196e26d1e73cccee4ee65`） | 小范围代码修订、未合并；不含 LoRA 合并修复 |
| LoRA fail-fast | #984（2026-06-24，head `467ee43e3e98097cf6cad2f5fa6798c4238f4181`） | 仅 guard、未合并；不能据此认为转换受支持 |
| 官方 benchmark 与验证范围 | NVIDIA 教程及其 overlay；OpenPI 固定 `15a9616a00943ada6c20a0f158e3adb39df2ccac` | 教程及脚本可读；overlay 非固定版本 URL，部署时记录摘要；未见本项目 JAX LoRA 验收 |
| 定制 FP16/NVFP4 经验 | openpi-thor HEAD `40c88146e5e6b82526db6583c8a75520e02e551e`；作者测试 JP7.0/L4T38.2.2/TRT10.13.3.x | 与本项目 JP7.2.1 不同；需重跑，默认门槛不可外推 |
| 低比特成功与失败配置 | FlashRT HEAD `054bea4d02ebc63f6a0c45991c6061b1e1caa46c`；详细报告 2026-08-05 | 作者报告，有任务试验和逐样本限制；未取得本项目 JAX 原模型对照 |
| JAX Thor 可行性与限制 | JAX 官方安装页；NVIDIA Thor 论坛 2026-03-05 至 03-16 | 官方平台说明 + 实际失败记录 + 构建指引；OpenPI/Flax/Orbax 组合未验 |
| 累加/类型与数值误差 | NVIDIA TensorRT accuracy / 10.x precision control；[PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html) | 框架数值行为依据；不是 YAM 容差或升级依赖的指令 |

本地审计基线 `09ce14f`；转换脚本 Git blob 为 `632c0b8782c1ecb5cb380130a30a3152b220eafd`，policy loader 为 `6570df05ed068f297d48326990a1cd10ee68ac5a`，Gemma Torch 为 `203b36be8ae4c525422c93aa50926c03ded7deb5`，LoRA 实现为 `3dfff5b4f22ff42f73298037a77df900c6b11d75`。本次未改这些代码，也未运行模型或采集机器人数据。

当次读取的 NVIDIA overlay SHA-256（对应第 4.3 节链接，便于识别后续网页更新）：

| 文件 | SHA-256 |
|---|---|
| `examples/convert_jax_model_to_pytorch.py` | `a496c143f11e9bc9cc160d7cf1889568f8074fb3404e9ed301524408e30af428` |
| `deployment_scripts/pi05_inference.py` | `b9c56d68964e34ff200fbd8bdf986443f7ca8ce7197df7cdeabfa0b165ef9946` |
| `deployment_scripts/pytorch_to_onnx.py` | `b0f7709de2fcd6c15b74edefa01b547161a00c5b4b7ad5300ad9b3ec256b4f1a` |

检索分两轮：先检索 OpenPI 的 conversion/precision/LoRA/Thor 报告及官方教程，再追踪 #958/#960/#978/#984、官方实际 overlay、FlashRT fidelity 报告与 JAX CUDA13 安装支持。GitHub #810 评论 API 在一次替代分页重试后仍返回 403，因此仅使用可见的 issue 正文，不推断关闭原因。关键机制已有代码互证，剩余缺口需要具体 checkpoint 与 Thor 实测，继续增加同类网页不会解决，因此结束文献检索。

记忆技能用于按证据修订 owner 并刷新 kernel；精度调研完成时记忆准入账本记录 `27,708 / 32,768` UTF-8 bytes，后续更新沿用同一账本继续计费。宿主完整请求 token 计量不可用，不能宣称该账本限制了全部对话/工具上下文。报告采用仓库 Markdown，执行结构与链接检查，不涉及 PDF/页面渲染验收。
