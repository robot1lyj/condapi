# 08 · Thor 端侧部署

逐步安装指导的冷记忆入口：[Thor 安装系列 00](reference/thor/00_start_here.md)。该组按 G0–G5 持有设备预检、ISO/USB 制作、固件/NVMe 安装、宿主/容器检查、Pi 工程闸门及故障交接；本页继续持有版本、精度决策和当前状态。指导 agent 不得把下文安装概览当成跳过目标确认的操作脚本。

本页是 NVIDIA Jetson AGX Thor IPC 的端侧系统、Pi0.5 转换/加速和推理验收的唯一 owner。训练仍在服务器 GPU 上进行；3588 IPC 负责相机采集、机械臂驱动、CAN、GUI、home pose、限位和急停，本仓库不读取或修改 3588 的实现。

## 1. 当前决策

- 默认推理目标是 Jetson AGX Thor Developer Kit（T5000 口径）；设备到手后仍需用 `jetson_release`/`cat /etc/nv_tegra_release` 核对实际 SKU。若实际是 T4000 或定制载板，不能直接套用开发套件 ISO。
- 官方系统基线选 JetPack 7.2.1 / Jetson Linux r39.2.1。系统盘制作介质是 Jetson ISO USB 安装盘，实际 BSP 安装到 Thor 的 NVMe，不把 ISO 当作 Live USB。
- 部署形式确定为 Docker + NVIDIA Container Toolkit，以 Docker Compose 按模型系列管理容器：Pi 系列共用一个服务，通过配置/checkpoint 选择模型，默认一次加载一个。首版在容器内验证原生 JAX；“原生 JAX”表示保留原模型实现与权重，不表示直接安装到宿主机。模型 Python 依赖、转换工具和开发环境均封装在系列镜像中。
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

### 制盘电脑上的 Etcher 安装包

2026-09-07 已按当前制盘电脑 Ubuntu 24.04 / x86_64 下载官方 Etcher 2.1.6 Debian 包，未安装或启动，未刷写任何磁盘。它安装在制盘电脑，不是在 ARM64 Thor 上。

- 本地路径：`/home/wuyan-lyj/thor-system/tools/etcher-2.1.6/balena-etcher_2.1.6_amd64.deb`（仓库外，不提交二进制）。
- 长度：`123910696` 字节；Debian 包元数据为 `balena-etcher / 2.1.6 / amd64`。
- SHA-256：`2bdebb46c9f750a9abf11c188ff69a405b4a4fed114333d634c3b3fe59a64057`，与 [官方 v2.1.6 Release 资产](https://github.com/balena-io/etcher/releases/tag/v2.1.6) 发布的摘要一致。
- 下载源：[官方 amd64 Debian 包](https://github.com/balena-io/etcher/releases/download/v2.1.6/balena-etcher_2.1.6_amd64.deb)。后续安装/制盘见 [冷手册 02](reference/thor/02_iso_and_usb.md)。

### 安装概览

本节为架构概览，真正指导用户操作须从 [冷手册 G0](reference/thor/01_preflight.md) 开始，逐阶段通过。尤其在 QSPI 更新进行中不得断电，在安装 NVMe 前必须确认唯一目标和备份；USB 与 NVMe 覆盖授权不能互相替代。

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

### 3.1 按模型系列隔离容器的部署约定

2026-09-05 用户指定容器化，并进一步明确每个模型系列一个容器，Pi 系列一个即可。Pi0、Pi0.5 及不同微调 checkpoint 作为该系列的模型选择项，逐一验证兼容性，不按每个模型创建常驻容器。本节为待实施的容器方案；尚未创建或验证 Dockerfile、Compose 文件或模型镜像。架构为：

```text
3588 IPC（相机、状态、机械臂控制）
  <== 直连以太网 / observation、action ==>
Thor 专用网卡 IP + policy 端口
  -> Docker 发布端口
  -> Pi 系列容器：OpenPI JAX + 当前选定的原始 checkpoint/LoRA
  -> NVIDIA Container Toolkit -> Thor GPU
```

| 层次 | 管理内容 | 生命周期 |
|---|---|---|
| Thor 宿主机 | JetPack/L4T、GPU 驱动、Docker、Compose、NVIDIA Container Toolkit、网卡 | 统一维护，不随模型安装 Python 包 |
| 版本化镜像 | 固定代码 commit、Python、JAX/Flax/Orbax、CUDA 用户态依赖及启动入口 | 构建后以不可变 digest 记录；依赖或代码改变时构建新版本 |
| 每系列容器 | 系列服务名、当前模型配置、端口与运行参数 | Pi 系列共用一个服务；模型通过配置/checkpoint 选择，默认一次加载一个 |
| 模型与运行数据 | 原始 checkpoint/norm/tokenizer、golden、日志和编译缓存 | 模型与 golden 只读挂载；日志和缓存独立持久化，重建容器不覆盖模型 |

具体实施规则：

1. 在系列镜像构建阶段通过项目约定的 conda/pip 安装固定依赖，记录基础镜像 digest、依赖清单和代码 commit；不使用 `latest` 作为生产版本，不在运行中容器临时升级依赖来形成不可复现的环境。同系列模型共用兼容的运行栈，只有依赖/代码变化才构建新镜像版本；Torch/TRT 实验作为 Pi 系列的新候选镜像临时验证，不按每个 checkpoint 新增常驻容器。
2. checkpoint 整目录以只读方式挂载，保持 `params/` 与 `assets/yam/norm_stats.json` 的绑定；tokenizer 和 golden 也只读。镜像内包含固定版本代码，生产不依赖可变的宿主源码挂载。编译/下载缓存按模型与运行环境版本隔离并显式指定可写位置；日志按模型与运行批次保存。
3. 用 Compose 为每个系列声明一个服务，显式记录 GPU 访问、挂载、环境变量、端口、启动命令和健康检查。Pi 系列模型切换通过更改配置/checkpoint 并重启该系列服务完成；随后重新预热、验证输出合同与精度，不假定已经实现动态热切换。不同系列同时运行必须另测统一内存占用与延迟；容器不是 GPU 资源完全隔离的虚拟机。
4. 首版使用 bridge 网络，将 policy 端口明确发布到 Thor 的直连网卡 IP；容器内服务监听 `0.0.0.0`，3588 连接 Thor IP，而非容器内部地址。IP/实际端口联调时确定；多候选容器使用不同测试端口。仅通过网线接收 observation，不给模型容器配置相机或机械臂设备透传。
5. 首次启动必须完成 GPU 实际运算、checkpoint/norm 校验、JAX 编译预热和固定样本推理，之后才报告模型就绪；健康检查不能只看容器进程或 TCP 端口。后续轻量健康检查读取就绪状态，不反复触发大模型冷编译。
6. 发布单元同时绑定镜像 digest、模型摘要、配置、Compose 参数和精度报告。先在独立候选容器完成 golden/延迟/网络验收，切换时由控制侧负责人配合暂停请求，再把生产端口交给通过的容器。保留前一版本镜像、模型及配置以回退，不删除已有模型资产。
7. 容器仍共享宿主内核、GPU 驱动与硬件。更换镜像不能绕过 Thor/CUDA/JAX 兼容检查，也不自动保持数值精度；镜像、依赖、驱动或编译设置变动后重新跑第 6 节验收。

实施交付物计划为系列 Dockerfile、版本锁定依赖清单、每系列 Compose 服务、模型选择配置及镜像/模型 manifest；待 Thor 环境核验后实现。已核实的官方示例镜像见第 3.2 节，生产镜像仍须实测锁定。依据：[NVIDIA Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)、[Compose GPU 支持](https://docs.docker.com/compose/how-tos/gpu-support/)、[Docker 端口发布](https://docs.docker.com/engine/network/port-publishing/)（2026-09-05 核对）。

### 3.2 官方容器的具体口径

2026-09-05 重新核对官方教程与实际 Dockerfile，需区分“官方基础镜像”“按教程本地构建的镜像”以及“本项目原生 JAX 候选”。

| 官方入口 | 镜像 | 含义 |
|---|---|---|
| Jetson AI Lab Pi0.5-on-Thor 教程 | `nvcr.io/nvidia/pytorch:26.05-py3` | 当前专门针对 Pi0.5 的官方教程基镜像；实际 `thor.Dockerfile` 的 `ARG BASE_IMAGE` 与页面一致 |
| 同一教程的构建命令 | `openpi-pi0.5:l4t-jp7.2` | 用户执行 Dockerfile 后生成的本地镜像标签，不是可以直接从 NGC 拉取的成品 Pi 模型镜像 |
| Thor 通用 Docker Setup | `nvcr.io/nvidia/pytorch:25.08-py3` | 通用 GPU 容器示例，不是当前 Pi0.5 教程的版本选择 |

来源：[Pi0.5 教程 Step 3](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/)、[实际 thor.Dockerfile](https://www.jetson-ai-lab.com/code-samples/openpi_on_thor/deployment_scripts/thor.Dockerfile)、[Thor Docker Setup](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)。教程基线是 JP7.2，项目系统计划是 JP7.2.1，仍需在目标系统验证。

官方 Pi0.5 教程验证的路线是 JAX 权重转换为 Torch，再走 TensorRT；镜像含有 JAX 依赖不代表已验证原生 JAX GPU 推理。因此本项目继续按 Pi 系列容器内原生 JAX 路线做精度优先验收，不因基镜像名称或教程存在而自动转格式。

NVIDIA 也有独立 [JAX 26.05 容器发布说明](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/rel-26-05.html)，列出 CUDA 13.2.1、JAX 0.10.0；该页未给出本项目 Thor/OpenPI/LoRA 的完整验收。实际选用的 ARM64 镜像 manifest、GPU kernels、Flax/Orbax 兼容性和镜像 digest 均须核验后确定；目前只能确认官方 Pi 教程的基镜像，不能宣称原生 JAX 生产镜像已选定并通过。

## 4. JAX 训练产物的精度深度调研

调研日期：2026-09-05。对象是本项目部署负责人，问题是 JAX 训练的 Pi0.5 LoRA 能否在 Thor 保持行为。结论依据官方文档、上游 issue/PR、作者实验记录和本仓库静态代码审计；本次没有运行 YAM checkpoint 的跨框架实验。以下区分已确认机制、社区测量和待验候选，不把社区阈值作为本项目验收标准。

### 4.1 先分清格式、存储精度与计算精度

JAX/PyTorch 是实现框架，Orbax/SafeTensors 是存储格式；BF16、FP32、FP8 等是数值类型。换容器格式本身不要求量化，但转换脚本可以同时改变权重数值、模型结构、预处理和算子行为。JAX checkpoint 也不能仅凭文件格式断定所有参数均为 FP32，必须读取各 tensor 的 dtype。

本仓库在调研基线 `09ce14f` 下的关键发现：

| 环节 | 静态代码证据 | 对本项目的影响 |
|---|---|---|
| 转换时提前降精度 | [转换脚本](../adapters/openpi/convert_jax_model_to_pytorch.py) 先以 FP32 restore，再用原 `model_config` 构造模型、加载权重，最后才 `.to(float32/bfloat16)`；[配置](../src/openpi/models/pi0_config.py) 默认 BF16 | 仅加 `--precision float32` 不能保证无损：FP32 值可能已在加载到 BF16 参数时舍入，之后升回 FP32 无法恢复 |
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

### 4.5 Thor 原生 JAX：安装前判断与实机验证边界

以下保留安装前判断供追溯；2026-09-07 原生 JAX 完整模型三精度回放已完成，最新结果以第 7 节为准。原生回放通过不代表与服务器训练版 JAX 完全等价，也不代表 YAM 实机任务验收通过。

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

### 2026-09-07 用户反馈：转向低延迟后端（最新决策）

用户明确不能接受当前原生 JAX 最快约 177 ms，要求完整推理约 100 ms 或更低，同时保证模型精度；下方 A/B/C 结果继续保留，但 C 不再作为可接受的最终部署选择。三组均已使用 MAXN + CPU/GPU/EMC 最高频率，不把重新开 MAXN 当作下一轮加速。下一轮同时记录完整 policy 的 P50/P95，优先无量化编译及保留高精度敏感运算；当前不以改小三相机、H50、10 去噪步来偷换对照合同。

执行优先级改为：复用现有 9 输入与已保存 JAX 输出 → FP32 权重转换核对 / PyTorch 编译 → 保留 FP32 敏感运算的 strongly typed TensorRT；并行核对 FlashRT 的 YAM/H50 适配。先筛出显著提速且误差受控的候选，再做 300 状态 × 3 种子扩样，不再优先重复大量慢速 JAX。准确性至少分为映射后权重、完整动作输出、微调后任务三个层次，不以单个总体余弦值替代逐维误差。

[FlashRT Thor 报告](https://github.com/flashrt-project/FlashRT/blob/main/examples/thor/README.md) 公布三视角 FP8 约 54.8 ms；[NVFP4 报告](https://github.com/flashrt-project/FlashRT/blob/main/docs/pi05_thor_decoder_fp4_e2e.md) 公布三视角约 31.74 ms，但对照为 FP8，不能直接证明原始 JAX 精度。[前端代码](https://github.com/flashrt-project/FlashRT/blob/main/flash_rt/frontends/torch/pi05_thor.py) 的 action 序列 `Sa` 固定为 10，另有 10 次去噪；先解决本项目 H50/14D/状态提示合同，才能做公平比较。查阅于 2026-09-07，以上均为社区结果而非本机新实测。详细候选顺序见 [下一轮方案](reports/thor/next_test_plan.json)。

代码同步改为工作站提交 → Gitea → Thor 快进拉取，不设后台双向覆盖。Thor 已生成 Gitea 专用 Ed25519 密钥、校验并固定 Gitea 主机公钥、配置 `origin` 和 `pull.ff=only`；首次访问仍因公钥未授权失败，等待用户添加仓库只读部署密钥。Thor 原 rsync 副本已初始化空 Git 元数据，但还没有首次 checkout/HEAD；原代码文件未覆盖，不能误记为已经同步完成。入口及后续接入见 [Gitea 同步](reference/thor/09_gitea_code_sync.md)。

### 2026-09-07 18:52 完整 Pi0.5 三精度基线回放

**当前固定可运行路径**：Thor JetPack 7.2.1 / L4T 39.2.1 → Pi 系列 Docker → NVIDIA JAX 26.05 ARM64 → 只读原始 `pi05_base` JAX checkpoint → 本地三路 RGB、14D state、prompt → YAM `(50,14)` 动作。只操作 Thor；这次容器使用 `--network none`，没有服务器实时数据流或机械臂执行。

三组均已真实完成：同一 3 条 Lego sorting 录像的早/中/晚共 9 个输入，每输入 2 次预热、20 次正式推理，种子 0、同一 `(50,32)` 噪声、去噪 10 步、horizon 50。实际加载参数叶子为 A/B 各 51 个 FP32，C 为 51 个 BF16；原始磁盘权重保持不变。所有正式输出有限、形状正确，固定输入与噪声下重复差异为 0。

| 模式 | 权重加载 / 计算 | P50 / P95（ms） | 相对 A 原单位 MAE / 最大误差 |
| --- | --- | --- | --- |
| A 数值参考 | 保留 FP32 / FP32，matmul highest | 1231.477 / 1232.370 | 0 / 0（自参考） |
| B 计算精度对照 | 保留 FP32 / BF16 | 239.761 / 240.805 | 0.002699 / 0.022904 |
| C 性能候选 | BF16 加载 / BF16 | 177.393 / 178.237 | 0.002170 / 0.018646 |

逐维、逐样本、归一化误差、首次编译、温度和内存详见 [中文 HTML](reports/thor/index.html) 与 [机器可读结果](reports/thor/status.json)。C 比 A 约快 6.94 倍、比 B 约快 1.35 倍；本批平均/最大误差略小于 B **不能推导出 BF16 参数普遍更准确**，各样本/维度并不单调。保留 A 参考，C 作为下一轮扩样的首选，B 用于分离计算与参数精度影响。

运行时 GPU 温度采样最高 A/B/C 分别约 54.9/55.8/57.8°C；GPU 实际频率 1572–1575 MHz。各次结束恢复 120W、动态频率、自动风扇，无模型容器常驻。MAXN 控制入口仍由下方约定持有，不设开机锁频。

**环境适配已验证**：JAX `0.10.0.dev20260415+c98e1bb97`、jaxlib `0.10.0.dev20260521`、Flax `0.12.6`、Orbax `0.11.39`。保留 NVIDIA 栈，仅将 PyTorch/FAST 相关导入按需加载，避免 JAX 推理强制拉入 LeRobot/Transformers 训练依赖；`restore_params` 兼容新 Orbax `StepMetadata.item_metadata`，不改变恢复 dtype/shape。容器中的 CPU Torch 2.7.1 用于现有 IO/类型依赖，**不是已适配的 GPU PyTorch 推理后端**。直接依赖已锁定本次实测版本。

实测镜像 ID 为 `sha256:876e8ae0cfb81c2b8f93735e44147ce14048898abaf4a503fd7324e2a5b9bfde`，保留标签 `openpi-pi:thor-jax-20260907`。这是 Pi 系列的一个运行环境版本，不是给每个 checkpoint 新开一套服务。重新构建会产生新的镜像 ID，不能沿用旧 ID 宣称已经测试。

**结果边界**：当前模型没有 YAM 微调、没有 LoRA；本次 norm 是从本地同一组录像计算的 benchmark-only 统计，关节未来 50 步均相对当前 state、夹爪 absolute、末尾按轨迹内末帧补齐。它不是未来训练 norm，物理单位仍按原数据保留，不宣称弧度或毫米。没有实机闭环/成功率、生产动作容差、服务器 JAX 等价或 Thor↔3588 联调验收。

**证据位置与重跑**：Thor `/home/wuyan-lyj/thor/pi/results/` 下 `pi05-A-20260907-r2`、`pi05-B-20260907-r1`、`pi05-C-20260907-r1`；原始 stdout/1 秒 tegrastats 在同级 `logs/`；工作站完整副本为 `/home/wuyan-lyj/thor-system/test-data/results/`。失败的 A/r1 Orbax 读取日志保留。操作步骤见 [真实录像精度回放](reference/thor/08_recorded_precision_replay.md)。

**下一轮方案**：先扩为每轨迹 100 状态 × 3 固定噪声种子，保留 10 去噪步与 horizon 50；然后单独验证 JAX→PyTorch FP32 转换的每个张量与前向输出，未来 LoRA 必须正确保留/合并后再转换。TensorRT 借鉴保留 FP32 敏感运算、strongly typed 的混合 FP16 路线，不全局 `.half()`；FP8、attention-side NVFP4 排在基线转换之后。详细顺序与理由见 [测试方案](reports/thor/next_test_plan.json)，不在这一轮自动开始转换或量化。

社区依据：[Jetson AI Lab 教程](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) 是 PyTorch 26.05 → ONNX → TensorRT 路线，其 LIBERO/horizon 10 数据不能充当本项目 YAM/horizon 50 实测；[xuweiwu/openpi-thor](https://github.com/xuweiwu/openpi-thor) 的定制 FP16 保留 FP32 敏感运算并使用 strongly typed，不能理解为任意纯 FP16 转换均可用。查阅日期 2026-09-07；作者验证条件与阈值不直接替代本项目验收。

### 2026-09-07 较早实机状态（安装与管理事实保留；推理进度以最新小节为准）

以下为当日 SSH、系统查询和传输实测状态，不是模型部署通过结论；临时地址、进程与下载状态使用前复核。

- **系统已安装并启动**：原厂 Jetson AGX Thor 开发套件，NVMe 根分区 `/dev/nvme0n1p1`；JetPack 7.2.1 / L4T 39.2.1、Ubuntu 24.04.4、内核 `6.8.12-1021-tegra`。`nvidia-smi` 可识别 NVIDIA Thor，驱动 `595.78`；这不等同于 JAX 内核或模型推理验证。
- **管理网络已可用**：用户名 `wuyan-lyj`，主机名 `thor`。USB 管理地址 `192.168.55.1`，工作站别名 `thor-usb`；Wi-Fi 当前 DHCP 地址 `192.168.110.108/23`，别名 `thor`。`thor-admin-wifi` 为系统连接，autoconnect=yes、重试次数 0（无限），NetworkManager 开机启用；已验证连接与配置，尚未重启实测回连。NTP 已同步。
- **SSH 已配置**：`ssh.socket` 开机启用，服务 active，工作站密钥经 USB/Wi-Fi 登录成功；禁用 SSH 密码登录及 root 登录。用户的本地登录/sudo 密码不写入文档。USB/Wi-Fi 仅作管理，生产 Thor↔3588 仍为网线直连；不操作 3588。
- **显示问题独立保留**：HDMI 状态为 disconnected、EDID 为空，未定位为硬件或驱动问题。系统安装成功不依赖 HDMI 修复；为支持 NoMachine 无屏桌面，已于当日关闭 GDM 本地登录服务，改用下面的远程显示。
- **NoMachine 无屏管理**：用户确认个人非商业用途，Thor 安装官方 `nomachine_9.8.3_1_arm64.deb`。`nxserver.service` enabled/active，启动模式 Automatic；`CreateDisplay 1`、`DisplayOwner "wuyan-lyj"`、`DisplayGeometry 1920x1080`。已核对 GNOME 会话运行，`:1001` 的 `nxoutput0` 实际为 1920×1080，USB/Wi-Fi 的 TCP 4000 均可达；客户端实际看到画面及重启后回连仍待用户确认。USB 地址 `192.168.55.1:4000`，Wi-Fi 当前 `192.168.110.108:4000`，使用系统账户登录，不把密码写入连接文件。工作站现有 Personal Edition 10.0.60 的 Player 可用作客户端；它的服务端订阅不是连接 Thor v9 的前提，不代表本机服务端已激活。
- **无屏配置恢复**：原配置备份 `/usr/NX/etc/server.cfg.before-thor-headless-20260907`。HDMI 恢复且希望切回物理桌面时，先断开 NoMachine，再恢复备份至 `/usr/NX/etc/server.cfg`、执行 `sudo systemctl enable --now gdm3`，最后 `sudo /usr/NX/bin/nxserver --restart`；不要在远程会话中无提示重启桌面。仅为远程桌面而关闭 GDM，不改 GPU 驱动和推理容器。NoMachine 10 服务端已改为订阅/试用许可，不能把旧版免费说明用于 v10，参见 [官方许可说明](https://kb.nomachine.com/AR03P00972)。
- **容器基础已安装**：Docker `29.1.3`（Ubuntu `docker.io` 包）、NVIDIA Container Toolkit `1.19.1`、Docker Compose `2.40.3`；用户已加入 docker 组，Compose 配置解析通过。Pi 系列只维护一个系列环境，不按 checkpoint 新建独立依赖栈。
- **JAX 镜像已导入 Thor 并通过 GPU 内核检查**：`nvcr.io/nvidia/jax:26.05-py3` ARM64，manifest digest `sha256:3f009a485f5ba64c177f2f8f7999adc900fcf14dce3f11b826f412ea19d23553`，77 层共 9,010,137,575 字节。工作站 `/home/wuyan-lyj/thor-system/images/jax-26.05-arm64` 下载完成后导出 gzip 归档，经 USB 传输 9,265,433,320 字节、约 29.47 MB/s，已完成 docker load。
- **实测 GPU 范围**：当日 18:15 使用 `maxn_session.py` 临时 MAXN，在禁网容器中执行 `scripts/docker/thor/jax_smoke.py`；FP32/BF16 的 JIT 矩阵乘法精确值检查、随机生成和非线性有限值检查全部通过，结束后实际自动恢复 120W/动态时钟。原始日志 `/home/wuyan-lyj/thor/pi/logs/jax-kernel-smoke-maxn-20260907.log`，工作站镜像目录留有副本。启动有 CUDA 驱动版本字符串格式告警但运算成功，不能把此小矩阵测试当作完整 Pi0.5 延迟/精度验收。
- **容器内版本实读**：Python 3.12.3、JAX `0.10.0.dev20260415+c98e1bb97`、jaxlib `0.10.0.dev20260521`、Flax 0.12.6、Orbax 0.11.39、NumPy 2.4.4，未装 torch。**OpenPI 依赖适配/完整 checkpoint 加载仍未完成**；不使用仓库训练 CUDA 12/JAX 0.5.3 锁定项直接覆盖 NVIDIA 容器栈。
- **模型资产已在 Thor**：`/home/wuyan-lyj/thor/pi/checkpoints/pi05_base`（12,441,749,581 字节），分词器 `/home/wuyan-lyj/thor/pi/cache/big_vision/paligemma_tokenizer.model`。Orbax 元数据实读 51 个参数叶子均 float32；原始磁盘权重未改写，没有已微调 YAM checkpoint。
- **测试输入已本地化**：`/home/wuyan-lyj/thor/pi/test-data/ABC-130k-two-tasks/lego_sorting` 保存 train 轨迹 95、96、97，共 7,962 帧观测、9 个视频文件；连同 parquet/原始 manifest 共 60,021,788 字节。11 文件在服务器、工作站、Thor 的 SHA-256 一致，本机完整解码与状态/动作结构审计通过。另保留转换小样本于 `test-data/lego_sorting/smoke-train-20260907`。原始 manifest 覆盖更多轨迹，不代表整库已下载；耳机任务尚未找到三路齐全轨迹。所有回放从 Thor 磁盘读取，不从服务器实时取帧。
- **精度对照仍待实测**：A 原始 FP32 参数 / FP32 计算 / highest；B 原始 FP32 参数 / BF16 计算；C 读取为 BF16 / BF16 计算。固定输入、噪声、10 次去噪与 50 步 horizon，分开首次编译和预热后完整调用计时。尚需生成固定输入和与 YAM delta 合同匹配、明确仅用于 benchmark 的 norm_stats；不套用 DROID 统计，不宣称基础模型具备 YAM 任务成功率。
- **工具与结果入口**：`scripts/thor/benchmark_pi05.py`、`scripts/thor/render_report.py` 和 `docs/reports/thor/index.html`；11 项相关单元测试通过，不代表模型/GPU 验收。转换到 PyTorch 后的键/形状/dtype/LoRA 覆盖、逐层和完整动作对照另行验收，之后再分组比较 BF16/FP16、TensorRT 与量化，不同时改变格式和精度后直接接受结果。

### MAXN 性能测试约定（2026-09-07 用户指定）

- 用户进一步明确：**只在推理/测试时启用 MAXN**；安装、下载、远程桌面和待机时使用日常 120W、动态调频及自动风扇，不新增 MAXN/锁频自启动服务。性能测试使用本机官方 `MAXN` / mode 0，不修改官方频率表进行额外超频。
- 使用 `sudo python3 scripts/thor/maxn_session.py -- <前台推理命令及参数>`：从 120W 保存当前时钟配置，临时切到 MAXN 并锁最高频率，前台命令结束后自动恢复已保存时钟与 120W；普通异常和 SIGINT/SIGTERM/SIGHUP 同样走恢复流程。禁止传入脱离前台的后台启动命令。断电、SIGKILL 或恢复命令本身失败不能保证清理，应按保留的时钟备份手动恢复并复核；不宣称全故障覆盖。
- 当日已实机生效、无需重启：CPU 14 核在线且 min=max=current=2,601,000 kHz；GPU min=max=current=1,575,000,000 Hz；EMC min=max=current=4,266,000,000 Hz。`nvidia-smi` 亦实读 GPU 1575 MHz。模式名与频率是运行时观察，重启或其他工具改变配置后重新检查。
- 曾尝试满速风扇：`jetson_clocks --fan` 在此 BSP 对 `pwm1_enable` 写入 255 报告 Invalid argument；当时实际 `pwm1=255`、`pwm1_enable=1`。已恢复自动风扇（nvfancontrol active，PWM 实读回到 78）；新启动器不使用有该告警的 `--fan`，保留自动散热并采集温度/频率，遇到降频不能宣称持续最高性能，不关闭硬件保护。
- 首次切换前备份保留于 `/home/wuyan-lyj/thor/pi/logs/clocks-before-maxn-20260907.conf`；已执行恢复并确认 120W、CPU/GPU min/max 不再锁为同值。每次临时启动器另保存独立 `/tmp/thor-maxn-*/clocks.conf`。手动恢复先 `sudo jetson_clocks --restore <本次备份>`，再 `sudo nvpmodel -m 1`，复核自动风扇恢复。
- MAXN 模式可跨重启保留，但不假设 `jetson_clocks` 的静态锁频也跨重启；每次推理计时前必须重新核对。报告绑定功耗模式、CPU/GPU/EMC 频率、温度和降频情况；桌面远程连接/后台负载也记录，避免混入精度差异。
- 官方明确 MAXN 仍会受到供电/热限制，不能保证每种负载都比 120W 快；本项目按用户指定 MAXN 实测，不把模式名本身当作性能结论。[对应 r39.2.1 官方说明](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonThor.html#supported-modes-and-power-efficiency)

### 早期安装记录（历史状态，不能作为当前未完成项）

- 2026-09-07：已编写并核对 [Thor 安装冷手册](reference/thor/00_start_here.md)，未操作 Thor/3588 或烧录磁盘。本地 ISO 重新检查类型、字节数与 SHA-256，结果与第 2 节一致；尚未取得发布者 ISO 校验和/签名匹配证据。系统/容器安装步骤为官方资料核对状态，不能宣称用户设备已实测通过；原生 JAX 镜像仍有依赖兼容与模型验收阻断项，见手册 G5。
- 制盘软件：官方 Etcher 2.1.6 amd64 Debian 包已下载并与 Release 资产摘要匹配，尚未安装；路径与身份见第 3 节。用户采用六步简洁指导主线，扩展诊断按异常展开，不重复要求正常步骤截图/填表。

- 官方 JetPack 7.2.1 ISO：2026-09-05 已下载，长度/类型已核对且本地 SHA-256 已记录，详见第 2 节；USB 制作与刷写未完成。
- 按模型系列隔离 Docker 容器方案：已确定，Pi 系列共用一个服务；首版候选为容器内原生 JAX。官方 Pi 教程基镜像已核实为 `nvcr.io/nvidia/pytorch:26.05-py3`，但项目 JAX 镜像选型、Dockerfile、Compose、构建与容器内验收仍待实施。
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

### 2026-09-07 加速迭代记录入口

本日原生 JAX A/B/C 之后已实际开展 FP32 跨框架对照、PyTorch 编译、三相机合批与 CUDA Graph 组合测试。最新配置、失败原因、动作差异和时延由 [Thor 系列 10](reference/thor/10_acceleration_execution.md) 持有，[中文 HTML](reports/thor/index.html) 展示实际结果。早先 A/B/C 的阶段状态不再代表只完成这三组；不得把最新延迟领先者自动视为精度通过或已达约 100 ms 目标。
