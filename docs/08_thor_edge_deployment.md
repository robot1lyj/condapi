# 08 · Thor 端侧部署

本页是 NVIDIA Jetson AGX Thor 端侧系统、Pi0.5 转换/加速和本地推理验收的唯一 owner。训练仍在服务器 GPU 上进行；YAM 机械臂驱动、CAN、GUI、home pose、限位和急停不属于本仓库。

## 1. 当前决策

- 默认推理目标是 Jetson AGX Thor Developer Kit（T5000 口径）；设备到手后仍需用 `jetson_release`/`cat /etc/nv_tegra_release` 核对实际 SKU。若实际是 T4000 或定制载板，不能直接套用开发套件 ISO。
- 官方系统基线选 JetPack 7.2.1 / Jetson Linux r39.2.1。系统盘制作介质是 Jetson ISO USB 安装盘，实际 BSP 安装到 Thor 的 NVMe，不把 ISO 当作 Live USB。
- Pi0.5 的权威模型仍是原始 JAX/Flax checkpoint；Thor 的首个生产候选是转换后的 PyTorch SafeTensors，再导出 TensorRT。JAX 只作为参考实现、转换源和数值 golden baseline，不作为首次端侧默认 runtime。
- 首次加速顺序固定为 BF16 PyTorch smoke → TensorRT FP8 → 通过真实 YAM 样本的误差 gate 后再试 NVFP4。Jetson AI Lab 的纯 FP16 导出警告必须保留；社区 `openpi-thor` 的 strongly-typed FP16 是独立扩展，不能与官方 recipe 混为一谈。
- 远程 WebSocket 只保留为调试、异机控制器兼容和回退路径；若采集、预处理和控制进程都在 Thor，默认走 Thor 本地进程/容器，避免把网络往返算进闭环。

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

本次下载的官方 ISO：

```text
URL:  https://developer.nvidia.com/downloads/embedded/l4t/r39_release_v2.1/iso/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso
PATH: /home/wuyan-lyj/thor-system/jetpack-7.2.1/jetsoninstaller-r39.2.1-2026-08-07-18-30-47-arm64.iso
```

ISO 下载完成后必须执行 `sha256sum` 并把结果写入本页和 `docs/07_change_log.md`；未完成校验前不得制作系统盘。ISO 是约 4.8 GB 的安装介质，制作 USB 需要至少 16 GB U 盘和至少 25 GB 的下载/制作空间。

## 3. 从零安装顺序

1. 在任意 Windows、macOS 或 Linux 主机下载 ISO，用 Balena Etcher 或同类工具写入至少 16 GB U 盘；不要把 ISO 文件简单复制到 U 盘。
2. Thor 插入 U 盘并上电，按官方 Quick Start 进入安装器；若提示 QSPI capsule update，确认 `Y`，否则新 ISO 与出厂 UEFI 可能不兼容。
3. 安装目标选择 `Install on NVMe`。安装结束后拔出 U 盘，再完成 `oem-config` 的用户、网络和时区设置。官方 USB 介质只用于安装 BSP，不能作为完整系统试运行盘。
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
6. 原生开发不是首选路径；若确实需要，在 JetPack APT 源中使用 `sudo apt install nvidia-jetpack` 或 `nvidia-cuda-dev`，不要安装 Ubuntu 的 `nvidia-cuda-toolkit`。Pi0.5 首次部署优先使用匹配 Thor 的 NVIDIA PyTorch 容器，减少 Python/CUDA/JAX 依赖污染。

## 4. Pi0.5 Thor 调研结论

| 案例 | 模型/源权重 | Thor runtime | 结论 |
|---|---|---|---|
| [Jetson AI Lab OpenPi π0.5](https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/) | `pi05_libero`；先下载原始 JAX checkpoint | JAX → PyTorch → ONNX（FP8/NVFP4）→ TensorRT | 目前最完整、最适合做第一版基线的官方流程；JP 7.2、horizon 10，PyTorch BF16 约 132 ms，TensorRT FP8 约 54 ms，FP8+NVFP4 约 49 ms。 |
| [openpi-thor](https://github.com/xuweiwu/openpi-thor) | 以 JAX checkpoint 作 reference，可处理训练配置 | PyTorch bundle/TensorRT，并对比 JAX | 对自定义 fine-tune 配置更有参考价值；要求先做 JAX→Torch/engine 的真实样本验证，不能只看 engine 能否启动。 |
| [FlashRT](https://github.com/flashrt-project/FlashRT) | 支持 Pi0.5 的 Torch 和 JAX frontend | Thor 专用 CUDA kernels、FP8/NVFP4、CUDA Graph | 社区高性能路线；公开 Thor Pi0.5 约 44 ms（FP8）及 23/27/31 ms（NVFP4+FA4，1/2/3 view），但需要额外适配 YAM 14D/三路图像/50 horizon，暂不作为本仓库首个部署依赖。 |
| [OpenPI issue #826](https://github.com/Physical-Intelligence/openpi/issues/826) 与 [NVIDIA forum](https://forums.developer.nvidia.com/t/how-to-run-a-pi0-or-pi0-5-model-on-agx-thor-device/362389) | Thor 上的 PyTorch/JAX 兼容性讨论 | CUDA 13、Python 3.12、SM110 | PyTorch 端已有可运行报告；直接构建 `jax==0.5.3` 在 Thor 上遇到 SM110/依赖问题，说明 JAX 直接运行不应作为第一条落地路径。 |

因此，对“Pi0.5 到底用 JAX 还是 PyTorch”的回答是：用 JAX 原版 checkpoint 作为源和对照，用 PyTorch 作为 Thor 上的可验证中间运行层，最终优先 TensorRT engine；不是把一份未经转换的 JAX 服务直接搬到 Thor。FlashRT 虽同时支持 Torch/JAX，但它是后续性能分支。

社区数字只用于判断路线，不用于宣称 YAM 性能。官方案例是 `pi05_libero`、7D action、horizon 10；本项目是 `pi05_yam_lora`、三路图像、真实 14D 输出、模型内部 horizon 50，必须重新导出、校准、比较和实测。

## 5. YAM 端侧落地计划

部署 bundle 必须绑定同一个 `pi05_yam_lora` 配置、训练 commit、checkpoint、`assets/yam/norm_stats.json` 和 YAM 数据版本：

1. 在服务器保存 JAX policy 的 golden 输出：三路实际图像、14D state、prompt，确认输出有限且为 `(50,14)`。
2. 在 Thor 的 NVIDIA PyTorch 容器中，将 JAX checkpoint 转为 PyTorch SafeTensors；转换后保留 norm assets，并先做 PyTorch BF16 本地 smoke。
3. 按真实 YAM 三路输入和 `action_horizon=50` 导出 ONNX，先构建 TensorRT FP8；不要把官方 `pi05_libero` 的 horizon 10 engine 直接复用。
4. 用相同输入和固定噪声分别比较 JAX reference、PyTorch 和 TensorRT，记录 cosine、MAE、最大误差、有限值和 `(50,14)` shape；任何误差 gate 失败都不能进入真机。
5. FP8 通过后再单独试 NVFP4；每种 precision 使用独立 engine、报告和实验目录。若采用 FlashRT 或 `openpi-thor`，也必须通过同一 YAM golden gate。
6. 在 Thor 本地执行真实 camera/state → policy → `(50,14)` 的 smoke；只有控制器在另一台设备时才启用兼容 WebSocket，并同时保留本地路径。

## 6. 端侧验收闸门

- 系统为 JetPack 7.2.1 / L4T r39.2.1，GPU/容器可见，Docker runtime 正确。
- checkpoint、config、norm stats、转换工具版本和 engine manifest 可追溯。
- JAX reference、PyTorch、TensorRT/FlashRT 在同一批 YAM 样本上完成数值比较。
- 输入保持 `observation.state`、三路 `observation.images.*` 和 prompt；输出为有限 `(50,14)`，不得把内部 32D padding 发送给机器人。
- 记录 Thor 温度、功耗模式、warmup、端到端时延和 action horizon；不得把 10 步 LIBERO benchmark 当作 YAM 闭环频率。
- 端口监听或容器启动不算通过；必须有真实本地推理结果，远程 WebSocket 仅在启用时额外做协议 smoke。

## 7. 当前状态

- 官方 JetPack 7.2.1 ISO：已下载到 `/home/wuyan-lyj/thor-system/jetpack-7.2.1/`；SHA-256 校验和待下载完成后补录。
- Thor 实机刷写、Docker GPU smoke、YAM checkpoint 转换、TensorRT engine 和真实三路输入 smoke：计划中，尚未宣称已验证。
