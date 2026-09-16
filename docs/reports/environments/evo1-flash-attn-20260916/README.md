# Evo-1 FlashAttention 环境修复与 GPU 复测

观察日期：2026-09-16（Asia/Shanghai）
目标环境：服务器 `/home/wuyan/.conda/envs/vla-evo1-train`
固定 LeRobot：`2774d9bddcbbda50e697e162e89e7eaada8d7105`

## 结论

Evo-1 环境原先找不到 `flash_attn`，固定源码因此会选择普通 `eager` 注意力。本次已在该独立环境安装 `flash_attn==2.8.3`，并在 GPU 节点 `gpu001.hlink.local` 的 RTX 4090 上完成导入、Evo dispatch 和 FlashAttention 前向/反向 smoke：检测结果为 `flash_attention_2`，不是回退路径。`pip check` 通过。

这只解决并验证了注意力依赖和 CUDA kernel；没有加载完整 Evo 权重、没有跑 YAM batch、没有启动正式训练，也没有修改 Thor 或 3588。正式训练前仍需做完整模型前向/反向、保存重载和真实数据 smoke。

## 下载与安装的内容

用户要求优先镜像，因此先检查了 PyPI 镜像。阿里云镜像提供的是 `flash_attn-2.8.3.tar.gz` 源码包，已下载并校验但没有安装：

```text
URL: https://mirrors.aliyun.com/pypi/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
SHA256: 1e71dd64a9e0280e0447b8a0c2541bad4bf6ac65bdeaa2f90e51a9e57de0370d
Size: 8,447,812 bytes
Remote receipt: /home/wuyan/lyj/evo1-install-2774d9b/flash-attn-mirror-20260916/flash_attn-2.8.3.tar.gz
```

由于服务器系统 CUDA 是 13.2，而环境中的 PyTorch 是 `2.10.0+cu128`，直接用系统 `nvcc` 构建会遇到 CUDA 主版本不匹配风险。PyPI 镜像也没有提供本组合的预编译 wheel，因此没有把源码编译结果混入正式环境。

最终安装的是与本环境精确匹配的 Linux x86_64/Python 3.12/CUDA 12/Torch 2.10/CXX11 ABI wheel。它由上游 FlashAttention 仓库 issue 中记录的社区构建提供；下载时使用 `gh-proxy.com` 作为传输镜像，下载完成后按 SHA256 校验。该传输站和 wheel 均不是 Dao-AILab 官方发布物，故保留完整文件和哈希作为交接凭据。FlashAttention 官方安装前置条件和支持的 Ampere/Ada/Hopper GPU 见[官方 README](https://github.com/Dao-AILab/flash-attention)，PyTorch 2.10/Python 3.12 对应 wheel 信息见[上游 issue #2299](https://github.com/Dao-AILab/flash-attention/issues/2299)。

```text
Wheel: flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
SHA256: d4a497a7bd837bf47f7a8f6a7aa6887695f2ea819fa597f307552552018ee9d7
Size: 253,640,641 bytes
Remote receipt: /home/wuyan/lyj/evo1-install-2774d9b/flash-attn-wheels-20260916/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
Install mode: pip --no-deps --no-index --force-reinstall
```

仅用于探测构建可行性的 `ninja` 和 `nvidia-cuda-nvcc-cu12` 已从 Evo 环境移除；它们没有作为运行时依赖留下。第一次直连 GitHub 下载留下的 22,962,176-byte 未完成文件也被改名保留，没有覆盖完整 wheel。

复现安装：

```bash
P=/home/wuyan/.conda/envs/vla-evo1-train/bin/python
export LD_LIBRARY_PATH=/home/wuyan/.conda/envs/vla-evo1-train/lib
"$P" -m pip install --no-deps --no-index --force-reinstall \
  /home/wuyan/lyj/evo1-install-2774d9b/flash-attn-wheels-20260916/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
"$P" -m pip check
```

## GPU 复测

Slurm GPU smoke 使用 `gpu001.hlink.local` 的 NVIDIA GeForce RTX 4090（compute capability 8.9），环境报告为 Torch `2.10.0+cu128`、Torch CUDA `12.8`、CXX11 ABI `TRUE`。最终环境复测结果：

- `transformers.utils.is_flash_attn_2_available()`：`true`
- Evo embedder 的同名检测：`true`
- Evo 选择：`flash_attention_2`
- `flash_attn` 前向结果有限；前向+反向结果和梯度有限
- `pip check`：`No broken requirements found`

在 BF16、`(B,S,heads,head_dim)=(1,2048,16,128)` 的 attention kernel 微基准中，FlashAttention 前向 P50 为 `0.263 ms`，强制 math-only SDPA 前向 P50 为 `3.733 ms`，约 `14.18x` 差异。每组首个样本未丢弃，因此 P95 含有 CUDA warm-up 离群值；这不是完整 Evo 模型或训练吞吐结论。

完整结构化结果见 [`gpu-smoke.json`](gpu-smoke.json)。服务器原始 Slurm 日志保留在：

```text
/home/wuyan/lyj/evo1-install-2774d9b/flash-attn-gpu-final-20260916.log
/home/wuyan/lyj/evo1-install-2774d9b/flash-attn-gpu-math-20260916.log
```

登录节点没有 GPU 可见时，单独执行 `is_flash_attn_2_available()` 可能返回 `false`；这属于探测上下文限制，不能替代 GPU 节点复测。本次正式判断以 Slurm GPU smoke 为准。
