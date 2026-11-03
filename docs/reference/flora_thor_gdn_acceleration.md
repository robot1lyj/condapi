# Flora / Thor 的 Gated DeltaNet 加速候选

日期：2026-11-03。本文保留官方源码与包制品研究，并记录同日 Thor 实测：`gdn-overlay-candidate-02` 已构建、导入和通过7组初步数值检查，原 Torch/Triton、BF16 主体及 GDN/LoRA 的 FP32 原值保持。依赖仅在本任务新 overlay 内，baseline 不变。详细运行状态归[执行回执](../reports/data-tshirt-fineart-design-20261103/execution.json)。

实施前结论：存在可直接探索的官方路径。FLA 0.6.0 提供纯 Python/Triton 内核；causal-conv1d 1.7.0 的源码已明确编译 `sm_110`，但其发布 CI 未覆盖当前 NGC Torch 2.12 alpha，需在原镜像内源码构建。安装成功不能代替 GPU 运算及 Flora 输出验收。

## 已核实的触发条件

执行方提供的当前环境为 Python 3.12、Transformers 5.8.1、PEFT 0.18.1、Torch `2.12.0a0+5aff3928d8.nv26.05`、Triton distribution `3.7.0+gitb4e20bb.nv26.5`（`triton.__version__` 实际为 `3.7.0`）、CUDA 13.2、ARM64/`sm110`；已有 `flash_attn`，缺少 `fla` 和 `causal_conv1d`。这些适用条件已由 Thor 运行时回执重新探测；最初只读研究没有进行该探测。NGC 26.05 官方说明与该 Torch commit、CUDA 13.2.1 和 Python 3.12 对应。[NVIDIA 26.05 发布说明](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-05.html)

Transformers 5.8.1 的 `Qwen3_5GatedDeltaNet` 直接导入以下符号，四个运算函数齐备时 `is_fast_path_available` 才为真：

| 模块 | 必需符号 | 用途 |
| --- | --- | --- |
| `causal_conv1d` | `causal_conv1d_fn` | prefill 的短因果卷积 |
| `causal_conv1d` | `causal_conv1d_update` | 缓存状态的单 token 卷积 |
| `fla.ops.gated_delta_rule` | `chunk_gated_delta_rule` | prefill / 多 token continuation |
| `fla.ops.gated_delta_rule` | `fused_recurrent_gated_delta_rule` | 单 token decode |
| `fla.modules` | `FusedRMSNormGated` | FLA 可见时选用的 fused gated norm |

`flash_attn` 不属于这个开关。现有 SDPA/FlashAttention 配置不能补上缺失的 GDN 函数。FLA 与 causal-conv1d 的选用分别生效；只安装一项可能出现部分加速，但整体 warning 仍在。FLA 的出现还会改变 gated norm 实现，因此后续比较须覆盖 norm，不能只测卷积。[固定版本的 Qwen3.5 源码](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L54)

可用性检查还要求 CUDA 可用，且 `fla` 版本至少 0.2.2。检查使用已安装 distribution metadata；必须在启动 Python **之前**设置 overlay 路径，不在已导入 Transformers 的 baseline 进程内追加路径。[5.8.1 import 检查](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/utils/import_utils.py#L810)

## 固定的候选制品

| 项目 | 固定版本 / release commit | 制品 SHA-256 |
| --- | --- | --- |
| `fla-core` | 0.6.0；FLA tag `v0.6.0` → `413ad2c655ece57b94bc9ddc51fdbd43cf09d984` | wheel `b0708b70e0b119cd58a0a5146ea56cb175cef6ed969d8ac60267e3e077ce3ebd` |
| `flash-linear-attention` | 0.6.0；同一 FLA release | wheel `6dec9735f8454523c13553622e98aff30152dde6586b4ac7d88010fbb7a06b5f` |
| `causal-conv1d` | 1.7.0；tag `v1.7.0` → `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` | sdist `3202758494eaa7b597ce1c282dfa188889506bfcb92cad3c407d26736bfcd32b` |

本次查询官方 PyPI 元数据，并实下载两个 FLA wheel 到内存核验 SHA 与 ZIP 布局：`fla-core` 约 1.09MB，包含 `fla/__init__.py`、`fla/modules` 和 `fla/ops`；`flash-linear-attention` 约 0.40MB，包含 `fla/layers`、`fla/models`，不含 `fla/__init__.py`。因此不能安装后者且配合 `--no-deps` 却遗漏 core。两个 wheel 应同一次 pip 事务进入全新的 target。只使用 HF 的上述 imports 时，core 已包含所需内核；完整组合更接近官方发行合同。[core 官方制品元数据](https://pypi.org/pypi/fla-core/0.6.0/json)、[完整包官方制品元数据](https://pypi.org/pypi/flash-linear-attention/0.6.0/json)、[conv 官方源码包元数据](https://pypi.org/pypi/causal-conv1d/1.7.0/json)

FLA core 的基础依赖为 `einops`；完整包另外要求 Transformers `>=5.6.2,<6`，当前 5.8.1 满足。Torch/Triton 在 backend extra 中，官方安装说明专门给了 nightly Torch/Triton 使用 `--no-deps` 的路线；本候选不使用 `[cuda]` 依赖解析。`importlib`、`importlib.metadata` 是 Python 标准库，无需安装同名第三方包。[FLA 固定 release 的安装说明](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/INSTALL.md)、[基础依赖](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/pyproject.toml)

causal-conv1d 1.7.0 支持 ARM64 平台名，在 CUDA `>=13.0` 的构建分支显式加入 `compute_110/sm_110`。但是发布 CI 的 Torch matrix 最高为 2.10.0，不能以“有 ARM wheel”推断某个 wheel 兼容 NGC 2.12 alpha/CUDA13.2/CXX11 ABI。源码强制本地构建更可审计。[固定 setup.py](https://github.com/Dao-AILab/causal-conv1d/blob/cd81f0413cad2fc1e6f17e785ac39f59aae690cd/setup.py#L179)、[固定发布 matrix](https://github.com/Dao-AILab/causal-conv1d/blob/cd81f0413cad2fc1e6f17e785ac39f59aae690cd/.github/workflows/publish.yaml#L39)

## 隔离构建入口

仓库脚本：[`prepare_flora_gdn_overlay.sh`](../../scripts/data/prepare_flora_gdn_overlay.sh)。它仅构建到一个全新绝对目录；已有目录一律拒绝，失败现场保留，不在同目录覆盖重跑。脚本本身没有执行 GPU import/smoke，更不会启用到正在运行的 baseline。

```bash
bash /deployment-condapi/prepare_flora_gdn_overlay.sh \
  --root /data/tshirt-fineart-20261103/gdn-overlay-candidate-01 \
  --python /opt/flora/bin/python
```

执行前应使用当前 Flora 镜像、相同 NGC 编译工具链，并保证 `g++`、`nvcc` 可用。缺少编译工具时可从**同一镜像 digest**派生一个构建候选层补工具；不另装 Torch、CUDA runtime 或 Triton。安装 script 为 CPU 构建，但 `MAX_JOBS=2` 的每个 nvcc 任务可能内部使用 4 个线程，不等于只占两条 CPU 线程。构建期间的数据不应作为推理延迟 A/B 样本。[官方 nvcc threads 设置](https://github.com/Dao-AILab/causal-conv1d/blob/cd81f0413cad2fc1e6f17e785ac39f59aae690cd/setup.py#L129)

执行方已反馈：`candidate-01` 在初始 guard 停止，原因是把 Triton distribution 的完整 vendor 版本与模块的 `__version__` 混为一谈；没有依赖安装或运行时替换。该目录保留，脚本改为分别记录/核验这两个字段后，用新 `candidate-02` 开始构建。`candidate-02` 随后构建及 import 退出码均为0；7组短内核数值排查通过，不能据此推断任意形状均等价。

脚本执行的关键约束：

- 先记录 Python/ARM64、Torch/Triton 版本与模块路径、Torch CUDA、实际 CXX11 ABI、`nvcc`/`g++`、安装版本 freeze；构建后重新记录并比较基础运行时身份。
- 只在模块缺失时把 helper 安装到新 `build-tools`：`setuptools==80.9.0`、`wheel==0.45.1`、`ninja==1.13.0`、`packaging==25.0`、`einops==0.8.1`。已有版本不替换，其确切版本进入 freeze。helper 下载与安装均 `--no-deps`，不触发 Torch 解析。
- 三个主制品下载前后都对固定 SHA；causal-conv1d 使用 `CAUSAL_CONV1D_FORCE_BUILD=TRUE`，并用 `pip wheel --no-deps --no-build-isolation` 接入当前 libtorch/header。
- 显式清除 `TORCH_CUDA_ARCH_LIST`，使用官方 setup 的 `sm110` gencode，不依赖 PyTorch 对字符串 `11.0` 的架构枚举。官方 setup 还会生成其他架构，不能把一个环境变量当作已限制为 Thor 的证据。
- 不设置 `CAUSAL_CONV1D_FORCE_CXX11_ABI` 去改 Torch 的 ABI 标志；保留实际 NGC ABI，由 extension 构建逻辑匹配。源码默认 ABI 的来源就是当前 Torch；不同 ABI 或 Torch commit 的 wheel 不混用。
- 所有主 wheel 同一次 `pip install --no-deps --no-index --target NEW_ROOT/site`。构建路径、source pins、制品 SHA、freeze、日志和生成的 `check_imports.py` 留在新 root。`BUILD_COMPLETE_NOT_RUNTIME_VALIDATED` 仅表示完成构建。

构建缓存也应指向新候选 root 下的独立目录：`PIP_CACHE_DIR`、`TORCH_EXTENSIONS_DIR`、`TMPDIR`、`TRITON_CACHE_DIR`；复用候选时保留这些证据。不要共享 baseline 的 Triton cache。若使用派生镜像，固定镜像 digest、构建脚本 SHA 和生成 wheel SHA，再把 overlay 只读挂入一个独立候选容器。

## 导入、ABI 与运行门槛

构建完成后，由执行方在空闲的 Flora GPU 临界区运行，使用与现有标注进程**相同的共享 GPU 锁**。以下锁路径沿用标注文档示例，执行前需与实际 runner 参数核对；不能用另一把锁假装互斥。

```bash
GDN_ROOT=/data/tshirt-fineart-20261103/gdn-overlay-candidate-01
GDN_GPU_LOCK=/data/.flora-annotation-gpu.lock
flock -n "$GDN_GPU_LOCK" env \
  PYTHONPATH="$GDN_ROOT/site:$GDN_ROOT/build-tools${PYTHONPATH:+:$PYTHONPATH}" \
  TRITON_CACHE_DIR="$GDN_ROOT/triton-cache" \
  /opt/flora/bin/python "$GDN_ROOT/check_imports.py"
```

该检查会加载 `causal_conv1d_cuda`，核实两库 API、`sm110` 设备和 HF fast-path 标志。必须记录扩展 `.so`、FLA norm 的真实导入路径。成功只证明能 import；尚未触发所有 Triton JIT 分支或全部 CUDA kernel。

随后进行三个递进检查，失败都保留候选现场并继续使用 baseline：

1. **内核小样本。** 比较 causal conv prefill 与 update（含 in-place conv state）、FLA chunk 与 recurrent（含 final recurrent state）、fused gated norm。reference 取同版本 Transformers 的 `torch_*` fallback，保持 BF16 输入、原 FP32 gate/state 及 `use_qk_l2norm_in_kernel=True`。覆盖 `T=1,63,64,65,128` 和真实输入中观察到的长度/形状，区分初始空状态、prefill 后 decode、分块续接；检查 finite、dtype、shape、状态更新、batch 与 padding。FLA 的 prefill 与 recurrent 分支都要实际触发 JIT，不能只执行一个短 matmul。[HF fallback reference](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L234)、[FLA 固定 GDN API](https://github.com/fla-org/flash-linear-attention/blob/413ad2c655ece57b94bc9ddc51fdbd43cf09d984/fla/ops/gated_delta_rule/__init__.py)
2. **原值与单层。** 保留 Flora 的 `_keep_native_gdn_fp32`、`verify_gdn` 和 TempSeg FP32 LoRA loader。加载后仍须证明 `A_log/dt_bias` 在最终 CUDA device 上为 FP32，并与保存 checkpoint 的数值逐项精确相等；先 BF16 舍入再转回 FP32 不满足要求。比较完整 GDN block 输出、prefill/cache/decode 和 norm，而非仅核对参数 dtype。当前本地只读依据为 Flora commit `129a2ada74c526ee4692f3d1b111bea6ca19d5b7` 的 `captioner/batch_utils/models.py` 与 vendor `production_gdn_fix.py`、`verify_gdn.py`；这部分继续服从原生 loader。
3. **同输入完整 Flora A/B。** 固定同一模型 revision、原生预处理输入 tensor SHA、prompt、采样帧、profile、greedy 配置和 token 预算；分别用没有 overlay 的 baseline 新进程与有 overlay 的 candidate 新进程。先覆盖普通窗口、低运动窗口和长 episode 接缝窗口；TempSeg 与 Captioner 都测。要求分段时间、raw generation token、caption 和 unresolved/retry 状态完全相同，计时字段单独比较；不以“意思相近”自动替换生产。任何分歧要定位并人工审核，不为通过而自动放宽采样或模型精度。

数学等价的 fused BF16 内核通常会改变舍入顺序，不能承诺 logits 逐 bit 相等。可用 `atol=1e-2,rtol=1e-2` 作为短随机算子的**初始排查阈值**，并记录最大误差、相对 RMS、NaN/Inf 和 final-state 误差；这不是官方 Thor 容差或生产放行证据。若阈值失败，先定位 reference、精度与归一化差异，不自动放宽。生产验收仍以固定窗口的 greedy 输出与标注合同为准。

至少分别记录首次 Triton 编译、warm prefill、warm decode、视觉编码、模型加载与完整窗口 wall time；CUDA 同步点、输入/输出 token、显存 allocated/reserved、host RSS 和并行进程状态一起归档。去掉首次编译后再评估收益，且 A/B 不与编译负载重叠。同一47.8s输入的冷 baseline 阶段合计1123.745s，GDN+常驻模型阶段合计513.295s；前者包含模型加载和并行构建/准备，后者是两输入倒序批次的第二条，不能当作受控端到端2×结果。Captioner逐段generation中位数为33.791s→19.849s，仍有一次约260s调用；长耗时来源尚未单独profile。

实际完整比较中，该输入的9条最终边界/caption/status/provenance，以及8次TempSeg、9次Captioner输入tensor digest均相同；**严格raw audit比较失败**：一处自适应重试生成多出后来被原生stitch舍弃的片段。因此未达到本节原定逐项raw完全相同标准，也不宣称bitwise parity。后续离线标注选择该backend的依据是7组数值检查、精度/原值核验与最终监督字段一致；该选择、raw差异及独立产物的hash记录在Thor `backend-selection.json`。扩大pilot仍需监测，原native路径可回退。

## 回退与边界

baseline 不设置候选 `PYTHONPATH`，在新的原镜像进程中直接恢复原路径；现有镜像、容器、模型权重和标注结果不覆盖。安装失败、undefined symbol、no kernel image、Triton 编译失败或输出 parity 失败都不应触发更换 Torch 或降低模型精度。对另一个架构的公开编译问题也不能推断 Thor 一定错误；本候选的 `sm110` 必须独立运行核验。

本方案未改变 attention 选择、视频帧率、分辨率、窗口、prompt、模型权重或量化格式，也不通过停掉并行 `yam-agent-perception` 来获得可用显存。当前任务的 CPU 下载/清洗与 baseline 标注可继续各自断点；当前已完成构建、7组内核排查及一条完整输入 A/B，按记录选择GDN用于离线pilot及批处理；没有把此结果外推为全部数据语义质量已验收。
