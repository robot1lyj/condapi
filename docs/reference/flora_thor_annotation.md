# Flora 在 Thor 上标注 T 恤：原生合同、断点与加速核查

更新：2026-11-03。本任务已在 Thor 完成 T 恤 pilot 的结构检查、单 clip 原生 GPU baseline、GDN overlay + 模型驻留候选，以及独立 9B selector 和语言列 materialization。下表是当前已取得的证据；后文保留的耳机计时与实施前方案明确标注了时点。训练数据发布仍未完成，不能据此声称 300 多集已全部标注或可训练。

| 本轮检查 | 实际结果 | 结论边界 |
| --- | --- | --- |
| pilot 输入结构 | 12 个完整父 episode、23 个标注窗口通过三视角/帧数/时间与来源结构检查 | 不是 12 集语义标注质量验收 |
| 原生 baseline | `tshirt-train-128691-w000`，47.8 秒，9 条 final caption，完整 wall 1125.69 秒 | 含冷加载及当时并行负载，不是稳定吞吐 |
| GDN + resident 同 clip | 完整 wall 513.295 秒 | 冷暖/加载条件与 baseline 不同，不能据此承诺固定倍数或单独归因 GDN |
| 两路径输出比较 | 最终 9 条 caption、边界、status、provenance 及已对齐输入 tensor digest 一致 | raw audit 有一处重试分支不同，严格完整比较失败，不能宣称 bitwise parity |
| 9B selector v3 | 真实处理 3 个冲突/缺口 atom：两处选择与该 clip 视觉诊断一致，0.5 秒缺口返回 `UNKNOWN` | 单 clip 校准，不是语义准确率指标 |
| 语言列实际写入 | 新 Parquet 保留 1434 个 action anchor：1419 个有语言、15 个双语言列为空 | 已 materialize 语言列，尚非完整 LeRobot v3 数据发布或训练 READY |

GDN 使用独立 `gdn-overlay-candidate-02`，原容器环境、模型权重与无 overlay 路径保留。当前批量路线是原生 Flora 阶段批处理/模型驻留，加程序 atom 约束与独立 selector；无需逐 episode 请求用户或 agent 审批。质量证据目前仍只有 1 条 clip 校准，分层抽样和批次语义质量验收尚待实测。

## 1. 已确认的原生模型与输入合同

审计源码：`/home/wuyan-lyj/data_clean/Flora`，提交 `129a2ada74c526ee4692f3d1b111bea6ca19d5b7`。

| 项目 | 原生行为 |
| --- | --- |
| TempSeg | Qwen3.5-9B + Flora LoRA；adapter 固定 revision `114a1076f70eb53f4a58ac045505db3087e79347`，base 固定 `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Captioner | Flora 全模型，基于 Qwen3.5-27B；固定 revision `fef0bbfe11829ca7f3539d58da7a6a2c2f109dc6` |
| 精度 | BF16 主体；Gated DeltaNet 的 `A_log`、`dt_bias` 保留 FP32；TempSeg 保存的 FP32 LoRA 不合并、不降精度，temporal compute 用原生 BF16 autocast |
| Attention | TempSeg 已用 SDPA；Captioner 的 text/vision 均已 SDPA；Captioner 不接受 CPU offload，原生检查全部参数在 CUDA |
| Processor | `min_pixels=1024,max_pixels=589824`；TempSeg processor `fps=4,max_frames=128`；Captioner `fps=8,max_frames=40` |
| 实际帧计划 | 原生 `temporal_frame_plan` / `plans_for_episode` 选源帧；processor 使用 `do_sample_frames=False`；不能把 processor 的 fps 当成任意改采样旋钮 |
| 三相机 | main/top、left_wrist、right_wrist；YAM 固定 main 在上、两腕在下的 mosaic，时间戳/帧数/时长必须完全同步且 CFR |
| 时间窗口 | `profile=yam` 默认原生 16 秒 temporal profile；`yam32` 是另一原生 profile，会改变推理输入，不能混入速度 A/B |
| 解码 | thinking disabled；greedy，`do_sample=False,num_beams=1,use_cache=True,max_time=300`；TempSeg 最多 1024 新 token，Captioner 最多 512；上下文最多 32768 token；必须 EOS 且文本非空 |
| 输入上限 | 每个输入视频不超过 120 秒、256 MiB、7200 帧、每边 4096 像素；源帧率至少 4；PyAV 单流完整检查 PTS |

来源：`captioner/batch_utils/{models.py,media.py,preprocessing.py,pipeline.py,config.py,models.json}`，`captioner/batch_utils/vendor/tempseg_fp32_loader.py`，TempSeg/Captioner model card。模型 manifest、LoRA/GDN 原值核对沿用原生逻辑。

超过 120 秒的完整训练 episode 不应因此删除或裁成短轨迹。Flora 需要单独的合规标注派生窗口及原始帧/绝对时间映射；训练仍使用完整原始 episode。接缝、重叠窗口和跨窗动作需要单独质检，不能直接把派生窗口边缘当成 subtask 边界。原生 Flora runner 只接收已准备、同步并通过原生检查的窗口；视频派生窗口由准备工具生成，语言接缝由第 6 节自动策略处理。

## 2. 实施前排期依据：2026-10-31 耳机旧记录

证据：`/home/wuyan-lyj/data_clean/reports/timing-summary.json`、`reports/test.log`。三集耳机输入总长 302.233334 秒，输出 93 个片段；当时尚无语义质量验收记录。

| 历史测量 | 数值 | 含义 |
| --- | ---: | --- |
| 处理时间 | 4655.66 秒 / 77.59 分钟 | 不包含全部下载等待 |
| 模型加载 | 1150.71 秒，24.7% | 每集 TempSeg、Captioner 各重载一次 |
| generation 区间 | 2910.44 秒，62.5% | 包括视觉编码、prefill、自回归输出，未分别计时 |
| 其余 | 594.51 秒，12.8% | 探测、解码、预处理、审计、拼接等的合计；不能全部归给 CPU 解码 |
| TempSeg generation | 343.19 秒 / 31 次 | 平均输入 5509 token、输出 82.4 token、11.07 秒/次 |
| Captioner generation | 2567.25 秒 / 93 次 | 平均输入 11698 token，最多 16792；平均仅输出 20.7 token；27.60 秒/次 |
| 历史峰值 allocated | 67.607 GiB | 与当前内存可用量分别观测 |

这组旧记录中 Captioner 占 generation 时间约 88.2%。由于输出很短、输入较长，这些计时不足以认定“慢在纯文本 decode”；后续受控测量仍需拆分视觉、prefill、decode。

旧日志第 12 行记录了缺少 fast-path 库、Gated DeltaNet 回退 Torch，并给出 `flash-linear-attention` 与 `causal-conv1d` 提示。这是当时研究候选的依据；此后已推进至 candidate-02 的构建、运行及单 clip 对照，当前证据见本文开头。该旧 warning 不再代表候选运行状态，单 clip 结果也不能覆盖所有长度/形状和语义场景。

实施前按这组耳机旧日志估算，完全消除加载的理论上限约 1.33 倍，10 小时视频线性外推约 154 小时。它仅记录当时的排期粗算；T 恤现已有单 clip 数据，后续仍需在相同冷暖条件下测多窗口吞吐，再更新批量预算，不能沿用旧外推承诺交付时间。

## 3. 固定官方源码现象与旧部署的重复工作

以下描述针对固定 Flora 官方源码及旧 worker 路径；本任务的 session-cache wrapper 已绕过其中的同模型重复 load，不应读成当前批处理仍逐视频重载。

1. `models.Session.load()` 每次无条件 `self.close()`。`pipeline.annotate()` 每 episode 新建 Session；旧 `deployment/run_test.py` 虽按阶段循环共享 Session，每次 `segment()` / `caption_three()` 仍会调用 load，因此仍重复加载。
2. 模型 manifest 校验已有进程内、带文件 inode/size/mtime/ctime 的安全复用。现有 viewer worker 每 job 起独立 CLI 子进程，该缓存无法跨 job 复用；不能用删除校验换速度。
3. `media.decode_selected()` 每次重新打开源文件，从开头串行解到最大目标帧；每路 `thread_count=1`。`caption_three()` 每 segment 调一次 `frames_for()`，三个视角反复解码同一源前缀。单视角 caption 路径已有 union 帧一次解码，三视角尚未采用。
4. `models.generate()` 对各输入 tensor 做 CPU 连续化和 SHA-256，记录 `input_tensor_digest`。这有审计价值，应保留，并用它验证候选帧缓存是否改变输入。
5. viewer worker 使用 catalog 的 `gpu-worker.lock`、lease 和 heartbeat。直接运行单个 Flora wrapper 的容器 fLock 只约束同锁进程；当前 host 串接入口已额外持有既有 viewer host GPU 锁，再由 Flora/selector 持有容器共享 GPU 锁协调执行，不改 viewer DB 或服务。等待下载时不持有 GPU 锁。

## 4. 已实现的低风险调度 wrapper

文件：[`scripts/data/run_flora_tshirt_batch.py`](../../scripts/data/run_flora_tshirt_batch.py)。默认 `--mode session-cache`，另保留 `--mode baseline` 供同输入 A/B。两个模式均直接调用原生 `DataEnginePipeline._inputs`、`segment`、`caption_three`、Session 和所有原生预处理/生成函数。

- `baseline`：按两阶段保存，但每 job 的 load 保持原生重载；`session-cache`：同 kind 已加载时返回原生 `(model,proc)`，否则调用原生 load。阶段结束 close，只驻留一个模型。
- 所有输入先进行原生 CPU 探测、解码、同步与 SHA-256 检查。记录输入文件 stat，在推理前后检查源文件未改变。
- `*.tempseg.json` 保存原生分段、temporal audit、identity 和计时；`*.captioner.json` 保存原生最终 payload、identity、TempSeg 文件 SHA-256 和计时。identity 包含视频 SHA、全部模型 revision、instruction、profile、yam32、相机角色和时长。
- `*.json` 仍是 Flora 原生 `flora_demo_annotation_v1`，包含 `pipeline.temporal_windows`、`pipeline.caption_inputs`、原生 status/provenance；没有伪造全部 complete。独占 hard-link 原子发布，已有文件不覆盖；恢复会重验 checkpoint identity，并核对 final 的精确字节。
- Captioner checkpoint 已保存但 final 尚未发布的崩溃，可恢复发布而不重新生成。旧 final 若无本 runner 的 identity-bound checkpoint，会明确拒绝作为断点。
- `--stage tempseg|captioner|all`、`--limit N` 限定处理。失败立即非零退出，已完成阶段保留；未保存的当前阶段需重跑。`--limit` 数的是 JSONL 窗口 job，不是完整原始 episode。
- 统一共享文件 fLock，避免本 runner 多实例同时用 GPU。JSON 日志前缀为 `FLORA_BATCH`；保留原生 `FLORA_RUNTIME`。
- 模型加载与每阶段计时在 CUDA 同步点测量；`native_work_wall_s = stage_wall_s - model_load_wall_s` 仍包含原生 CPU/GPU 工作，不能当纯模型前向耗时。CPU validation 单独计时。

这个调度 wrapper 自身没有加入帧缓存、混合批量 generate、新的 attention kernel、量化或 torch.compile，也不更改原生重试、提示词、采样、窗口 profile 或 token 限制。独立 GDN overlay 是另一个明确选择的运行后端，其实际比较结果见开头，不能与 session-cache 的调度改动混为同一项。

实施前的本地检查覆盖 AST/CLI 和 CPU 模拟协议：两 job baseline 共 4 次 load、session-cache 共 2 次 load、完整断点零新调用、caption checkpoint 后恢复发布、指令/最终结果篡改拒绝、fLock 互斥和原子不覆盖。之后已执行本文开头列出的真实 GPU baseline/候选对照；完整 raw-audit 严格比较未通过，当前只确认已列明的最终标注/输入 digest 一致项。

## 5. Thor 挂载、依赖与命令

本轮沿用下列基础容器与挂载；候选 overlay 和版本化工具写入本任务新目录，不替换原环境。每次启动仍按实际运行回执与绑定文件确认所选后端：

- Thor root：`/home/wuyan-lyj/thor/flora`；容器 `flora-annotation`；镜像 `flora:thor-20261031-r2`。
- 原有挂载：`code:/workspace/Flora:ro`、`deployment:/deployment:ro`、`data:/data`、`logs:/logs`、`results:/results`；新 runner 通过新增只读挂载或复制到该任务的独立部署目录接入，不能覆盖官方 code。
- 调度 runner 使用 Python 标准库，时间列读取/materializer 另外使用已有 PyArrow；原生推理仍使用 Flora 模型依赖。运行解释器沿用 `/opt/flora/bin/python` 或该容器已经验证的同一 Python 环境。保留 NVIDIA PyTorch 26.05 ARM64 配套 Torch `2.12.0a0+5aff3928d8.nv26.05` / torchvision `0.27.0a0+6214cb20.nv26.05`，不安装官方 x86 Torch 2.11/cu128 替换它。
- 其余原有固定依赖：Transformers 5.8.1、PEFT 0.18.1、accelerate 1.12.0、huggingface_hub 1.33.0、safetensors 0.7.0、AV 16.0.1，NumPy<3、Pillow>=11,<13；部署脚本另含 pyarrow 23.0.1。
- 环境：`HF_HOME=/data/hf`、`FLORA_MODEL_CACHE=/data/models`、`OMP_NUM_THREADS=4`；既有镜像使用国内 HF/PyPI mirror，仍须固定 revision 和 manifest。鉴权走既有环境，不能把 token 写进 jobs、命令日志或仓库。
- GDN 候选路径：`/data/tshirt-fineart-20261103/gdn-overlay-candidate-02`。候选新进程通过自身 `PYTHONPATH`/编译缓存选择 overlay；native 新进程不带该 overlay。原基础环境保留，构建和选择证据留在任务目录。

原生 JSONL 输入（相对路径以 JSONL 所在目录为基准；lineage 放独立 sidecar）：

```json
{"id":"tshirt-000123-window-000","video":"windows/000123/000/top.mp4","left_wrist":"windows/000123/000/left.mp4","right_wrist":"windows/000123/000/right.mp4","profile":"yam","yam32":false,"instruction":"fold and stack the t-shirts"}
```

以下保留为实施前的单阶段诊断命令模板；当前整批自动流程使用第 6 节两个入口，不能把这里的占位路径当作已执行回执。

原生 CPU 检查和原生 CLI baseline 模板：

```bash
docker exec -w /workspace/Flora/captioner flora-annotation /opt/flora/bin/python -m batch_utils --input-jsonl /data/abc-tshirt-pilot/jobs.jsonl --validate-only
docker exec -w /workspace/Flora/captioner flora-annotation /opt/flora/bin/python -m batch_utils --input-jsonl /data/abc-tshirt-pilot/jobs.jsonl --output-dir /results/abc-tshirt-native-baseline --include-audit
```

假设新 runner 以只读文件提供在 `/deployment-condapi/run_flora_tshirt_batch.py`，建议先限定少量窗口；以下是命令模板，不是已经执行的回执：

```bash
docker exec flora-annotation /opt/flora/bin/python /deployment-condapi/run_flora_tshirt_batch.py --flora-root /workspace/Flora --input-jsonl /data/abc-tshirt-pilot/jobs.jsonl --output-dir /results/abc-tshirt-pilot-cache --validate-only --limit 3
docker exec flora-annotation /opt/flora/bin/python /deployment-condapi/run_flora_tshirt_batch.py --flora-root /workspace/Flora --input-jsonl /data/abc-tshirt-pilot/jobs.jsonl --output-dir /results/abc-tshirt-pilot-baseline --mode baseline --lock-file /data/.flora-annotation-gpu.lock --limit 3
docker exec flora-annotation /opt/flora/bin/python /deployment-condapi/run_flora_tshirt_batch.py --flora-root /workspace/Flora --input-jsonl /data/abc-tshirt-pilot/jobs.jsonl --output-dir /results/abc-tshirt-pilot-cache --mode session-cache --lock-file /data/.flora-annotation-gpu.lock --limit 3
```

baseline 与 cache 必须不同输出目录；否则已存在结果会复用，不能测推理收益。正式后台运行由 owner 用有退出码回执的任务日志或 tmux 管理，避免把容器 running 当作完成。

## 6. 自动批量标注与语言列串接

host 入口为 [`run_tshirt_annotation_pipeline.sh`](../../scripts/data/run_tshirt_annotation_pipeline.sh)，显式选择 `--backend native|gdn`。它绑定本任务工具/策略/后端证据，依次运行 pilot 原生标注与自动语言处理，等待下载完成后准备 train/dev 窗口，再继续同样批处理；reserve 保持下载备用。host 与容器 GPU 锁分别协调 viewer 和原生/selector 进程，等待下载时释放 GPU 锁，不以停止其他服务换吞吐。

批次语言入口为 [`run_fineart_auto_batch.py`](../../scripts/data/run_fineart_auto_batch.py)。它冻结 lineage、父 audit 的同目录字节快照，以及本批 policy/工具版本，然后串接：

1. 原生结果转 draft，程序把全部 start/end/core 事件映射为父帧半开 atom，保证帧覆盖和时间唯一。
2. 单一有效 active caption、仅空白差异的同 caption 自动规则接收；冲突、缺口和窗口接缝交独立 9B selector。模型通过受限 ID/`UNKNOWN`选择 primary，不能改时间轴或自由生成新 caption；hold/smooth 等并发信息留 sidecar。
3. 从原 Parquet 实读 `episode_index/frame_index/timestamp`，核对原 data SHA、完整帧数和 30FPS；时间值不由 `frame/30`伪造。只有存在 selector tasks 才加载/调用 9B。
4. apply 根据哈希绑定的模型/提示词/图片回执生成明确视图，materializer 只向新目录逐帧写两语言列，检查正常帧完整已知 timeline 和 UNKNOWN 帧双空列表，原始非语言列不变。

每阶段都有 stdout/stderr、退出码、状态、ownership、来源 hash 和失败回执；任何非零止步，已完成原件/结果保留，不无限重试。`--skip-materialize`可明确只停在视图阶段。无需为每条 episode 手填 review 或 timestamp JSON，也不要求逐条用户或 agent 审批；少量超阈值异常进入显式队列，其他已解决父 episode 继续处理。

当前policy允许显式 `native-empty-anchor-fallback`：仅真实 selector `UNKNOWN`，每父累计未知帧 **≤2%且合并后的连续run≤30帧**，可以保留整父动作和视频。这是初始工程阈值，不是论文结论或已验证准确率。小 UNKNOWN anchor 的 `language_persistent=[]`、`language_events=[]` 触发固定原生 task→action fallback，不监督空 assistant/subtask 文本；flow 与原生启用的 FAST 动作监督仍保留。不能用null-content row或把`UNKNOWN`写成指令。超阈值、大缺口、缺decision或hard source问题保留pending，不强行填词，也不阻断其他父episode。

这保留原task、绝对14D state/action、完整父episode和H50；标注窗口不变成训练episode。当前单 clip 实际写出1434个 action anchor，其中1419个有效语言anchor、15个双空列表（35–35.5秒）；动作coverage100%，有效语言coverage约98.954%，不能写成语言100%或当作accuracy。真实9B对该clip的两个重叠分别选择重新spread和straighten，与已有视觉诊断一致；0.5秒gap诚实返回UNKNOWN，而非自动继承前一句。

语言列产物状态为 `LANGUAGE_COLUMNS_MATERIALIZED_NOT_DATASET_RELEASED`：已实际写Parquet不等于完整LeRobot v3 metadata/发布完成，`training_ready=false`。后续还需分层批次语义抽样、窗口接缝质量、数据身份/切分、训练集norm和FAST审计；`task_success`保持`NOT_REVIEWED`。当前仅1clip做过selector校准，不能据12pilot结构通过或单clip输出一致，声称12集/319集已完成语义验收或全量标注。

策略、原生fallback依据及训练准备门槛归 [FineART训练说明](fineart_tshirt_training.md)；机器合同归 [tshirt-fineart-v1.json](../../configs/annotation/tshirt-fineart-v1.json)。GDN候选的源码/制品研究记录归 [GDN说明](flora_thor_gdn_acceleration.md)，其中实施前“未实测”陈述保留为当时研究阶段，当前运行结论以本文及本任务后续回执为准。

## 7. 已做实验与后续验收

1. **现有结果与受控复测。** 12pilot/23windows的输入结构检查和1clip baseline/GDN+resident已完成；raw-audit严格比较的一处重试差异保留，不按bitwise验收通过。下一步覆盖普通、低运动和长episode接缝，分开冷加载与warm run，记录总wall、load、generation token/耗时、显存、host RSS和并行负载，再评价稳定吞吐及语义质量。
2. **精确帧缓存候选。** 三视角按原生 wanted union 一次解码、按原生 canonical RGB 与 mosaic 缓存；保持源 PTS、颜色转换、resize、每个 frame index 和模型输入 tensor SHA 一致。先 CPU 比较所有所需图像像素、几何与输入张量；控制缓存上限，不能把全部 10h RGB 放内存。未经 parity 不接生产。
3. **CPU/GPU 流水候选。** 单 GPU 仍串行原生 generate，提前 CPU decode/processor 下一输入，保持原生 `batch_concurrency=1` 和调用顺序。测实际 CPU 空档与 GPU 利用率后再做，不先改线程数或采样。
4. **Gated DeltaNet后续覆盖。** candidate-02独立overlay和原环境均保留，已有构建/运行及1clip证据。继续核查真实长度/形状、重试路径、FP32 GDN/LoRA原值、NaN/显存以及更多clip的最终标注/完整audit差异，不能把单clip的最终相同项扩成所有输入bitwise承诺。SDPA已开启，改FlashAttention选项不能代替GDN验证。
5. **编译与 batch generate 候选。** 仅当前项收益不足时研究 torch.compile、prefill/decode 分阶段计时和多 caption batch；需要核验 Qwen hybrid cache、padding/position、显存峰值与 greedy 输出一致性。不是当前已实现能力。

不把降低帧率/分辨率、修改 prompt、减少输出预算、切换 yam32、跳过 unresolved retry、量化或降 FP32 LoRA/GDN 精度列入保持原生标注语义的加速方案。
