# 14 · OpenWAM Thor 独立部署实验

## 范围与状态（2026-09-22）

用户最初限定先准备镜像和权重，2026-09-22准备验收后允许暂停Pi、独占MAXN做GPU测试。最新指示覆盖此前的测试后恢复要求：**先保持Pi停止，集中解决OpenWAM加速部署，不再自动恢复Pi**。不操作3588、不把RoboTwin模型下发YAM。GPU测试状态与结果必须以对应报告为准。测试/推理使用MAXN，非推理空闲恢复低功耗；这与是否恢复Pi服务是两个独立行为。

官方源码：<https://github.com/OpenWAM-Official/OpenWAM>，固定提交 `90e94ae31efddd64b59e00365cfc501d9a972eb1`。它不同于另一个同名 `OpenWAM/OpenWAM` 项目。本地其他任务的 `third_party/openwam`、`adapters/openwam` 不作本轮构建来源、不覆盖或提交。

Thor 根目录 `/home/wuyan-lyj/thor/openwam`，与 Pi 分开。镜像配方见 `scripts/thor/openwam/Dockerfile`，使用已安装 `nvcr.io/nvidia/pytorch:26.05-py3`，保留 NVIDIA Torch/vision/triton/numpy，不照搬上游 CUDA12.8 x86 依赖锁。仅安装推理依赖，不装训练用 DeepSpeed；完整训练依赖审计不在本次范围。

选择官方可部署的 `OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full`，约24.8GB（下载器估计，实际以元数据为准）。Foundation 官方标记 finetune-only，不拿它冒充可直接部署的策略。下载脚本固定HF revision并核验LFS SHA256，输出 `thor-download-receipt.json`；没有回执就不能宣布下载验收。

Thor直连HF及hf-mirror超时，本机HF可通。临时SSH远程动态转发只监听Thor的 `127.0.0.1:18890`，CPU下载容器使用SOCKS代理并禁用Xet，文件直接落Thor。依赖本机及SSH连接在线，不是长期网络配置；下载完关闭该转发。初次失败容器 `openwam-checkpoint-download` 保留诊断，重试为 `openwam-checkpoint-download-relay`。

## 不可套用 Pi 的合同

镜像 `openwam:thor-20260922-r1` 已构建；无GPU/无网络的官方CLI `--help`、`BaseInferenceEngine` 与模型工厂导入通过。Torch=`2.12.0a0+5aff3928d8.nv26.05`、transformers=`5.17.0`、diffusers=`0.40.0`。这只验证依赖导入，不等于权重加载或GPU推理通过。构建日志 `logs/build-r1.log`、CPU检查 `logs/cpu-cli-smoke.log`，完整依赖在镜像 `/opt/openwam/thor-environment.txt`。

HF检查点已固定revision `04b96af53eeeb111c64631f822efcbb82f4b186e`；主权重文件24,813,767,464字节，Wan2.2 TI2V5B骨干。转发下载经历多次断连后断点恢复，2026-09-22 17:14 CST已完成9个文件并生成回执，主权重SHA256与官方LFS记录一致。

文件头显示2089个张量全部为BF16（不是依据训练配置猜测）；下载后CPU扫描2089张量未发现NaN/Inf，容器退出0。检查入口 `scripts/thor/openwam/inspect_safetensors.py <文件>`，支持未完成文件但不能替代下载回执。官方config确认 `joint_self_attn`/`mutual`、三相机、384×320、`action_mode=eef`、80维动作/状态、min-max。分词器无网络/无GPU真实编码通过，Pi健康检查仍为OK。回执与完成边界见[准备报告](../../reports/thor/openwam-20260922/README.md)。

Alpha官方合同为80维统一槽位、32个action（33源帧，video_stride4），包含双臂EEF位置/rot6d及夹爪，min-max归一化。不是Pi的14维关节/50步/分位数归一化。官方RoboTwin模型仅用于原生推理运行验证；不下发YAM，也不声称测得乐高任务成功率。后续输入形状和dtype以下载的config核实。

## GPU获准后实验顺序

**2026-09-22首轮GPU结果已完成**：r4四组均有限32×20输出；MAXN下baseline1254ms、提示词缓存1253ms、DiT缓存520ms、compile964ms（各3次稳态P50）。后两者有输出差异，未验收任务精度或YAM部署。Pi已恢复原8000服务与MAXN、healthz=OK。当前可运行镜像为 `openwam:thor-20260922-r2`，r1镜像缺h5py，不再推荐；具体数值、合成输入局限和失败记录见[GPU报告](../../reports/thor/openwam-20260922/gpu-r4.md)。

**后续r6–r9**：用户要求Pi停止；BF16官方组合在Thor MAXN下约404–408ms，`reduce-overhead`确认CUDA Graph重放、`max-autotune`没有加速。7步单独编译能比10步快，但加官方DiT缓存后均只跑4次完整前向，7步没有增益且动作差增大。GPU profile显示矩阵乘法约249ms、联合masked SDPA约60ms，是后续热点。细节、原始回执和2026-09-23 Thor管理网断连情况见[BF16优化报告](../../reports/thor/openwam-20260922/bf16-optimization.md)。下次先恢复Thor连接，运行已准备的同形状注意力后端探针；探针不等于完整模型验收。

历史r5选择性FP8实验已单独归档为[量化报告](../../reports/thor/openwam-20260922/quant-r5.md)，当前按用户指示优先BF16路线。

2026-09-22 GPU首测：r1在模型加载期间终止，修正测试输入必须按官方ObsPreprocessor把三相机拼成单帧，未产生有效推理数据。r2加载走到归一化阶段失败，官方dataloader注册连带导入h5py，r1镜像缺依赖；Pi退出恢复已验证。修复镜像 `openwam:thor-20260922-r2` 补齐h5py/av等原生依赖，并通过完整normalizer的CPU检查，20D状态→80D模型空间、32×80→32×20原始EEF动作。r3开始四组对照：baseline、prompt_cache、dit_cache、compile。脚本 `scripts/thor/openwam/benchmark.py` 为单一合成输入、固定seed42、10步，每组1次冷调用+3次计时；P95仅描述小样本，不是可靠尾延迟保证或任务精度。原始目录 `logs/benchmark-r3`、宿主日志 `logs/benchmark-r3-host.log`。`run_benchmark.sh` 在MAXN包装器退出后恢复Pi，最多1800秒；后续运行使用新run_id，不覆盖旧结果。

续查下载达到约18GiB后，已启动一次性CPU校验容器 `openwam-download-audit`：等待下载完成回执（最多4小时），然后扫描BF16指数位检查NaN/Inf，输出 `logs/finite-audit.json`。入口 `scripts/thor/openwam/audit_download.py`，限制1CPU/512MiB、无网络、无GPU、checkpoint只读；不是定时推理或自动上线。检查有限性不代表模型任务精度。下载超时或校验失败需读取容器日志，不得当作通过。

1. 独占MAXN，固定官方checkpoint/config、输入/噪声/种子，记录原生dtype与10步基线，关闭compile和近似DiT缓存。真实观测与人工构造smoke分开标注。
2. 同输入测试提示词缓存、`decode_video=false`、torch.compile/SDPA。跳过视频**解码**不等于删除内部video DiT；动作仍可能依赖它。
3. 单独启用官方DiT velocity cache：属于近似跨去噪步复用，报告动作逐维差异、连续性及实际跳步率，不当成无损。
4. 再试减少去噪步和分模块量化（先视频骨干Linear，再决定是否扩展），每项单独比较误差与P50/P95、峰值显存、编译冷启动；不先承诺200ms。
5. FP8/INT8/INT4只有Thor可用内核与实测才有意义。动作头、归一化和敏感时间条件先保留原精度。TensorRT导出和CUDA Graph需验证固定shape与控制流，当前未实现。

## LoRA 与压缩

LoRA减少训练参数/每任务增量文件，不减少基础骨干的推理参数量。可合并的标准Linear LoRA部署路线：原base+adapter → 较高精度合并 `W + scale·BA` → 与未合并输出对照 → compile/量化 → 再对照。还须包含训练过的action expert及其他modules_to_save，不能只保存adapter丢掉额外训练模块。保留原base与adapter及完整配置以便重做。

若训练采用量化基础权重，合并到未量化原base不一定复现训练时结果，必须以实际训练/未合并模型为参考；不是把QLoRA文件直接交给TensorRT。

截至本轮官方源码检索，尚未核实Alpha完整PEFT LoRA注入/保存/恢复/合并入口。Cosmos代码中AdaLN-LoRA是架构内部机制，不能当作用户LoRA微调支持。需要训练侧明确target_modules、rank/alpha、base revision、额外训练模块，推理侧再接入；本轮不生成或启动训练。

依据：[官方Alpha合同](https://github.com/OpenWAM-Official/OpenWAM/blob/90e94ae31efddd64b59e00365cfc501d9a972eb1/assets/openwam_usage_docs/openwam-alpha-finetuning.md)、[官方部署开关](https://github.com/OpenWAM-Official/OpenWAM/blob/90e94ae31efddd64b59e00365cfc501d9a972eb1/configs/deploy.yaml)、[PEFT LoRA](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora)。以上加速路线为候选，不是Thor实测结果。
