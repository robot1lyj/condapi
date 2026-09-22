# 14 · OpenWAM Thor 独立部署实验

## 范围与状态（2026-09-22）

用户要求构建镜像、下载官方检查点并探索加速与 LoRA 后的压缩；随后明确**先准备镜像和权重，暂不暂停 Pi**。因此本阶段不加载 GPU 模型、不运行基准、不改变 Pi 服务和 MAXN 生命周期，不操作3588。尚无 OpenWAM Thor 延迟或精度结论。

官方源码：<https://github.com/OpenWAM-Official/OpenWAM>，固定提交 `90e94ae31efddd64b59e00365cfc501d9a972eb1`。它不同于另一个同名 `OpenWAM/OpenWAM` 项目。本地其他任务的 `third_party/openwam`、`adapters/openwam` 不作本轮构建来源、不覆盖或提交。

Thor 根目录 `/home/wuyan-lyj/thor/openwam`，与 Pi 分开。镜像配方见 `scripts/thor/openwam/Dockerfile`，使用已安装 `nvcr.io/nvidia/pytorch:26.05-py3`，保留 NVIDIA Torch/vision/triton/numpy，不照搬上游 CUDA12.8 x86 依赖锁。仅安装推理依赖，不装训练用 DeepSpeed；完整训练依赖审计不在本次范围。

选择官方可部署的 `OpenWAM/OpenWAM-Alpha-Sim-RoboTwin-Full`，约24.8GB（下载器估计，实际以元数据为准）。Foundation 官方标记 finetune-only，不拿它冒充可直接部署的策略。下载脚本固定HF revision并核验LFS SHA256，输出 `thor-download-receipt.json`；没有回执就不能宣布下载验收。

Thor直连HF及hf-mirror超时，本机HF可通。临时SSH远程动态转发只监听Thor的 `127.0.0.1:18890`，CPU下载容器使用SOCKS代理并禁用Xet，文件直接落Thor。依赖本机及SSH连接在线，不是长期网络配置；下载完关闭该转发。初次失败容器 `openwam-checkpoint-download` 保留诊断，重试为 `openwam-checkpoint-download-relay`。

## 不可套用 Pi 的合同

镜像 `openwam:thor-20260922-r1` 已构建；无GPU/无网络的官方CLI `--help`、`BaseInferenceEngine` 与模型工厂导入通过。Torch=`2.12.0a0+5aff3928d8.nv26.05`、transformers=`5.17.0`、diffusers=`0.40.0`。这只验证依赖导入，不等于权重加载或GPU推理通过。构建日志 `logs/build-r1.log`、CPU检查 `logs/cpu-cli-smoke.log`，完整依赖在镜像 `/opt/openwam/thor-environment.txt`。

HF检查点已固定revision `04b96af53eeeb111c64631f822efcbb82f4b186e`；主权重文件24,813,767,464字节，Wan2.2 TI2V5B骨干。转发下载曾在约100MB断开，已加入断点重试；当前仍下载中，未生成完成回执。

Alpha官方合同为80维统一槽位、32个action（33源帧，video_stride4），包含双臂EEF位置/rot6d及夹爪，min-max归一化。不是Pi的14维关节/50步/分位数归一化。官方RoboTwin模型仅用于原生推理运行验证；不下发YAM，也不声称测得乐高任务成功率。后续输入形状和dtype以下载的config核实。

## GPU获准后实验顺序

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
