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

已启动 Thor 直拉与本机下载备选，后者保存在 `/home/wuyan-lyj/thor-system/images/pytorch-26.05-arm64/`，必要时经 USB 传输。`scripts/docker/thor/pytorch.Dockerfile` 目前是候选配方，尚未构建验证；保留 NVIDIA 的 torch/CUDA/TensorRT 约束，不安装项目服务器训练整套依赖，不执行教程脚本来覆盖本项目代码。

下一次实测先跑少量既有 9 输入，检查 FP32 / BF16 输出差异和实际同步延迟；候选速度有价值才扩大样本。FP32 禁用 TF32，记录实际参数 dtype、编译开关、输入与权重摘要、首次编译时间。当前没有新的 PyTorch / TensorRT 延迟结果，原 JAX 约 177 ms 最快结果仍未满足用户目标。
