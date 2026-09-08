# 2026-09-08 · Pi0.5 四卡全参数测试与 YAM norm

4×RTX 4090 24GB 已通过 Pi0.5 全参数容量短测。全量训练数据 norm 已完成。
本报告证明短步数计算可行，不证明长期稳定、任务收敛或正式训练吞吐；未启动正式长训。

## 实测

Slurm 2064 / gpu001，分配截止 2026-09-15 11:15:32 +08:00；结束时保留原资源作业。
正式共享环境 `/home/wuyan/.conda/envs/condapi-yam`：Python 3.12.12、JAX/jaxlib 0.5.3、
Flax 0.10.2、Torch 2.7.1、LeRobot 0.5.1、NumPy 2.2.6。4 卡均完成 JAX 设备识别和计算。
驱动 595.45.04；GPU 之间为同 NUMA 的 PCIe NODE 路径，不是 NVLink。

| 项目 | 真实数据短测 |
| --- | --- |
| 配置 | `pi05_yam`，全参数，FSDP=4，global batch=4，EMA=None |
| 参数 | 3,353,433,872，全部属于 trainable filter，无 LoRA |
| 优化器/精度 | 原配置 AdamW；FP32 参数和优化器状态，BF16 模型计算 |
| 输入 | train episode 0 的前 4 帧，三相机 224×224，文本桶 200，state 32D，action 50×32D |
| 迭代 | 同一真实 batch 重复 10 次；每步同步等待计算完成 |
| 第一更新 | 63.02 秒，包含 JIT 编译 |
| 后 9 步 | 平均 0.6660 秒/步 |
| loss | 全部有限，首步 0.05589，末步 0.04842；不可据此推断收敛 |
| 每卡活跃显存高水位 | JAX allocator 13.6211 GiB |
| 每卡内存池高水位 | JAX allocator 16.0020 GiB；不含全部驱动/通信库开销 |
| 产物 | [逐步指标](real/metrics.jsonl)、[完整日志](probe-real-b4.log)；不保存模型 checkpoint |

### 初始化修复及对照

原 `train.init_train_state()` 把基础权重作为 replicated JIT 输入，导致每卡先放一份完整权重，
再分配分片训练状态，初始化 OOM，见 [失败日志](probe-b4.log)。
本次先按 FSDP 规则放置 checkpoint 输入，再初始化优化器；单卡仍使用 replicated placement。
模型、损失、优化器和参数冻结设置不因此改变。

修复后合成输入完成 3 步，见 [指标](synthetic/metrics.jsonl)。该早期合成测试沿用 FakeDataset 的
空 mask；结论以随后真实三相机、真实 prompt、真实 norm 的 10 步测试为准。

本地 norm 单测 6 项通过。原训练 smoke/保存恢复测试和 norm 单测在计算节点共 7 项通过
（105.37 秒）。本地训练测试曾以 137 退出，未视为通过；转计算节点后通过。
新增脚本 Ruff 通过，`git diff --check` 通过；`train.py` 原有 21 项 E402 来自提前初始化 JAX 后的导入，
本次不调整该顺序，忽略已有 E402 后无其余问题。

## 数据与统计资产

完整数据版本、统计路径和合同见 [数据 owner](../../../04_data_contracts.md#2026-09-08-全量发布与归一化完成)。
[数值加载器对照](loader_equivalence.json) 检查 3 条代表轨迹各 4 个位置，包括 horizon 边界和最后帧；
数值路径与 LeRobot→YamInputs→DeltaActions 逐值一致，同时成功解码真实三路图像。
[计算日志](norm.log) 记录全部进度；覆盖 10,374,181 个 state 和 518,709,050 个 H50 action 向量。

仅统计 train。包含不足整批的尾部帧，episode 尾部重复最后 action，关节减当前 state、夹爪 absolute。
使用原 OpenPI RunningStats 5000-bin 近似分位数、float64 累计；不是精确排序分位数，也不声称与旧脚本
float32/不同 batch 次序的近似直方图逐位一致。无视频全量重解码、无原始数据重写。
耗时 164.27 秒。每个 Parquet 的 SHA-256、计算前后 size/mtime 以及 manifest/info 不变性检查均保留于
资产的 `provenance.json`；这不替代转换阶段的全部视频完整性检查。

## 社区与官方核查

- [OpenPI 官方 README](https://github.com/Physical-Intelligence/openpi#requirements) 给出单卡全量微调
  >70GB，并明确支持 `fsdp_devices` 分片；不能把这个单卡值简单除以卡数当作实测。
- [OpenPI issue 677](https://github.com/Physical-Intelligence/openpi/issues/677#issuecomment-3327382894)
  有 4090 用户报告改用 8-bit Adam 后可启动；发帖者明确没有保证训练质量。这是 PyTorch 社区报告，
  不等价于本项目原 AdamW/JAX 的四卡验证；本次未采用 8-bit 优化器或 CPU offload。
- [RoboTwin 原始文档](https://robotwin-platform.github.io/doc/usage/Pi0.html) 的 full/batch32 表格列的是
  4×A100 40GB 或 2×A100 80GB。搜索摘要曾错误出现 4×4090，原文不支持那个结论，且该页主要针对 Pi0。

## 复现和溯源

本次独立执行目录：`/home/wuyan/lyj/YAM/env-transfer/pi05-full-norm-20260908.mgkPSsmT`。
使用本地基线 commit `edd5cf95d8eaab667258adeded668db47bbea22c` 的代码快照加本次改动，
未覆盖服务器旧 Git 工作区。真实运行文件 SHA-256：

```text
scripts/train.py 27b1f0c271690d6539aa4bc6ee9a542efe472d4226f5c8011bdfd557089a7760
scripts/benchmark_pi05_full.py 9c73e3069c5a5ba4ebef0cdb862d2fac257e75d21a25a74c39168b859179385c
scripts/compute_yam_norm_stats.py a15bf0b26a4e84fe5d5d863307690365a2c1772a2f8e7e215163feae2f6338a6
src/openpi/training/config.py 300ee18be50ae0e4702324e0f8762f85a85d75b6999cf8f23cc35bc394889894
src/openpi/transforms.py d31a2288fb50258a758fd1a3e991e689c15271ffca7636fb7be38f738cc30e0f
```

基础权重 `/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`，已由实际加载器读取。
此前上传只做文件清单/大小比对，未补称全权重端到端 SHA 验证。初始化 seed=0，更新 RNG key=1..10。
运行无 checkpoint 保存和推理部署，相关部署/校准/任务成功率 gate 均未执行。

在现有分配内执行的真实测试命令（`RUN` 表示上述独立执行目录）：

```bash
RUN=/home/wuyan/lyj/YAM/env-transfer/pi05-full-norm-20260908.mgkPSsmT
srun --jobid=2064 --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:4 --time=00:30:00 \
  env PYTHONPATH="$RUN/code/src:$RUN/code/packages/openpi-client/src" \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
  OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
  /home/wuyan/.conda/envs/condapi-yam/bin/python "$RUN/code/scripts/benchmark_pi05_full.py" \
  --checkpoint /home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --output "$RUN/probe-real-b4" --steps 10 \
  --dataset /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train \
  --assets "$RUN/assets"
```

复跑必须使用新 `--output`，脚本拒绝覆盖。长任务由登录节点 tmux 持有 srun，实际工作在 gpu001。
正式训练可复用同一 norm 资产，但 batch 扩大、EMA 开启、持续 dataloader、checkpoint 保存恢复及长期稳定性
需要独立验收；此速度不含这些开销。

记忆维护按 mlops-memory 限定读取 owner，未建立平行缓存。预算账本已准入30,945字节，余1,823字节；
该字节计数不是全部对话 token 计数，宿主完整请求的 tokenizer 强制预算不可见。
