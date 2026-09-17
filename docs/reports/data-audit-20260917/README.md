# 大数据集与随机 10h RTC 子集数据审计（2026-09-17）

> **后续复核更正：请先读 [FOLLOWUP.md](./FOLLOWUP.md)。** 本文是首轮历史报告，不能作为最终验收。后续已取回 1872 + 184 个成功窗口，并完成全量低维分布扫描；首轮 FD 失败与后续显式设置 `ulimit -n 65535` 的成功结果必须一起解读。本文“已完成审计”的说法范围过大，完整视频、RTC mask 和多 worker 对照仍未完成。

本报告只做只读核查，不启动训练，不修改或删除原始数据。比较对象是同一发布版 LeRobot v3 train：完整 4458 集与 seed=42 随机抽取的 468 集 RTC 子集。结论按“已证实 / 可疑 / 当前证据不支持”区分；历史报告的通过结论只适用于同一文件指纹。

## 结论

现有证据**不支持把大集训练异常归因于“Parquet 中存在 NaN/Inf、维度错或明显文件损坏”**。更符合证据的首要差异是训练配置和运行环境：大集故障实验使用 batch=64、workers=8、全量 norm、无 RTC、peak LR=2.5e-5；10h RTC 使用 batch=32、workers=2、子集 norm、RTC delay 0..10、peak LR=1e-5→1e-6。两者还不是同一个训练目标，不能把“子集稳定”当作数据完整性证明。

大集历史故障最早落在 GPU/NCCL/非有限梯度类别，而不是数据读取器报告：

| 运行 | 最早可定位现象 | 证据 |
|---|---|---|
| 旧全量 run | step 16031 后 `NCCL illegal memory access`，GPU1 Xid13/Xid43，Python 段错误 | `docs/reports/training/pi05-recovery-20260909/README.md` |
| r2 全量重开 | step 271 后相同 `NCCL`/CUDA illegal access 与段错误 | 服务器 `training-runs/control/lego_full_b64_r2_20260909/train.log` |
| r3 全量重开 | step 9361 检测到 `grad_norm=inf`，loss 仍有限，随后退出；此前 step 9351 的梯度有限 | 服务器 `training-runs/control/lego_full_b64_r3_20260909/train.log` |
| r4 诊断续训 | 5001、5101 的每个参数叶梯度均有限；约 step 5191 再次发生 NCCL/illegal access | `docs/reports/training/pi05-r4-diag9361-20260910/README.md` 及服务器诊断 JSONL |
| 后续 b32 低压续跑 | `batch=32, workers=2, peak LR=1.25e-5` 的 full-data 续跑在 step100450 仍为有限指标，用户在约 step100531 主动暂停；它从已有 step100000 checkpoint 续跑，只有约 450 个新 step | 服务器 `training-runs/control/lego_full_b32_stable_20260916/srun.log`、`pause_20260916.json`；不能当作完整长跑通过 |
| RTC 10h | 2140→2141 已恢复到约 step 29681，loss/grad_norm/param_norm 全部有限，显存约 9.4 GiB/GPU；当前作业仍由服务器状态决定 | `docs/reports/training/rtc-base-10h-20260916/README.md`、服务器 `metrics.jsonl` |

这组故障的形态不能由“某个视频读失败”解释：没有固定文件的解码错误；唯一明确的 DataLoader 事件是后续的文件句柄耗尽。r2 在约 271 步就触发 GPU 非法访问，r3 则在长时间正常更新后出现 grad overflow。GPU1、CPU DIMM_B1 corrected ECC 和驱动/通信仍是未决因素；历史 memcheck 超时，不能当作根因已定位。

按故障类别核对后，最早位置和证据边界如下：

| 类别 | 最早/最清晰位置 | 判读 |
|---|---|---|
| 读取/解码 | 没有发现固定文件的 Parquet 或 PyAV 解码错误；另有一次 `DataLoader worker OSError: [Errno 24] Too many open files`（2026-09-11 20:34:15，batch32、workers8） | 这是文件句柄/并发资源故障，不是数据内容损坏；同一 run 更早已有 GPU/NCCL 事件 |
| 卡顿/内存 | batch96/128 的短容量测试 OOM；长跑中没有“训练显存 OOM”证据 | 容量边界与长跑故障不能混为一谈；workers、视频并发和预取压力仍需单独复现 |
| loss/梯度/参数非有限 | r3 全量 step9361 首次 `grad_norm=inf`（step9351 有限）；后续 batch32 链 step134061 首次被训练代码检测到 `grad_norm=inf`，loss/param 仍有限 | 非有限点没有 episode/frame 来源，不能回溯到具体样本 |
| CUDA/Xid/NCCL | r2 step271；随后 batch32 在 step37761（GPU1 Xid13/MMU）、65531/67670/77490 等重复 `illegal memory access`/段错误 | 发生前 loss、grad、param 多为有限；这是当前最重复的长跑故障类别 |

另外，完整链曾出现有限但极大的梯度（如 step69111 `1.74e18`），RTC 子集也在 step24351 出现 `2.17e9` 后继续运行；因此“梯度很大”本身既不是大集独有，也不能当作坏样本证据。

可复核的服务器原始记录包括：`lego_full_b32_xid13_recovery_20260911/recovery_from.json`（step37761/Xid13）、`lego_full_b32_xid13_recovery_20260911/auto_resume_20260911T204200+0800.json`（FD 耗尽）、`lego_full_b32_xid13_recovery_20260914_nccl_129501_hl10/failure_event_2026-09-14T171303_0800.json`（step134061 非有限），以及 `lego_full_b32_xid13_recovery_20260912_nccl_65531_hl/recovery_event_20260912T094700+0800.json`（step65531 NCCL）。这些记录均保留在 `/home/wuyan/lyj/YAM/training-runs/`，未复制或改写。

## 审计对象与指纹

- 发布 train：`/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train`，4458 episodes、10,374,181 frames、30 FPS、单任务 `sort the legos into containers by color`。
- 发布 train `meta/info.json` SHA-256：`ba88887ff84544eb130708cdf8c0ce965b53f099fdad07286a57f41883211851`。
- 发布 train `conversion_manifest.json` SHA-256：`9844c0b86b93fd40adf251275d770899ba413800d2a0cba11d4e9bdae40544dc`。
- RTC 子集：468 episodes、1,082,232 frames、10.020667 h；selection 的源 manifest SHA-256 `9844c0b86b93fd40adf251275d770899ba413800d2a0cba11d4e9bdae40544dc`，子集 norm `norm_stats.json` SHA-256 `606d5c69e56aadb273ed3882ba9b62e11978a4222d8ff1e827e541bddd112893`。
- 源映射保持转换后的 `episode_index` 与 `source_episode_index` 两列，未把源 ID 当作连续索引。完整集减子集为 3990 episodes、9,291,949 frames。
- 转换 manifest 声明 10 个视频修复记录，均为已完成的逐帧数不足修复：源 episode `9750, 9873, 15193, 20234, 28675, 30059, 79470, 92573, 113716, 119779`；对应发布 episode `323,324,536,737,1016,1058,2734,3187,3913,4131`。RTC 子集与修复集交集只有发布 episode `3187`，因此子集并非完全绕开修复数据。

## 两次实验的训练合同对齐

| 项目 | 大集故障链（r2/r3） | RTC 10h（2140→2141） |
|---|---|---|
| 代码/数据 | `a53bb001…`；同一发布 train 根目录，4458 集 | `3dd9c820…`；同一根目录但显式 468 集清单 |
| 初始权重 | 新开链从 Pi0.5 base `/home/wuyan/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`；后续 r4/恢复链多从 checkpoint 接续 | 从同一 Pi0.5 base 开始，2141 从 10000 checkpoint 接续 |
| norm | 全量 norm `c496b738…d62305be9` | 子集 norm `606d5c69…ddd112893` |
| RTC | `rtc_training_max_delay=0` | `rtc_training_max_delay=10`（0..10 前缀/mask） |
| batch/workers | 初始 r2/r3 `64/8`；后续对照链有 `32/2` | `32/2` |
| 优化器/LR | Adam（b1=.9、b2=.95、clip=1）；r2/r3 eps=`1e-8`，peak LR=`2.5e-5`→`2.5e-6` | Adam（b1=.9、b2=.95、eps=`1e-6`、clip=1）；peak LR=`1e-5`→`1e-6` |
| 精度/并行/节点 | BF16、FSDP4、`gpu001` 四卡 RTX4090；失败步从 271、9361 到后续恢复链 | BF16、FSDP4、`gpu001`；2140/2141 已完成 30k，2147 正在接续 |

因此“同一 base 权重”并不等于同一实验：大集历史链包含不同 checkpoint、batch、workers、eps、LR 和节点故障状态；RTC 还改变了 norm、episode 集合和 RTC 前缀处理。

## 已完成的完整性证据

1. 2026-09-09 服务器全量低维审计：4458 train + 69 val 全部 `validated_lowdim`；10,374,181 train frames；Parquet SHA-256 与 norm provenance 一致；state/action 均为 Nx14 且有限。原始审计回执见 `docs/reports/training/pi05-crash-audit-20260909/data-audit.json`。
2. 发布转换 manifest 的 14D schema、30 FPS、task、episode 长度和 frame/index 约束一致；转换使用 `action_mode=absolute`，训练 transform 再做关节 delta，夹爪保持 absolute，mask 为 `(6,-1,6,-1)`。
3. 完整 norm provenance 与子集 norm provenance 都记录了 H=50、仅 train、对应 manifest/info 指纹和尾帧重复规则。完整集 norm SHA-256 为 `c496b738470e432b9da02b22a45c8c309b3db8412d73723944a7cbdc62305be9`；子集 norm 仅用于 RTC 合同。
4. 已有真实 LeRobot loader 等价性抽样各 12 个：完整集覆盖发布 episode 0/2229/4457 的首、中、`length-H` 和末帧；RTC 子集覆盖 episode 3/2330/4456 的同样位置。三路图像均为 `(224,224,3)`，state/action 与独立数值路径逐元素相等，task 映射正确。
5. 用真实发布 train 数据重算 RTC 468 集、1,082,232 帧的 norm，Parquet、episode 边界、帧序号、时间戳和 loader 遍历均无错误；固定默认 `block_size=1024` 后，重算 norm 与 29000 checkpoint 记录逐值一致（输出 SHA-256 `606d5c69e56aadb273ed3882ba9b62e11978a4222d8ff1e827e541bddd112893`，仅末尾换行可见差异）。`block_size=4096` 会改变近似 quantile，比较 norm 时必须固定 block size。
6. 对 10 个历史修复 episode 的 30 路发布视频做了独立 PyAV 全量解码（21.73 s）：每路帧数都等于 episode length，PTS 单调，无 `missing_pts`、解码异常或越界。发布 train 当前有 4458 个 Parquet、13,374 个视频文件（三路齐全）；修复集的这次复核不能替代全量视频复核。

## 覆盖、失败与未检查清单

| 对象 | 覆盖/结果 | 证据边界 |
|---|---:|---|
| 发布 train Parquet | 4458/4458 episode，10,374,181/10,374,181 帧 | 既有低维完整性通过；本轮 H50/norm 复算只覆盖子集 |
| RTC 子集 | 468/468 episode，1,082,232/1,082,232 帧 | 选择清单、源 ID 映射和 norm 指纹一致 |
| 完整集减子集 | 3990 episode，9,291,949 帧 | 已建立映射，尚未对补集做完整视觉/变换遍历 |
| 历史修复视频 | 30/30 路（10 episode×3）逐帧解码通过 | 不能外推到全量视频 |
| 真实 loader 窗口 | 24 个持久化等价窗口（完整集12、子集12）通过 | 覆盖首/中/`length-H`/末帧；不是全量窗口 |
| 468 集全选择 loader probe | 读到完整选择时触发 `OSError: [Errno 24] Too many open files`，未生成 summary | 只读复现了 FD/reader 压力；尚无 workers=0/2/8 对照，不能单独归因训练实际 worker |
| 全量三路视频（首轮） | 30/13,374 路有独立当前运行时回执 | 转换 manifest 声明过 `full decode verified`，但原始回执不完整 |

全选择 probe 的失败日志为 [loader-probe-2146.log](./loader-probe-2146.log)（服务器原路径 `/home/wuyan/lyj/YAM/training-runs/control/yam-loader-probe-20260917/slurm-2146.log`）；失败发生在读完整选择之后、写摘要时，未指向某个固定 episode/frame。它与历史 `workers=8` 的 FD 耗尽相互印证“读取压力”方向，但仍需在同一版本、显式关闭 reader、workers=0/2/8 下做最小对照。

## 尚未完成或只能判为可疑的部分

- 独立的全量三路视频 audit 产物 `full.json` 不存在，保留的 `full.json.incomplete` 为空；不能用它声称 13,581 路视频都已被独立报告覆盖。转换 copy 流程在首次复制时会逐路 `validate_video`，manifest 将其记为“full decode verified”，但保留的 `conversion.log` 缺少 `SPLIT_VERIFIED=train`/`CONVERSION_COMPLETE` 回执，因此应把“转换时逐路检查”与“可复核的独立全量报告”分开记录。
- 现有 loader 等价性只覆盖 24 个窗口，未遍历完整 4458 集的所有 H50 起点，也没有在相同配置下对照 workers=0/2/8 的资源压力。
- 全选择 468 集 probe 已在读完整选择后因 FD 耗尽退出，未能保存每个窗口的最终张量统计；需要修正 reader 生命周期后再做分层遍历。
- 尚未取得 step 9361 的实际 batch episode/frame 来源；r4 诊断只到 step 5101。因而不能把某个具体 episode/frame 指为 grad overflow 根因。
- 尚未完成按 episode/维度的相邻帧跳变、action-state delta 分位数、归一化后范围和视频质量分布比较；已有“全部有限”不等于“无语义异常”。manifest 仍保留 287 集 state gripper 略超名义上限的 warning，不应自动裁剪或删除。
- `scripts/audit_yam_subset.py` 的输入是旧原始布局 `raw_root/{manifests,train,data, videos}`，直接对发布 v3 train 目录运行会返回 `pending_upload`，不是发布目录的有效完整性结论。对发布 v3 应使用转换 manifest、`compute_yam_norm_stats.py` 和真实 LeRobot loader。

## 最小复核命令（服务器、只读）

以下命令不会启动训练。大规模视频解码应在获准计算资源执行，先在新报告目录保存 stdout/stderr；不要覆盖旧报告。

```bash
# 原始布局：检查 manifest、Parquet、三路视频是否缺失/近期变更
python3 /home/wuyan/lyj/YAM/code-snapshots/rtc-10h-3dd9c82/scripts/audit_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting \
  --inventory-only --progress-every 100

# 原始布局：低维全量复核；不解码视频
/home/wuyan/.conda/envs/condapi-yam/bin/python \
  /home/wuyan/lyj/YAM/code-snapshots/rtc-10h-3dd9c82/scripts/audit_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting \
  --lowdim-only --progress-every 100

# 发布 v3：只读计算对应清单的 H50 / delta mask norm，并核对 provenance
/home/wuyan/.conda/envs/condapi-yam/bin/python \
  /home/wuyan/lyj/YAM/code-snapshots/rtc-10h-3dd9c82/scripts/compute_yam_norm_stats.py \
  --dataset /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train \
  --train-episodes-file /home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/episodes.json \
  --output /tmp/yam-audit-norm-20260917

# 发布 v3：按 episode/维度统计跳变和 action-state；输出必须是新文件
/home/wuyan/.conda/envs/condapi-yam/bin/python \
  /home/wuyan/lyj/condapi/scripts/audit_yam_distributions.py \
  --dataset /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train \
  --selection /home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/episodes.json \
  --output /home/wuyan/lyj/YAM/training-runs/control/data-audit-20260917/numeric_distribution.json

# 发布 v3：真实训练输入链的边界窗口探针；不创建 trainer/optimizer
/home/wuyan/.conda/envs/condapi-yam/bin/python \
  /home/wuyan/lyj/condapi/scripts/probe_yam_loader.py \
  --dataset /home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train \
  --selection /home/wuyan/lyj/YAM/training-assets/lego_rtc_10h_20260916/episodes.json \
  --episode-stride 100 --output /home/wuyan/lyj/YAM/training-runs/control/data-audit-20260917/loader-probe
```

真实管线 probe 的最低验收标准是：`LeRobotDataset(..., episodes=...)` → `YamInputs` → `DeltaActions` → `Normalize` → `ResizeImages/TokenizePrompt/PadStatesAndActions`，在每个抽样窗口保留 `episode_index/frame_index`，检查 H50 末尾重复、三视角 shape、每个中间张量的 finite 和 norm 后范围；先 workers=0，再在资源允许时只改 workers=2/8，分别记录错误、延迟、CPU/共享内存和文件句柄。不能把这个 probe 叫训练 smoke。

## 归因与下一步

- **已证实**：大集和 RTC 子集不是单变量实验；完整低维 Parquet 通过既有全量审计；历史故障首先表现为 GPU/NCCL/Xid 或 grad_norm=Inf；真实 loader 的 468 集全选择只读 probe 复现了 `Too many open files`；10 个修复 episode 中有 1 个进入 RTC 子集且修复视频可完整解码。
- **可疑**：reader 生命周期与 workers/batch/显存、视频并发压力；全量 norm 与更高 LR；未被当前 468 集覆盖的 3990 集中可能存在视觉或相邻帧异常；GPU1/驱动/主机 ECC。FD 复现支持读取压力方向；b32/workers2 的低压续跑可连续产生有限指标，但历史仍有更晚的 NCCL/Xid，说明降压可能缓解而非证明根因；每项都需要单因素证据。
- **当前证据不支持**：把所有问题归为某一坏 episode、把有限值当成语义正确、把 Xid 当成数据损坏、或把子集训练稳定当作全量数据已验收。

下一步按收益排序：先完成真实 loader 的分层窗口 probe，并保存异常源 episode/frame；再对同一 probe 做 workers=0/2/8 的资源对照；若仍无数据异常，再把 GPU/NCCL/节点健康作为主线。只有发现可重复、固定到具体文件/帧的失败，才进入单条视频隔离或按仓库规则修复；不凭离群值批量删除。
