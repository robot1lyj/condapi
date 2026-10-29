# 服务器存储盘点（2026-10-29，只读）

用户要求列出占用，供其选择哪些不再需要。**本轮没有删除、压缩、截断或迁移任何服务器数据、权重、缓存和日志。** 下列大小统一用十进制 GB/TB，来自 `df -B1`、低优先级 `du -x -B1` 和文件 `st_blocks×512`。活动目录仍在增长，数值为本轮观察快照，不是配额承诺。

## 总体

| 范围 | 占用 / 容量 |
|---|---:|
| `/home` 共享文件系统容量 | 15.87 TB |
| `/home` 全部用户已用 | 14.04 TB，94% |
| `/home` 当时可用 | 1.03 TB |
| 我们的 `/home/wuyan/lyj` | 7.69 TB |
| 其中 `YAM` | 7.12 TB |
| 其中 `YAM/training-runs` | 6.71 TB |
| 其中 `YAM/YAM_data` | 372.40 GB |
| 另计 `/home/wuyan/.conda` | 73.27 GB，其中环境70.10 GB、包缓存3.17 GB |
| 另计 `/home/wuyan/.cache` | 21.19 GB |

父目录包含子目录，上表不能逐行相加。共享盘总体使用量不等于我们的使用量。此次没有扫描其他账号目录。

## 优先供用户选择的大项

以下相对路径统一以 `/home/wuyan/lyj/` 为根。

| 项目 | 路径 | 当前占用 | 判断与保留边界 |
|---|---|---:|---|
| 历史 CUDA 崩溃转储，27份 | `YAM/training-runs/control/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022/incidents/`内的`.nvcudmp` | **2.755 TB** | 最优先审查，属于诊断证据，不是模型权重；可由用户选择保留代表性样本。27份都属于旧作业2574–2635，没有当前2637的转储；不能因此删除整个incidents目录。 |
| 两个历史巨型训练日志 | `YAM/training-runs/control/lego_full_b32_after_xid13_20260910/train.log`；`.../lego_full_b32_xid13_recovery_20260911/train.log` | **114.98 + 50.63 GB** | 可考虑保留必要证据后压缩或选删。尚未独立核查全部后续分析对它们的引用。 |
| 乐高50h历史检查点 | `YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921` | **1.580 TB** | 共49个数字step目录，每份约32.3 GB；按部署、评估、续训引用选择保留版本，不能自动整组删除。 |
| 旧耳机50h检查点 | `YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_20261012` | **774.24 GB** | **114000/params 是当前耳机作业2637的初始参数路径，必须保留。** 其他历史step和32.27 GB的`85000.orbax-checkpoint-tmp-16`可单独审核；临时目录名本身不构成删除授权。 |
| 乐高10h检查点 | `YAM/training-runs/pi05_yam/lego_pi05_rtc_base_10h_20260916` | **516.22 GB** | 旧实验，是否保留取决于后续基线/回放比较需要。 |
| 乐高专项10k检查点 | `YAM/training-runs/pi05_yam/lego_pi05_rtc_separation_r1_10k_20261014` | **322.47 GB** | 保留仍用于部署/比较的版本，其余step可供用户选择。 |
| XR-1运行产物 | `xiaomi-robotics-1/runs` | **460.98 GB** | 正式`lego-50h-eef-v1-20261010`203.10 GB；capacity试跑三组122.48/67.70/67.70 GB，若不再需要可优先审核。 |
| OpenWAM检查点smoke产物 | `openwam-runs/yam-20h-official-checkpoint-smoke-20260923-2174` | **24.84 GB** | 旧试跑，按后续验证需要决定。基座模型另占59.04 GB，不混同为普通日志。 |
| 旧batch容量测试副本 | `YAM/env-transfer/pi05-batch-limit-20260908.eLaflYGQ/b64-stream` | **32.34 GB** | 历史测试目录，需确认是否仍有引用；`env-transfer`合计36.96 GB，并非全部是安装包。 |
| 旧未清洗耳机派生尝试 | `YAM/YAM_data/processed/earbuds_lerobot_50h_v1_20261012` | **14.31 GB** | 旧尝试现场，现用为clean-v2；按保留原件/失败现场边界，仅列候选，不删除。 |

[candidates.json](candidates.json) 保存27个实际CUDA转储和两个巨型日志的精确完整路径、已分配字节、逻辑大小和mtime。转储合计2,755,454,279,680字节，两个日志合计165,604,872,192字节；两组理论合计约 **2.921 TB**。这只是供选择的最大量，没有执行释放空间操作，也不默认删除所有诊断证据。

## 当前应保护的目录

- 当前耳机训练2637的输出`YAM/training-runs/pi05_yam/earbuds_pi05_rtc_base_50h_clean_v2_excl_pantheon_retry4_20261022`约322.49 GB，不整组删除。
- 当前数据`YAM/YAM_data/processed/earbuds_lerobot_50h_clean_v2_20261013`53.28 GB；原始双任务`ABC-130k-two-tasks`173.54 GB；乐高派生`lego_lerobot_v1_20260907`93.19 GB；`derived`37.70 GB。来源、失败现场和既有下载仍遵守04保留边界。
- 当前norm资产`YAM/training-assets/earbuds_rtc_50h_clean_v2_excl_pantheon_20261021`和旧耳机`114000/params`。路径实际从作业2637启动文件读取，不从旧文档猜测。
- 当前incident目录`.../incidents/job-2637-attempt-20261029T005619Z`仍在写入；其中`nccl.gpu001.2517947.log`观察时约30.68 GB，`numerical/steps.jsonl`约1.18 GB。不删除或截断正在写的文件；本轮未更改诊断级别或训练配方。
- 本次折纸盒`YAM/YAM_data/ABC-130k-paper-box-20261029`仍下载/筛查中，不列清理项；状态见[下载回执](../data-download-paper-box-20261029/README.md)。
- 当前训练环境`/home/wuyan/.conda/envs/condapi-yam`和下载数据环境`/home/wuyan/.conda/envs/yam-xr1-data`。

## 其他占用

`pld-grasp`7.08 GB，`openwam-install-7c5861e`4.43 GB，`evo1-install-2774d9b`2.66 GB。YAM代码快照合计约0.43 GB，代码主目录约0.09 GB，清它们收益很小。

缓存中OpenPI12.45 GB、Hugging Face7.00 GB、pip1.63 GB；Conda包缓存3.17 GB。模型缓存可能是训练/离线恢复需要的资产，不作为普通下载缓存自动删。若需要小范围释放空间，可另行选择确认不再需要的pip/Conda安装包缓存，但它们远小于历史CUDA转储和巨型日志。
