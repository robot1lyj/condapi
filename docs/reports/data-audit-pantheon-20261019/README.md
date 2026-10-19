# Pantheon ABC-130k 标记与现用 YAM 数据核对

**结论：有重合。** 本次核实 7 条 Pantheon 隔离记录：乐高完整 train 中 4 集、乐高 val 中 1 集、耳机 clean-v2 train 中 2 集；其中现用乐高 50h 含 2 集，耳机 50h 含 2 集。按用户“有问题就不用”的要求，已登记整集排除，并生成独立的剩余 ID 清单；原件、旧选集、norm 和历史运行记录保留。

**范围：存在性核查，不是全库清洗。** 读取了 128 个标注分片的首条身份头，进一步检查首条属于两个目标任务的 7 个分片。抽样由分片边界决定，不是随机抽样；以下数量均为已确认下限。未逐条遍历其余约 291 GB 标注，未匹配 episode 的质量未知，不能把剩余清单称为 clean 或 DATA_READY。200 集专项和旧 10h/20h 在这 7 条记录里没有命中，不代表其余标注没有问题。

## 结果与索引

| 任务 / split | 本地发布索引 | 上游 LeRobot 源索引 | 原始 UUID 前缀 | 当前选集命中 | Pantheon 标记 |
|---|---:|---:|---|---|---|
| 乐高 train | 255 | 8096 | `3ad71652` | 完整来源池 | 时间戳缺口、重复帧、画面冻结 |
| 乐高 train | 565 | 15948 | `4792c2ab` | 50h | 时间戳缺口、动作不连续、视觉变化与动作不符 |
| 乐高 train | 3490 | 101616 | `0a4dea08` | 50h | 相机与状态时间戳缺口 |
| 乐高 train | 3919 | 113728 | `cadb3707` | 完整来源池 | 时间戳缺口、视觉变化与动作不符 |
| 乐高 val | 17 | 455 | `3189040f` | 69 集原验证集 | 时间戳缺口、视觉变化与动作不符 |
| 耳机 train | 158 | 15503 | `7e704c2d` | clean-v2 50h | 相机与状态时间戳缺口 |
| 耳机 train | 556 | 52311 | `dfd5f5fc` | clean-v2 50h | 相机与状态时间戳缺口 |

7 条的 `quality_status` 均为 `quarantined`。完整 flags、版本、UUID、来源文件散列和数值证据见 [identity-matches.json](identity-matches.json)。标记来自 Pantheon，未在本次独立重跑全部 QC；其 `_corrected` 表示派生有效性 mask，不证明本仓库使用的原文件已修复。本次按用户要求保守整集排除，不把这些自动标记全部断言为已独立证实的上游损坏。

## 身份核验方法

现用来源固定为 `lerobot/abc_130k_v3_train@68651e4929d9fb00f798937b2d62617cab5c771d`、`lerobot/abc_130k_v3_val@46ca817c39d3df08115a75c31b3cd3867e3d875a`。原始 UUID 未随 LeRobot 端口发布，不能用分片数字、数字索引、UUID 排序或任务名直接关联。实际读取已反证“相同任务内按 UUID 排序即可对应”。

用户开通访问后，使用本机保存的 HF 身份对原始库进行获准的范围读取；服务器 SSH 可连接，但没有保存 HF 身份且直接 Hub 连接失败。本次没有把 token 发送或保存到服务器。原始 MCAP 固定为 Pantheon 所声明的 `XDOF/ABC-130k@29136bc9b9e38d320b00ffcddbbe4cd0e3278c58`。

以任务、split 和时长 ±0.5 秒筛出 LeRobot 候选，在服务器只读取候选低维数据。每条最终唯一匹配，比较首 3 帧、中间 3 帧、尾 3 帧的 state/action：共 9×28=252 个标量，每个臂/夹爪向量均能在对应原生通道时间窗内找到，转 FP32 后最大误差均为 **0**。绑定原始 MCAP 的 Hub LFS SHA256、读取的源 Parquet SHA256，以及源 manifest SHA256。

这是数值抽样建立的轨迹身份，**不是**全轨迹逐值验证、原始采集同步验收或实机标定。原生 MCAP 与 Pantheon canonical 流、LeRobot 30fps 流的帧号不能直接互换。本次没有使用其 EE delta 动作替换现用关节 action，没有迁移其帧 mask 或裁帧。

## 排除后的独立清单

| 池 | 原集数 | 本次排除 | 剩余集数 | 剩余帧数 |
|---|---:|---:|---:|---:|
| 乐高完整 train | 4458 | 4 | 4454 | 10,364,926 |
| 乐高 50h | 2337 | 2 | 2335 | 5,395,985 |
| 乐高 val | 69 | 1 | 68 | 162,730 |
| 耳机 clean-v2 train | 1384 | 2 | 1382 | 5,397,092 |
| 耳机 val | 17 | 0 | 17 | 62,173 |

[impact.json](impact.json) 绑定父数据/选集身份，并指向 [retained-episode-ids/](retained-episode-ids/) 的新清单。ID 均属于各自父 LeRobot 数据集，没有重新编号。未补集、重算 norm 或改变训练配方；新清单均标为 `KNOWN_EXCLUSIONS_APPLIED_NOT_FULLY_SCREENED_NOT_TRAINING_READY`。后续采用前还需完成所需质量审核、norm 合同和该轮配方确认。

排除登记归 [yam-excluded-episodes.json](../../../configs/datasets/yam-excluded-episodes.json)，以源 repo/revision/split/source index 精确匹配。选择器跳过已核实异常；清洗入口在解码前排除已登记的 train/val 成员，并把原因写入新 selection；Pi 原生入口在写入训练合同前拒绝仍包含异常的完整数据或子集，不静默改变训练成员。没有来源映射的 legacy/native 数据不在该检查的认证范围内。

本次实现与报告保存在当前功能分支，未同步服务器代码、重启/取消作业或部署设备。服务器上现有不可变代码快照尚无新拦截逻辑；采用新代码和经审核的新清单后才生效。历史 checkpoint 已见过的训练数据不会因新排除记录自动消失。

## 来源与验证

- [原文](https://pantheon.inc/research/we-looked-at-the-data)。
- [固定标注 schema](https://huggingface.co/datasets/sambhav-pantheon/XDOF_ABC-130k_pantheon_annotated/blob/08345b7cbbb19581bb04d91940ef2e31e273f5d2/v1/README.md)。标注 revision：`08345b7cbbb19581bb04d91940ef2e31e273f5d2`。
- [抽查的 QC 原字段](public-qc-examples.json)、[全部分片的首条身份头](shard-headers.json)、[数值身份与来源散列](identity-matches.json)。完整范围读取窗口保存在本地被 Git 忽略的 `artifacts/pantheon-audit-20261019/`。
- 31 项轻量元数据/配置/源审核测试通过，覆盖版本与split隔离、错误ID混用、异常夹带阻断、正常成员保留及不解码已排除来源。6 个涉及的 Python 文件 Ruff 检查与格式通过，`git diff --check` 通过；7 条身份、8 份剩余清单、来源计数、拦截和报告本地链接复核通过。未运行训练循环或训练 smoke。
