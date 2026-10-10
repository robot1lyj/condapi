# 乐高分拣 HIL1009：RL 准备审核（2026-10-10）

## 本轮决定

用户最终选择：第一轮以约 20 条**固定 π0.5 检查点、无人干预的真实 rollout** 为主。连续保存成功、部分成功、失败；每集结束标开始待分拣数 N、最终新增正确数 C、分错数 W、结束原因。未完成数由 N−C−W 得到，正常任务终点奖励 C/N；紧急停止、故障和待审核均保留未知奖励。原有 15 条 HIL 保留作补充，不默认混入首次训练。没有抽取/转换/混入旧公开成功示范。

原生 Pi 全量训练继续运行。本轮没有停止作业、占用训练 GPU、执行本地训练或训练 smoke，也没有改动/部署 Thor 或 3588 控制代码。当前无需用户停 4090。

## 已完成数据准备

原始来源：3588 的 `/data/YAM/data/episodes/9fc4ed25-cba0-423a-b8e6-2f4b3d15e3aa`，仅只读访问明确授权的数据目录。复制 15 个已关闭集、155 个 manifest/已提交 segment 文件，1,693,414,850 字节。按来源快照核对 SHA256，原件保留。

本地独立目录：`/home/wuyan-lyj/condapi-data/rl/lego-hil1009-20261010-mvd2kson`。完整审核归 `audit_v2/audit.json`、原件身份归 `source_snapshot.json`、RL 结果标签归 `audit_v2/rewards.json`。原始 manifest 的 `outcome=unknown` 未改写；旁路 R=1 来自用户“保存的完成 HIL 成功集”的约定，不能称独立视觉认证，也不代表每帧动作最优。

| 项目 | 结果 |
|---|---:|
| 原始集数 / 帧数 | 15 / 37,602 |
| 名义录制时长（30FPS） | 20.89 分钟 |
| 无人工帧的集数 | 4 |
| 含人工帧的集数 | 11 |
| 人工控制 intervention ID 数 | 27 |
| 键盘 command ID 数 | 151 |
| 可用于 DAgger 的有效人工帧 | 2,812 |
| 无效观测/状态/动作等被屏蔽帧 | 481 |
| 不跨坏帧、缺 tick、epoch 切换的相邻 transition | 36,661 |
| 能对齐末尾有效观测的集数 | 14 |
| 无后续人工接管且终点可对齐的校准 transition | 15,561 |

每个已提交 segment 的三路视频全部解码，核对帧数、PTS 单调性、索引范围与尺寸；保留所有原始等待帧及真实单调时钟。没有把 151 次键盘命令压成 151 条连续快速动作。RL 使用实际 `submitted_action`，不是测量下一状态，也不是未经约束的模型输出。14D 顺序和单数 action 保持原 YAM 合同，未套用其他硬件单位。

外部只读 YAM 转换工具：`/home/wuyan-lyj/YAM/yam-abc-reproduce/scripts/convert_lerobot.py`，Git HEAD `2dc603595daf5ebe33c58c5aa0279c52fcb9740c`。使用其既有数据环境，完整轨迹模式（非 expert-only），写新目录 `lerobot_full/`。相关转换文件无本地修改，具体文件 SHA256 归 `converter_provenance.json`。15/15 转换完成，37,602 帧。额外逐项核对转换后的 action、有效 observation.state、来源、tick、control_time；45 路集级视频的 H264 packet payload 在拼接前后 SHA256 一致，没有重编码图像。验证归 `conversion_verification.json`。

## 两项异常，不能隐藏

1. `669afd1a7b604c52b0b47edc17cfbde2`：最后两帧观测/状态无效，没有可对齐的终止 transition。原始数据和前面的有效 transition 保留；其 `reward_valid=false`，不能把终点奖励悄悄挪到较早帧。后续审核不能靠插值伪造最终观测。
2. `95fb37024bdd449981eb6f988e87f09c`：原帧 2549–2551，保存的 policy_action 与对应原始回复行在右夹爪维（索引13）最大差约0.0015304，单位为归一化开度，关节维无差异。实际提交动作有保存，可保留作物理 action critic 候选；不能把这些行直接认证为严格的历史 actor 行为残差。其余策略行核对没有发现同类差异。

所有清洗只生成派生索引/数组；原件未删、未裁帧。没有用 HIL 整集 R=1 作为“原策略无人帮助的成功率”；人工救场前的 MC 回报不作为自主完成的校准证据。

## 代码和验证边界

新增 `packages/residual-rl/`，独立于旧 PARTS/RLT。参考 [PLD](https://arxiv.org/abs/2511.00091)、[Cal-QL](https://arxiv.org/abs/2303.05479)；实现冻结 actor 的校准保守 critic 初始化、物理 action 双 Q、零均值残差 actor、后续 SAC 更新、真实秒数折扣、版本与服务器计算节点限制。**这是项目适配，不是原论文逐项复现**；未实现 BC actor 预热。

新增数据脚本：`audit_hil.py`、`verify_conversion.py`、`label_rollouts.py`。终点计数不要求每次抓放标注；标签与 episode 身份对应。纯数组/数据合同测试 8 项、限定 Ruff/AST/差异检查；没有 Torch 前向、反向或优化器执行。GPU 数值检查尚未进行。

当前 `training_ready=false`，不能加载生产残差 actor。缺失条件包括：新无人 rollout 与每集结果、固定检查点/norm/推理模式身份、真实冻结三视角特征、同版本参考动作缓存、审核后的物理残差范围，以及短执行段/RTC 队列与实际动作约束的接口验收。不能用零填充/随机视觉特征替代，也不能把旧 30Hz 相邻记录伪装成 H50 已执行决策。

## 论文与数据选择依据

[PARTS](https://arxiv.org/html/2609.21788v2) 用离线专家数据先 SFT 基础 π0.5，再在线采集瓶颈局部尝试训练 TD3+BC 残差，周期性成功重加权重训。不能称其第一轮就是拿未参与 SFT 的 20 条公开成功示范离线训练专家。[PLD](https://arxiv.org/html/2511.00091v1) 的实验则用每任务 50 条基础策略成功 rollout 初始化，之后进行在线 RL。成功数据能起步；这些结果不证明只用 20 条成功数据、完全没有后续探索就能学出可靠纠错策略。

服务器现有 train split 实际 4,458 集，当前 50h 合同用了 2,337 集；剩余 2,121 集、约 46.05h。仅说明未进该份训练合同，不声称从未被历史模型见过。原验证集未动。用户本轮改为优先新的无人 rollout，所以未进一步抽取旧示范。

## 已完成上传和填写模板

服务器新目录：`/home/wuyan/lyj/YAM/YAM_data/derived/rl/lego_hil1009_20261010_mvd2kson/bundle`。打包原始15集、原生LeRobot转换、清洗数组、审核与来源身份，共327个文件。服务器重新验证2,936,760,320字节归档与327个成员SHA256，通过后保存父目录 `UPLOAD_RECEIPT.json`。归档SHA256为 `9dabd3900d81f697711ce66bfbc9156348d08f20406f56408717e769c475e9e3`。上传前从IPC重新核对155个来源文件均未变化。回执记录本地到服务器的路径映射；没有改写原始provenance以冒充服务器原生采集。

用户填写模板：`/home/wuyan-lyj/condapi-data/outputs/lego-rl-20261010/乐高分拣_rollout记录表.xlsx`。20行，每集填录制身份、N/C/W、结束原因；未完成数与奖励自动算。检查点整批填一次。正常结束/到时结束允许部分成功，紧急停止/设备故障/待审核不自动生成奖励。验证过空值、零正确、全部正确、部分成功、错误计数和首/中/末行引用；导出保留40个公式、3项输入校验和冻结表头，渲染无截断。表格中的检查点简称仍需与实际模型资产核对。

准备包不是可直接训练或部署的READY。
