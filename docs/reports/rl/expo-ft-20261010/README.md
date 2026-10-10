# EXPO-FT 论文审读与旧RL退役（2026-10-10）

## 结论与入口

推荐以EXPO-FT作为新RL算法基线，并针对已有30 Hz、training-time RTC架构研究Real-Time EXPO-FT适配；首版采用固定联合行为版本、episode间异步学习发布。两任务分别为乐高按颜色分拣、蓝牙耳机入充电盒。服务器训练、Thor推理、YAM控制/记录三方分离。**本轮交付设计与删除旧代码，不声称新RL已可运行。**

- [系统架构与实施阶段](../../../01_system_architecture.md#expo-ft-强化学习架构2026-10-10设计草案)
- [学习机制、公式、参数差异与评估](../../../03_training_and_evaluation.md#expo-ft-学习与评估设计2026-10-10)
- [双任务数据与Replay合同](../../../04_data_contracts.md#expo-ft-双任务数据合同2026-10-10)
- [两阶段推理、传输与行为束接口](../../../05_inference_and_rollout.md#expo-ft-三方接口草案2026-10-10)
- [客户端重构提案与待审核任务正文](../../../reference/expo_ft_client_proposal.md)
- [交互架构图](../../../diagrams/expo-ft-architecture.html)

## 下载与一手来源

| 来源 | 固定版本与本地文件 |
|---|---|
| EXPO-FT | [论文2605.25477v1](https://arxiv.org/abs/2605.25477v1)；[下载PDF](../../../reference/expo-ft/2605.25477v1.pdf) |
| 实时扩展 | [论文2609.18207v1](https://arxiv.org/abs/2609.18207v1)；[下载PDF](../../../reference/expo-ft/2609.18207v1.pdf) |
| 作者实现 | [pd-perry/expo-ft](https://github.com/pd-perry/expo-ft/tree/21fe3d3b7d913c80836496817965932f49e0aedf)，固定commit `21fe3d3b7d913c80836496817965932f49e0aedf`；[源代码归档](../../../reference/expo-ft/expo-ft-21fe3d3.tar.gz) |
| 下载身份 | [sources.json](../../../reference/expo-ft/sources.json)，原始字节sha256/大小；论文CC BY 4.0、源码MIT（LICENSE在归档中） |

未安装作者依赖或运行作者脚本。源码归档只是研究资料，不能当成已经接入的后端。原论文PDF重点核读第3–5页方法、第8–10页结果与讨论、第15–19页任务/优化附录；渲染检查了第4页架构与公式。实时论文重点核读快慢两阶段、延迟训练和noise-Q备份，未对其真机结果做独立复现。

## 论文为何值得借鉴，哪些不能照搬

EXPO-FT把难以直接求策略梯度的大型流策略作为动作提议器：轻量Edit在其附近寻找高Q行为，同时保留基础候选参与竞争，降低完全从零探索的负担。环境中真正执行的行为进入Replay，VLA用熟悉的监督目标持续吸收经验；这才形成闭环，而不是只在冻结Pi外接一个长期残差专家。人工接管在不修改整个算法目标的前提下提供有效探索。

“基础候选参与竞争”有助于保持先验，但Q不准确时仍可能挑错；它不是安全保证。高UTD和多候选会放大Q过估计、视觉奖励误判和离线分布偏移，需保留失败样本、独立验证奖励并限制物理编辑幅度。连续成功样本不足时VLA success池会空或严重偏置，应显式跳过VLA更新/保留已审核示范比例，而不是强行把失败标成功。

论文结果是8项任务各30次评估全成功，19.1分钟是在线机器人交互量；还有示范、复位、人工和训练时间。双臂颜色分拣与耳机入盒不在这些实验中，属于新任务和新硬件分布。我们有更大量历史数据，但其来源图像/物料、动作单位和成功标签仍需审核，不能以小时数替代可用RL经验。

原论文基于LoRA任务起点并冻结基础图像编码器，不能据“finetune VLA”推断全参数训练。实时扩展又改变了观测时刻、动作窗口和训练backup；普通EXPO加上现有RTC并不自动等价于Real-Time EXPO。

## 已核对的代码证据与适配风险

以下链接均绑定同一个作者commit，不追随main：

| 源码位置 | 已核对事实 / YAM适配要求 |
|---|---|
| [expo_ft.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/expo_ft.py) `sample_actions/update_critic/_update_jit` | N基础与编辑动作Q选择、按C折扣、UTD循环critic再一次VLA/Edit更新；N=1直接base-only，不是便宜版完整EXPO |
| [expo_ft_pi_config.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/configs/model/expo_ft_pi_config.py) | 10Q/2min、8基础/8编辑、冻结Pi encoder、训练critic encoder、`actor_success_only=True`；这些是上游默认，不是YAM已审配方 |
| [pi05.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/vla/pi05.py) `process_transformed_outputs` | 用dummy zero state反变换，适用于其DROID动作约定；若直接用于YAM joint delta会丢失absolute基准，必须接真实state锚点与YAM输出变换 |
| 同上 `train_step/train_step_p1_prefix` | 原生flow loss、OpenPI trainable_filter、prefix条件loss；不能把RL配置actor_lr等同于VLA实际lr或擅换checkpoint结构 |
| [batch_utils.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/batch_utils.py) | 默认critic两相机、channel stack；YAM需明确第三相机与14D，Pi内部32D不能泄漏到物理控制 |
| [replay_buffer.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/data/replay_buffer.py) | C步奖励聚合；`add_offline_data`默认demo成功，`restore_success_marks`按末尾reward阈值恢复；YAM的未知/部分成功/assisted不能套用这些默认 |
| [realtime_expo_ft.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/realtime_expo_ft.py) | 慢 `sample_pre_cache` + 最新观测 `sample_actions`，窗口[d:d+C]，d≤C且2C≤H；noise-Q减少backup解码，需双时刻Replay与正确prefix |
| [agent.py](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/expo_ft/agents/alg/agent.py) | 上游checkpoint初始化支持overwrite删除目录，本项目adapter必须禁用覆盖，恢复与新输出目录分离 |
| [README](https://github.com/pd-perry/expo-ft/blob/21fe3d3b7d913c80836496817965932f49e0aedf/README.md) | 作者依赖修改版OpenPI/DROID、uv双环境；本项目只移植模型侧能力，不采用DROID控制和uv安装，也不替换YAM唯一控制层 |

单Thor并行慢Pi与快速视觉/Edit/Q会争抢算力；训练中Pi/encoder可更新使特征与权重缓存过期。设计采用联合bundle与episode边界切换，先测完整链路，不把单次基础Pi延迟外推到8候选系统。不默认把每轮更新都编译TRT，也不以降BF16/FP8解决未经验证的延迟问题；后端转换和数值/排名等价仍待实现。

## 本地旧代码退役

删除清单包含50个被Git跟踪的文件，见[retirement.json](retirement.json)。覆盖：

- `packages/parts-rl`（含RLT/vendor及测试）、`packages/residual-rl`（含测试）。
- `adapters/parts`、`adapters/rlt`、`configs/parts`、`configs/rlt`、`environments/parts.yml`。
- `scripts/parts`、`scripts/rlt`、`scripts/residual_rl`。
- `vla_platform.parts`及Thor PARTS/RLT policy和eager服务。
- 同时移除普通WebSocket的旧模型扩展注入、TRT的`--parts-manifest`，留下对off/shadow的base兼容与collect/eval明确拒绝。

历史恢复基线为 `b9e89ad9aa17fd35d8858a1be5acec39e3c7db52`，源码可由Git恢复。未删除原始数据、权重、服务器环境、实验报告；未停止下载或训练。`probe_rlinf.py`及其报告属于Pi模型核/权重转换的离线对照工具，不是RL learner，所以保留。旧文档在原owner中标历史，不把历史实验痕迹当成活动实现。

用户原有 `configs/dashboards/pi05-rtc-metadata.json` 修改未纳入本任务提交。

## 当前未完成的实现项

本轮以用户要求的深度设计和架构图为边界，未新建EXPO训练框架。下一阶段需完成YAM适配器、两时刻动作变换、数据assembler、端侧联合模型加载、上下游延迟/版本协议，再安排获准GPU与真机验收。耳机需先完成转换、单位/视觉审计和任务SFT基线。客户端提案已写出完整待审正文，尚未取得“发送给客户端任务”的审核授权，因此未发送。

验证结果与图回执见[validation.json](validation.json)及[图验收记录](../../../diagrams/README.md#expo-ft-强化学习架构)。
