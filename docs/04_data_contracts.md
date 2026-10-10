# 04 · 数据合同

## EXPO-FT 双任务数据合同（2026-10-10）

本节为新设计；旧PARTS/RLT/整轮残差章节保留历史，不自动成为EXPO训练合同。原始数据、原始奖励标注和转换目录均未修改。本次通过既有 `yam-server` 只读访问登录节点，读取manifest、文件存在性及每split一个Parquet首行；未进行训练、视频全量解码或数据修复。证据见 [server-data-audit.json](reports/rl/expo-ft-20261010/server-data-audit.json)。

### 两个任务已经有数据，不以记忆旧状态代替现状

根目录：`/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/`。

| 任务 / split | manifest集数 | 帧数 | 30 fps小时 | 三路非空视频齐全的集数 |
|---|---:|---:|---:|---:|
| `lego_sorting/train` | 4,458 | 10,374,181 | 96.057 | 4,458 |
| `lego_sorting/val` | 69 | 165,142 | 1.529 | 69 |
| `earbuds_charging_case/train` | 2,095 | 8,233,457 | 76.236 | 2,095 |
| `earbuds_charging_case/val` | 17 | 62,173 | 0.576 | 17 |

乐高prompt：`sort the legos into containers by color`。耳机prompt：`insert the wireless bluetooth earbuds into the charging case`。耳机指“放入充电盒”，不擅自推成耳罩/外壳压装。train源revision为 `68651e4929d9fb00f798937b2d62617cab5c771d`，val为 `46ca817c39d3df08115a75c31b3cd3867e3d875a`。视频目录名 `top/left_wrist/right_wrist` 映射为既有 `observation.images.top_rgb/left_rgb/right_rgb`，不可按字典排序改变语义。

本次每split首个Parquet抽样均有 `observation.state` 与单数 `action`，首行长度均为14；这不是全量shape/NaN/单位审核。文件存在性不证明可完整解码、视频帧数对齐或轨迹成功。尤其耳机尚未在 `processed/` 发现已发布的同类LeRobot目录；需用已有转换/审计工具写新目录后才确定READY和专属norm。已推翻09-08“耳机未找到三路齐全轨迹”的旧观察，仅更新当前证据，不删除历史。

### 三层存储与训练池

1. **raw（不可变）**：原始示范和在线episode包；保存三图时间索引、物理absolute action/state、实际提交值、设备反馈、人工/策略来源、提示词、原文件哈希。只写新接收目录，不覆写来源。
2. **audited events / transitions**：验证身份、完整性、tick/epoch、双时刻观测与prefix、实际执行、奖励/终止和分组划分；审核结果独立sidecar。隔离缺数据/未知奖励，不能自动当0。
3. **训练视图**：任务隔离的Q/Edit replay、已审核专家/成功VLA池、独立评估池。训练时按该Pi的delta mask `(6,-1,6,-1)`、norm及32D内部padding变换，raw保持14D绝对物理语义。derived标签变更产生新版本，不改raw。

### 在线最小记录

| 类别 | 必须表达的事实 |
|---|---|
| 身份 | schema、task/reward版本、run/episode/request/candidate ID、epoch、行为bundle ID、Pi/norm/Edit/Q/encoder版本、客户端源码/运行配置身份 |
| 观测 | 慢生成与快决策各自的三图frame ref、曝光/采样时间、state及采样时间、单调控制tick；不能以视频帧率反推真实耗时 |
| 行为 | 慢Pi各候选/随机种子与生成身份、edit、Q评分/选择ID、物理选中动作、已承诺prefix及其哈希、目标tick、有效长度/期限 |
| 执行 | 实际提交动作（与反馈state分开）、逐tick source、人工mask、限幅/保护改动、被丢弃或未执行行、执行回执、降级原因 |
| 转移 | 起止观测、实际执行步数、逐step reward事件及label版本、terminated/truncated/abort原因、next observation有效性 |
| 发布 | 分片哈希、episode闭合、outbox水位/幂等ACK、来源group、train/eval用途、审计状态及缺口 |

慢候选payload可在Thor保存一次，通过request/candidate ID与sha关联YAM执行记录，避免每视频帧重复H50×N；只有两端引用闭合后服务端才能建Replay。不同主机monotonic时间不能直接相减；tick属于YAM时轴，网络耗时分别记录本机send/receive或显式校准的时钟映射。

### 转移与奖励的正确性

- Critic使用**实际执行动作**重编码的窗口；human、安全裁剪、基础降级不能仍归到原“候选已执行”。未执行的H50后缀不作为发生过的动作训练Q。完整C窗口的R和discount按[03](03_training_and_evaluation.md#expo-ft-学习与评估设计2026-10-10)一致构建。
- terminated任务成功/正常失败可定义已知终点；time-limit若是任务合同内预算耗尽可为明确失败，若仅是外部截断且有效next observation可按合同bootstrap。设备故障、丢帧、人工停止的未知终点不能默认failure=0或挪奖到前一帧。
- 人工介入是有来源的行为数据；可参与Q和受控示范池，但不可计入自主成功率。人工接管导致epoch切换时不跨epoch拼C，保存边界与实际动作，不复制未发生的基础动作。
- 原20条无人rollout的C/N标签、旧PARTS局部抓取reward、旧token缓存均保留为原语义。没有明确映射时不复用为EXPO主reward，不重新挂一个schema就声称兼容。
- 颜色和槽位判断必须依据视觉/任务语义及复核，不能把FK低位或gripper effort当成任务成功。未经审核的历史示范只进VLA监督池，不触发上游默认 `is_success=True`。
- 以episode、采集会话、物料/摆放组划分留出；相邻窗口/复制轨迹不得跨split。reward classifier训练集也遵守同一分组隔离，避免分类器给自己训练样本打分造成假提升。

## 整轮 rollout 终点标签与离线初始化（2026-10-10）

当前第一批为固定检查点的约20条无人rollout，不筛成功。每集记录唯一集身份（跨会话时含会话）、开始待分拣数N、最终新增正确数C、分错数W、结束原因；未完成数N−C−W由工具计算。R=C/N仅在正常任务终点发一次。正常结束允许部分成功或全失败；预设任务预算结束可作为明确终点。紧急停止、设备故障、混色待审核先保留未知奖励，不自动记0。不要求每次抓放按键，不绑定固定框的位置/颜色；评分类别不明确时保留终点图像复核。

逐集结果由 `scripts/residual_rl/label_rollouts.py` 校验为 `yam_rollout_label_v1`，再由 `audit_hil.py --labels` 对齐原始episode。基线身份整批固定，由检查点、norm和推理模式核对；不能只凭表格中的简称判断权重一致。原有自动/人工来源、实际14D动作、观测、三图、tick/单调时间和RTC记录继续保留。人工介入和中断单独记录，不能默认为自主成功。

备份原件、原生LeRobot完整转换、清洗replay分别存放。相邻transition不得跨无效观测、缺tick、epoch变化；最后一条没有下一观测的动作不伪造next_state。没有有效最终观测时不向前挪奖励。保留HIL整集用户成功约定，同时标assisted；人工救场前的回报不当自主完成校准证据。新代码采用物理action critic，不从人工/测量状态猜历史基础策略残差。颜色判断需真实视觉特征；当前数组不含视觉及同版基线缓存，不能直接称训练READY。详细审核归 [HIL1009报告](reports/lego_hil1009_rl_audit_20261010.md)。

## RLT 派生特征与训练数据（2026-09-30）

RLT沿用下面的原始发布包、14D物理动作、活动臂六关节残差、真实决策间隔、RTC队列与结果奖励合同；新增特征版本不能覆盖原始包或旧READY。入口与训练顺序归 [03](03_training_and_evaluation.md#rlt-服务端实验接口2026-09-30)，推理归 [05](05_inference_and_rollout.md#rlt-eager-实验服务2026-09-30)。

| 阶段 | 不可变产物与检查 |
|---|---|
| 原始包 | publication.json完整、mock=false；布局在train组、split_role=train且非eval；request的state、tick及video_refs与原HDF5/episode匹配 |
| 观测导出 | `yam_rlt_observations_v1` observations.json；每观测一个NPZ，含三路`observation.images.*_rgb`（HWC uint8 RGB）、`observation.state`（有限14D）、`prompt`（原始提示词）；成员有path/SHA、observation_key、group_id、context、video_refs，清单记录publication哈希与split |
| 冻结prefix | `yam_rlt_prefixes_v1` PREFIX_READY.json；NPZ仅含prefix=`(S,2048)` FP32、mask=`(S,)` bool，有限且至少一个有效token；S不超过显式架构上限，三视图预期768，实际形状由hook/cache核对 |
| token快照 | `yam_rlt_token_v1` token.json + `yam_rlt_token_weights_v1` token.pt；固定上游commit、架构、prefix schema、checkpoint/norm SHA、cache SHA、train/holdout组及权重SHA；原生encoder/decoder均有限FP32，部署只加载encoder |
| RL回放 | 仍为`yam_parts_replay_v1`，状态前D维为选定token的z，后1527维不变；左右READY的feature ID/D匹配token，且同reward/contract/holdout；保持accepted_candidate_plan_v1行为残差 |
| Actor快照 | `yam_rlt_training_v1` training.json、每臂`yam_rlt_actor_v1`；绑定token清单SHA、feature/state schema、物理合同、fixed_std、train统计、replay/成员哈希；不兼容旧PARTS actorbundle |

`feature_schema_id = rlt-<canonical SHA256>`由上游commit、token架构、checkpoint/norm身份、prefix schema及权重成员身份确定。换token/checkpoint/norm另开版本；行为目录复制原清单和相对权重路径后保持同一ID。token预训练和RL必须保留同一holdout布局组，不把评测图像用于token训练。成员路径受目录边界和哈希检查，不读取pickle；不完整输出没有READY。

现有原始数据可用于token重建，不要求完整颜色分拣任务；RL仍要求完整的局部下降→闭合→终止/实际交还attempt。取消、失效力矩和不连续来源等排除规则保持。旧池化feature的READY不能直接训练RLT actor，本轮未实现旧replay自动重编码迁移；新RLT collect包的同次token由服务端回传，再复用replay assembler。不能根据FK或后续测量倒推动作。

`prepare_observations.py`只读原数据，按MP4引用解码，不宣称恢复压缩前逐字节像素。PyAV为可选数据依赖，模型/GPU依赖留在Pi服务器环境。本轮用fixture验证字段与来源连接，未审核实际采集包。

## PARTS 抓取 RL 回放数据合同（2026-09-30方案）

**状态：服务端原始包审核和replay构建已实现，未接收/发布真实RL训练集。** 用户澄清当前失败为下降高度不足；需采集下降接近→关爪的完整尝试，下降进入信号待核对。`abs(effort_nm)>0.65`仍为关爪后的结果判据；状态机和权责归 [05](05_inference_and_rollout.md#parts-左右抓取的服务端与客户端合同2026-09-30方案)，学习算法归 [03](03_training_and_evaluation.md#parts-左右抓取残差学习方案2026-09-30)。本合同不是LeRobot专家SFT/HIL导出规则。

### 已有原始记录与缺口

2026-09-30只读检查本地YAM `docs/hil_dataset_fields.md`、`hil/policy.py`、`hil/rtc_protocol.py`、`hil/grasp_diagnostics.py`。原始记录是 `yam_hil_v2`：每集manifest、每段HDF5和top/left/right MP4。现有字段包括三图帧号/同步质量、观测与测量state、策略/选中/实际提交action、request/reply、epoch/tick、RTC `policy_selection`来源、可选夹爪力矩快照。仅据源码列能力，不声称现场已经启用新版本。

夹爪快照 `grasp_diagnostics.followers[left,right]` 包含 `position/velocity/effort_nm/sdk_updated_at/sampled_at/feedback_age_s/valid`，采样阶段为before_command。该行反馈发生在该行新命令之前，应关联前一已提交命令及其来源。`effort_nm`为有符号Nm，判据按用户要求取绝对值。重复SDK快照不得累计为多次有效确认；SDK的Unix更新时间不能直接与IPC monotonic相减。

现有记录没有逐次抓取reward、terminated/truncated、残差候选/应用值、对应behavior/feature身份。现有离线Detector还带速度与空夹位置条件，只输出候选；不能直接把contact_candidate升为0.65规则的成功标签，也不能把expert_valid或episode_success当局部reward。

### 原始库、请求表、attempt表

拟复用原始HDF5/MP4，新增字段使用details/sidecar；不覆盖原件、不从客户端模型进程重复录制图像。数组按request只存一份，逐帧使用引用。

| 记录层 | 必须保存的内容 |
|---|---|
| run manifest | schema/contract SHA、task、实际基础checkpoint/norm/engine身份、Pi和YAM版本、mode、feature schema、左右actor snapshot、action_dt=1/30、活动臂关节mask/六维rad边界B、各臂实际h_entry_m（用户初始值0.05）/h_goal及高度参考/标定身份、奖励公式/权重/版本、下降进入与确认/超时规则、数据传输完成标志；模型身份可存交接sidecar，不强制改旧YAM产品字段 |
| request表 | `(run_id, session_id, epoch, request_id, obs_id)`；三相机源帧/时间和state；观测tick、RTC committed prefix、其逐tick原来源、队列状态；完整基础H50、左右候选U、feature或可校验feature_ref；探索实现后的behavior、snapshot与噪声来源 |
| 每tick记录 | 原生policy_selection.request/model_index/target_tick、attempt_id/arm/phase、是否启用残差、基础目标与残差候选引用、应用的physical residual、selected/bounded/submitted/measured各自值与时间、当前高度/目标高度/误差及有效性、height_reward/grasp_reward/total_reward、力矩原值与反馈有效性、客户端约束/裁剪标记 |
| attempt表 | arm、下降进入及首次实际残差tick、闭合开始tick、reward proposal/确认tick、handback effective tick、成功/失败/取消原因、reset边界、grasp_reward=0/1或null、高度辅助与总奖励分量、terminated/truncated、trainable与排除原因、来源request/视频区间 |
| 发布清单 | 完整attempt成员、源文件哈希、来源组/train-val分区、数组维数/有限性、下降进入/关节残差/reward合同、逐tick执行掩码与客户端完整标志；服务器审计后生成READY，只写新目录 |

数据传输和大文件落盘由已有后台录制/传输路径承担，不在30Hz提交路径等待网络/磁盘。客户端生成所有控制tick、epoch、attempt与奖励事件；Thor原样回显关联token并将自身特征/残差记录与之绑定。跨机monotonic保持原时钟域，不伪造对齐。

RTC回复的前d行是既有实际承诺动作，可能已带旧request残差，不能当本request的未修正基础动作。request表标记这些行的基础引用不可用；通过原policy_selection来源找到生成该target的旧基础/残差。请求返回但迟到/被discard的候选及另一只未选中arm的候选不进入已执行动作样本。

### reward、边界和transition组装

| attempt结果 | 抓取结果分量grasp_reward | 回放处理 |
|---|---|---|
| 按05确认、在实际交还边界仍满足力矩成功判据 | 局部终止处1，其余0 | 有效成功attempt；终止不bootstrap |
| 有效反馈下重新张开且未成功，或到显式抓取任务时间预算 | 0 | 有效失败attempt；任务结束，终止不bootstrap |
| 断联/人接管/反馈缺失或过旧/epoch切换/保存中断 | null | canceled/invalid；原件保留，默认不入自主RL回放，不冒充失败 |
| 训练重置/人工摆放 | null | 与前后attempt分隔；不作为抓取动作或next state |

同一attempt的抓取结果分量总和为0或1；只在已开始关爪后用力矩给下降抓取尝试的结果。下降阶段提供独立的高度误差辅助分量，具体公式/权重待定，总reward不限定为0或1。一个持续力矩高的片段不能按每帧产生多个抓取+1。同一SDK快照只判定一次。交还后、明确释放之前的力矩下降可另记 `post_grasp_loss`，与局部抓取reward和整任务正确放置指标分别存储。

用户确认 `h_entry`可配置、初始50 mm（内部0.05 m），以及高度辅助奖励与抓取结果奖励的组合设计。录制必须保存各臂实际生效 `h_entry/h_goal`、当前高度及目标误差，绑定同一参考点/坐标/单位和标定版本；高度奖励、抓取奖励、合成总奖励分别保存，并绑定公式/权重/版本。奖励数值配方尚未确定，不能从当前文档推导默认高度权重。达到高度与抓取成功分别记录。

初始录制从下降接近进入前保留观测，到关爪后的结果和实际交还结束，不能在夹爪开始闭合才开录。若记录派生末端高度，另存FK模型身份、frame、单位及桌面法向标定；仅有base系FK不等于已有桌面相对高度，不把未标定Z写成真实下降毫米数。

transition只组装控制来源和记录连续的区间，状态包括冻结视觉特征、反馈state、基础参考与RTC调度队列；按真实决策tick差保存 `h_i`、reward发生位置、bootstrap和执行掩码。H50预测长度不代表本request执行50步。重叠队列/承诺动作来源缺失、next state不完整或feature版本不匹配时 `trainable=false`；不能仅靠mask补齐不存在的轨迹。

行为标签来自记录的残差候选和客户端实际应用链，不用measured_state代替动作、不把submitted_action当U直接输入网络、不从动作差中猜未知B。约束改变了行为时保留原候选、最终应用差和标记；学习器采用的动作表示固定在本run合同，并核对超出归一化[-1,1]的样本。首次发布先对逐值来源和实际调度回放验收。

成功/失败attempt都需保留。重训练的成功重加权只是派生采样清单，不删除失败原件。同一原始rollout/重置布局组放同一split，左右arm来自同组不能分开跨train/val；holdout组不进入online buffer或后续success-reweighted retraining。旧HIL数据只有满足完整来源、力矩、特征和动作合同才可能派生，不默认给整集U=0或局部成功标签。

### PARTS 原始发布格式

这是本轮客户端构建的拟实施物理格式，新增层schema为 `yam_parts_raw_v1`；已有episode继续保留 `yam_hil_v2`，运行配置/发布规范以本节为准。客户端实施步骤见 [交接计划](reference/parts_client_handoff.md)。服务端已有审核器与派生replay构建；YAM客户端录制实现及现场完整性由客户端交付验收，不能从本地代码推定真实包齐备。

```text
<新采集根目录>/<run_id>/
  run.json
  requests.jsonl
  requests.h5
  events.jsonl
  attempts.jsonl
  publication.json
  episodes/<episode_id>/
    manifest.json                  # 原yam_hil_v2结构，补充parts来源引用
    <原有segment目录>/
      samples.h5                   # 原14D列和UTF-8 details，新增details.parts
      top.mp4
      left.mp4
      right.mp4
```

JSON为UTF-8；JSONL每行一个完整对象，未设置/无效值用null并保存valid标志，不写NaN/Inf。HDF5沿用客户端已有依赖，数值保留源精度，不将物理动作重新归一化；三路像素仍只保存在MP4。大数组通过HDF5路径引用，跨文件引用使用run根目录下的相对路径。原始tick/epoch/IPC单调时间及相机时钟域保留，视频frame_index不替代控制tick。

| 文件 | 必须保存的内容 |
|---|---|
| run.json | schema、run_id/session_id、task、mode、mock、control_hz/action_dt、layout_group_id与split_role、两侧配置、h_entry_m/h_goal_m、FK模型/末端参考点/table frame/转换与标定身份、reward schema和数值是否已固定、contract SHA、behavior/feature manifest引用 |
| requests.jsonl | context（run/session/epoch/request/observation）、观测tick与本机发送/接收时间、三图的episode/segment/frame引用、state与FK/高度快照、请求时调度队列来源、parts能力/合同/mode、HDF5数组路径、接收错误/迟到/discard、behavior与feature引用 |
| requests.h5 | 每个request独立group，保存actions_native、左右u/B_rad/editable_mask、RTC committed_prefix与原来源引用；实际数组shape/dtype写在请求索引。没有收到的数组不创建，记录缺失原因 |
| events.jsonl | 唯一event_id、单调event_seq、context/attempt/arm、event类型、控制tick与IPC时间、原因、原始SDK快照引用、奖励分量及valid；记录进入、首个残差、关爪、目标到达、成功确认、交还、失败、取消与重置 |
| attempts.jsonl | 每个已终止attempt一条结果：attempt_id/arm、run配置引用、episode/来源区间、进入/首个残差/关爪/终止/实际交还tick、成功/失败/取消及原因、grasp_reward、height_reward/total_reward汇总与有效性、terminated/truncated和已采用请求区间 |
| publication.json | publication_id、run_id/schema、client_complete、文件相对路径/bytes/SHA256、完整episode/attempt成员、记录缺口、producer代码SHA及传输状态。只在队列消费完、文件关闭并校验后标client_complete=true |

requests.h5的group拟命名为 `/requests/e<epoch>_r<request_id>`，JSONL必须显式给出group路径，不能依赖命名猜测关联。`actions_native`为50×14；`left/u`、`right/u`为50×6，`B_rad`为6，`editable_mask`为50，RTC前缀为d×14；`scheduler/targets`为50×14、`scheduler/valid_mask`及`committed_mask`为50。请求JSON中移出的大数组由assembler按HDF5路径还原。对RTC前d行标记原前缀来源，它们不是该request未修正的base；缺少parts的基线请求仍存actions_native，并记录候选/特征缺失，不能填伪造零候选。

视觉feature由Thor生成并按本request/observation绑定，可返回数组或feature_ref。Thor返回数组时，客户端按原dtype存request group并注明feature_schema；仅返回引用时存引用和可核验身份，由condapi合并对应feature文件。无feature出口时如实标missing，不用关节state冒充图像特征；基线/早期shadow包用于标定和接口审计，是否可派生训练由模型侧另行审核。

每行 `details.parts` 至少包含下表逻辑字段。无活动attempt可以为null；两臂反馈独立保存，active_arm只选一个。

| 字段 | 定义 |
|---|---|
| schema/run_id/mode/contract_sha | 新增层身份、实际模式和锁定合同 |
| active_arm/attempt_id/phase/eligible | 当前活动臂/尝试/阶段/接近资格及其来源；持物和非接近下降不自动启用 |
| arms.left/right | 每侧FK position/orientation及frame/参考点/采样时间/valid、height_m/h_entry_m/h_goal_m/error_m、原力矩快照引用与新鲜度 |
| 自动选择器诊断（拟新增） | 每侧selector_state、selector_schema/config_sha、eligible/empty_hand/source、reason_codes、entry_armed及table_position_m/frame；run绑定规则配置、抓取区与标定哈希，events记录状态切换/关爪/释放/回撤证据；无活动attempt也保存，详细拟实施规则归05/客户端交接计划 |
| selection | 生成最终采用目标的epoch/request_id/observation_id/model_index/target_tick，是否继承RTC前缀，request数组/feature引用；继承行继续指向旧request |
| base_target/candidate_ref | 残差叠加前经客户端原映射后的14D基础目标；候选的request_epoch/request_id/model_index/arm/actor_snapshot_id/behavior_snapshot_id，必须独立记录，不能从submitted或未知夹爪映射倒推 |
| residual_applied/physical_residual_rad | 是否实际启用及最终物理14D差值；只有该活动臂六关节可非零，shadow实际值全0 |
| constraints | 候选/组合/最终提交来源、裁剪/连续性/高度边界结果及拒绝原因；原动作与反馈保留在原14D列 |
| reward/event_refs | height_reward/grasp_reward/total_reward、valid、reward_schema引用、对应event_id；未固定公式/目标或取消时不伪造总奖励 |

本节 `error_m`统一定义为 `height_m - h_goal_m`；未设置h_goal时为null。每个event和attempt需能定位原始反馈与上一条已提交命令，before_command语义不变。request和逐tickselection分别记录“生成候选”与“实际采用”，未执行/丢弃的候选不能计入行为动作。

客户端在episode manifest增加 `parts` 引用，包含run_id、contract_sha、mode、run相对路径及本episode的attempt成员；不覆盖既有action_semantics、时钟、视频同步和等待切点说明。发生恢复/写入中断时将未完成attempt记canceled，文件收尾不完整则client_complete=false。服务/采集mock必须mock=true，split_role记录mock/eval/holdout/train或未分配，不能自动把所有包放入train。

发布走后台持久outbox和独立传输路径，按publication_id幂等；源包收尾后保持不可变，重传核对相同哈希。接收侧ack只代表文件接收，客户端client_complete只代表生产完整。condapi另在新派生目录做时序/残差/奖励审计、分组split并生成READY与transition清单；训练只消费READY中的有效自主区间，不能把H50整块默认当成已执行50步。

### 服务端派生replay v1

入口 `scripts/parts/audit_publication.py`为标准库文件审核，只返回files_verified_not_training_ready。`scripts/parts/prepare_replay.py`使用独立数据/模型环境中的NumPy/HDF5，在新输出目录保存audit.json及左右独立 `READY.json + transitions.npz`。builder目前只消费请求HDF5中已解析的features/z，不自动解析外部feature_ref。成功、取消、缺失字段和mock检查均有离线fixtures；没有读取真实现场包。

`READY.json` schema=`yam_parts_replay_v1`，含arm、state_schema=`yam-parts-state-v1`、D与state_dim=D+1527、feature_schema_id、contract_sha、reward_recipe（含显式height_scope）、每控制tick gamma、原始publication哈希、holdout组、attempt结果及NPZ哈希。action_semantics=`accepted_candidate_plan_v1`：行为动作为实际被采用请求的归一化H50×6规划候选，环境含真实裁剪/提交/队列；不是把整个H50标记执行完。终止next_state只是不用的零占位，bootstrap=false。

| NPZ字段 | shape/语义 |
|---|---|
| state/next_state | N×(D+1527)，同一个编码器；下一有效自主决策或终止占位 |
| action/action_mask | N×300，原候选u/该计划可编辑且B非零的维度；critic以带mask动作估计回报 |
| executed_mask | N×300，命令tick区间[k,next_k)实际采用的位置，仅这些位置参加成功BC/可选失败锚定 |
| next_action_mask | N×300，下一计划的可编辑维度；终止全false |
| reward/elapsed_steps/bootstrap | N，反馈区间(k,next_k]的逐tick折扣和、真实tick差、是否bootstrap |
| success | N bool，对整个已审核attempt标记成功，途中行可bootstrap；不是每行成功奖励 |
| attempt_id/group_id | N文本，run_id:attempt_id及原始布局组；组不跨train/holdout |

入口前READY候选只有后来被某attempt实际采用、来源与整个决策区间连续时才可组装；进入前基础反馈reward为0，执行mask仅标实际活动命令，未采用候选仍不进入replay。active_descent高度分量使用前一提交命令的phase，终止结果与前一命令反馈对齐。

builder验证source SDK有效性、单调/不重复确认、新鲜终止反馈和有效handback；obs_id/state与原始row、三路视频引用/索引/文件、分段committed_rows、队列数组、候选身份与实际提交差值须可追溯。视频这里只检查索引与文件身份，不解码像素或独立证明现场同步/标定。缺少独立base_target、next队列/特征、原始连续tick或reward配置的attempt写排除报告；不补造数据。重复run/attempt或发布损坏拒绝，重传不能重复计入。

正式奖励作用阶段与数值尚待固定。服务器逐反馈tick复算注册配方，客户端汇总只保留来源；双方需明确height_scope及时间对齐后才交真实READY。当前YAM只读源码中base_target在编辑记录存在但record转发未包含该字段，这一客户端对齐项已放待用户审核清单，未发送追加任务。

## 模块化数据 inventory 与 split（2026-09-24）

控制层 v0.2 只处理数据**身份与成员清单**，不替代 LeRobot/XR-1 loader 或逐帧有效性审计。`configs/datasets/*.toml` 声明源版本、机器人合同、三相机顺序及动作空间；审核后的 JSONL 输入逐条标明 `episode_id`、原始 rollout/场景 `group_id`、`task_id`、`frames` 和所有会影响解码/标签的源文件，包括元数据。`vla inventory create` 在新路径计算每个源文件 SHA256，再把相对路径→哈希的规范 JSON 映射取 SHA256 作为该 episode 的 `source_sha256`；因此它不是单个 Parquet 或视频的直接哈希。XR-1 派生数据另有 `asset_uri` 指向原生 JSON，必须包含在该 episode 源文件列表；其引用视频也必须列入。工具不自动发现清单遗漏。生成的清单不修改原始数据。

`vla split create` 固定 seed、数据声明和源清单哈希，将同一 `group_id` 的所有 episode 放在一个 train/val/test 分区，拒绝重复 ID、缺失文件、源文件变化与输出覆盖。它不自动判断片段有效、任务成败或既有官方 val 是否可回流训练；原有 val 与各轮 DAgger val 的隔离规则仍有效。训练计划还核对后端原生 episode 选择与冻结 train 清单一致，统计须由对应训练集单独产生。操作与限制见 [10](10_vla_platform.md#模块化配置后端-v02)。

## DAgger 纠正数据发布合同（2026-09-21方案）

状态：拟实施，尚未验收现场新数据或导出器。采集/训练操作见 [DAgger手册](reference/lego_dagger_playbook.md)。已有14D、三路RGB、absolute动作原值和Pi训练时delta规则继续适用。

原始完整rollout保留自主、人工、过渡、暂停与复位；训练版本只选择审核通过的连续专家片段，不覆盖原件。发布为当前LeRobot v3兼容数据＋来源/审核sidecar；sidecar不是模型必需输入，拟定字段不是现有实现声明。

| 内容 | 合同 |
|---|---|
| 观测 | 三路 `observation.images.top_rgb/left_rgb/right_rgb`；`observation.state[14]` 为观测对应的follower反馈状态 |
| 专家动作 | `action[14]` 为时间对齐、审核通过、实际提交follower的absolute目标；不是下一帧反馈，不是未经映射的主臂读数；训练时才做delta |
| 原始动作链 | 分别保留policy预测、人工目标、仲裁/过渡目标和实际提交目标，缺失值显式标记，不相互冒充 |
| 时间 | 帧索引、各相机时间、state时间、命令提交时间、policy tick及跨设备时钟映射/误差；不能以读日志时间冒充电机写入时间 |
| 控制权 | `policy/expert/transition/hold/reset`；单臂接管须逐臂标记。首版完整14D专家监督要求两臂标签语义均有效，否则排除或另案做维度mask |
| 来源 | round、原始rollout、纠正事件、连续片段、起止原帧/时间、布局/采集组、采集模型及控制版本、norm指纹 |
| RTC审计 | control epoch、request ID、观测tick、chunk起始tick、承诺前缀与实际采用动作索引；可用单独请求表关联，避免逐帧重复存整块 |
| 结果 | 纠正原因、恢复是否成功、整任务结果、无效/排除原因；整任务失败不自动否定其中已成功的局部纠正 |

split按原始rollout/场景采集组隔离，同源片段不可跨train/val；原验证集和各轮val不进入后续训练。片段内不得有观测缺口、控制权切换或人工改场景。H50首版仅取未来50步真实标签均有效的起点，长度L贡献max(0,L-49)，每个标签有源帧映射；不拼接过滤缺口、不用复制末帧凑专家标签。短片段留在原始库，待有效位mask实现并验收后再使用。

M0配套norm可作为适配坐标系继承，C1统计只用于范围审计；这不同于宣称norm由混合新数据计算。manifest分别记录norm来源与训练数据来源。任何单位/夹爪方向不一致、明显越界先阻断发布并复核；不静默改变单位或重算norm。完整新训练版本需审核清单、视频解码、时序检查和真实loader边界检查，当前状态保持planned。

## 2026-09-16 · 现场同款kit与数据范围复核

用户确认现场为ABC-130K同款官方YAM kit、三台D405、各640×480，优先复现原数据任务。这是用户确认的硬件条件，不等于安装视角、像素尺度、标定及时间合同均已独立验收。

随后用户明确现场普通2×2小颗粒底面约16 mm。14:51最新记录三路视频均640×480；追加抽取ABC六集共18帧显示物体形态/相对尺度、背景、指形和容器存在差异，源积木物理尺寸及全量比例未知，不能仅因同kit就认定同任务分布。抽样出处和224几何预览归 [最新证据](reports/training/redesign-20260916/README.md#乐高分拣3最新异步测试)；现场数据如何转换为专家标签、同步、14D绝对目标、30fps与H50边界需独立验收，不能把本次失败的policy_action直接当专家示范。

服务器metadata：train为4458集/10,374,181帧、val为69集/165,142帧，均30fps、1个任务，文本为 `sort the legos into containers by color`。train约96.06小时，含val约97.59小时。三路已发布视频均224×224；抽查source episode95（转换后episode0）frame100，均有上下近黑补边，与640×480等比缩放到224方形相符；一集不能代表全量或现场图像一致。快照、图像和局限见 [证据](reports/training/redesign-20260916/README.md)。未改数据、单位或norm；小集实验设计归 [03](03_training_and_evaluation.md#2026-09-16--训练重设计先复现乐高分拣再比较模型)。

## 2026-09-08 全量发布与归一化完成

实时核对 v3 续跑日志：`CONVERSION_COMPLETE` 为 2026-09-08 07:20:07 +08:00。
已发布版本 `/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/{train,val}`；
train 共4,458轨迹、10,374,181帧。本节替代后文9月7日“尚在转换”的历史状态。

`scripts/compute_yam_norm_stats.py` 通过只读数值列、逐集 H50 分块和同一 DeltaActions 计算全部训练帧，
不包含 val，不裁掉最后不足 batch 的帧，不修改原始数据。12个代表性首中尾/边界样本与真实
LeRobot→YamInputs→DeltaActions 逐值一致，三相机解码成功。统计为14D，模型变换随后补到32D。
所有关节 delta，夹爪 absolute；源 metadata 合同为关节 radian、夹爪0闭1开，raw→port 单位保持
仍是发布者文档与数值范围支撑的推断，不是独立机械臂标定结论。

持久化资产目录（训练应设置 `data.assets.assets_dir=.../pi05_h50`，`asset_id=yam`）：

```text
服务器 /home/wuyan/lyj/YAM/training-assets/lego_lerobot_v1_20260907/pi05_h50/yam/norm_stats.json
本地   assets/lego_lerobot_v1_20260907/pi05_h50/yam/norm_stats.json
SHA256 c496b738470e432b9da02b22a45c8c309b3db8412d73723944a7cbdc62305be9
```

`assets_dir` 指 `yam` 的父目录。本地资产不进入 Git；数值文件、逐文件来源哈希、
加载器对照与进度记录同目录保存。训练集 conversion manifest SHA256：
`9844c0b86b93fd40adf251275d770899ba413800d2a0cba11d4e9bdae40544dc`。
state计数10,374,181，action计数518,709,050；全量计算164.27秒。
分位数使用原 RunningStats 的5000-bin近似直方图，累计改用float64。
该数值路径只支持本仓库已发布的单episode/Parquet布局，遇到其他布局拒绝运行。
旧转换 manifest 内 `norm_stats=not_computed` 是发布时的历史快照，保持不改；新 norm 的来源由独立
`provenance.json` 记录。[详细证据](reports/training/pi05-full-20260908/README.md)。

本页是当前 YAM 训练数据的唯一语义 owner。服务器路径见 [02](02_installation_and_environment.md)，训练流程见 [03](03_training_and_evaluation.md)。OpenArm 16D 合同只在历史文档中保留，不能套用到 YAM。

## YAM 双臂合同

```text
state key:  observation.state
action key: action
state/action shape: (14,)
layout: [left 6 joints, left gripper, right 6 joints, right gripper]
images:
  observation.images.top_rgb
  observation.images.left_rgb
  observation.images.right_rgb
model action: internal 32D, horizon 50
policy output: real YAM 14D
```

RTC时间索引：当前 `create_torch_dataset` 按 `[0,1,…,49]/fps` 加载50步 `action`，YAM发布集为30fps，故样本中 `action[0]` 与 observation/state 对齐同一 LeRobot 行时间戳，步距约33.33ms。这是训练数据索引，不证明真实部署中相机曝光→控制命令的物理延迟为0。3588异步部署应把同步观测归属的30Hz policy tick 显式传给Thor，完整协议见 [RTC冷手册](reference/thor/13_trained_rtc_inference.md)。

YAM 与 YAM-ABC 使用同硬件配置；本仓库只实现训练数据和 policy transform，不实现机械臂控制。YAM state/action 的物理单位、正负方向、夹爪开闭范围必须从当前数据的 `meta/info.json`、feature metadata 和样本审计中确认；在证据完成前不写成 degree、弧度或归一化值。

## 图像和 prompt

LeRobot dataset 每行至少应能提供三路 RGB 图像和 state/action：

| 原始键 | OpenPI 槽位 | 要求 |
|---|---|---|
| `observation.images.top_rgb` | `base_0_rgb` | 必须存在 |
| `observation.images.left_rgb` | `left_wrist_0_rgb` | 必须存在 |
| `observation.images.right_rgb` | `right_wrist_0_rgb` | 必须存在 |
| `observation.state` | `state` | 14D |
| `action` | `actions` | `(horizon, 14)` 或单帧 14D |
| LeRobot task | `prompt` | `prompt_from_task=True` 时映射 |

视频读取后可以是 CHW 或 HWC，`YamInputs` 会统一到 HWC；正式数据应固定编码、fps、时间戳和 RGB 语义。缺少任一路图像不应静默补黑图，先修复数据或明确建立新版本。

YAM 配置默认视频后端为 `pyav`，便于 conda 打包后使用随 PyAV 提供的 FFmpeg 库；
若计算节点另行通过 TorchCodec/系统 FFmpeg 导入与首中尾解码验收，可通过 `--data.video-backend=torchcodec` 切换。

## Action transform

当前 `LeRobotYamDataConfig` 默认假定 LeRobot action 是绝对目标：

```text
per arm: [6 joint delta, 1 gripper absolute]
two arms: (6, -1, 6, -1)
```

训练输入先由 `YamInputs` 将 `action` 改名为 OpenPI 的 `actions`，再由 `DeltaActions` 相对当前 `state` 处理 6 个关节；模型输入通过 `PadStatesAndActions` 补到 32D。输出路径先恢复 absolute，再由 `YamOutputs` 裁回 14D。若实际 action 已是 delta，必须在独立数据审计中确认后关闭 `use_delta_joint_actions`，不能重复转换。

## LeRobot 目录和版本

当前代码依赖 `lerobot==0.5.1`，优先使用 LeRobot v3 namespace。一个可训练数据版本应自洽地包含：

```text
<dataset>/
  meta/info.json
  meta/tasks.parquet 或等价 task metadata
  meta/episodes*.jsonl 或 v3 对应元数据
  data/...
  videos/...
```

实际目录结构必须以当前 LeRobot 版本的 metadata loader 为准。新增/删除 episode、重写 parquet、修改视频或改变 feature/unit 都创建新数据版本；不原地覆盖下载的 raw 目录。

服务器已经发现 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks`，但它在完成 metadata、feature、shape、视频和单位审计前只是原始/待审计数据，不能仅凭名称直接训练。

## Norm stats 和资产

YAM 资产 id 固定为 `yam`，训练和 checkpoint 约定为：

```text
assets/yam/norm_stats.json
```

norm 必须针对同一数据版本、同一 train episode split，并在 YAM action delta transform 后计算。禁止跨 OpenArm/Piper/YAM 合同或跨单位复制 stats；服务优先读取 checkpoint 内资产。

## 数据发布 gate

### ABC 乐高子集：上传期间的清洗流程

2026-09-07 只读核验：`ABC-130k-two-tasks/lego_sorting` 是任务筛选后的原始导出，
不是可直接加载的 LeRobot v3 数据集。manifest 声明 train 4458、val 69 episodes，30 FPS；
布局是 `manifests/{train,val}.jsonl`、`{train,val}/data/source-file-NNN.parquet` 和
`{train,val}/videos/{top,left_wrist,right_wrist}/episode-NNNNNN.mp4`。
parquet 保留原始 episode_index、frame_index、index、task_index，以及 language_persistent/events。
抽查 train/source-file-000.parquet 有 11082 行，state/action 均 14D；语言列可为空。
当次清点 4527 条全部为 `pending_upload`，首条 train episode 95 缺 top/right_wrist 视频；
这是上传中的瞬时观察，不能作为后续完成状态。清点清单指纹：train manifest SHA-256
`1fabf4a2b83224f85a724146fb37ccf17928f91c00bd0a697e470980e54fb3a6`，val
`9927aa682e78662bec5584e83728c2e9eb5fd86323ab4bc46a90294f255bfe08`。

清洗按以下顺序推进，每一步保留原始 train/val 分离：

1. **上传清点（已实现）**：`scripts/audit_yam_subset.py --inventory-only` 对照 manifest 检查每条 episode
   的 parquet 和三路视频；缺失、空文件、近期修改或读取期间变化标为 `pending_upload`。
   报告包含 manifest SHA-256、源 repo/revision、episode ID、文件长度和 mtime；mtime 稳定仅是预检，
   不是上传完成证明。冻结时仍须上传方确认完成并复核报告。
2. **结构清洗（已实现审计，不修改原始文件）**：完整模式验证每个 episode 行数、14D 有限数值、
   连续 frame_index、30 FPS 时间戳，逐帧解码三路视频并核对时间戳和帧数。
   已稳定但失败的样本标为 `rejected`，人工核查/重传后重跑；不自动裁短视频、补零、插值或删帧。
3. **语义确认（待完成）**：确认左右顺序、关节单位、夹爪范围、action 是否绝对目标。
   manifest 的 `task` 用于统一 prompt：`sort the legos into containers by color`。
   数值大小只能提示单位，不能证明单位；`validated_structure` 不等于可训练。
4. **发布转换（已实现并通过合成数据回读；真实完整 episode 待验收）**：只将验收通过的完整 episode 写入新版本目录，
   train/val 各自生成独立 LeRobot v3 数据集。使用当前 LeRobot writer 生成 metadata/task/episode 表，
   重编号索引并保留 `(source_repo, source_revision, split, source_episode_index)` 映射。
   相机映射为 top→top_rgb、left_wrist→left_rgb、right_wrist→right_rgb。
   manifest 视频时间戳指向原始长视频；当前逐 episode 视频必须使用本地 PTS，不重复裁切原始区间。
5. **发布验收**：真实 loader 验证首中尾/跨 episode action chunk，核对源 split 无泄漏，
   norm 仅使用 train；再做短训练。正式版本记录纳入/拒绝清单、源指纹和转换参数。
   DAgger 后续追加独立版本与来源标签，不混入本轮 holdout。

可在上传期间运行清点（仅标准库；JSON 写到 stdout，由调用方留存到原始数据目录外）：

```bash
python3 scripts/audit_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting --inventory-only
```

上传完成后的结构审计应放到计算节点，用项目环境运行同一命令并去掉 `--inventory-only`。
可先加 `--limit 2`，表示每个 split 最多审计两条，不代表全量验收。
报告始终保留 `trainable=false`，直到独立的语义、转换和 loader 发布 gate 完成。

- `info` 中的 episode 数、task metadata、parquet 索引和视频清单一致。
- state/action 所有样本最后一维为 14，顺序是 `[左6+夹爪, 右6+夹爪]`，无 NaN/Inf。
- 三路图像首、中、尾样本可解码，时间戳不越界，颜色通道和 shape 一致。
- task/prompt 映射不为空，train/val split 明确且可复现。
- 物理单位和夹爪语义已写入数据版本的 audit/manifest；未确认前标记为待核验。
- norm stats、config、训练 split 和 checkpoint asset id 一一对应。

## 转换工具操作

`scripts/convert_yam_subset.py` 使用已安装的 LeRobot 0.5.1 writer 生成 v3 metadata、task 表、episode 表、
parquet 和视频；train/val 分别转换，默认要求该 split 的全部 episode 完整。
原始 parquet 的 state/action 以 float32 原值写入，不在转换时做 delta 或单位缩放；delta 仍由训练 transform 处理。
空语言列由 manifest 的非空 task 补入标准 LeRobot task。

转换必须提供经审计的 JSON 合同：`state_action_names` 为脚本 `JOINT_NAMES` 的完整左6+夹爪/右6+夹爪顺序，
`joint_unit`、`gripper_unit` 为已确认单位，`action_mode` 为 `absolute` 或 `delta`，`evidence` 标明确认依据。
工具只能检查声明是否完整，不能代替硬件/来源文档核验；不能用测试中的 synthetic 单位发布真实数据。
若声明 action 已是 delta，训练配置必须关闭 `use_delta_joint_actions`。

```bash
"$PYTHON" scripts/convert_yam_subset.py \
  /home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting \
  /home/wuyan/lyj/YAM/YAM_data/audited/lego_train_v001 \
  --split train --contract /path/to/reviewed_yam_contract.json
```

首轮可通过 `--episode-ids` 显式选择少量已上传完整的源 ID；该子集选择会写入 provenance，不会自动跳过缺失 episode。
不存在、近期修改、空文件、14D/时间戳异常、视频帧数不匹配均停止转换。
工具对源文件记录并复查 SHA-256，在 `<output>.incomplete` 完成写入、finalize 和 loader 首中尾回读后才改名为输出目录。
失败目录保留，不自动覆盖或续写；重跑使用新版本路径。原始数据只读。

当前版本视频通过临时 PNG 和 LeRobot 默认 H.264/yuv420p（CRF30）重新编码；这不是视频无损复制。
临时 PNG 会占用额外磁盘空间，适合先小样本验证；正式全量前需核验画质、吞吐和可用空间。
`conversion_manifest.json` 记录 split、原始 repo/revision/episode 映射、源文件散列、合同和编码方式。
转换成功不意味着训练就绪：norm 尚未计算，真实 GPU smoke 与最终训练验收另行完成。

## 旧合同隔离

OpenArm 的 16D `[右臂7, 右夹爪, 左臂7, 左夹爪]`、HQ degree 语义和 Piper 14D transform 都是历史/legacy。它们不能与 YAM 的 14D `[左6, 左夹爪, 右6, 右夹爪]` 混合，也不能复用其 norm stats、动作顺序或训练结论。

## 2026-09-07 Lego 实测清洗状态

用户后续明确收窄范围：本轮“清洗”仅指开源数据完整性验收，不做语义质量筛选、夹爪裁剪、
静止帧删除、重新划分 train/val 或全量视频重编码。范围警告不影响完整性通过判定。
保留现有原始数据，逐集检查 Parquet 数值/帧序列及三相机完整解码、帧数和时间戳；
发现损坏只报告具体文件，不自动删除或替换。没有上游逐文件校验值时，不宣称与上游字节级一致。

本节为本次上传后的最新实测，不把先前的 pending_upload 清点当作当前结论。
原始路径仍为 `/home/wuyan/lyj/YAM/YAM_data/ABC-130k-two-tasks/lego_sorting`，原始文件只读。

- 4458 train + 69 val，共 4527 集、10,539,323 帧：文件齐备；全量 14D、有限值、帧序号和时间戳检查通过。
- 287 集的 observation.state 夹爪值略超名义上限，最大约 1.003627；action 夹爪均在 [0,1]。
  此项标记为 warning，不裁剪读数、不删除整集；不能从数值范围独立证明物理单位。
- train/95（3428 帧）与 val/421（2567 帧）完成全部三路视频解码验证。
- train/95 的实际 LeRobot 转换与首/中/尾帧回读通过。小样本发布到
  `/home/wuyan/lyj/YAM/YAM_data/processed/lego_sorting/smoke-train-20260907`，不是全量训练集。
- 全量视频逐帧审计已在 tmux `lego-full-audit` 启动，限制单核、nice 19、最长 24 小时；尚未完成。
  报告目录 `/home/wuyan/lyj/YAM/env-transfer/lego-full-audit-20260907`，
  `full.json.incomplete` 不作为完成结果；`full.json` 发布后还须检查 counts，不能仅凭进程结束宣称全部合格。

全量数值报告为 `/home/wuyan/lyj/YAM/env-transfer/lego-clean-20260907/lowdim.json`。
新增 `audit_yam_subset.py --lowdim-only --progress-every 100`，仅验证数值，状态为 validated_lowdim，
绝不冒充 validated_structure；warning 不自动转成 rejected。默认 full 模式才逐帧解码全部视频。

本次转换合同保留在同目录 `contract.json`，副本写入小样本的 conversion_manifest.json。
[XDOF 原始格式说明](https://huggingface.co/datasets/XDOF/ABC-130k/blob/main/README.md) 明确弧度和夹爪 0=闭、1=开；
[固定版本 LeRobot 字段定义](https://huggingface.co/datasets/lerobot/abc_130k_v3_train/blob/68651e4929d9fb00f798937b2d62617cab5c771d/README.md)
确认左右臂 14D 顺序。将原始单位用于该 LeRobot port 仍是与样本范围一致的推断，尚未做 raw-to-port 逐值对照或实机标定。
当前转换保留数值、不做尺度转换，输出 training_verified=false；norm stats 与训练验收仍未完成。

## 2026-09-07 全量 LeRobot 转换启动

用户已授权全量转换，但禁止覆盖原始数据。当前任务在完整性验收之外增加格式转换，不增加语义筛选。
`scripts/convert_yam_subset.py --video-mode copy` 使用 LeRobot 0.5.1 的公开 metadata API 重建 v3 数据：
视频独立复制、目标 SHA-256 校验并完整解码，不重编码、不建软/硬链接；数值不缩放、不裁剪、不删帧。
逐集重建连续索引、任务标签和来源映射；video image stats 不伪造，OpenPI norm stats 仍需另算。
旧的 reencode 模式保留供小样本/显式使用，批量入口固定使用 copy。

批量入口 `scripts/run_yam_conversion.sh`，远端 tmux `lego-convert-v1`，单 CPU 核、nice 19、最长 48 小时。
按 val → train 顺序转换，发布目标：
`/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/{val,train}`。
运行期间父目录带 `.incomplete`；两个 split 都通过回读才发布父目录。既有目录拒绝覆盖，失败产物保留排查。
日志 `/home/wuyan/lyj/YAM/env-transfer/lego-convert-v1-20260907/conversion.log`；
只有 `CONVERSION_COMPLETE=...` 才代表整个版本发布完成。当前已启动，不能声明全部完成。

转换使用已验证的 `/tmp/condapi-yam-smoke.CzQI6oAj/env`，输入输出均在共享盘；正式共享环境安装独立继续。
约 92.6GB 输入（视频约 91.5GB），输出另占空间。每路视频复制时记录 SHA-256，复制后检查目标 SHA-256，
全部源文件在发布前重查 size/mtime，Parquet 另重查 SHA-256；不是对视频做发布时二次源文件全量哈希。
不变性检查不能防御外部刻意改内容并恢复同 size/mtime 的操作，转换期间原始数据应冻结。

验证：本地 28 项相关测试通过，服务器 7 项转换测试通过；真实 val/421 复制转换及回读成功。
全量作业确认逐集推进后，停止旧 tmux `lego-full-audit` 的重复检查，保留旧日志和未完成报告，
其 `full.json.incomplete` 不能当作全量通过证据；完整性验收由新转换流程执行。

## 断点续跑（2026-09-07 当前入口）

全量任务已切换为支持 `--video-mode copy --resume` 的版本。既有 val/69 集经源/目标哈希、数值和索引
复核后复用，训练集从带检查点的新流程开始；没有重做或覆盖已完成验证集，也没有改动原始数据。

- 检查点在共享盘的 `train.incomplete/resume_identity.json` 和 `resume_records/`（相对于版本暂存目录），
  不放在临时环境目录。记录源 manifest、文件 size/mtime、Parquet 哈希、合同、episode 选择和 LeRobot 版本。
- 每路视频复制、SHA-256 和全帧解码通过后，fsync 数据，再原子写入检查点；续跑校验目标哈希后复用，
  不重复复制/解码。没有有效检查点或校验失败的派生文件移到 `.interrupted-*` 备份后重做，不删除原始数据。
- 中断的 metadata 不直接追加：从校验后的文件重建，原有 metadata 代次保留；已有 Parquet 先逐值比较，
  一致则复用。未知旧 `.incomplete` 目录不自动接管，源数据或配置改变时拒绝混用。
- split 和整个版本各有文件锁，禁止并发写入；已发布 split 只读复核，损坏时拒绝覆盖。重启不保证自动启动，
  但检查点保留；可用下面的入口恢复，已有同名 tmux 会拒绝重复启动。

在本地执行：

```bash
ssh yam-server 'bash /home/wuyan/lyj/YAM/env-transfer/lego-resume-v2-20260907/scripts/resume_lego_server.sh'
```

入口优先使用已出现 INSTALL_COMPLETE 的正式环境，否则使用已验收的临时环境；二者均不可用时停止，
需恢复环境或显式指定 `YAM_ENV_PREFIX`，不会因此删除检查点。tmux 仍为 `lego-convert-v1`，当前日志改为
`/home/wuyan/lyj/YAM/env-transfer/lego-resume-v2-20260907/resume.log`。新版代码在独立执行快照，不改服务器 Git 仓库。
进度 `reused_videos` 表示本集复用视频数；最终仍以 `CONVERSION_COMPLETE` 为整个版本发布标志。

验证：本地 32 项相关测试、服务器 11 项转换/续跑测试通过，覆盖中途异常、派生视频损坏、配置变化、
未知目录、并发锁、完成版本复核和批处理二次运行。实际服务器已输出 REUSED_COMPLETED_SPLIT=val 并继续训练集转换。

## OpenWAM YAM 数据投影

XR-1 使用单独的末端动作合同，不继承本节 OpenWAM 的 joint-action 投影；其字段、split 与 FK 审计门槛见 [10 的 XR-1 原生训练入口](10_vla_platform.md#2026-09-24--xr-1-原生训练入口)。原始 YAM 14D `action` 保持关节/夹爪语义，派生末端 JSON 只能写新目录，并记录来源 episode、帧对齐、坐标系和单位。

### YAM HIL 人工专家帧到 XR-1 原生训练数据

数据侧接口 owner 是只读参考 `/home/wuyan-lyj/YAM/yam-abc-reproduce/yam_abc_reproduce/hil/xr1_dataset.py`。先在具备该版本 YAM 数据环境的主机用 `python -m yam_abc_reproduce.hil.xr1_dataset <源episode> --output <新sidecar目录>` 导出 sidecar。它按 `expert_valid`、`source=human`、观测有效性、等待边界、连续 tick/epoch 和源存储片段切段，并使用官方 YAM `linear_4310/grasp_site` FK 分别计算同帧观测与**实际下发**目标。每个 sidecar 保留原时间、tick 和三相机帧号；不能把过滤后的行再次无条件拼接。

`adapters/xr1/prepare_hil.py` 接收该 sidecar 和对应原始 HIL episode，只允许 `mock=false`、无错误、已完成的 30 Hz 录制，拒绝 aborted/discarded；源结果非 success 时原生标签为 `ongoing`，不冒充成功。调用者显式选择 `train` 或 `val`，并提供与源 manifest 的 `task` 完全相同的 instruction。程序将每个不少于 30 帧的连续专家段转换成独立 XR-1 JSON 和从原始三路视频重编码的连续专家帧视频；摄像头源帧重复时仍保留重复帧。产物位于新目录，不写回源 HDF5、视频或 sidecar。末端位置以米、旋转以旋转矩阵表示，均位于各臂底座坐标系；动作来自 `submitted_action` 的 FK，关节和夹爪仅用于 proprio 与夹爪目标，腰部/底盘填固定零。原始帧号、时间、tick 和源文件摘要保存在产物 `provenance.json`/`manifest.json`。

每个源 rollout/session 的 train/val 划分须先经数据审核并在外部清单中固定；不得将同一源 rollout 的不同专家段分入两侧。少于 30 帧的段只计入 `skipped_short_segments`，不生成 XR-1 样本。`normalize.json` 只对 train JSON 使用固定上游 `third_party/xr1/tools/compute_normalize.py` 计算，并把每个训练 JSON 的 SHA256 记录到 `train_json_sha256`。`fk_audit.json` 的单位、坐标和目标时序真值仍需针对真实数据核验，转换脚本不会自行把它标成通过；训练入口继续执行该 gate。真实 HIL 数据位置与 split 尚待提供，当前本机 YAM 目录仅发现 mock 录制，不是可发布的真实训练版本。

### 50h 乐高 LeRobot 数据的 XR-1 末端派生版

2026-09-24 用户指定将服务器既有 50h 乐高分拣选集用于 XR-1。它是已发布 LeRobot v3 数据，不含 HIL 的 `expert_valid`、`source_tick` 或 `source_video_index`，因此不能调用上述 HIL sidecar 筛选并把所有帧宣称为新采集的纠正专家帧。源 train repo 为 `/home/wuyan/lyj/YAM/YAM_data/processed/lego_lerobot_v1_20260907/train`；50h 选集固定在 `/home/wuyan/lyj/YAM/training-assets/lego_rtc_50h_20260921/episodes.json`，共 2,337 集、5,400,685 帧。原始 train/val 已分仓；验证集使用同一发布版本的 `val` 仓，不从 50h train 选集中再抽帧。

`adapters/xr1/prepare_lego.py` 对每个选中 episode 的同一行 `observation.state` 和绝对 `action` 各执行官方 YAM + `linear_4310/grasp_site` 的 MuJoCo FK，产出米制末端位置与 3×3 旋转矩阵、原 6 关节与夹爪 proprio/目标。模型束在 `third_party/yam-fk-model/`，来源 revision、XML/mesh 哈希和 MIT 许可同目录。服务器 XR-1 的 `decord==0.6.0` 实测不能打开源 AV1 视频，因此每集将源视频按 v3 `start` 帧偏移提取、以 H.264 `yuv420p/crf18` 重编码到新目录，并逐集核对输出帧数；这是有损图像派生，不是逐像素无损副本。原 `timestamp`、episode/frame 索引和源 `start` 仍在溯源文件。源数据和 Pi 50h 资产只读，派生输出写新目录；中断后仅允许相同 source/selection/FK 身份执行 `--resume`。源集没有逐集成功标注，派生 `trajectory_type` 设为 `ongoing`，不得在审核前宣称全为成功示范。

发布前核对总集数/帧数、逐集三视频偏移和原生字段。原转换 manifest 对 joint 单位为 rad、夹爪 0 闭 1 开的结论基于发布者合同及数值一致性，仍未完成独立 raw→port/真机标定；因此 FK 单位与同帧目标时序审计不能自动写为训练 gate 的 true。`adapters/xr1/compute_stats.py` 在 train/val 完整产物写成后，按上游相同的完整 30 步窗口与旋转向量编码流式累积 30×60 均值/标准差、14 个有效状态槽的精确分位数，并将每个训练 JSON 的 SHA256 写入 `normalize.json`；不读取原 val。验证集保留独立资产身份。

服务器转换于 2026-09-24 21:15+08 完成（exit=0），仅train统计于21:19+08完成（exit=0）。派生根目录为 `/home/wuyan/lyj/YAM/YAM_data/derived/xr1_lego_50h_eef_v1_20260924/`；转换和统计日志分别为 `/home/wuyan/lyj/xiaomi-robotics-1/install/prepare_lego_50h.log`、同目录 `prepare_lego_50h_stats.log`。2026-10-10复核：train为2,337集/5,400,685帧，val为69集/165,142帧，两个manifest和全部episode JSON均存在；统计绑定的train JSON摘要映射及train manifest摘要一致，完整30步窗口数5,332,912。首条train episode 0的三路H.264各3,428帧，历史decord首末帧读取通过。**转换与统计已完成，训练审核尚未完成**：复核时仍无 `fk_audit.json`，不能将转换成功解释为单位/坐标/目标时序已核实，也未在此次复核重新逐帧解码全量视频。

2026-09-22 接入 `adapters/openwam/data.py`；输入支持 LeRobot v2.0/v2.1 独立 episode 文件与 v3.0 共享 Parquet/MP4 分片。服务器正式数据实测为 v3.0、14D、30fps；共享 Parquet 按 episode_index 筛选，视频根据每相机 metadata 的 from_timestamp×fps 计算文件内起点，并核对片段时长。原始数据只读，不自动转换版本、不写回统计。14D 次序仍为 `[左6关节, 左夹爪, 右6关节, 右夹爪]`，状态读 `observation.state`，监督读同一行的单数 `action`；要求调用者确认数据为绝对目标，本次不推断物理单位、不做 Pi delta 或 EEF/FK 转换。两个夹爪保留连续值，不翻转、二值化或重标单位。

三相机按 `top_rgb / left_rgb / right_rgb`（完整键前缀 `observation.images.`）排列为上部全宽、左下/右下各半宽的原生 OpenWAM L 形 RGB 拼图，训练和离线推理复用同一函数。禁止缺失腕部相机时静默填黑。动作窗长为 `num_frames - 1`，视频每 `video_stride` 抽帧；例如 33/4 得 H32 与 9 帧视频，与 Pi H50 不同。窗口不跨 episode；末尾动作零填充并屏蔽 loss，视频复制最后一帧并用 `video_mask` 标明填充；proprio 仅为窗口起点状态。

动作/状态分别使用所选训练 episode 的 min-max 或 z-score 统计。统计 JSON 带源 metadata/parquet 哈希和清单，原生 checkpoint 保存 `normalization_stats.npy` 的 `joint`（动作）与 `joint_state`（状态）两个区块。推理使用原生定向 normalizer：状态归一化使用 state 统计，动作反归一化使用 action 统计，保留 fps 为返回动作周期的来源。

14D 模型不 padding；80D 预训练模型要求配方明确指定 14 个唯一槽位。先在原始 14D 空间归一化，再 scatter 到 80D 并构造维度 mask，推理先 gather 再反归一化。此映射不宣称与上游 EEF 槽位有相同物理语义；没有已核对的槽位合同不得声称 pretrained policy 可直接控制 YAM。动作头维度不一致的微调会拒绝，不能悄悄随机重建头。配置与验收状态归 [10](10_vla_platform.md#2026-09-22--openwam-微调接入)。
