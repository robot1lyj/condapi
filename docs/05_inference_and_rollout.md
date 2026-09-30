# 05 · 训练后 policy 端侧与 IPC smoke

本页保留 YAM 训练后 policy 的输入输出协议、Thor 本地推理验收和 Thor↔3588 的直连以太网推理通道。模型运行在 NVIDIA Jetson AGX Thor；3588 负责相机采集和机械臂控制，YAM 机械臂控制、CAN、GUI、home pose、控制频率和真机安全不在本仓库适配范围，也不要从独立 YAM-ABC-Reproduce 代码推断本项目合同。

Thor 系统、容器、Pi0.5 转换和 TensorRT 方案见 [08 · Thor 端侧部署](08_thor_edge_deployment.md)。

## RLT eager 实验服务（2026-09-30）

新增 `scripts/thor/serve_pi05_rlt_eager.py --rlt-manifest ...`，复用eager Pi服务与RTC适配器；训练及快照选择见 [03](03_training_and_evaluation.md#rlt-服务端实验接口2026-09-30)。这是尚未GPU/延迟/真机验收的实验入口，未改生产TRT。清单指定`feature_extractor=pi_eager_rlt_final_prefix_v1`时，读取同次冻结Pi prefill最终image prefix，经固定token encoder输出z后调用左右actor；不重复调用基础Pi，不部署token decoder。加载时核对behavior/token/actor/基础checkpoint/norm/feature ID与有限FP32成员。

外部协议仍为 **yam-parts-v1**，同H50×14基础动作、左右H50×6的u、B_rad、editable_mask、features.z、context和behavior snapshot字段。客户端按原自动规则决定当前arm是否应用 `B_rad×u`；服务端不决定抓取阶段、不控制夹爪、不生成奖励，不增加VLM或阶段网络。50 mm为客户端可配置介入高度，目标高度独立；关爪后新鲜、连续的`abs(effort_nm)>0.65`用于结果确认，达到高度不自动判成功。持物/放置禁止重入仍由客户端裁决。

- **off：** 原基础推理，不提取token或运行actor。
- **shadow：** 验证token出口，候选为零，客户端执行基础策略。
- **collect：** `u=tanh(mu+fixed_std*epsilon)`，只采样一次；初始零均值actor同样使用tanh前高斯噪声，不叠加旧PARTS的tanh后噪声。固定seed/context/behavior/arm可重现候选，std与训练快照一致。
- **eval：** `u=tanh(mu)`，无探索；必须有learned actor。

RTC前d行u为零且不可编辑，原已承诺物理动作及RTC回退边界保持。旧PARTS `--parts-manifest`及均值池化/TD3路径保留；actions-only TRT仍仅off/shadow，不能据此宣称RLT collect可用。

启动参数复用原服务：`--checkpoint`、`--norm`、`--warmup-sample`、`--rtc-mode off|trained`、`--num-steps`、`--max-joint-step-rad`及host/port；trained另需匹配的rtc_manifest.json与`--warmup-rtc`。新contract的feature ID取自选定token，客户端按审核后的manifest声明配置新run，无需协议新增字段。发布快照不会热替换运行模型。本轮未同步Thor/客户端或自动发送任务；真实样例的token/动作数值、延迟、RTC与跨IPC验收待完成。

## PARTS 左右抓取的服务端与客户端合同（2026-09-30方案）

**状态：服务端首版已实现，离线协议/数据验收通过，未远端部署。** 用户已确认旧夹爪、动作相位和左臂问题解决，并澄清当前失败是下降高度不足；用户确认客户端已有末端位姿/FK。用户确认将介入高度、目标高度和关爪后的抓取判据分开，并指定 **`h_entry` 可配置，初始值为50 mm（0.05 m）**；目标高度和高度测量坐标/参考点待核对。采用高度误差辅助奖励与抓取结果奖励的设计，具体公式/权重待定。抓取结果用用户确定的 `abs(effort_nm)>0.65`，到达高度不自动发抓取成功奖励。学习目标归 [03](03_training_and_evaluation.md#parts-左右抓取残差学习方案2026-09-30)，数据与字段归 [04](04_data_contracts.md#parts-抓取-rl-回放数据合同2026-09-30方案)。

客户端开发的执行入口为 [YAM客户端实施计划](reference/parts_client_handoff.md)，含构建范围、采集阶段、双方依赖、验收交付和可直接执行的指令。此前按用户授权交给客户端任务；2026-09-30用户追加要求：后续客户端任务指令须先展示具体内容并经用户审核，禁止自动发送新任务/补充指令。condapi当前优先完成服务端。

**自动资格现状：** 用户要求不人工标记、不调用额外VLM，采用本地规则自动切换，并反馈客户端已实现。此前只读核对的提交为 `714e9c409cab9fc29655f2900fff2e2cf7f978e1`（`fix(parts): allow preclosing grasps without spatial or opening gates`），不再沿用`98e8dee`显式标记首版作为当前结论。客户端生成资格并跟踪持物/释放及回撤，服务端消费协议字段、不训练阶段判断器。本轮未修改客户端、发送新任务或验收现场规则。[原改造计划](reference/parts_client_handoff.md#本轮客户端改造自动规则选择器待审核的实施方案)保留设计来源，实际参数以采集run及现场验收为准。

### 本地接口证据与责任分配

本次仅检查condapi本地源码，以及只读参考 `/home/wuyan-lyj/YAM/yam-abc-reproduce` 的客户端协议/录制文档和对应 `hil/policy.py`、`hil/policy_process.py`、`hil/rtc_protocol.py`、`hil/grasp_diagnostics.py`。没有读取或操作3588运行系统/机械臂控制/相机实现；没有将YAM控制代码复制进condapi。

| 所属侧 | 责任及拟新增能力 | 可以复用的接口 |
|---|---|---|
| YAM/3588客户端 | 唯一的高度门控/进入/退出/奖励裁决；在实际目标tick启用活动臂动作残差；提交动作、RTC承诺、源索引、末端位姿/力矩与三图录制；重置执行与操作员事件 | 单写入者、30Hz提交、PolicyWorker/ProcessPolicyClient、RtcPolicyClient、原始HDF5/MP4、grasp_diagnostics快照；FK存在为用户确认 |
| Thor服务端 | 冻结Pi基础推理和同checkpoint视觉特征；左右下降抓取actor候选、collect探索、token回显及请求资产记录；服务端不生成SDK/复位控制指令 | Pi系列容器、物理H50/14D逆变换、RTC sampler、现有WebSocket端口 |
| GPU训练服务器 | 发布后数据审计、独立残差TD3+BC/twin critics、成功重加权、snapshot导出 | 独立Conda prefix、Slurm/tmux、现有模型资产/清单管理；训练不在工作站或Thor执行 |

本项目采用**Thor预计算候选，客户端按tick决定是否应用**。这是针对本地高度/力矩与RTC时间轴的工程设计，不称PARTS原作者已公开的服务端/客户端分工。不能把力矩成功判定放在约一次Pi推理周期之后才响应；候选应在下降入口之前准备好，不能等已经关爪才开始请求修正下降轨迹。

### 进入、退出与单臂状态机

高度统一约定为末端参考点沿桌面法向、相对桌面的距离。`h_entry`在基础策略仍能到达的下降段，用来启动修正；`h_goal`是待核对的可抓取深度，用来指导下降，两者不共用一个值或事件。目标需核对夹爪参考点/指尖偏移、积木高度和抓取姿态；统一的接触前平面只有在各目标适用同一抓取高度时才能共用。若目标还在抓取位置上方，应作为接近阶段的中间目标；是否在该高度交还须单独验证。

运行参数 `h_entry_m` 在客户端 `98e8dee` 配置及状态机中已实现，左右初始均为 `0.05`；界面/交接显示50 mm，协议及网络输入统一使用m。每个run记录实际生效值及高度参考点/标定版本，活动attempt内锁定参数；后续调整通过配置生效，不写死在状态机或网络中。50 mm是用户指定初值，不代表现场已验证可到达性或坐标标定；不能直接把未转换的机器人base系Z与0.05比较。`h_goal`独立配置，目前没有指定数值。

每臂独立跟踪 `READY → ACTIVE_DESCENT → ACTIVE_CLOSURE → EXIT_PENDING → WAIT_REARM → READY`。首版同一tick最多选中一只arm的actor，保持论文单一active bottleneck语义；同时满足下降进入条件时用明确、可记录的轮换优先级处理，未选中arm按基础策略执行，不伪造其RLattempt。

- **进入：** READY且处于空手接近目标时，客户端按FK末端位姿判断手臂正在下降并跨入 `h_entry` 对应门控区间，启用ACTIVE_DESCENT；关爪开始再进入ACTIVE_CLOSURE，两段属于同一attempt。`h_entry`初始为0.05 m，下降跨越阈值的候选规则为前一有效高度>阈值、当前有效高度≤阈值，具体抗抖/初始已在阈值内的处理待实现合同确定。高度必须使用约定末端参考点、桌面法向和m单位；关节q中的某一维不是高度，内部控制phase也不能自动充当任务下降阶段。`h_entry`需位于基础策略仍能可靠到达、但未接触积木的位置；如果入口低于基础策略的停住高度，RL永远不会启动。下降速度判定/抗抖、工作区及空手判定待定。客户端记录进入、关爪、首次实际残差tick；候选迟到时保留真实延迟。
- **成功确认：** 只在ACTIVE_CLOSURE阶段判断有效、新鲜、SDK更新时间严格推进的样本是否 `abs(effort_nm)>0.65`，不在尚未关爪的下降段直接判成功。建议以连续时间 `confirm_s=0.1` 过滤单次尖峰，以 `max_feedback_age_s=0.1` 过滤旧反馈；这是当前离线诊断已有默认值的工程候选，未获本轮数值确认。按实际时间累计，不能把三个30Hz采样点误称跨度恰好0.1秒；缺帧/重复快照不继续累计。
- **成功退出：** 条件确认后进入EXIT_PENDING，停止新的探索残差并请求基础策略依据实际修正后的状态/RTC前缀生成交还计划。不能在未承诺区域把关节offset生硬清零；只有新计划的最终关节连续性检查通过，才能在承诺边界交还。`handback_effective_tick`用新鲜力矩再次检查仍 `>0.65`，才给一次抓取结果分量 `grasp_reward=1`；总奖励另含按合同计算的高度辅助分量。条件已丢失则在预算内返回ACTIVE_CLOSURE。普通模式若需要残差逐渐退场，整段实际退场动作与来源同样记录，不能隐藏成纯基础动作。随后基础策略负责抬升/搬运/放置。
- **失败退出：** 有效反馈下未成功就重新张开，或达到显式下降抓取预算/已审查的最低高度边界，结束attempt，`grasp_reward=0`，保留实际高度辅助分量。预算/边界属于子任务终止条件，训练不bootstrap；具体timeout/最低高度不按论文的一般3–15秒或任意毫米数拍定。事件终止tick和RTC旧残差实际结束tick分别记录。客户端故障/断联/人接管/反馈失效属于取消，reward=null，不能当抓取失败训练。
- **重新进入：** 成功/失败后进入WAIT_REARM。自动规则方案要求发生新的实际张开/释放确认、末端回撤到入口上方，且原attempt及旧残差承诺结束，再回READY；当前首版仍以提交的张开目标和高位反馈判断，实测张开确认待客户端改造。持物后的持续关爪命令不能重复触发新attempt。成功交还后、明确释放前的力矩下降另记post_grasp_loss供整任务评测，不重复改写已终止的局部reward。

0.65阈值不会自动增加速度/空夹位置条件。现有离线Detector还含这些判定且只输出contact_candidate，不能直接用其结果替代用户规则；复用的是原始反馈字段、时间检查方法和可选确认参数。

### 动作和RTC配合

用户已确认保留当前六维关节增量动作空间；高度/目标/误差用于状态、门控和奖励。下面的joint_delta_rad/H50×6合同继续作为采集、训练和执行依据，自动资格改造与动作维数选择分别管理。

Thor返回的 `actions` 仍是有限 `(50,14)` **基础物理目标**；当前关节空间候选方案的 `parts` 返回左右 `U_left/U_right`，各为 `(50,6)`，以及六维rad边界 `B_left/B_right`、适用tick和contract SHA。基础Pi先完成原norm/delta/32D逆变换，再叠加物理关节增量；两种delta不混用。B待审查，不给默认猜值。若选单维高度delta路线，须另换残差shape/单位/IK映射合同，而不是复用旧 `(50,1)` 夹爪合同。

客户端计算 `a_final = constraints(a_base + M_arm ⊙ B_arm ⊙ U_arm)`，候选mask为左0–5或右7–12；另一臂与两夹爪按基础路径。记录原U、应用差和最终提交；最终组合后的关节跳变、现有动作限位和高度边界必须重新检查，不能仅依赖Thor已经检查过的基础动作。服务off返回原服务语义，shadow实际残差为0，collect有探索，eval无探索。基础与残差对照保持同一最终映射，RTC继续不叠加TDA/普通夹爪变换，后续编辑均记录来源。

RTC已有 `(d,14)` 前缀在两侧均不可重写：Thor候选前d行不参与本次叠加，客户端只修改未承诺后缀，然后把**约束之后的最终目标**承诺进RtcTimeline；下一请求送回该实际修正后的前缀。前缀继承原request来源，不能把旧残差归到新request。关闭/退出残差只影响未承诺目标；HOLD/介入/故障按现有客户端合同取消时间轴，并把受影响attempt标为取消。

候选随基础H50提前返回，在下降入口对应tick启用并持续到关爪结果。高度门控使用所约定参考点的实际反馈位姿，同时记录基础目标位姿；不把计划已下降误当实际已到位。不能重写已承诺tick“提前介入”，保留真实first_residual_tick。网络时延及控制周期沿用现有合同，不靠放慢30Hz或强制执行整H50补数据。

### WebSocket扩展和客户端透传

拟沿用直连Pi地址及msgpack-numpy，在原 `type=infer/obs/rtc` envelope增加可选 `parts`。不建立左右臂各一个Pi服务，不让训练服务器成为机器人请求链路的必需节点。以下字段已由服务端v1实现；不能据此推定Thor或客户端现场已部署。

| 消息 | PARTS字段 |
|---|---|
| handshake | 新增metadata.parts声明协议/模式、contract SHA、per_arm索引/边界、shape/rad语义、feature schema与snapshot引用；确切字段见下节，不得以旧夹爪contract启动新collect/eval |
| infer请求 | 新增parts.context/arms/scheduler关联run/session、epoch/request/observation、arm阶段/attempt/高度/FK/力矩及RTC队列；obs/rtc原结构保持 |
| infer回复 | 原actions，parts.context原样回显、左右候选u/B_rad、behavior snapshot/探索标记、feature_ref、contract SHA及可编辑mask |
| attempt反馈 | 异步可靠回执：event ID、进入/实际残差开始/终止/交还tick、结果、reward/null、已采用request与model_index区间；丢包重发，event ID幂等 |

现在 `_split_request`显式提取obs/rtc/parts，旧 `_split_payload`二元返回保持；WebsocketPolicyServer的可选 `parts_extension`提供握手与候选。未配置时off/旧请求仍走原路径，shadow按无能力处理，collect/eval明确拒绝。`scripts/thor/rtc_onnx_sampler.py`及TRT adapter现在只导出/接受actions，新增feature出口必须使用新的、经数值与延迟验证的导出版本，不能假设加字段即可取到视觉特征。

YAM `ProcessPolicyClient`已经转发原始response，但 `PolicyWorker`构造Reply时只取actions、timing和plan，会丢掉parts。客户端实现需贯穿 `infer_rtc` envelope、Reply、录制details和每tick动作来源引用。不能只在WebSocket收包处保存候选，实际提交时却没有可验证的来源。

反馈/大数组记录不另占Pi单在途socket做阻塞往返。首版使用本地持久outbox和已完成episode/请求表的异步发布，在独立的后端传输路径作ack/重试；即时进入/退出由本地完成。若后续增加独立反馈API，其服务不持有SDK，ack和learner更新都不在提交路径等待。模型身份可以保存在condapi交接manifest与sidecar，保持既有YAM产品不强制记录每帧模型名称/指纹的边界。

### PARTS 消息字段规范

本节是本地服务端已实现的v1字段，客户端和远端部署仍须验收。传输沿用msgpack-numpy；原metadata/obs/rtc/actions和时钟语义不变。`parts`均为新增可选对象，off保留旧调用；shadow在旧服务缺少能力时仅记录unsupported并按原基线执行。collect/eval切换前必须完整握手。

| 位置 | 字段和约束 |
|---|---|
| metadata.parts | protocol=`yam-parts-v1`、contract_sha、supported_modes、residual_space=`joint_delta_rad`、horizon=50、state_dim=14、action_dt=1/30、per_arm.left/right.indices、per_arm.left/right.B_rad、feature_schema_id、behavior_manifest_ref、behavior_snapshot_id、state_schema、feature_dim |
| infer.parts | protocol/contract_sha/mode、context、active_arm（null/left/right）、arms.left/right、scheduler |
| reply.parts | protocol/contract_sha/mode、原样context、behavior_snapshot_id、candidates.left/right、features |
| 发布event | schema/event_id/event_seq、context/attempt_id/arm/tick/time、kind/reason、奖励/原反馈引用；经原始包与独立outbox发布，不新增必须实时应答的Pi反馈请求 |

`context`包括 `run_id, session_id, epoch, request_id, observation_id, observation_policy_tick`，客户端生成、服务端完整回显；不改变已有客户端token。observation_policy_tick与RTC原字段相同，非RTC也记录其观测控制tick。错epoch/request/observation/contract回复不应用；迟到仍按原时间轴处理，不强行从第0行执行。

请求 `arms.left/right` 包括attempt_id、phase、eligible及其来源、height_m/h_goal_m/error_m、pose_frame/pose_ref/pose_sampled_at/pose_valid、force_feedback_ref或原始effort_nm与更新时间/年龄/valid，以及显式elapsed_s、confirmation_s、force_valid/effort_nm。所有数值沿用本机时钟域和本合同m/rad/Nm；误差为height_m-h_goal_m。null表示未设置/无效，不转为0。`scheduler`保存targets（50×14）、valid_mask/committed_mask（50 bool）及本观测时未完成/已承诺目标来源，RTC前缀的数值仍在原rtc对象；还记录是否已无活动attempt与旧残差承诺，供snapshot边界判断。

自动规则改造拟增加独立的selector_state、selector_schema/config_sha、reason_codes与entry_armed诊断；原phase枚举和网络状态维度保持。选择器规则/参数纳入新行为合同与contract_sha，待双方实施和联调；不能用当前握手通过推定自动选择器已验证。

每侧 `candidates`包括 `u`（50×6，有限且在[-1,1]）、`B_rad`（6个非负有限边界，与握手/manifest一致）、`editable_mask`（50个bool）、actor_snapshot_id和exploration_applied。索引只使用握手per_arm，左为0–5、右为7–12；本版本两夹爪和另一臂不在活动mask内。editable_mask前d行必须false，候选始终在所回显context和该次actions时间轴下解释；实际执行仍需客户端eligible/phase/承诺检查。collect探索由Thor在生成u时完成，客户端不再加噪声。

`features`包含feature_schema_id、feature_ref及校验身份；可选z数组有确切shape/dtype。特征绑定本次observation_id和冻结Pi资产。当前导出缺少feature时shadow记录missing；collect发布的可训练区间需能由condapi校验完整特征。behavior_snapshot_id引用含左右actor、基础Pi/norm/engine、B、feature/reward合同的不可变manifest，客户端在run.json留引用，无需逐帧输入模型名称。

parts字段错形状/非有限/合同或snapshot不匹配时按模式处理：shadow拒绝候选并记录原因，保持零实际残差；collect/eval取消受影响attempt并沿用既有协议故障HOLD流程。实际提交链及记录故障仍归YAM。握手未提供parts能力时不得启动collect/eval；不能将RTC悄悄降为普通模式。

### 服务端入口与能力边界

- `scripts/thor/serve_pi05_rtc_trt.py --parts-manifest ...`可选扩展现有TRT入口，只声明off/shadow，基础actions保持；当前engine无视觉feature出口，禁止用该入口collect/eval。
- `scripts/thor/serve_pi05_parts_eager.py`新增显式eager FP32候选入口，支持普通off或原训练式RTC adapter，禁用compile/TF32；冻结PyTorch Pi，作用域hook从同次embed_prefix获取三视图各自token均值并连接，不重跑Pi。要求实际checkpoint/norm SHA、feature维数/方案和warmup输入；collect/eval还须完整队列/高度状态，learned actor须同state/feature/动作合同及FP32/哈希核对。它尚未通过GPU、JAX数值、时延或真机验收，不能替代现有生产TRT的验收状态。
- `scripts/parts/build_behavior_manifest.py`仅标准库，在新目录复制固定snapshot的actor并生成behavior.json，或显式 `--zero-residual --feature-dim D`初始化。必须提供contract.json与base-identity.json；collect还须显式std/seed及非零已审查B，eval拒绝零初始化。learned snapshot按training.json与actors SHA核对，加载时继续核对bundle。它不会部署、启动服务或通知客户端。
- 可选TRT shadow manifest至少含contract、behavior_snapshot_id、feature_dim=0；它的supported_modes只能off/shadow。eager manifest另外固定feature_extractor=`pi_eager_image_prefix_mean_three_views_v1`、state_schema、实际D、base_identity的checkpoint_weights_sha256/norm_stats_sha256、左右actors及探索配置。示例/未填数值不意味着正式合同已确定。
- metadata.parts另声明behavior_snapshot_id/state_schema/feature_dim；回复z为同观测float32数组及SHA。actor候选单独返回，不在Thor改写基础actions；已承诺前d行u=0且editable=false。相同run/session/epoch/request重复调用拒绝，避免重抽行为；失败请求由客户端分配新ID。
- `rtc_prefix.py`保留控制端已提交prefix的原数值精度，model编码仍走原归一化；返回必要时提升容器dtype以避免float64目标被float32重写。delay=0保持原输出dtype，trained RTC拒绝仍不降级普通推理。

发布候选模板（只建文件，不启动设备）：

```bash
python scripts/parts/build_behavior_manifest.py --output /data/parts/behavior_initial \
  --contract /data/parts/contract.json --base-identity /data/parts/base-identity.json \
  --zero-residual --feature-dim <经验收的D> --exploration-std <显式std> --seed <固定seed>
python scripts/parts/build_behavior_manifest.py --output /data/parts/behavior_v1 \
  --contract /data/parts/contract.json --base-identity /data/parts/base-identity.json \
  --snapshot /data/parts/train_v1 --exploration-std <collect时的std> --seed <固定seed>
```

跨IPC smoke和GPU测试未执行。状态机及执行链属于YAM，服务端不能验证未记录的现场动作；当前待用户审核的客户端对齐项集中在交接计划末节，未自动发送。

### 部署和联调交付

1. **condapi本地实现：** 标准库协议/数据检查放控制层；独立残差网络/learner放模型环境，薄adapter调用；Thor候选wrapper接冻结Pi并增加feature出口。已创建 `adapters/parts/`、`packages/parts-rl/`、标准库 `vla_platform.parts`及 `scripts/thor/parts_policy.py`。普通/RTC旧入口及parts=off保持可用。
2. **YAM客户端交接：** 提供本节消息/状态机及04字段合同，由YAM项目实现本地selector/verifier、候选按tick叠加、RTC来源和recording扩展。condapi本次不修改外部只读YAM或3588。
3. **静态/回放：** 本地只验协议、状态机和纯数组时序。覆盖高度入口可到达/坐标方向、双臂入口竞争、下降→关爪、过旧/重复力矩、epoch/attempt错配、迟到候选、最终关节连续性、前缀不变、退出退场及取消不标失败；不运行训练smoke。
4. **Thor shadow：** 现场状态观测后按08部署候选，先做本地真实输入及跨IPC直连smoke、记录基础动作逐值不变和附加feature/actor时延。尚无actor权重时只允许零残差/shadow。
5. **采集与训练：** 完整批次/READY审核后，在服务器训练两个独立残差策略；B/确认/timeout/训练配方固定后方可collect。原始记录异步复制到新目录；训练server返回snapshot，Thor只在attempt和旧残差承诺均结束时加载。
6. **eval：** 固定snapshot关闭探索，完成03的局部与完整任务对照。自动物理重置的实现/验收归YAM；不以Thor服务代码替代机械臂控制。

## 1. 部署前 gate

必须同时满足：

1. checkpoint 参数元数据完整，且有与训练数据绑定的 `assets/yam/norm_stats.json`。
2. 当前全量微调配置是 `pi05_yam`，数据和 checkpoint 属于同一 14D 合同；旧 LoRA 配置不是默认入口。
3. Thor 已核验 JetPack/L4T、GPU、Docker runtime 和 Pi 系列容器内的 CUDA/JAX；采用其他候选后端时额外核验其依赖。镜像、只读模型挂载与端口发布按 [08](08_thor_edge_deployment.md) 记录。
4. 原 JAX checkpoint、golden 输入/噪声/输出已保存；任何候选后端均需按 [08](08_thor_edge_deployment.md) 做分阶段精度验收，不能只与转换后的 Torch 比较。
5. Thor 本地推理 smoke 确认输入键、输出 shape、有限值和 checkpoint/norm 绑定；随后必须做 3588↔Thor 的真实直连以太网 smoke，验证 observation/action 往返。

## 2. Thor 本地推理边界

默认数据流为：

```text
3588 camera/state/prompt
  == direct Ethernet / WebSocket or agreed transport ==>
Thor Pi 系列容器：YamInputs + norm
  -> 已完成离线原 JAX 对照的 TensorRT policy（当前 RTC 30000）
  -> YamOutputs + absolute action
  == direct Ethernet / action response ==>
3588 controller: 有限的 (50,14) YAM action chunk
```

端侧 bundle 必须来自当前指定的 `pi05_yam` 全量微调 checkpoint。成熟案例的 `pi05_libero` 是 7D、horizon 10；不能直接复用其权重资产、TensorRT engine 或 49/54 ms benchmark 来代表 YAM。YAM 需要三路图像、模型内部 horizon 50 和真实 14D 输出，详见 [数据合同](04_data_contracts.md)。

首个端侧运行顺序固定为：

1. 原 JAX policy 以实际 YAM 样本和同一份噪声数组生成 golden；保留原始全量 checkpoint。
2. 在 Pi 系列容器内核验 Thor 原生 JAX 可行性；若转换到 Torch，先审计 FP32 构造/存储、norm 绑定和未量化计算对齐。不同 checkpoint 通过配置选择，不各自建立常驻容器。
3. 精度通过后再按延迟需求决定是否导出 engine；FP8/NVFP4 和定制 FP16 是独立候选，必须与原 JAX 比较。
4. 在 Thor 本地用回放样本直接调用 policy，验证三路图像、14D state、prompt 和 `(50,14)` 输出；再用 3588 的真实 observation 做跨 IPC 直连 smoke。

原生 JAX、经审计的 Torch 和加速 engine 均须有 Thor 实测证据；具体精度、环境限制和验收层次由 [08](08_thor_edge_deployment.md) 持有。保持 `action_horizon=50` 与默认去噪 `num_steps=10` 分别记录。跨 IPC smoke 只证明传输和模型可运行，不能替代任务成功率验收。

## 3. YAM 输入输出合同

YAM policy 输入使用和 LeRobot 导出一致的键：

```text
observation.state                  float array, shape (14,)
observation.images.top_rgb         RGB image
observation.images.left_rgb        RGB image
observation.images.right_rgb       RGB image
prompt                              scalar string（或由 task 注入）
```

图像可由 `YamInputs` 兼容 CHW/HWC，正式 smoke 建议使用 HWC `uint8`。推理返回：

```text
actions: float array, shape (50, 14), all finite
```

模型内部 padding 到 32D 只属于 OpenPI 模型边界，不应把 32D 直接当作 YAM 真实动作发送给机器人侧。单位、限幅、执行频率和安全检查由已核实的机器人侧系统负责。

## 4. 本地 smoke 验收

不得以容器启动、端口监听或只返回 metadata 作为成功。按顺序检查：

- 同一批三路实际图像、14D state 和 prompt 在 JAX reference、PyTorch、TensorRT（或 FlashRT）中都能完成推理；
- 输出严格为 `(50,14)`，所有值有限，action 顺序和 norm asset 绑定到 YAM；
- 记录 engine/backend、config、checkpoint、训练 commit、norm 路径、JetPack/L4T、CUDA/TensorRT、warmup、时延和功耗模式；
- 端侧本地路径通过后，才允许进入低速、限位和人工急停可用的机器人侧测试。

旧 100000 服务使用 [serve_pi05_trt.py](../scripts/thor/serve_pi05_trt.py) 与 [smoke_pi05_ws.py](../scripts/thor/smoke_pi05_ws.py)；其独立真实回放和原 JAX 对照见 [100000 报告](reports/thor/100000.html)。当前RTC30000见[训练时RTC冷手册](reference/thor/13_trained_rtc_inference.md)。不要把旧 OpenArm 16D 工具重新作为默认入口。

## 5. Thor↔3588 网络推理通道

Thor 服务端在 Pi 系列容器内加载已验收的 TensorRT engine 与训练 norm，3588 通过 Thor 直连网卡上发布的 policy 端口发送 observation 并接收 action。Pi 系列固定 `ws://192.168.250.1:8000`，切 checkpoint/普通或RTC模式时更改服务配置并重启、重新预热和验收，**不因模型切换另设客户端 URL**，也不假定支持热切换。旧100000 Docker `pi05-infer` 已停；当前RTC30000为 `pi05-rtc-infer`、`--restart no`，**尚未实施 Compose**。GPU 接入与端口规则见 [08](08_thor_edge_deployment.md)。

### 2026-09-16 · 100000 服务历史验收（2026-09-17 已停）

- Thor 直连 URL：`ws://192.168.250.1:8000`；只绑定该网口，不绑定 Wi-Fi 管理地址。宿主 `curl http://192.168.250.1:8000/healthz` 返回 `OK`；`docker inspect pi05-infer` 为 running、restart=unless-stopped、当次重启0。运行状态在使用前复核。
- 入口为 `python /service/serve_pi05_trt.py`，Pi v6 镜像，独立服务目录 `/home/wuyan-lyj/thor/pi/services/pi05-100000-20260916`；容器只读挂载对应 100000 FP32 checkpoint、训练 norm、100000 W/80 engine、真实回放，tokenizer 缓存本地挂载。新请求由 Thor 生成 `(50,32)` FP32 噪声，客户端只发标准 observation；服务关闭 RTC。
- 握手 metadata 声明 `checkpoint_step=100000`、`backend=tensorrt`、`norm_stats_sha256=044aad51…4dcc`、`engine_sha256=70b366c5…9d54`、`precision=BF16 main/FP32 sensitive+time cache, no quantization, TF32 off`、`action_horizon=50`、`robot_action_dim=14`、`denoising_steps=10`。完整字段及哈希在 [Thor 本机 WebSocket smoke 回执](reports/thor/evidence/20260916/100000-service/local-ws-smoke-20260916.json)。未核实第 0 步相对 observation 的物理时间偏移，不在握手中臆造 `action_dt_s`/时间偏移。
- 返回 `actions` 为训练逆变换后的 50×14 **绝对目标**，顺序 `[左 6 关节, 左夹爪, 右 6 关节, 右夹爪]`。根据当前数据发布合同，关节数值按弧度、夹爪名义 0 闭 1 开；这是数据语义，不宣称硬件标定/限位。模型输出未裁夹爪；机器人侧不能把本服务输出视为已做安全约束。
- Thor 本机经直连地址发送一条真实记录的三路 224×224 RGB、14D 状态和 prompt，普通 WebSocket 握手及推理成功，返回有限 50×14。初次 `120W` 的服务端推理 `174.71 ms`、请求往返 `177.07 ms`；按用户 2026-09-16 明确要求切入 **MAXN 推理/测试阶段**，同一路 WebSocket 复测服务端 `106.10 ms`、请求往返 `107.16 ms`，GPU 1575 MHz、EMC 4266 MHz 锁到该模式上限。MAXN 原始回执见 [本轮 MAXN smoke](reports/thor/evidence/20260916/100000-service/local-ws-smoke-maxn-20260916.json)；这仍只是单次在线协议 smoke，不等于离线 180 次 P50 `104.34 ms`，也不是 3588↔Thor 跨 IPC、真机闭环或任务效果验收。
- **电源阶段规则**：以后推理和测试阶段启用 MAXN＋`jetson_clocks`；准备、下载、安装和日常空闲为 120W/动态调频/自动风扇。当时 Thor 宿主用 `maxn_session.py -- docker wait pi05-infer` 保持模式；容器后续重启暴露该等待方式会卡住，已停用并恢复120W。若主机断电/SIGKILL，Python 清理无法执行，重启后先检查 `nvpmodel -q`。旧100000 Docker容器虽有自动重启策略，但目前手动停止；不能只看旧容器配置推断当前服务。
- 当时 W/80 engine 仅接受有效 token ≤80 的 prompt；更长文本需为**本次 100000**另建 200 桶引擎，当时没有自动路由。历史日志可用 `ssh thor 'docker logs --tail 100 pi05-infer'` 查看；这不是当前RTC服务的日志入口。

### 2026-09-17 · 固定8000的 RTC 30000 服务

> 2026-09-17 晚间勘误：本节记录的是**事故前历史**，不是当前在线状态。旧 RTC 前缀误用均值/标准差，而训练/输入transform使用分位数；JAX旧对照也有同一错误。`pi05-rtc-infer` 已停止，8000无监听、Thor日常120W。正确前缀下的事故回放和9例重测、运行闸门见[事故报告](reports/thor/rtc-prefix-quantile-incident-20260917.md)；不得直接重启旧容器或旧10步回退。

> 2026-09-18 更新：上条是9月17日事故处理时状态；用户要求测试后，**新版**`pi05-rtc-infer`已在固定8000运行，Thor为MAXN。Thor本机9例两轮和事故4例协议smoke通过，握手`rtc_prefix_norm=quantile`及`rtc_joint_step_guard_rad_per_tick=0.2`。第二轮9例有4次本机往返约390ms，原因尚未确认；3588端到端和真机未验收。回执与当前运行边界仍见[事故报告](reports/thor/rtc-prefix-quantile-incident-20260917.md#2026-09-18-更新新版服务与-thor-本机协议测试)。

- 旧100000容器已停、权重与引擎保留。当前 `pi05-rtc-infer` 只监听 `ws://192.168.250.1:8000`；曾短暂试验的8001已停且无监听。后续Pi系列模型共用此固定URL，通过握手metadata区分实际 checkpoint/backend/模式。
- 握手 `rtc_mode=trained`、`backend=tensorrt_cuda_graph`、`rtc_max_delay_steps=10`，`checkpoint_weights_sha256=fc60f64cfd05906edda9f446e113c159e1df6ece4ea479e27ba22b01791d5f60`、`norm_stats_sha256=b38ac082a729a4825c1a946c965183125382f10ccb5a75da89c2427019d1fefe`，`action_dt_s=1/30`、第0步相对**observation数据行的30Hz policy tick**为0；相机曝光到物理控制tick的偏移未知。输出为 `(50,14)` 绝对目标，关节rad、夹爪连续值名义0闭1开，未裁剪。
- 2026-09-17 18:40 CST，用户指定的7步版本已替换固定8000上的旧10步FP32服务：checkpoint仍为30000，FP32权重、TF32计算、80-token文本桶、时间条件缓存、CUDA Graph，握手 `denoising_steps=7`、`precision_mode=fp32_weights_tf32_compute`、engine SHA256 `0b5cc53e2c4eaba58021b4018d471c3d8954633507d4d477088731f4a12455ce`。旧10步容器/引擎以 `pi05-rtc-infer-fp32-10step-preserved-20260917` 保留并停止；8001预检容器也已停止，无8001监听。上线前临时8001与上线后固定8000各完成9/9真实三相机/状态/prompt、d=0/1/10协议smoke：有限50×14、前缀逐位不变。固定8000本机MAXN服务/往返P50约194.26/195.03ms；首个请求约211.96ms往返，9例P95含该首请求约205.41ms。原10步FP32约1033ms为历史结果，不是当前服务。原始收据在Thor `/home/wuyan-lyj/thor/pi/artifacts/rtc-30000-trt-tf32-cache80-7step-20260917-r1/ws-smoke-fixed-8000-20260917-r1.json`；数值与回退见[RTC冷手册](reference/thor/13_trained_rtc_inference.md)。3588→Thor跨IPC和真机闭环尚未验收。
- 当前MAXN由临时 `thor-pi-maxn-30000.service` 监视容器，2026-09-17 18:39 CST 实测 active、MAXN、GPU 1575MHz/EMC 4266MHz 锁频；切换旧容器时曾自动恢复120W，已在新容器启动后重新建此临时服务。停止 `pi05-rtc-infer` 后会恢复120W。容器 `--restart no`，临时 systemd unit 也不是持久开机配置；重启机器后必须重新启动容器和MAXN会话并核对模式。

### 2026-09-14 直连网络基线

用户授权本轮仅对 3588 的独立网口进行联调，不读取或修改其机械臂、相机和控制实现。两端 NetworkManager 均已保存 `yam-thor-direct`，自动连接并绑定物理接口和 MAC：

- Thor `enP2p1s0`：`192.168.250.1/24`；
- 3588 `lan1`：`192.168.250.2/24`。

直连配置不设 gateway、DNS 或附加 route，`ipv4.never-default=yes`、IPv6 disabled、MTU 1500、自动协商。实机插线后两端均为 `UP/LOWER_UP`，协商 2500 Mb/s、full duplex；双向各 10 次 ICMP 为 0% 丢包，Thor→3588 平均 0.245 ms、3588→Thor 平均 0.235 ms，双方 TCP/22 均可达。两台机器到公网的路由仍分别使用 Wi-Fi，绑定 Wi-Fi 接口的 HTTPS 请求均返回 HTTP 200。Thor 当次 Wi-Fi DHCP 地址为 `192.168.110.250/23`，3588 为 `192.168.110.140/23`；DHCP 地址是临时观察值，直连服务应只使用 `192.168.250.0/24`。

这是 2026-09-14 的网络层基线；当时尚未启动 policy 端口，也没有完成 observation/action 或机器人闭环 smoke。上面的 100000 服务是 2026-09-16 的新事实，不能回写为网络基线当天的结果。旧 `scripts/serve_policy.py` 载入的是普通 policy，不是本次 W TensorRT engine，不作为 100000 默认启动入口。

使用真实 `openpi-client` WebSocket 协议（或后续核定的等价直连协议）从 3588 发送三路图像、14D state 和 prompt，检查：

- 输入无缺少 image/state/prompt 错误；
- 输出严格为 `(50,14)`，无 NaN/Inf；
- metadata 的 `robot_action_dim/output_action_dim=14`、`model_action_dim/action_dim=32`、`action_horizon=50` 与 checkpoint config 一致；
- 日志记录 commit、checkpoint、config、端口、prompt 和 smoke 结果。

跨 IPC 直连 smoke 由 3588 侧测试者执行并保存报告；本仓库只提供 Thor 推理服务和上述 Thor 本机回执。不要触碰 3588 的系统、控制进程、相机进程或其他用户任务。

## 6. 停止条件

Thor 本地或跨 IPC 直连路径出现以下任一情况，停止 rollout 并回到 checkpoint/data gate：

- shape 不是 14D 输入或 50x14 输出；
- 动作非有限、checkpoint/norm 不匹配、转换产物未通过 JAX reference 比较；
- 三路图像缺失、通道/时间同步不明；
- 把 OpenArm/Piper 单位、顺序或 transform 混入 YAM；
- 只看到端口监听或服务进程存在，尚无 Thor 本地推理和 3588↔Thor 网络结果。
