# FineART T恤10小时：原生训练准备合同

状态：**PREPARATION_NOT_TRAINING_READY**。这份材料准备独立 FineART 全量微调配置，不表示下载、数据质检、语言标签、训练环境或服务器资源已验收。未启动本地/服务器训练；服务器维护期间不访问训练服务器。Thor 的下载与标注由数据工作流负责，这里的工具不连接设备，不导入 Torch/JAX/LeRobot，不安装环境。

## 配置资产与固定版本

- 原生配置模板：[fineart-tshirt-10h.json](../../configs/native/fineart-tshirt-10h.json)。JSON 字段属于上游 `TrainPipelineConfig`；大写路径占位符在准备阶段不可直接运行。
- 准备清单：[fineart-tshirt-10h.preparation.json](../../configs/native/fineart-tshirt-10h.preparation.json)。记录状态、版本、候选规模、资产哈希及未满足条件；它不是训练器输入。
- 标准库解析器：[prepare_fineart_tshirt_training.py](../../scripts/prepare_fineart_tshirt_training.py)。只校验与生成配置、来源 lock，不调用训练入口。

固定上游 [huggingface/lerobot@f5c9bc02137c331b009eccd44ece6f11b62187a8](https://github.com/huggingface/lerobot/tree/f5c9bc02137c331b009eccd44ece6f11b62187a8)，固定基座 [lerobot/fineart_vla_base@b6cf4faf2630f971dd390bc32365e9d9c330da8e](https://huggingface.co/lerobot/fineart_vla_base/tree/b6cf4faf2630f971dd390bc32365e9d9c330da8e)。2026-11-03 核对旧名 `lerobot/FineART-VLA-subtask-midtrained` 也解析到同一 revision。不得默默跟随 `main` 或用另一个同名目录代替。

准备清单固定了 checkpoint config、保存的 preprocessor、完整 `action_tokenizer/` 和 `model.safetensors` 的 SHA256。模型约9.35GB，解析时完整验哈希可能较慢；这是读取准备资产，不是训练。权重哈希来自固定 revision 的 HF LFS metadata，其余小文件从同 revision 读取计算。当前没有为本方案下载权重。

文本 tokenizer 也固定为 `google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c`，准备清单记录两个大文件的 LFS SHA256和三个小文件的Git blob SHA1。解析器要求独立本地snapshot并验哈希，避免词表跟随main；Git blob SHA1使用含`blob <bytes>\0`头的Git算法。该上游标为manual-gated，未来使用已有获准缓存；本次仅查询公开metadata，没有下载或申请访问权限。

## 数据规模和发布边界

源任务仅 `fold and stack the t-shirts`，源 `lerobot/abc_130k_v3_train` revision `68651e4929d9fb00f798937b2d62617cab5c771d`。候选文件由数据 owner 存放于 `docs/reports/data-tshirt-fineart-design-20261103/`。

| 候选池 | episode | 帧数 | 小时 | 用途 |
|---|---:|---:|---:|---|
| train | 319 | 1,080,000 | 10.0000 | 通过质检/复核后训练 |
| dev | 29 | 107,946 | 0.9995 | 独立验证与选型 |
| reserve | 63 | 216,000 | 2.0000 | 按相同时长层补充被拒绝样本 |

这些是元数据候选，不是已发布训练集；最终帧数与实际替换必须由 release manifest 记录。train 含123集超过120秒，合计6小时。保留完整父 episode，Flora 分窗只影响标注输入；不能把窗口或每个语义 subtask 变成新训练 episode。原生 H50 可以跨正常 subtask 转换，只在 episode 边界 padding；真实损坏/缺口不能直接拼接。

预期 Thor 数据工作根目录为 `/home/wuyan-lyj/thor/flora/data/tshirt-fineart-20261103/`，未来派生 `release/train` 与 `release/dev`。训练服务器上的路径尚未分配，解析器通过 release manifest 读取具体绝对路径，不假定 Thor 文件已经在服务器可见。

原始 ABC 文件只读。release 使用仓库 YAM 三个规范图像键 `observation.images.top_rgb`、`observation.images.left_rgb`、`observation.images.right_rgb`；源文件中的 `top/left_wrist/right_wrist` 名称在派生数据层适配，保留来源映射。`action` 和 `observation.state` 保留原始绝对14D数值，名字为 `left_j1..left_j6,left_gripper,right_j1..right_j6,right_gripper`；单位必须由 metadata/audit 证明。不得预先做 OpenPI delta、32D padding 或套用已有 OpenPI norm。

训练模板仅在 native processor 中将三图像键 rename 为 checkpoint 的 `cam_high/cam_left_wrist/cam_right_wrist`。state/action 不 rename；它清空保存的 ALOHA feature/name 申明，让原生 `make_policy` 从发布的 YAM metadata 推导目标特征。不会把224像素图像谎称为360×640，也不为匹配旧 metadata 而重编码视频。

## Flora → FineART 的监督接口

本地 Flora 参考为 `/home/wuyan-lyj/data_clean`，上游源码固定 `129a2ada74c526ee4692f3d1b111bea6ca19d5b7`。已查看的原始结果 `reports/results/abc-val-000526.json` 有 `segments[].start_s/end_s/caption/status`，TempSeg 结果还保留 `concurrent_intervals` 和窗口 provenance。这个原始区间结构不是最终的单一 persistent 时间线；`status=complete` 表示推理输出完成，不等于语义正确标签。

派生语言示例：

```json
{
  "language_persistent": [
    {"role": "assistant", "style": "subtask", "content": "Spread the T-shirt flat on the table.", "timestamp": 0.0, "camera": null, "tool_calls": null},
    {"role": "assistant", "style": "subtask", "content": "Fold the left side of the T-shirt toward the center.", "timestamp": 12.4, "camera": null, "tool_calls": null}
  ],
  "language_events": []
}
```

完整 persistent 列表广播到该 episode 每帧；时间与源 parquet 的 episode 帧时间一致。每条从开始时间持续生效，直到下一条替换，因此不能删除 unknown/mistake 行后让上一指令无意延续。原始重叠、窗口时间偏移、置信度、审核状态保留 sidecar；最终时间线要解决冲突、重复同刻标签、空白和错误物体/左右手描述。只有一个 `style=subtask` 的有效指令序列，宏观/原子多层结果不同时塞进该 style。

保留原任务字符串，标注实际发生的目标；不强制每件衣服有固定段数，也不把握持/支撑手的每次小动作机械拆段。语义完整、可执行的简洁英文优先；text-length 审计需覆盖原生 recipe 两分支及低层附加状态后的真实 token 序列，不能只统计 caption 单句长度。

原生30/70 recipe 每个样本按确定性权重选择一个分支：30% task→subtask 文本目标，70% subtask 条件动作执行。不能解释为同一帧或同一 batch 的每个样本都同时有两种监督。完整语言覆盖仍然必要；空标签回退普通 task→action 不等价于规划器训练。

### 默认自动化标注合同

批量处理采用[自动化合同](../../configs/annotation/tshirt-fineart-v1.json)：`review_method=automatic_candidate_selector`。程序接受无冲突标签，独立VLM只处理需要选择的局部区间；不要求319集逐段由用户或人工批准。下面的review表是统一机器接口，规则/模型流水线可自动填写`APPROVED`和证据记录，agent视觉复核是少量异常的处理方式之一。当前合同固定I/O与质量等级。对47.8s pilot 的派生 Parquet 回看后发现：selector v3 在两处冲突都选错了——11.5–16.5s 把“重新铺平”选成继续折叠，20.5–24s 把“整理近边”并入折叠远边；另有35–35.5s共15帧按保守策略返回 UNKNOWN。1434帧动作均保留，1419帧有语言、15帧两语言列为空，非语言列及源metadata逐值保持；98.95%只是语言覆盖率。该样本的自动selector语义验收失败：对照已有三相机视觉复核，两个冲突都选错候选。整体语义质量与长窗仍未通过分层抽样，不能视为训练合格样本。当前全量主集语言归一尚未完成。

该JSON作为planner的实际policy输入，plan保存完整`policy_contract`和`policy_source`路径/SHA；apply重新核验同一policy文件。当前实现只接受本版支持的参数和硬约束，修改策略需形成可追溯的新policy/plan版本，不会静默放宽覆盖、候选集合或原始episode边界。

每批先冻结lineage与父audit字节快照，再生成draft/plan，保留原件持续下载/准备的流程。当前适配器按各自文件的父目录解释window/raw相对路径，因此快照使用**原目录里的新同级文件名**，不要搬到review目录后让路径错位，也不要改写audit记录的artifact路径而破坏parent_record hash。新增episode或来源/策略变化形成新批次版本，不跳过源哈希变化检查。

时间轴由程序确定。将全部有效Flora start/end映射成父episode整数帧，加入core边界及0/完整长度，离散为连续、无重叠的半开atom；上下文片段的事件也保留。`core_owned_interval`决定active候选，context-only片段只作为邻近候选与来源。模型没有写时间戳、移动边界或生成新caption的权限；逐帧覆盖由程序校验，不靠自然语言回答猜测。

| atom情况 | 自动路由 | 结果含义 |
|---|---|---|
| 唯一有效complete active caption | `RULE_SINGLE_ACTIVE` | 自动接收该候选，保存raw来源；未独立验证其语义准确率 |
| 多条active仅空白规范化后完全相同 | `RULE_DUPLICATE_CAPTION` | 确定性分组，保留全部来源；不做同义词或大小写合并 |
| 多个不同active caption | 独立selector | 从提供的ID中选primary，或返回`UNKNOWN` |
| 没有有效active caption/空白 | 独立selector | 在受限邻近候选与`UNKNOWN`中选择，不能默认续上一句 |
| window core接缝存在跨窗不同caption/相邻目标切换 | `WINDOW_SEAM`强制selector | 即使当前只有一个core-active候选，也不能只按机械owner自动决定语义 |
| 源视频/结果身份不一致或缺资产 | source repair队列 | 先修复来源；选一句caption不能证明视频正确 |

active候选完整保留；neighbor默认最多4种不同caption、最大距离150帧（5秒），按距离/稳定raw ID选取，不使用“先折哪边”的固定先验。允许重新展开、重新铺平和多次尝试。hold/support/smooth等并行动作留在sidecar，原始并发标签不会丢弃；native时间线上每个atom只有一个primary。这些动作是否主要目标取决于上下文，不能把所有straighten/smooth用关键词强制降为辅助。

`WINDOW_SEAM`覆盖内部core边界前后150帧：若不同window的有效raw区间完整覆盖当前atom且caption不同，即使只有一个core-active也请求selector；atom直接邻接core边界、两侧core-owned有效caption不同时也触发。同caption跨窗仍可规则接收和合并。任务保留`reason_codes/seam_boundaries`，`plan.seam_scope`说明机械core覆盖不等于所有接缝已经做过语义核验。

selector plan使用`schema_version=condapi_fineart_auto_timeline_v1`的JSON envelope及`tasks[]`。每task包含`task_id/atom_id/from_frame/to_frame`、active/neighbor候选ID、`candidates[{candidate_id,caption,source_label_ids,origin}]`、`evidence{videos,parent_frame_indices,frames}`、`evidence_frame_ids`及`task_sha256`。候选ID是任务内`C000/C001/...`，不能跨atom只比较字符串`C000`就认定同一个目标。

独立refiner只输出一个候选ID或字面量`UNKNOWN`，由token Trie硬约束可生成集合；wrapper生成`schema_version=condapi_fineart_selector_decisions_v1`的decision JSON。模型不自由生成边界、caption或native语言格式。decision envelope用`task_manifest_sha256`绑定plan原始文件bytes；每项记录task/atom/task hash、`selected_id`、实际证据frame IDs、输入视频SHA和推理receipt。selector身份记录`model_id/model_revision/implementation_version/prompt_sha256`，receipt中的实际图片路径与SHA由apply重新核验。具体canonical task hash算法和字段集合见配置，不能仅凭一个未绑定任务的`C001`接受缓存结果。

默认证据是atom首/中/末帧及前后各30帧（1秒）上下文，三视角均有实际frame descriptor，绑定父帧、源视频帧、视频SHA和原始offset。规则分支记录规则与raw来源，不冒造VLM调用证据。VLM分支只有在ID、模型/提示词身份、任务和实际图片证据全部匹配时记为`VLM_SELECTED_CANDIDATE`。图片哈希绑定消费版本，像素与源帧的对应仍由CPU提取器正确处理PTS/offset并记录，不能把哈希本身当作独立图像来源证明。相邻选择以空白规范化后的相同caption合并，同时保存各atom的任务内选择ID、来源、规则或模型决策与证据；不跨episode合并。

`UNKNOWN`先进入局部补标流程：扩充该atom附近证据并重选；仍无合适候选时，通过独立局部caption补标阶段注册新候选及来源，再以新hash任务重选。selector本身仍只能选ID，不能悄悄生成补标文本。配置草案限制每atom两轮selector和一轮局部补标，最后少量问题进入明确异常队列，可由agent视觉复核或更强模型处理，不默认转成用户审批。恢复/重试不会阻断其他已解决episode继续处理。

当前规则/apply层的pending队列是局部修复入口；扩充证据重试、局部caption补标调度及自动抽检预算仍需工作流接入和实测，不能把这些合同策略描述成已完成批量修复。

默认`strict-parent`导出要求父episode完整时间线已解决。未解决atom保留在sidecar中，不能删除后让persistent文本carry-forward，也不能写成`UNKNOWN`训练指令或把atom切成新的episode。若先发布已完成父episode，必须明确included/pending父列表，生成对应且哈希绑定的subset artifacts，并保留全量plan/draft来源。下述显式`native-empty-anchor-fallback`提供保留小缺口父episode的原生降级；它使用既有空语言列fallback，不增加loss/mask训练循环。

原生适配不变：程序填`reviewer`为实际pipeline/selector标识，`review_method=automatic_candidate_selector`，逐atom/合并interval记录机器复核结果，并为原始overlap/gap issue写rule/selector/repair处理记录；`APPROVED`只是该机器合同中的已解决标记。最后仍调用`validate_review`，使用实际parquet timestamp写assistant/subtask persistent rows。数据发布、raw绝对14D、train-only QUANTILES、FAST审计及checkpoint合同不由自动语言选择器替代。

`Cxxx/UNKNOWN`只属于控制和审计sidecar。原生行的`content`必须由程序取所选候选的真实指令文本，不能把ID当语言目标写入FineART训练数据。

[单集边界/选择ID回归总结](../../configs/annotation/tshirt-fineart-v1.regression.json)只保存此前128691例的规则总结，不提交原Flora JSON、caption全文或图片。10个atom的期望raw标签ordinal是`[0,1,2,3,4,5,6,7,7,8]`，相邻L7合并后9段：345–495帧保留真实重新铺平，615–720帧先整理边缘，1050–1065帧选择邻近的下一折叠目标。这是自动规则/selector的回归对应，不是319集的准确率指标，也不是强制所有episode采用相同顺序。

持续记录各等级的atom/时长、冲突与缺口、selector与UNKNOWN比例、局部修复及最终异常数量、缓存/耗时，并按时长层及规则/模型等级做自动视觉抽检。唯一且完整的帧覆盖是程序保证的结构性质，不等于语义准确率100%；不得据此推断stack成功，`task_success`默认仍为`NOT_REVIEWED`。

### 少量UNKNOWN anchor：原生空语言列降级

固定`f5c9...`原生消费端支持这种显式派生视图：**仅UNKNOWN anchor帧**的`language_persistent=[]`、`language_events=[]`，其他帧保持完整已知persistent时间线，两列key每帧都存在。原task非空时，`RenderTrainingMessagesStep`把空语言anchor渲染成原task的low-level执行，`target_message_indices=[]`；mixed batch也按每样本执行同样fallback，不返回None、不丢掉这些有task的样本。[单样本与批路径](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/processor/render_messages_processor.py#L89)、[fallback内容](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/processor/render_messages_processor.py#L321)

FineART tokenizer对无target span的样本把`text_labels`保持全`-100`，同时`predict_actions=True`。这些anchor没有subtask文本CE监督，原生flow和启用的FAST action-token CE继续；代码可能仍经过文本CE函数，但ignore-index路径没有有效subtask目标，其sum/有效token数clamp处理贡献为0。这不是关闭整个语言头或FAST梯度，也不修改30/70 recipe：已知anchor继续按原30/70分支，未知anchor降为task→action。[tokenizer掩码](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/policies/fineart_vla/processor_fineart_vla.py#L403)、[CE ignore路径](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/policies/fineart_vla/modeling_fineart_vla.py#L308)

**不能使用null-content subtask行作为mask。** Arrow struct允许nullable content，但官方writer要求content/tool_calls至少一个非None，`subtask(content=None,tool_calls=None)`被拒绝。即使直接存入，resolver返回的是非空row对象，`if_present`只看binding是否None；它仍会渲染空assistant目标（包括EOS），或渲染空低层指令而非原task。[Arrow类型](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/language.py#L77)、[writer检查](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/writer.py#L152)、[if_present/替换](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/recipe.py#L267)

原生reader按当前帧读language列，HF transform跳过language转换，language-aware collate保留每帧不同的Python列表；所核对消费路径没有跨帧“必须广播内容相同”的硬校验，只有batch中两列key须一致存在等检查。官方annotation writer的常规输出约定仍是全帧广播，且会把整集slice重新写回每帧；因此这种逐anchor空列表是**显式的consumer兼容派生模式**，需要专门materialize，之后不能再让官方writer无条件广播覆盖这些空anchor。[reader](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/dataset_reader.py#L502)、[HF转换](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/io_utils.py#L267)、[collate](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/utils/collate.py#L29)

初始工程阈值是每父episode UNKNOWN累计≤2%帧、合并后每连续run≤30帧（1秒），不是论文结论或已校准准确率；只接受有真实hash-bound selector `UNKNOWN`决策、无hard source issue的情况，缺decision不算UNKNOWN。128691的15/1434帧约1.046%可保留整个47.8秒动作/视频/H50；有效语言coverage约98.954%，动作coverage仍为100%。保留0..1434全帧，明确记录1050..1065帧两列为空；不能宣称语言coverage=1.0。

自动apply的该模式目前只产带`valid_language_rows`与`empty_language_anchor_intervals`的sidecar视图，状态必须是`NOT_MATERIALIZED_NOT_TRAINING_READY`、`materialized=false`、`training_ready=false`。它不是可直接喂训练的v3数据集。未知区的persistent延续必须在实际逐帧两列写入时被明确清空；无publisher/真实Parquet审计时，继续停在该状态。

训练准备器增加显式release `language_supervision_mode=native-empty-anchor-fallback`分支，默认仍为`strict-parent`。fallback语言审核回执必须声明`mode`及`materialization_status=MATERIALIZED_AUDITED`，分开给`action_coverage`和`effective_language_coverage`，列出全部train/dev父episode及其`length/empty_language_anchor_intervals`（每段含from/to和两列[]），并提供全行`parquet_scan`：`checked_all_rows/empty_columns_exactly_at_unknown_anchors/normal_anchors_keep_full_timeline/only_language_columns_changed=true`、两split的实际empty计数及全部派生data Parquet路径/SHA。标准库prepare核对阈值、coverage算术、全集scope与每个审计过的实际文件hash；全行Parquet扫描由数据审计工具完成，不把一份sidecar或手填PASS当成已materialize。原14D/timestamp身份、norm/FAST等其他release门槛继续保留。

可复现纯协议检查：[verify_fineart_native_fallback_protocol.py](../../scripts/verify_fineart_native_fallback_protocol.py)。它校验11个固定上游源码SHA，执行recipe/render/collate/formatter原函数的AST，替换的只有包依赖、注册器/类型和非语言tensor collate，不导入Torch/JAX/Transformers/datasets/PyArrow/NumPy，也不做训练。验证1000个index两分支、mixed known/empty batch、known→empty→known、null/EOS行为；这不是实际Arrow/HF Dataset加载或数值模型测试。

```text
python3 -B scripts/verify_fineart_native_fallback_protocol.py
  --source-root <fixed-checkout/src/lerobot-or-new-snapshot-directory>
  --download-source
  --output <new-protocol-result.json>
```

`--download-source`仅在显式给出时从固定GitHub revision取得11个小源码文件，按SHA检查，不安装环境；已有正确snapshot可省略该参数。prepare门槛的纯标准库receipt-interface测试：`python3 -B -m unittest discover -s scripts -p 'prepare_fineart_tshirt_training_test.py'`，其中合成文件不被声称为真实Parquet扫描。

### 直接对照基线：LeRobot官方annotation pipeline

[官方说明](https://huggingface.co/docs/lerobot/main/en/annotation_pipeline)提供`lerobot-annotate`，可作为Flora+selector的直接标注基线。以下核对固定于同一上游`f5c9bc02137c331b009eccd44ece6f11b62187a8`，没有安装或运行GPU。它是当前LeRobot工具实现；本次没有证据将其等同于FineART论文原作者实际生成论文数据的完整流程。

`plan`模块先看带时间戳的contact sheets生成可见行为description，再把description和图像交给VLM产生新subtask文本及start/end；它可以提出Flora候选池中缺失的标签/边界，适合作为更好标注的候选方法。默认2FPS、60帧/次prompt，超预算按连续窗口处理，窗口内时间归零后加回父episode时间。默认plan只看所选单相机；不能把它描述成原生三视角联合标注。[生成流程与窗口实现](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/modules/plan_subtasks_memory.py#L446)

**stitch本身是确定性格式处理**：首段start拉回首帧，每段end改成下一段start，末段延到末帧；同帧start冲突会推进到下一个空帧，尾部无空帧则drop。它不在拼接处额外看图决定缺口属于前一目标还是后一目标，也不提供我们selector的UNKNOWN决策。可选seeded-relabel看前/当前/后一段图，修正文案但保持边界固定。完整覆盖只在生成了非空有效spans时成立；空响应/JSON重试失败可返回空标注，官方validator也没有语义正确性或非空subtask覆盖的检验。[stitch与distinct starts](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/modules/plan_subtasks_memory.py#L641)、[validator](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/validator.py#L115)

官方span的末end是最后一帧timestamp（该例47.766666秒），native persistent行仍覆盖这一末帧；转换为我们半开帧区间作比较时，末`to_frame`应取完整长度1434，而不是把这个秒值直接ceil后得到1433而漏掉末帧。

只训练FineART当前subtask recipe时，可仅开`plan`模块、只产subtask：

```text
plan.enabled=true
plan.emit_plan=false
plan.emit_memory=false
plan.n_task_rephrasings=0
plan.task_aug_axes.enabled=false
plan.derive_task_from_video=off
plan.subtask_describe_first=true
plan.subtask_seeded_relabel=false
interjections.enabled=false
vqa.enabled=false
push_to_hub=false
```

`derive_task_from_video=off`使prompt沿用原task；该选项本来也只影响prompt，不改`meta/tasks.parquet`。`plan_max_steps=8`及`min_subtask_seconds=1.5`属于生成prompt约束，不能当成程序已经严格限制段数/最短时长；窗口模式的max steps还分别作用于每窗。[PlanConfig](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/config.py#L48)

固定源码的CLI仅支持`backend=openai`，调用OpenAI-compatible chat completions；`transformers/vllm`进程内backend已明确移除。可连接本地兼容endpoint并设`auto_serve=false`，无需外部付费API；本次没有调用模型API或新建服务。`annotations` extra声明dataset、transformers-dep及`openai>=1.40,<2.0`，不强依赖vLLM；框架本身仍依赖Torch/Pillow及数据读取用PyArrow/Pandas/视频backend，不能称纯标准库工具。[client工厂](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/vlm_client.py#L193)、[依赖](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/pyproject.toml#L285)

**已有本地Transformers9B可在Python层注入**：`PlanSubtasksMemoryModule`构造器接受任意实现`VlmClient.generate_json(messages_batch, *, max_new_tokens=None, temperature=None)->list[JSON对象]`的对象，以及`FrameProvider`。适配器消费upstream给出的PIL contact-sheet image blocks，调用既有已加载9B并返回description/subtasks/label对应的JSON；可使用固定上游`_GenericTextClient`的JSON解析/一次格式重试包装，或实现同一协议。该接口是代码级dependency injection，CLI没有现成`--callback`或`backend=transformers`开关。candidate-ID Trie的受限输出不能原样用于这条自由description/segment基线，应单独记录prompt、响应和格式重试。[VlmClient协议与文本包装](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/vlm_client.py#L54)、[模块注入点](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/modules/plan_subtasks_memory.py#L92)

1clip最小对照建议用128691这条47.8秒数据，复用相同固定9B，不反复逐段人工review。直接构造读取真实parquet timestamp的`EpisodeRecord`、适配已有原视频路径/offset的`FrameProvider`，调用upstream `plan.run_episode(record, new_staging)`，先只写独立staging。当前`dataset/raw/train/episode-*`不是完整LeRobot v3 root，不能不经适配直接当CLI输入。对照harness应断言实际图像非空：官方frame-provider构造失败会回退null provider，不能让一次无图的task-only生成冒充视觉标注。[record](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/reader.py#L46)、[provider fallback](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/frames.py#L345)

首轮可用subtask-only、2FPS、`max_frames_per_prompt=120`（此clip约97帧，避免30秒处恰好切窗）、`plan_max_steps=16`避免过早用8段上限掩盖恢复动作；这些是对照草案。记录原始VLM spans、clean后spans和stitch变化，再与Flora+selector的相同clip比较：真实respread是否保留、两处原冲突如何定目标、35秒接近下一折叠是否错归前段、调用数/耗时/视觉输入量。之后再以默认60帧预算检验自动长窗接缝。原生默认top-only与Flora三视角存在输入差异，需在结果中明示，不能把差异全部归因算法优劣；9段agent样例只作诊断参照，不当人工真值或319集准确率基准。

如调用完整executor，**必须指向新派生dataset目录**：它会原地重写`root/data`并更新`meta/info.json`；`new_repo_id`只改变push目标，不会先复制本地root，`push_to_hub=false`也不会阻止本地重写。共享shard的部分episode运行还可能重建未标注episode的空language列。因此1clip对照优先独立staging，不调用writer；以后需要完整数据产物时，在新目录校验raw数值身份再发布。[CLI](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/scripts/lerobot_annotate.py#L87)、[writer](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/annotations/steerable_pipeline/writer.py#L208)

### 标准库的语言准备工具

[prepare_fineart_subtask_review.py](../../scripts/prepare_fineart_subtask_review.py)只生成审阅sidecar与原生语言行，不写完整LeRobot数据集、不加载模型、不修改Flora原输出。输入是实际annotation `lineage.json`、父episode实际长度metadata（推荐下载审计`audit.jsonl`）和原生Flora最终结果目录。不能把设计用`annotation-window-design.json`当作实际lineage。

```text
python3 scripts/prepare_fineart_subtask_review.py --draft
  --lineage <actual-annotation/lineage.json>
  --parent-metadata <data-root/audit.jsonl>
  --results-dir <native-Flora-results-directory>
  --output <new-review-directory/draft.json>
```

它输出`draft.json`与`draft.review.json`，拒绝覆盖旧件。draft保留lineage receipt、父metadata与结果文件哈希、模型版本、每条完整raw segment、窗口local区间、映射到父episode的区间和core所有权；`.tempseg.json`若存在也保存来源哈希。输入`[start_s,end_s)`对应满足`f/30`落在区间中的视频帧，边界用ceil并只消除精确帧边界的微小浮点误差；从`parent_from_frame`映射回父episode。上下文片段仍保留原件，只有core拥有的部分参与待审建议。

待审状态为`DRAFT_REQUIRES_SEMANTIC_REVIEW`，每条初始建议为`UNREVIEWED`。并发/重叠标签不会自动择一；对应区间`content=null`并保留全部source label引用。空白、unknown/mistake、incomplete、缺最终结果、缺core覆盖、标签缺口、结果输入哈希/时长不符等均生成明确issue。两层标签不会直接写进互相竞争的`subtask`行。不得通过删除unknown或补齐上一句来消除问题。

在`draft.review.json`中记录真实reviewer、时间和`review_method`，例如`human_video_review`、`model_assisted_review`或`agent_visual_review`；reviewer可以是实际参与复核的人或agent/模型标识。方法必须如实，不得把agent或模型辅助结果冒称human。用户已经授权清洗标注，这个数据复核过程不要求用户逐条审批。

复核执行方把`intervals`修成从0至完整父episode长度的**单一、连续、无重叠、半开整数帧区间**；每段填`APPROVED`、reviewer及决策/实际视觉复核证据note。`APPROVED`是语义复核的结果标记，不表示用户审批。所有原始issue需独立记录resolution和reviewer。允许经有证据的语义复核后重标原来无有效标签的区间，保留raw draft、问题及处理记录；工具自身不会补标或carry-forward。

draft范围只包含这份实际lineage中出现的父episode：它能发现已出现父episode的缺窗/缺core，不替代全319集的覆盖统计。整集尚未进入lineage的情况由数据发布审计与候选对账处理，不能从一次pilot draft成功推定全数据集语言已就绪。

真实导出还需要从原始parquet读取的每帧timestamp输入。JSON形状如下，示例仅3帧，不是本任务的真实时间数据：

```json
{
  "schema_version": 1,
  "parents": [{
    "pool": "train", "source_episode_index": 7, "fps": 30,
    "timestamps": [0.0, 0.03333333507180214, 0.06666667014360428],
    "timestamp_source": {
      "kind": "actual_parquet_timestamp_column", "column": "timestamp",
      "path": "/actual/source/data.parquet", "sha256": "ACTUAL_PARQUET_SHA256"
    }
  }]
}
```

时间输入应由既有Parquet读取步骤按父episode与frame_index顺序提取完整`timestamp`列，不得用`frame_index/30`重建来冒充真实值。这个标准库工具校验输入帧数、严格递增、30FPS节奏及源parquet哈希；**它不解码Parquet，文件哈希绑定不独立证明JSON数值确实来自该文件**，提取步骤仍需审计。时间缺失时停在审阅sidecar，不生成原生语言行。

```text
python3 scripts/prepare_fineart_subtask_review.py --validate-reviewed
  --draft-file <immutable-draft.json>
  --reviewed <completed-draft.review.json>
  --frame-timestamps <actual-timestamps.json>
  --output <new-language-rows.json>
```

校验逐父episode的完整覆盖、每段语义复核结果、实际review_method、全部issue处理、raw label引用、draft hash和实际timestamp后，输出assistant/subtask的`language_persistent`行；每条timestamp直接取其开始帧的真实值，`camera/tool_calls=null`，`language_events=[]`。这些行仍需由后续派生数据发布步骤广播到父episode每帧。成功状态仅`LANGUAGE_ROWS_VALIDATED_DATASET_RELEASE_PENDING`、`training_ready=false`，不替代完整数据/归一化/FAST/语言语义质量验收。

轻量fixture测试：`python3 -B -m unittest discover -s scripts -p 'prepare_fineart_subtask_review_test.py'`。测试只核对frame映射、core所有权、问题保留和校验门槛；使用合成源文件，不声称做过真实Parquet提取或语义质量验收。

## FAST 和归一化的明确选择

首轮基线保留 bundled tokenizer：`auto_fit_fast_tokenizer=false`，`action_tokenizer_name=<固定snapshot>/action_tokenizer`，offset128、最多384动作 token、文本128 token。当前源码新配置默认 offset1152/动作256，不能拿默认值替代旧 checkpoint 合同。此基线不加入与 offset128占用区冲突的 VQA `<loc>` 训练。

当前原生 trainer 在非 resume 且 recipe 非空时**重建 processor**，不会直接复用 checkpoint 保存的处理器。解析器先读取真实 checkpoint config，再合并模板 override，因此保留其未显式更改的结构参数；recipe 显式嵌入固定的30/70版本，FAST 路径显式指向已验证的 bundle。

重建的 normalizer/unnormalizer 从发布 train 数据集 metadata 得到14D绝对 state/action 的 `q01/q99`。train 目录只能包含 train；不得用train+dev拼接目录的总统计量。独立 dev 评估将使用训练产物保存的 normalizer；当前未实现这一评估入口，也不在 dev 上重拟合归一化。

保留 bundle 是首个对齐基线的设计选择，不代表其对 YAM 分布已经合格。需在 train-only 绝对动作 QUANTILES 空间验证 FAST 重建：至少256个验证窗口，RMSE≤0.1、各维最大RMSE≤0.2，并对全部 train anchor 做长度审计、无超384截断。当前上游重建 processor 的 `allow_truncation=True` 与旧保存处理器不同；这里通过前置长度审计阻断静默截断，不修改/复制一套 processor。若 codec 不合格，应单独版本化 train-only refit 的资产、阈值和配方后再生成配置。官方当前推荐新机器人自动拟合，但这不能变成无记录的隐式变更。

## 准备与原生入口

仓库内可运行的轻量静态检查：

```bash
python3 scripts/prepare_fineart_tshirt_training.py --check-template
```

返回 `static_template_check=PASS` 时状态仍是 `PREPARATION_NOT_TRAINING_READY`。若在已有固定版本上游 checkout 上加 `--lerobot-root /absolute/path/to/lerobot`，还会只读核对 git HEAD/跟踪文件状态，并以 AST 校验 native dataclass 字段及默认 recipe。它不 import ML 包。

未来数据和资产就绪后，在获准的训练资源上通过同一标准库工具生成 resolved config；参数需要真实路径，不能照抄占位符：

```text
python3 scripts/prepare_fineart_tshirt_training.py --resolve
  --lerobot-root <fixed-commit-checkout>
  --checkpoint-snapshot <verified-base-snapshot>
  --text-tokenizer-snapshot <verified-paligemma-tokenizer-snapshot>
  --release-manifest <QC_RELEASED-release.json>
  --train-output-dir <new-server-training-output-directory>
  --config-output <new-resolved-native-config.json>
```

解析器拒绝覆盖 config/lock、拒绝已有训练输出目录，完整核验固定 checkpoint 资产，要求 train/dev release 和哈希化审核回执，再生成纯 native JSON 与 `.lock.json`。lock 状态仅 `NATIVE_CONFIG_RESOLVED_NOT_EXECUTION_VALIDATED`，不会伪装 GPU/训练已通过。原生入口为 `python -m lerobot.scripts.lerobot_train --config_path=<resolved-json>`，必须使用未来独立 FineART/LeRobot Conda prefix；控制层不导入模型，不复用 OpenPI 训练循环。**本次没有运行该入口。**

模板配方是 full FT、30k步、每进程batch8、LR1e-5、warmup1000、末端LR1e-6、每1000步保存，vision/VLM/action expert 全部可训练，原生 knowledge insulation 开启。关闭 compile 与额外 fused backend，保留基础 gradient checkpointing。它是保守 draft：GPU数量、每卡显存、全局batch、梯度累积和吞吐未验算；不代表batch8能在未知服务器运行。服务器维护结束、资源重新观测后再定执行拓扑及最终配方，不在工作站做 training smoke。

独立29集dev不通过原生 `eval_split` 接入：固定上游该字段只提供分数式自动拆分，不能直接指定这里的29集。模板设置 `eval_split=0`、`eval_steps=0`、`env_eval_freq=0`，只消费 train root。后续需要接入独立dev语言/动作离线评估和真实rollout，再按两者选型；不要把已保留dev写成已执行评估。

## Release manifest 与回执接口

解析器不是数据/标注审计器，不能替代逐帧审计。数据工作流完成质检和审核后写下列索引；所有审核对象需有实际 JSON 文件及SHA256，禁止为了让配置通过而手填虚假的PASS。路径相对 `release.json` 解释；train/dev root 使用未来训练机器上的绝对路径。

```json
{
  "schema_version": 1,
  "status": "QC_RELEASED",
  "language_supervision_mode": "strict-parent",
  "source_revision": "68651e4929d9fb00f798937b2d62617cab5c771d",
  "task": "fold and stack the t-shirts",
  "contract": {
    "action_semantics": "absolute",
    "state_action_names": ["left_j1", "left_j2", "left_j3", "left_j4", "left_j5", "left_j6", "left_gripper", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6", "right_gripper"],
    "joint_unit": "REPLACE_WITH_AUDITED_UNIT",
    "gripper_unit": "REPLACE_WITH_AUDITED_UNIT",
    "raw_numeric_identity_verified": true,
    "episode_boundaries_preserved": true
  },
  "datasets": {
    "train": {"root": "/replace/release/train", "repo_id": "yam/tshirt-fineart-train-20261103", "source_episode_indices": [137]},
    "dev": {"root": "/replace/release/dev", "repo_id": "yam/tshirt-fineart-dev-20261103", "source_episode_indices": [779]}
  },
  "audits": {
    "numeric_identity": {"path": "audit/numeric.json", "sha256": "REPLACE"},
    "language_timeline": {"path": "audit/language.json", "sha256": "REPLACE"},
    "split_leakage": {"path": "audit/split.json", "sha256": "REPLACE"},
    "normalization": {"path": "audit/normalization.json", "sha256": "REPLACE"},
    "fast_codec": {"path": "audit/fast.json", "sha256": "REPLACE"},
    "text_length": {"path": "audit/text.json", "sha256": "REPLACE"}
  }
}
```

上例只说明字段，示例ID/路径/单位/哈希不是release结果。`source_episode_indices` 必须完整映射最终全部episode；解析器要求train来自train/reserve候选、dev来自dev/reserve候选，实际集合互斥。replacement 原因、局部索引映射和原件哈希记录在数据 owner 的发布证据中。

| 审核JSON | 解析器要求的机器字段 | 回执还应保存的证据 |
|---|---|---|
| numeric_identity | `status: PASS`，`raw_numeric_values_unchanged: true`，与contract一致的`joint_unit/gripper_unit`，`rows_checked`等于train+dev全部帧数，完整`source_episode_indices` | 完整14D值/单位/时间戳/视频身份核对及源哈希，不能只做稀疏抽查后称全量相等 |
| language_timeline | strict模式`coverage: {train: 1.0, dev: 1.0}`；显式fallback模式提供实际materialization审计及独立effective-language/action coverage；两者均要求`status: PASS`和已知行无歧义 | 覆盖统计、原始Flora来源、如实记录方法/证据的语义复核及分窗合并记录；fallback不能把空anchor当语言100% |
| split_leakage | `status: PASS` | episode、相近场景/会话/衣物身份审查；source index带不是会话；未知身份应保留风险和处理决策 |
| normalization | `status: PASS`，`fit_split: train`，`action_semantics: absolute`，`stats_sha256`，完整`source_episode_indices` | 全训练集统计过程、无dev/官方val参与，统计值来源 |
| fast_codec | `status: PASS`，`audit_split: train`，`stats_sha256`，`tokenizer_sha256`，`fast_skip_tokens: 128`，`max_action_tokens: 384`，`validation_samples >= 256`，`truncated_chunks: 0`，`length_audit_scope: all_train_anchors`，`reconstruction_rmse/max_dim_rmse` | 抽样种子、样本及维度误差，长度分布，episode尾部padding与原生H50一致 |
| text_length | `status: PASS`，`max_length: 128`，`truncated_samples: 0`，固定PaliGemma词表的`tokenizer_sha256` | 原生两分支完整序列的token长度，含状态前缀；标签不能为过检而盲目裁字 |

当前这些回执尚未存在；解析器不会创建占位PASS。数据发布、环境安装、GPU验证、实际训练和独立dev评估仍是不同验收阶段。

## 可核验上游依据

- [固定版本配置](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/policies/fineart_vla/configuration_fineart_vla.py)：嵌入recipe、FAST默认、知识隔离和优化字段。
- [原生训练器 processor 重建](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/scripts/lerobot_train.py#L653)、[FineART processor](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/policies/fineart_vla/processor_fineart_vla.py#L427)：目标数据统计与FAST处理。
- [H50 episode查询](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/dataset_reader.py#L298)、[边界padding](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/datasets/utils.py#L146)。
- [官方语言schema与persistent语义](https://huggingface.co/docs/lerobot/main/language_and_recipes)、[FineART说明](https://huggingface.co/docs/lerobot/main/fineart_vla)。这两页跟随main，仅作解释；实际代码合同以上述固定commit为准。
