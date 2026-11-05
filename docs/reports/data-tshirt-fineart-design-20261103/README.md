# ABC-130K T-shirt → FLORA → FineART：10h 数据方案

2026-11-03。用户选择普通 T 恤折叠、明确保留并增加 >120s 轨迹；随后授权在 Thor 下载、清洗、开始标注及探索 FLORA 加速，同时准备 FineART 原生训练配置。训练服务器维护期间不访问服务器或启动训练。本报告的候选名单不是质量发布清单，运行进度见[执行回执](execution.json)。

## 规模与选择

唯一任务为源 prompt `fold and stack the t-shirts`，不混入翻面 T 恤、长袖、背心、毛巾等其他任务。固定 train revision 下有 11,009 集、291.177083h；官方 val 为 82 集、2.047546h，保持独立。源 metadata 的 SHA、revision 见 [source-provenance.json](source-provenance.json)。当前未对候选进行任务成功视觉认证。

**10h 指最终合格训练数据；开发集和备用池另计。** 按小时分配，保留完整父 episode：

| 原 episode 时长 | 训练目标 | 当前候选集数 |
|---|---:|---:|
| ≤60s | 1h | 70 |
| (60,120]s | 3h | 126 |
| (120,180]s | 3h | 77 |
| (180,240]s | 2h | 35 |
| >240s | 1h | 11 |
| 合计 | **10h** | **319** |

主集 1,080,000 帧@30fps；>120s 为 123 集、6h，占时长60%，其中4集>360s，最长512.6s。另有29集/0.9995h开发候选、63集/2h同层备用。完整非padding H50 anchor 数为1,064,369；原生loader可补尾，不能把此统计误作自定义删尾规则。

名单：[训练候选](train-candidate-episodes.json)、[开发候选](dev-candidate-episodes.json)、[备用候选](reserve-candidate-episodes.json)、[完整分层统计](candidate-design.json)。质检不合格从同一时长层替换；不裁掉正常原始帧以凑精确10h。2h备用是初始余量，若不够再扩池。

当前完成了 episode 互斥检查，**尚未完成采集会话、场景、衣物身份的组间隔离**。源索引区段分散只是抽样方法，不能替代会话分组。下载后优先恢复这些 group；同组及其所有窗口进入同一 split，必要时重新平衡候选。官方val不参与归一化、tokenizer拟合或梯度。

## 存储口径

当前选择现成 LeRobot 版本：三路224×224、AV1、30fps。9集、27路、约30min的MP4 sample-table实测外推：10h三路视频约5.87–11.77GB；加14D state+action约121MB和容器/Parquet等余量，**最终10h净导出估计6–13GB，预算15GB**。这是样本外推，不是最终导出实测。未实测原生高分辨率MCAP，不能套同一数字。

共享源分片与目标episode净大小差别很大。按固定HF revision逐文件元数据精算：

| 下载方式/集合 | 去重源文件数 | 整分片大小，十进制GB |
|---|---:|---:|
| train 319集 | 1,095 | 106.64 |
| train + dev | 1,187 | 116.99 |
| train + dev + reserve | 1,389 | 137.16 |

因此 Thor 工作空间按**200GB可用空间**预算，覆盖全源缓存、约13h派生数据、窗口和临时文件；模型权重/训练checkpoint另计。按有界Range抽取并复用块缓存，可显著减少实际下载与占用；不承诺一定等于15GB网络流量。[视频抽查](storage-estimate.json)、[源分片精算](storage-source-shards.json)、[逐源文件size/LFS SHA](storage-source-shards-path-metadata.json)。

## 清洗与发布门槛

1. **身份与结构，全量检查**：固定源revision及文件size/SHA；原14D `observation.state` / `action` 有限值、shape、顺序；连续帧号及30fps时间戳；三相机帧数/时轴一致；原AV1包无重编码抽取，三路逐帧解码。网络、权限、容量、下载中断属于操作错误，保留现场续跑，不标为损坏。
2. **视频异常与任务质量分开审核**：冻结、曝光、低对比、重复帧先生成review标记；持衣、等布料落稳、30fps重采样可产生静止/重复，不因静止1秒自动剔除。确认损坏则整集排除并保留原件。长轨迹优先保留完整展开、连续折叠、叠放和有效整理/重抓；排除无效等待堆时长、持续故障、任务不符、未完成且无法说明的轨迹。
3. **自动语义归一与批次抽样**：FLORA的`complete`只代表生成完成。按[运行策略](../../../configs/annotation/tshirt-fineart-v1.json)对全部轨迹建立整数帧时间轴：唯一候选或同文重复由规则处理；不同候选、空白及跨窗接缝交给本地9B视觉selector，只能选择候选ID或UNKNOWN，不能编造文本或时间。未知占比≤2%且连续≤30帧时，显式使用FineART原生两语言列`[]`跳过这些anchor的文本监督，保留全部动作/H50；这些是工程初值，不是论文质量指标。较大未知或来源问题进入父episode补标队列，合格父集继续导出。无需用户或agent逐集审批。局部扩证据/重标的预算已设计，自动修复循环尚未实现，不能把队列称为已修复。语义准确率仍须按时长、规则/selector路由、接缝及返工分层抽样测量；程序覆盖率不能替代准确率。首批建议从319集分层抽16–20集；这是质量验收计划，尚未执行。出现集中错误时调整该类规则并重跑相应层，不要求逐条复核所有episode。2026-11-05回看pilot派生Parquet发现，selector v3在11.5–16.5秒把重新铺平选成继续折叠，在20.5–24秒把整理近边选成折叠远边；35–35.5秒15帧以UNKNOWN空列回退。当前两个冲突的自动选择与既有agent三相机复核参考均不一致，因此单clip自动归一**语义QA未通过**，此前“两个选择正确”的记录作废。此结果只覆盖一个calibration pilot，不外推为354个raw窗口错误率；主集FineART归一汇总仍未生成。
4. **release**：原始抽取、结构审核、FLORA原JSON、语义修订及FineART转换分别留版本。最终发布冻结父episode、split/group、帧数、图像映射、动作单位/顺序证据、完整语言时间线、源及输出哈希。未达标不生成训练READY。当前候选不因下载成功就变成10h合格数据。

原数据值不裁剪、不猜测单位、不做OpenPI delta处理。图像分辨率沿源224，不放大到640冒充高分辨率。离线转换仅写新目录。

## FLORA标注：窗口用于输入，完整episode用于训练

FLORA固定官方commit及模型revision、prompt、YAM三视角profile、采样/解码合同。≤120s直接输入整集；>120s仅生成标注窗口：每个核心区间110s、左右各最多5s上下文，最大120s，相邻正常窗口重叠10s。整数帧划分：core=3300帧、context=150帧；半开区间，边缘截到父episode。所有三相机同步处理。

主集319集对应465个计划窗口；输入含上下文约10.405h，但独立训练时长仍是10h。[逐窗口设计](annotation-window-design.json)。12集pilot覆盖全部时长层，合计32.37min；源ID见[pilot清单](pilot-12-metadata-candidates.json)。

每个窗口记录父episode、源revision、起止frame、核心区间、原始任务、原三图引用和local→parent时间映射。FLORA原JSON不改；合并时映射回父episode真实帧时间戳，以核心区间确定拥有关系，合并跨窗口一致的目标，冲突由约束selector处理，未解决项入补标队列；pilot抽样已发现selector会选错候选，因此任何自动选择结果都需纳入分层语义抽样。不将重叠片段作为重复训练小时，也不将窗口发布成独立训练episode。

FineART H50在普通子任务边界自然跨越；只按原生episode末端裁夹/padding，不在每个语言边界额外截断。

## 对齐FineART公开checkpoint

固定 [`lerobot/fineart_vla_base`](https://huggingface.co/lerobot/fineart_vla_base/tree/b6cf4faf2630f971dd390bc32365e9d9c330da8e)，revision `b6cf4faf2630f971dd390bc32365e9d9c330da8e`；固定LeRobot代码`f5c9bc02137c331b009eccd44ece6f11b62187a8`。旧`FineART-VLA-subtask-midtrained`地址当前指向相同模型身份。后续不要仅用漂移的main/default恢复配置。

语言按官方[language columns](https://huggingface.co/docs/lerobot/main/language_and_recipes)存放：`meta/tasks.parquet`保存原任务；`language_persistent`为完整episode时间线，在已知anchor广播；显式允许的小UNKNOWN anchor把两语言列都写为`[]`以原生回退到task→action。无稀疏事件时`language_events=[]`。例子只是schema示意，不能当本批实际标注：

```json
{
  "role": "assistant",
  "content": "Fold the left side of the T-shirt toward the center.",
  "style": "subtask",
  "timestamp": 12.4,
  "camera": null,
  "tool_calls": null
}
```

`timestamp`必须映射到父episode的真实帧时间；此项在下一次subtask替换前持续有效。Flora `end_s`、置信度、窗口、人工修订等留审计sidecar。不能直接删`mistake/unknown`后让上一指令错误延长；不能让粗/细两套相互冲突的标签同时作为当前subtask。

动作保持YAM真实14D绝对值，使用此模型对应的train-only QUANTILES统计；FineART本checkpoint `use_relative_actions=false`、H50。三图按语义显式映射：源`top/left_wrist/right_wrist`→本仓库`top_rgb/left_rgb/right_rgb`→processor需要的`cam_high/cam_left_wrist/cam_right_wrist`。不靠字典排序，也不复用OpenPI delta norm或基座其他机器人的统计。

公开checkpoint的FAST offset=128、max action tokens=384，与新建配置默认1152/256不同；加载checkpoint自身配置后再审查覆盖项。[官方FineART说明](https://huggingface.co/docs/lerobot/main/fineart_vla)。当前固定代码的[原生trainer](https://github.com/huggingface/lerobot/blob/f5c9bc02137c331b009eccd44ece6f11b62187a8/src/lerobot/scripts/lerobot_train.py#L653)在非resume且有recipe时会重建processor，`auto_fit_fast_tokenizer=true`也会在新数据拟合FAST。因此“加载权重就自动保留原processor/tokenizer”不成立。

准备配置保留基座30% task→subtask语言监督、70% subtask条件动作监督及KI路线，避免只有task→action而不训练规划语言。FAST先比较bundled tokenizer的YAM重建误差和token长度，明确保留或train-only原生重拟合，再冻结资产与offset128。开发数据可以评估，不能参与拟合。训练配置与就绪门槛归[FineART训练准备](../../reference/fineart_tshirt_training.md)。

## 本轮执行范围

Thor新目录：`/home/wuyan-lyj/thor/flora/data/tshirt-fineart-20261103/`。复用现有独立FLORA容器进行数据准备/推理；不重启其他服务、不操作3588、不访问维护中的训练服务器。先下载并审核pilot，再按相同断点任务覆盖主集、开发和备用池；未解决的质量项不自动进入训练release。

已实现原生Session复用、两阶段断点及独立GDN overlay；同条47.8s输入的最终监督字段一致，但raw重试audit有差异，严格raw parity未通过，不能宣称逐位等价或受控倍速。单条自动链写出1434帧派生Parquet，其中1419帧有语言、15帧为空列回退，原动作和非语言列逐值一致；回审发现两处冲突选择都与视觉参考不一致，所以98.95%只是语言覆盖率，且样本语义QA失败。批处理入口为[`run_fineart_auto_batch.py`](../../../scripts/data/run_fineart_auto_batch.py)，Thor host编排为[`run_tshirt_annotation_pipeline.sh`](../../../scripts/data/run_tshirt_annotation_pipeline.sh)。当前批量进度归[执行回执](execution.json)，原始预测与图像留Thor，不提交仓库。加速与实际限制归[FLORA Thor标注](../../reference/flora_thor_annotation.md)。
