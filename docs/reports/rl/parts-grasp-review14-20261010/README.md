# 14 集逐次抓取候选审核集 · 2026-10-10

用户要求保留成功、失败与不确定候选，生成后先人工检查，**不开始 RL 训练**。本轮只做数据提取、Thor 离线 SAM3 推理和审核页面；没有训练更新、生产推理接入或 3588 操作。当前 PARTS 算法仍为论文的夹爪残差 TD3+BC 路线，LWD/DIVL 讨论没有改变这一分工。

## 数据与审核入口

本地派生目录：`/home/wuyan-lyj/condapi-data/rl/lego-grasp-review14-20261010-v1/`，打开 `index.html`。源目录为既有只读 `lego-rollout14-20261010-parts-v1/`；14 集顺序、20w 用户声明及前两集资产来源冲突仍沿用[原审核](../parts-rollout14-20261010/README.md)。未将整轮 121/140 正确分拣直接转换成逐抓取奖励。

共 **273 个闭爪候选**：左臂108、右臂165；**3,846 张唯一腕部图像**，另有273段顶视/左右腕同步视频。第一集的10个候选为 `validation`，其余263个为 `train_candidate`；这表示后续用途，不是训练就绪。重复尝试、可能的非抓取调整、观测中断与片尾候选都保留；闭爪启发式不能保证找全所有实际抓取。

审核文件：

- `candidates.json`：完整行、epoch/tick、时间、原视频引用、几何证据、自动建议、原因。
- `review.csv`：可填写的中文友好 UTF-8 BOM 表；`review_label` 初始为空。
- `index.html`：离线筛选、播放、原图/叠图切换、人工成功/失败/不确定/非抓取、备注、JSON导入导出；不把自动建议当成人工审核。
- `vision/`：逐图 boxes/scores 与压缩 bit masks，保留模型与脚本身份；`image-jobs.json` 记录源视频 PTS/time_base 和派生图 SHA。
- `clip-manifest.json`、`clip-audit.json`：源时间窗、视频 SHA、完整解码及时间轴检查。

自动建议为 **成功2、失败0、不确定271**；成功项也未经人工确认。逐集结果见[summary.json](summary.json)。**自动审核未通过，不能直接生成RL奖励。** 全部 `approved_reward=null`，人工已审核数为0，`ready_for_training=false`；没有把不确定样本改成负例，也没有从审核按钮自动发布训练奖励。

## 可解释的候选规则与限制

测量夹爪开度使用开/闭迟滞找候选；遇到 tick/epoch 或时间断裂即切断。每0.20秒及进入/闭爪/结束关键行取腕部图像；视频按记录的 `video_indices` 对齐，不能把控制行号当视频帧号。相机时间来自 `camera_host_received_at`，不是已经验证的硬件曝光时间；本批该字段没有非有限值。

最终相对高度使用**候选进入行**的各臂 base-Z，不使用未来最低点。初始提取曾保存闭爪附近最低点；原 `candidates-unscored.json` 与提取脚本快照保留，最终文件中的 `earlier_provisional_reference` 只作溯源，不参与最终建议。进入点只是闭爪启发式的进入行，仍需验证是否对应正式 selector；向上轴尚未标定。所有高度结果均为候选证据。

SAM3 使用 `LEGO brick`、`plastic storage bin`、`robot gripper fingers` 三个文本提示。结合指间图像区域、容器排除、分割面积/位置连续性、夹爪状态、新鲜且不同的帧，建议成功需相对进入行抬升≥5cm、连续持物≥1秒。零检测、模糊、遮挡或快速放置均不足以判失败。空手负例还要求明确的夹爪正检和连续空隙证据；当前提示词在这些 YAM 夹爪图像上识别不足，不能为了得到负例降低门槛。失败尝试会因此留在不确定项中等待审核。

这些规则不是已验证的奖励模型，没有人工 precision/recall 或在线验收；遮挡、白色积木、容器边缘及桌上积木可能误检。人工审核还需检查候选边界与参考点，不能只核对叠图是否分割到了积木。审核结果不能补出尚未执行过的专家残差。

## 推理与验证

Thor 环境和官方同 SHA 镜像权重沿用[视觉环境报告](../../thor/parts-vision-20261010/README.md)。本轮独立容器 `parts-grasp-labels-20261010`，网络关闭、无端口或控制设备映射；batch4、FP32 参数、上游 BF16 autocast。使用固定 SAM3 源码的原生 batch grounding，3张图与单图高置信 box 对照最大差0.1161像素；不称逐像素等价。正式结果见[vision-run.json](vision-run.json)，原始日志和退出状态保留在 Thor 与外部派生目录。

273段视频全部完整解码，PTS严格递增，与源首末时间差的最大误差 **0.000500秒以内**。首版编码器按10Hz量化不规则时间导致写入失败；失败派生片段保留在兄弟目录 `lego-grasp-review14-20261010-v1-failed-clips-v0/`。修正编码 time_base 为1ms并重新导出，未改原录像。

纯数组/监督测试 **29 passed**；限定 Ruff、Python AST、页面 JavaScript 语法与 `git diff --check` 检查。独立 QA 数据集验证了视频解码、标签与备注刷新恢复、未审核筛选、JSON导出与导入恢复；QA身份与本批审核集不同，没有代填用户标签。本地未导入 Torch、执行模型前向或训练循环。真实 SAM3 前向仅在 Thor 执行。

入口为 `scripts/parts_rl/prepare_review.py`、`predict_review.py`、`render_clips.py`、`build_review.py`；后处理纯数组规则归 `packages/parts-rl/src/parts_rl/candidates.py`。大文件保留在外部数据目录，不加入 Git。

## 用户反馈后的判定试验

第一版只用左右腕部，夹爪仅一条 `robot gripper fingers` 提示；主视角只用于回放。这是不充分的提示词/视角探索，不能据此判断相机看不到夹爪。按用户建议，另选12张腕部原图、对应12张下半裁剪图和2张同步顶视图，测试13个提示，包含 `black triangular gripper fingers`、`black ribbed triangular fingers`、`black triangular plastic wedges`、`black wedge-shaped jaws` 等外观描述。原提示在本试验仍0检出（max score≥0.5）；纹路三角夹指提示原图2/12、裁剪2/12，只是覆盖率而非准确率。积木改用 `plastic building block` 在原图11/12有检测，仍需排除容器和桌面误检。另做SAM3框视觉示例与实例点提示，不能把任何高分mask视作正确夹指；原始输出、叠图和独立摘要保留。

试验目录为 `/home/wuyan-lyj/condapi-data/rl/lego-grasp-prompt-probe-20261010-v1/`，Thor同名目录。脚本为 `probe_prompts.py`、`probe_visual_boxes.py`、`probe_points.py`，只推理不训练，不能据小样本宣布自动奖励已验收。论文[附录D](https://arxiv.org/html/2609.21788v2#A4)明确省略标定、辅助函数等实现细节；其视觉使用SAM3，但未公开本地可直接复用的完整检测配置。本文独立实现不冒称作者未公开的trick。

有效改进是**SAM3实例点提示，不是新文字词**：固定点在闭爪后会落到桌面，造成表面上“夹住反而分不出”。改为每半幅底部暗色长连段自动选择夹指正点，另一夹指和中上区域作负点，启用同一权重的 `enable_inst_interactivity=True` / `predict_inst`。12张样图的左右输出均主要落在黑色夹指（各mask暗色比例≥0.65），叠图定性检查明显改善；部分模糊图仍缺失指面细节，**这不是12/12抓取判定准确率**。固定点、框示例和自适应点均保存独立结果；最后一次自适应点批次退出0、无OOM，未修改初版273个建议或批准奖励。执行脚本SHA见[摘要](prompt-probe-point-adaptive-summary.json)，外观/面积检查见[诊断](prompt-probe-point-adaptive-audit.json)。

![自适应点提示：夹指分割小样本](point-adaptive-contact.jpg)

用户观察“夹住后分割消失”可作为待检验的时序特征，不能直接作成功reward。需要区分积木遮挡、运动模糊、出画、掉落与提示点落空；抓取前目标身份、闭爪后遮挡、抬升时随动/重现共同支持持物。单帧未检出保持unknown，不因漏检自动奖1或罚0；尚未实现或验收遮挡跟踪器。

用户反馈1秒过长，先比较5cm相对抬升加短时新鲜持物证据，不改正式监督器默认。纯物理抬升且闭爪窗口≥0.2秒有197个，≥1秒113个；沿用v1稀疏视觉规则对应54和2个，仅诊断，不重标成功。v1每0.2秒抽帧，不能假装已验证0.2秒内3–5张新图。之后的光流原型在8段212帧试验中全部保持不确定，未形成有效自动奖励；最新用户要求撤下自动跟踪，转用下节尖端ROI试验。旧光流代码及输出只留在外部 `lego-motion-probe-20261010-v1/run-scripts/retired/` 等证据目录，未提交进本轮工作树，不删除远端环境或实验原件。窗口摘要见 `window-diagnostics-summary.json`。

原始文件后缀只有H5/JSON/MP4，未保存深度流。SAM3分割可以与真实对齐深度结合，但本批不能追补传感器深度。[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)的通用权重输出相对深度；可用作辅助线索，不能直接把预测当5cm测量或唯一空手判据。本轮尚未下载或运行深度模型，未向客户端发送采集改造要求。

## 夹指尖端连接区域：20例固定80%门槛，无跟踪

用户授权测试“两片夹指尖端之间的小连接区域，彩色占比大于80%”，并明确去掉自动跟踪。新派生目录为 `/home/wuyan-lyj/condapi-data/rl/lego-fingertip-roi20-20261010-v1/`；独立审核入口 `http://127.0.0.1:8758/index.html`。原v1审核集及ZIP不改。页面包含20例原图、SAM3指面、自动尖端/窄带叠图、两种占比、逐例目视备注及三视角回放；初次结果与后续探索性对照分别留档。

在计算占比前选定20个不同候选：E01第一左/右候选、已有两个持物建议E02-L004/E12-L005、E04左右各前5个可抬升候选、E03/E07/E10左右各首个可抬升候选；具体ID和规则见[fingertip-roi20-selection.json](fingertip-roi20-selection.json)。每个候选在闭爪行后、开度<0.60、进入行相对抬升≥0.05m的已有采样中选最大抬升帧。实际相对抬升6.9–17.9cm；这个参考不是绝对离桌高度，向上轴仍未独立标定。此为定向诊断集，不是随机评估集或独立holdout。

**几何与规则：** 复用有效的SAM3自适应实例点提示，每图独立推理两片夹指，不跨帧跟踪。选包含暗色正提示点的连通分量，用mask像素纵坐标1%分位和其后5行的横坐标中位数估计上端尖点；以两个点连线为轴画总宽12px窄带，扣除指面后作为分母。间距<8px、有效区域<48px、有效区域不到窄带一半、mask quality<0.5或指面暗色比例<0.6时为unknown；质量分数不作为成功概率。20例均通过几何检查，叠图尖端定性核对合理。

固定判据是 `彩色像素数/有效区域像素数 > 0.80`，不是≥。首轮保守色彩定义为 `maxRGB>=60`、`chroma>=45`、`chroma/maxRGB>=0.25`；观察到阴影漏判后，另记探索性宽松对照 `maxRGB>=30`、`chroma>=12`、`chroma/maxRGB>=0.20`。两者都不改80%门槛；后者根据本批结果提出，不能冒充独立验证。脚本为 `scripts/parts_rl/analyze_fingertip_roi.py`，快照及SHA留档。

助手检查20张原图、几何叠图及所选时刻前后约0.2秒静态视频帧，记录**当时持物13、当时指间为空7**。这是快照持物状态的定性审核，不等于逐attempt成功/失败：例如E01-R001所选时刻可能已放置，E04-R003更早有红色积木经过指间。未代填用户的273条审核结果。逐例依据见[fingertip-roi20-snapshot-audit.json](fingertip-roi20-snapshot-audit.json)，完整测量见[fingertip-roi20-roi-results.json](fingertip-roi20-roi-results.json)。

| 色彩定义（均>80%） | 持物检出 | 持物漏判 | 空手误报 | 空手未触发 |
|---|---:|---:|---:|---:|
| 首轮保守 | 4/13 | 9/13 | 5/7 | 2/7 |
| 阴影宽松，探索性对照 | 11/13 | 2/13 | 6/7 | 1/7 |

主要反例：E04-R001指间为空，但后方蓝框让占比达保守99.5%、宽松100%；E12-L005确实夹住黄色积木，占比89.2%/97.5%，两者都能跨过同一门槛。E04-L005和E10-L001确实夹着蓝色积木，宽松占比也只有78.4%和68.5%。因此保留尖端ROI作为定位工具，**单凭彩色充满80%不足以自动判断持物**；简单排除蓝色又会漏掉蓝色乐高。下一步若继续无跟踪路线，应在该局部区域区分前景积木与背景容器，而不是仅改占比门槛。白色积木也无法靠彩色占比可靠识别。

Thor独立容器 `parts-fingertip-roi20-20261010` 执行20图SAM3推理，18.78秒、退出0，沿用同一镜像与权重SHA；本地只做CPU图像几何与审核。4项纯图像反例/几何测试通过，限定Ruff和`git diff --check`通过；20个页面条目和60个原图/叠图/视频引用HTTP检查通过。汇总归[fingertip-roi20-audit-summary.json](fingertip-roi20-audit-summary.json)。没有自动跟踪、RL训练、生产接入或3588操作；所有 `approved_reward=null`，`automatic_reward_ready=false`。
