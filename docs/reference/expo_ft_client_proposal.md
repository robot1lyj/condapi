# EXPO-FT 客户端重构提案（待用户审核，未发送）

状态：2026-10-10只读核对本地 `/home/wuyan-lyj/YAM/yam-abc-reproduce` 的RL模块、协议、集成调用点及文档，HEAD为 `2dc603595daf5ebe33c58c5aa0279c52fcb9740c`。外部工作树有其他未提交工作，本轮未修改。未访问3588系统、SDK、CAN或相机设备，未重启服务。此文是condapi内的设计交付，**不是已发送给客户端任务的指令**。

服务端架构归[01](../01_system_architecture.md#expo-ft-强化学习架构2026-10-10设计草案)，双任务定义归[04](../04_data_contracts.md#expo-ft-双任务数据合同2026-10-10)，线上接口草案归[05](../05_inference_and_rollout.md#expo-ft-三方接口草案2026-10-10)。

## 核心重构：从局部抓取残差，改为任务级RL会话

旧 `hil/parts/` 围绕下降高度、持物锁、单活动臂、六关节修正和attempt奖励组织。它能表达局部抓取实验，但无法作为“正确分拣/耳机入盒”的整个任务目标。重构后RL页面负责实验与数据闭环，物理控制仍归既有Runtime。保留已有暂停/接管/录制/RTC机制，不建立第二个机械臂写入者。

可复用的是能力边界：观测关联、实际提交动作、epoch、队列承诺、HIL、HDF5/MP4、outbox/ACK。重写的是任务状态、奖励协议、旧单臂资格、旧actor候选与parts数据schema；不将旧sidecar直接改名为expo。

## 模块与逐文件迁移建议

| 本地YAM现有位置 | 提案 |
|---|---|
| `hil/parts/` | 退役RL专用selector/attempt/残差合成；只读历史解析保留为迁移工具；新建 `hil/rl/` 的session、protocol、journal、labels、outbox |
| `hil/run.py` | 移除 `_build_parts/_new_parts/rotate_parts_session` 及RL阶段注入；新增任务会话挂钩，仍由现有主循环和仲裁者提交实际动作 |
| `hil/session.py` | 把parts资格/attempt生命周期替换为episode/bundle/epoch与取消语义；接管和暂停沿用现有权限顺序 |
| `hil/policy.py` | 旧 `PolicyJob/RtcJob.parts` 改为版本化propose/select协议适配；明确单在途worker的串并行、背压、取消和过期回复 |
| `hil/rtc_timeline.py` | 保留承诺前缀/target tick模型；移除 `pending_residual_arms` 与旧PARTS编辑回调的耦合，使用候选ID/来源标记；已承诺tick不可改 |
| `hil/workbench.py`、`hil/web.py`、`hil/device_client.py` | 替换 `/parts/grasp`、parts_config和局部attemptUI；提供任务选择、行为版本、episode、终点标签/复核与上传状态；接口改名与版本兼容一起验收 |
| `hil/data_session.py` | 换任务仍在HOLD的数据事务中旋转会话；不可把两种任务放进同一episode，不因换页重连设备 |
| `configs/parts_client.json`、旧RL页面与tests | 移除旧实验默认；以新任务/协议配置及独立回放测试替换。默认off，旧数据读取不触发运动 |

以上是读取调用点得到的集成范围，不是修改控制算法的授权；实际客户端开发在其独立worktree完成，遵循YAM自身AGENTS及用户审核。本仓库不复制YAM控制源码。

## 用户流程

1. 在HOLD选择“乐高颜色分拣”或“耳机入充电盒”，显示任务定义、固定bundle、奖励版本、连接/时间轴/录制就绪状态。
2. 运行模式区分off、shadow、collect、eval；切页、查看Q曲线或查看训练进度不改变模式、不发动作。模式与task/bundle锁定到episode结束。
3. 开始后仍用现有inference控制；人工接管复用Leader/HIL，记录逐tick来源。RL页面不提供绕过物理约束的第二条动作路径。
4. 结束后标注任务终点。乐高保留初始数量、正确新增、错分和遮挡未知；耳机记录槽位完成/释放后稳定等合同字段。可离线人工复核，不要求每次抓放按键。故障与未知不能默认失败。
5. 页面展示本地episode完成、上传ACK、服务端审计、训练可用四个不同状态。训练进度只读显示，不允许客户端自行宣称READY或启动远端训练。
6. 新策略到达时仅显示候选版本和评估摘要，按批准的运行合同在episode边界切换；不会因服务器有更新就替换现场动作。

奖励首版采用人工终点复核和规则/视觉候选标签；自动detector须先独立验证。两个任务需要颜色与槽位语义，不继承旧“只看高度/力矩即可判成功”的假设；也不未经选择就引入额外VLM依赖。

## 首次实现的验收清单

- 普通base-only与原RTC数据流回归；off模式下没有新模型修正；shadow只记录比较，不应用候选。
- 同时出现迟到回复、人工接管、epoch变化、版本更新时，旧动作不再提交；已承诺prefix逐值不变。
- 三图与state身份、慢/快双时刻、控制tick、实际action与反馈state分离可重放；人工/限幅/基础降级准确归因。
- 断网、重启、重复包、磁盘队列满与半包恢复：不中断控制优先级，不丢数据后伪称完整；上传幂等且控制线程不等待哈希或HTTP。
- 真正两任务episode切换、奖励修订、eval隔离和留出数据保持；未知结果不能自动变成Q标签。
- 离线模拟/协议回放通过只代表软件合同；真机动作、接触、相机同步、现场时延与reset另行验收，不在本轮执行。

## 供用户审核的客户端任务正文

> 请在YAM独立worktree设计并实现EXPO-FT客户端适配，按本提案及condapi文档05的版本化接口草案联调。将旧PARTS局部抓取残差页面、selector、attempt和协议退出活动路径，保留历史数据解析与原始文件；新建任务级RL会话，支持乐高按颜色分拣及蓝牙耳机入充电盒。复用现有唯一Runtime写入、RTC承诺前缀、Leader/HIL与录制通道，不另建SDK/CAN写入者。支持慢候选/快选择的请求身份、期限、取消与bundle锁定；保存真实执行、双时刻观测、人工与约束来源；实现终点标签复核和幂等outbox。先完成模拟/协议/数据回放及普通base/RTC回归，展示精确变更和验收结果；未经设备操作授权不连接、运动、部署或重启现场。与服务端对齐接口后再进入真机阶段。

该正文已经在交付文档中完整展示；依项目2026-09-30规则，**用户明确审核前不发送到客户端聊天**。本轮只完成服务端本地清理、论文分析与双方方案，未执行此客户端任务。
