# 客户端延迟改造建议（待用户审核后交客户端执行）

复用现有YAM的PolicyWorker、ProcessPolicyClient与原RTC队列。当前已确认前缀10步，最新一集0迟到；不修改机器人控制频率、RTC时钟映射、图像/状态配对、动作限位、HIL和原始录制，不加入新的客户端抓取门控。

第一步只补耗时拆分，先定位再改执行路径。在同一RK主机monotonic时钟下，记录worker从队列取任务、父进程send前后、子进程recv完成、RTC请求准备完成、打包完成、WebSocket发送/接收、解包完成、子进程发送回执、父进程接收回执、validate_reply完成、回复队列ready及主控制循环处理时间。现有token.created_at使用控制tick的now，并不等于真正发送时刻；保留其语义，新增实际submit/worker时间，不重新赋值。保留server_timing整字典；新Thor会增加request_decode_ms/base_infer_ms/pld_postprocess_ms，不能将不同机器的绝对monotonic时间相减。

静态合同、模型身份格式和模式能力在握手、重连、配置/模型切换时验证后缓存。当前PldClient已经如此处理；不要为“优化”删除每包的轻量身份相等检查。每包必须继续验证run/episode/epoch/request/observation、RTC前缀与来源、动作形状/有限值、trace与顶层动作一致、上一真实命令和observation绑定；30Hz最终下发保护、迟到丢弃、退出/超时/HOLD逻辑保留。

若实测确认Pipe序列化是主要成本，优先试“请求只编码一次”的小改动：由后台worker构造并验证RTC envelope后编码为msgpack bytes；进程间使用有边界的bytes消息传递，子进程只负责WebSocket I/O，不再把3路RGB数组pickle到子进程后重新msgpack。响应同理尽量只解码一次。握手、错误、超时、进程退出和重连消息需要明确类型/长度及回执绑定，不能混入旧回复；只在离线fake-server测试与暂停状态连通性测试通过后启用，保留旧路径回退。若收益不明确，不先引入共享内存环形缓冲的所有权复杂度。

OpenPI codec改动`22fcebf`只为C连续数组把tobytes中间分配换成memoryview，网络字节完全相同；若客户端升级这一SDK文件，需对连续、跨步、Fortran、大小端、标量/空数组及原实际消息逐字节回归。Thor此项仅省约0.07ms，不能宣称解决了RK侧13ms；RK实测后决定是否保留。图像仍为原始RGB，不缩图或有损压缩来改变输入分布。

验收在新目录保存改前/改后原始分段计时、P50/P95/P99、迟到数、模型/合同身份和完整回执；故障注入需证明迟到、错epoch、非有限动作、前缀篡改、子进程退出仍被拒绝。目标是降低真实端到端延迟，同时执行动作和对齐结果不变。没有向客户端任务自动发送本建议，也没有修改3588。
