# 乐高50h Pi0.5 RTC 240000：只下载，等待用户继续授权

## 当前状态

2026-10-13，用户先要求暂停`pld-rl`并开启240000；随后明确收窄为：**先下载，下载完成通知，取得新的明确授权后才能继续。** 最后这条指令优先。本次仅下载推理必要文件及做传输完整性校验，尚未转换、生成JAX参考、构建ONNX/TensorRT、测试推理或上线。

**2026-10-13 15:34:48+08下载结束，专用任务退出0，18项逐文件SHA全部通过。** 含合同共18个文件、12,440,565,864 bytes（约11.59GiB）；原始校验输出见[download-only.log](download-only.log)。15:35:48+08再次复核：Thor的`pld-rl`已停止、其MAXN watcher退出0/inactive，宿主120W；原Pi200000、230000备份也停止，8000无监听，没有运行中的容器。

24万FP32转换目录、有限性审计文件和TensorRT目录均不存在，证明未进入后续模型阶段。**下载完成不等于权重数值有限、转换准确或推理就绪；继续仍须用户新的明确授权。** 所有既有权重、引擎、容器和服务器源保留；未读取或修改3588，未发送客户端任务。

## 源与目标

- 源：`yam-server:/home/wuyan/lyj/YAM/training-runs/pi05_yam/lego_pi05_rtc_base_50h_20260921/240000`，只读核对正式Orbax完成元数据、params元数据和checkpoint内norm。
- Thor目标：`/home/wuyan-lyj/thor/pi/checkpoints/lego-pi05-rtc-base-50h-20260921/240000`；独立于200000/230000，不覆盖旧版本。
- 只取`params/`、`assets/`、`_CHECKPOINT_METADATA`，共17个源文件、12,440,542,241 bytes；另取同run控制目录的`training_contract.json`。**不取train_state，不是完整训练续跑备份。**
- 完成元数据SHA256 `49c4288ca7fc4f260ce2025d7e39e4ea736e2604a914ea7fd7af4fc91ac0edfe`；norm SHA256 `fd0d9cb2db3ce475c1175c7f21a3a0dd3aa7f59d36edc680e7044cd11b9f1edf`；训练合同SHA256 `924c8bfacede0e0bd83328199b4d1fb2e88df7c401ac884fb99f30a7ff08aecb`。18项完整清单[source.sha256](source.sha256)已全部核验。norm/合同与此前同run20万、23万相同，但它们的引擎/参考不能代替24万自身的后续验收。

## 下载边界与原自动流程的截断

初始完整接入脚本只进入服务器源文件SHA扫描，还没有实际下载或转换时，收到用户的新边界。立即SIGSTOP精确暂停本任务编排shell PID2709011，使其不可能自动进入CPU转换；源SHA扫描子进程2709014继续完成。随后独立下载专用任务复用这次扫描结果，核对源清单含所有必需元数据后，只终止该冻结编排shell，不终止数据传输；原SSH会话因此退出137。这是主动遵守新指令，不是权重损坏或下载失败。

唯一下载任务为[下载专用脚本](rtc50h240000-download-only.sh)，六路rsync由Thor直接读取服务器，通过当前SSH会话临时转发已有agent认证，不复制私钥、不在仓库保存凭据。任务末尾只有逐文件SHA校验和`DOWNLOADED_ONLY_50H_240000_WAIT_FOR_USER_AUTHORIZATION`，没有任何Docker模型执行命令；CPU转换及GPU部署单元没有启动。

Thor stage为`/home/wuyan-lyj/thor/pi/probes/rtc-50h-240000-20261013-r1`。其中此前准备的自动prepare/deploy脚本**未获新的继续授权，不得执行或恢复**。下载专用脚本也绑定本次源扫描PID，仅作操作证据，不应当成通用重跑入口。200000/230000备份和PLD容器保持原样，未经用户指令不清理或重新启动。

## 获准继续后的范围

必须先取得本轮新的明确授权，再按[RTC接入手册](../../../reference/thor/13_trained_rtc_inference.md)执行原始参数有限性检查、FP32保真转换、checkpoint自身norm的真实d=0/1/10回放、MAXN七步JAX/ONNX/TF32 TensorRT对照、同地址协议测试与上线。继续保持分位数前缀、H50/14D绝对目标、0.2rad/tick跳变拒绝；本报告不构成自动执行许可。此时也不能把“下载校验通过”写成数值精度或服务已就绪。
