# Thor · 08 真实录像精度回放

本页负责重跑操作；系统、精度结论和当前状态统一见 [Thor 部署第 7 节](../../08_thor_edge_deployment.md#7-当前状态)，结果见 [中文 HTML](../../reports/thor/index.html)。不是机械臂驱动指导，也不操作 3588。

## 已准备的内容

以下命令在 Thor 主机执行。代码目录 `/home/wuyan-lyj/condapi`；模型系列目录 `/home/wuyan-lyj/thor/pi`，其中 `checkpoints/pi05_base` 为原始只读 JAX 权重，`cache` 存分词器，`test-data/pi05-replay-v1/suite.json` 是本轮固定真实输入清单。不要把 benchmark-only norm 写入原始 checkpoint。

简单核对即可，不需要反复审计安装：

```bash
cd /home/wuyan-lyj/condapi
docker image inspect openpi-pi:thor-jax-20260907 --format '{{.Id}}'
nvpmodel -q
```

日常应为 `120W / 1`。如果上次异常断电/SIGKILL 导致 MAXN 未恢复，先按 [恢复指导](07_recovery_and_handoff.md) 查看遗留时钟备份，再恢复日常模式；普通报错和可捕获退出由启动器自动处理。

## 一次运行三组

`--batch-id` 每次使用新的名字；已有结果不覆盖。以下 `manual-01` 只是示例，第二次改为 `manual-02`。Thor 最初通过 USB/rsync 接收代码；当前 Git 首次接入状态见 [Gitea 同步](09_gitea_code_sync.md)，没有有效 HEAD 时不能在 Thor 用 `git rev-parse` 获取版本。`--code-commit` 填工作站同步源码的提交；它是代码来源基点，启动器还会记录实际文件指纹和镜像 ID。下面已填本轮基线实现提交；以后更新源码时，从工作站仓库获取新提交并替换它。

```bash
sudo python3 scripts/thor/run_suite_host.py \
  --image openpi-pi:thor-jax-20260907 \
  --batch-id manual-01 \
  --modes A B C \
  --code-commit 058c1f7ef838874f444639d687fdcc1b166ab335
```

只重跑 C 时改成 `--modes C`。每个模式各自加载模型一次，依次完成固定 9 输入的预热与正式调用；MAXN 只包围前台推理命令，结束/报错恢复 120W。期间自动记录自己的 `tegrastats` 进程，结束时只关闭这一进程；不停止其他用户任务。不使用 MAXN 开机服务。

推理容器断网，模型与录像均从 Thor 本地挂载，`checkpoints`/`test-data` 只读；Pi 系列共用相同镜像和入口。不要把下载、装依赖、桌面会话包在 MAXN 启动器内。

## 查看结果与比较

结果在 `thor/pi/results/pi05-模式-批次/`：`result.json` 仅在整组完成后产生；`started.json` 单独存在不代表完成。`actions.npy` 为 `[9,20,50,14]`，`normalized_actions.npy` 保留内部 32D。每个输入另外保留动作与耗时记录，发生后续失败也不丢掉已完成证据。宿主日志、遥测和退出码在 `thor/pi/logs/`。

工作站收回结果后，在现有项目 Python 环境中运行（不是让 Thor 宿主机安装模型依赖）：

```bash
python scripts/thor/build_suite_report.py \
  --runs /完整路径/A结果目录 /完整路径/B结果目录 /完整路径/C结果目录 \
  --logs /完整路径/宿主日志目录 \
  --plan docs/reports/thor/next_test_plan.json \
  --json docs/reports/thor/status.json \
  --html docs/reports/thor/index.html
```

报告生成器检查三组模式、实际参数 dtype、动作文件哈希、suite/noise/norm/config/版本一致性和宿主退出码；不接受缺失结果，也不会用理论性能补零。新实验要先审阅旧 `next_test_plan.json` 是否仍适用，再更新推荐，不要机械沿用旧判断。浏览器直接打开 HTML，无 CDN、无遥测。

## 当前还没有做什么

没有转换成 PyTorch/TensorRT，没有 FP8/NVFP4，没有 YAM 微调模型任务测试，没有操作机器人或 3588。先用这条原生路径保留数值参考，后续实验按 [下一轮方案](../../reports/thor/next_test_plan.json) 逐步改变一个因素。
