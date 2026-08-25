# Piper Legacy 指引

Piper 不是本项目主线。OpenArm 新训练、数据清洗、norm stats、服务和 rollout 一律从 `docs/00_handoff_index.md` 的 00–07 文档开始，不要从 Piper 配置复制命令。

仅在以下情况查看 Piper：维护旧 checkpoint、复现历史结果，或确认旧代码行为。配置名仍在 `src/openpi/training/config.py` 中，以 `pi0_piper_dual` / `pi05_piper_dual` 开头；它们使用自己的数据维度、transform 和 action space。请在执行前锁定旧数据目录、旧 norm stats 和旧 checkpoint，不与 OpenArm 16D degree/HQ 夹爪合同混用。

旧 Piper 长篇训练、微调、推理和 RTC 文档已从当前仓库移除，避免接手者误把 legacy 当默认流程。需要新建 Piper 维护说明时，应单独提交并从本页链接，不要把 Piper 内容写回 OpenArm 主线文档。
