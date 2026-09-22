# Condapi 架构图

使用已安装的 Archify 2.17 生成，类型为 `architecture`，主要语言为中文。

- [交互 HTML](architecture.html)：主题切换、缩放、来源查看与导出由 Archify 提供。
- [JSON 图源](architecture.json)：后续更新的编辑入口；不要直接修改生成 HTML。
- [确定性交付回执](architecture.delivery.json)：9/9 showcase，0 errors，0 warnings，19 处源文件引用通过验证。
- [浏览器回执](architecture.visual-check.json) / [截图对照](architecture.visual-check.html)：四种桌面尺寸均无横向或纵向溢出。

架构事实仍归 [01](../01_system_architecture.md)，模型可用性归 [10](../10_vla_platform.md)。本图是总览，省略重复的日志连线；看板独立读取训练日志。数据进入 OpenPI 的连线仅代表 Pi 路径，不暗示其他模型套用 Pi 数据变换。checkpoint 节点概括模型交接资产，不表示 Pi 使用 LeRobot processor，也不表示模型包完整性验证已实现部署。未读取或操作 3588，未运行训练或远端部署。

来源固定在 `30a270d4bb0517ca17f554d95a366444219a1136`；本次同时修正架构 owner 中仍称 LoRA 为默认的旧描述，依据 AGENTS.md 及正式训练入口。

## 验收

- specification_sha256: `6ce84dee27b0da73ebde45c41fb8775057247490297d9ff6d5790662f34d90d6`
- artifact_sha256: `e60c71b6f4b21999e00bb168003cd16c878d77aaf37db42cea1e5f223dfc0eda`
- browser_evidence: passed
- visual_review: passed（实际查看 2048×1320 浅色与 1440×900 深色截图；节点与标签完整，主线清楚，无页面裁切）
- correction_rounds: 1（关系标签避让；未修改交付 HTML）

自动检查覆盖 1440×900、1600×1000、1920×1080、2048×1320，以及最小/最大尺寸的深浅主题截图。截图目视检查不代表已逐项测试搜索、导出等交互功能。首次浏览器发现失败，指定本机已有 Chromium 后重试通过；最终回执绑定上述 HTML 哈希。

## 重新生成

在仓库根目录使用安装目录中的 CLI；图源带有源代码版本约束，更新图源时应核对来源并更新 revision。

```bash
node ~/.codex/skills/archify/bin/archify.mjs validate architecture docs/diagrams/architecture.json --repo-root . --quality showcase --json
node ~/.codex/skills/archify/bin/archify.mjs deliver architecture docs/diagrams/architecture.json docs/diagrams/architecture.html --repo-root . --quality showcase --json
node ~/.codex/skills/archify/bin/archify.mjs visual-check docs/diagrams/architecture.html --json
```

浏览器不在 PATH 时设置 `ARCHIFY_CHROME` 指向已有 Chromium 可执行文件。
