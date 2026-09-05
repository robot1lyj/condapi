# 00 · 参考资料索引

`reference/` 不是 YAM 当前操作入口，只保存不适合放在主线文档中的实现参考和 legacy 指引。

## 当前可参考

- `legacy/piper.md`：Piper 唯一保留的维护指引。

## 明确不作为默认值

- `legacy/piper.md` 中的旧配置仅用于维护/复现，不能作为 YAM 数据、norm stats、prompt 或服务命令。
- RTC、服务器和 YAM 合同只看 `docs/02_installation_and_environment.md`、`docs/04_data_contracts.md`、`docs/05_inference_and_rollout.md` 与源码；不要在 reference 下新增重复说明。
- 任何旧参考与编号化当前文档冲突时，以 `docs/01`–`docs/06` 和代码中的当前 config 为准，并在变更历史中说明原因。
