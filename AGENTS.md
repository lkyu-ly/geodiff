# GeoDiff 项目记忆

- 每轮交互结束前，在根目录 [GeoDiff迁移实验记录.md](GeoDiff迁移实验记录.md) 追加记录，静态调研也记。日期用 `## YYYY-MM-DD`，同日复用；每轮用 `### 工作内容`。关键数值和实测条件直接写入单文件，不能只链接 `runs/`；不贴长日志，未运行就注明。
- `geodiff/` 是作者的 Torch 参考源码。日志保留原有指标，打印间隔可调；精度对齐逐 step 打印，不写 TensorBoard 事件。核心前向、损失、反传或更新路径遇到兼容问题，先核对作者环境；数学上等价的补丁也须用户确认，带补丁的结果单独标明。Paddle 镜像与其并列，不向 Torch 上游贡献。
- 任何时候都不使用 SHA 标记文件。用可读的路径、分支、日期、配置和直接内容比较说明来源。
- 设计实验、改代码和运行测试都选能回答当前问题的最小规模。一个样本或小数据集足够就不扩大；不为预计中的需求加工程层。若确实需要大改逻辑或依赖，先说明影响，再与用户确认。不做国产显卡适配。
- 面向 PaddleMaterials 交付时，按[注释与措辞规范](../工作总结/ppmat贡献交付注释与措辞规范.md)与[贡献报告](../工作总结/PaddleMaterials贡献流程与CINN接入调研报告_2026-09-07.md)区分本地记录与上游文档。上游注释用简洁英文、保留必要署名，不写迁移过程、内部票号或个人路径；实测数字写明条件，论文结果注明出处。

## 阶段进度输出要求

多步骤任务中，用 `write` 写入 `out://` 向用户发简短进度，发完继续工作。
确认关键发现、准备转入实施或验证、遇到阻塞或改换方案时，先发一条。
每条 1–2 句：说清已确认什么、遇到什么问题、接下来怎么验证。
简单任务直接回答；不用逐条复述工具调用。

## Agent skills

### Issue tracker

本地 Markdown，规格和工单放在 `.scratch/<feature-slug>/`。见 [issue-tracker.md](docs/agents/issue-tracker.md)。

### Triage labels

使用默认五个 triage 标签。见 [triage-labels.md](docs/agents/triage-labels.md)。

### Domain docs

单上下文，按需维护根目录 `GLOSSARY.md` 和 `docs/adr/`。见 [domain.md](docs/agents/domain.md)。
