# 回填并验证正式链接

Status: needs-info

## What to build

收到用户转交维护者后取得的正式链接，用套件的预训练与数据设施注册，并真实下载验证可用。

## Acceptance criteria

- [ ] 在不预置缓存的情况下通过真实地址下载、解包、按名加载，并验证采样和数据切分读取。
- [ ] 仅填包外地址，不重打包、不伪造链接；不提交剩余的 PaddleMaterials 改动或推送，不代提 PR。

Blocked by: 05；另需用户提供正式权重和数据链接。

参考：[接入规格](../spec.md)、[对应任务](../plan.md#任务回填并验证正式链接)。

## Comments

实际两包都已在 GeoDiff 根目录并通过解包验证。等待用户手动交维护者后提供权重和数据正式 URL；当前没有登记占位地址，也没有向任何 PaddleMaterials 远端提交、推送或创建 PR。

用户决定把上传和登记延到后续调试全部定版。当前以真实本地 ZIP 验证下载、解包和模型加载，不等待正式 URL。

双预设的最新四个候选包都在 GeoDiff 根目录，已通过回环 HTTP 与真实套件下载。正式上传和地址登记仍延后；届时使用 QM9、Drugs 各自独立的权重与数据地址，见[双预设调研](../../../docs/GeoDiff_QM9与Drugs双预设调研.md)。

2026-11-26：任务页 `structure_generation/README.md` 的 Models Matrix 已加 GeoDiff 列，这是官方归纳的登记方式，不再另起第 3 节；模型页按 DiffCSP、MatterGen 的骨架对齐，架构图放在 `structure_generation/docs/geodiff.png`。仍待正式地址后处理：`MODEL_REGISTRY`、根 README 四份的表格、`docs/models_overview.html` 的注册权重表、模型页的下载链接。

2026-11-26（提交）：用户授权后，工作区代码已合成一个提交 `feat: add GeoDiff for molecular conformation generation`，与 `feat: support named optimizer groups` 一起在本地分支 `feat/geodiff` 上，共两个提交；没有推送到 fork 或官方。集成工作树已删除，后续在 `/home/lkyu/baidu/PaddleMaterials` 检出该分支。登记类事项不变，仍待正式地址。
