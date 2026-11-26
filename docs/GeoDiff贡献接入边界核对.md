# GeoDiff 贡献接入边界核对

**本地首版已完成；正式发布尚未就绪。** 对照本地 `upstream/develop`，`feat/geodiff` 新增的是一个 `GeoDiff`、一套 GEOM 读取与拼批、两个预设，不是两个模型。套件已能本地训练、原生续训、从有键 NPZ 生成 SDF；双 Adam 分别调度、统一裁剪。既有记录验证了 QM9 固定输入去噪、loss/梯度、续训和两步 SDF 采样；Drugs 验了加载、一步训练/验证和两步采样，未做精度对齐。两套完整作者 train/val/test 和各自发布权重共四包，已从空缓存经回环 HTTP、经套件解包加载。两步实测不等于跑过完整 5,000 步或复现论文指标。两个提交（公共分组训练、GeoDiff 本体）都在本地分支 `feat/geodiff` 上，未推送；集成工作树已于 2026-11-26 删除，后续在 `/home/lkyu/baidu/PaddleMaterials` 检出该分支，旧记录里指向 `PaddleMaterials-geodiff/` 的链接对应这个分支。[首版规格](../.scratch/geodiff-ppmat/spec.md#L11-L25) · [双预设计划](../.scratch/geodiff-dual-presets/plan.md#L5-L22) · [实测记录](../GeoDiff迁移实验记录.md#L170-L194)

## 作者工作流：当前覆盖范围

- **COV/MAT 已按原口径接入。** [套件命令](../../PaddleMaterials-geodiff/structure_generation/evaluate_geodiff.py)评估新生成且带来源标签的 SDF，按参考预算截取，计算重原子 RMSD 的双向覆盖与匹配。它与训练用 `GeoDiffValidationLoss` 分开。公开命令已在受控夹具和两个预设的真实小样本上对齐未修改的 Torch 作者结果，未运行整集论文基准。标签、旧 SDF 和不一致参考拓扑的处理见[使用说明](../../PaddleMaterials-geodiff/structure_generation/configs/geodiff/README.md#evaluating-conformer-coverage-and-matching)。
- **性质评估不贡献。** 用户已决定关闭这项可选工作，不在套件引入 Psikit/Psi4，也不额外分发性质切分。独立镜像与探针的历史结果保留在下节，不作为上游功能或精度证据。
- **采样实现与公开选项要分开。** 两个预设公开固定整数构象数和步数，默认 LD；底层仍有 `generalized`、`ddpm_noisy` 与轨迹选项。作者的权重/裁剪等旋钮（含 README 建议的 QM9 `--w_global 0.3`）、按分子切段、续采和轨迹不在套件公开接口；这不是默认 LD 缺失。PyMol/绘图、自制 GEOM 切分、独立训练回放也不是首版承诺；输入是有键 NPZ，不是任意 SMILES。`drugs_1k_default` 是消融训练配方，不是第三份发布权重；不要求恢复作者 Torch 优化器。[作者 README](../geodiff/README.md#L32-L64) · [作者采样参数](../geodiff/test.py#L16-L50) · [套件采样实现](../../PaddleMaterials-geodiff/ppmat/models/geodiff/geodiff.py#L300-L395)

## 性质评估实跑结果

从作者的性质切分取一个 16 原子分子，用独立 Paddle 镜像和发布权重完成 5,000 步采样。未改脚本时，两边的 Psi4 单点能都算出，但 Psikit 0.2.0 调用 Psi4 1.9.1 的旧偶极矩接口，现有异常处理丢弃了整个构象，最终对空数组取最小值。只在 `runs/` 的运行器中换用现行偶极矩读取接口，原 SCF/6-31G** 和统计公式不变，一对真实参考/生成构象算出了五项有限差值（[失败证据](../runs/property-probe.yw7DeF/eval-full.log)、[通过证据](../runs/property-probe.yw7DeF/eval-compat.log)、[实测条件](../GeoDiff迁移实验记录.md#工作内容性质评估实跑与贡献接入查漏)）。这证明计算路径可跑，还未验证套件 SDF 的分子配对、构象分组或失败处理；不能直接合入镜像脚本。

## 发布待办与后续阶段

- **发布门槛：** 两个预设的正式权重地址和数据地址共四个。类已可由 YAML 构建，不等于 `MODEL_REGISTRY` 已登记；模型 README 目前给本地用法，没有正式数据获取链接。拿到地址后只填包外登记，从空缓存验证按名下载、解包、加载并采样，以及数据 MD5 与切分读取，不重打包。英文模型页已对齐同类模型页的结构，任务页 Models Matrix 已加 GeoDiff 列；根 README 四份和 `docs/models_overview.html` 尚未列 GeoDiff，宜随发布同步，未见要求所有语言首页都先改的硬门槛。[发布票](../.scratch/geodiff-ppmat/issues/06-release-links.md) · [包契约](../../PaddleMaterials-geodiff/docs/model_package.md) · [新任务清单](../../PaddleMaterials-geodiff/docs/trainer_docs.html#m-newtask) · [模型页](../../PaddleMaterials-geodiff/structure_generation/configs/geodiff/README.md) · [模型总表](../../PaddleMaterials-geodiff/docs/models_overview.html#L201-L220)
- **CINN 与测速都未做，且互有先后。** 交接先要求验证 `denoise_step` 的真实编译路径和 eager/CINN 数值，再测同卡同输入的首调用及 warm 延迟；二者都明确不在首版规格内。[交接阶段](handoff/GeoDiff接入PaddleMaterials交接.md#L12-L16) · [首版约束](../.scratch/geodiff-ppmat/spec.md#L11-L18)
