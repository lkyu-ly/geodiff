# COV/MAT：同输入作者基准与独立评估入口

Status: resolved
Blocked by: 无。已同意在新 SDF 里写入来自原始分子图的 `GEODIFF_SOURCE_SMILES`；读入后仅以该字段配对，有序拓扑不符时报错，读回立体身份改变只计数不筛除。

1. 先读规格。对照 `geodiff/utils/evaluation/covmat.py`、`geodiff/utils/chem.py`、作者 `env.yml` 与 `geodiff_paddle/utils/evaluation/covmat.py`。先留存作者 Torch 原函数在非方形构象集合上的结果；若 `np.float` 在当前 NumPy 不再存在，仅可用已有或隔离的兼容 NumPy 版本跑原函数，不修改源文件或给模块打猴补丁。若需要重建复杂历史栈或改全局环境，停。
2. 可用的 Torch 基准先于套件业务代码。随后在同一个 `feat/geodiff` 工作树为 COV/MAT 的纯指标逻辑写一项会失败的行为测试；新增模块限于 `ppmat/metrics/geodiff_covmat.py`、对应独立测试。父任务负责共用 NPZ/SDF 适配、公开薄命令和文档。每个分子消费 `(key, rdkit_mol, ref_positions, generated_positions)`，数组按构象、原子、xyz 排列；接口 `compute_covmat(groups, *, ratio=2, thresholds=None)` 返回 JSON 可序列化的逐分子值、原阈值与四项指标汇总、参与/过滤计数。不要自行改变输入方以凑结果。
3. 默认 60 个阈值（Å）、ratio=2、去氢、双轴取最小；与作者原函数返回对象逐项对照。补齐输入不足、同分子过多构象及全空报错的真实行为。证据区分原实现与受环境影响的版本；不改模型或公共 sampler。

通过公开命令对齐才 resolved，遇停线则 needs-info。未经用户进一步授权不作套件 commit。

作者 COV/MAT 基准已跑通：2 个参考构象、5 个生成构象，按原规则取前 4 个，MAT-R 为 0.06669377942626072 Å，MAT-P 为 7.175751770612638 Å。完整双向覆盖数组及运行环境见 `runs/evaluation-parity.pISHWq/covmat/oracle.json`。共用输入触及停线，未实现套件指标或执行公开命令对照；红测仅保留在同一证据目录。

独立 Paddle 3.4.0（NumPy 1.26.4）与未修改作者 Torch 端（隔离 NumPy 1.23.5），CPU 单进程 worker，同一 RDKit 2024.09.1。读同一 NPZ 与 SDF 的坐标，默认 60 个阈值的 COV-R/P 数组、MAT-R/P 及其均值/中位数/标准差均相同；0.5 Å 阈值下 COV-R/P 为 `1.0 / 0.5`。夹具混有两条真实套件输出和已知坐标对照，结果仅验证算法，不是模型得分。[独立对照记录](../../../runs/evaluation-parity.pISHWq/covmat/paddle-mirror.log)。

**完整测试切分的数据核对（2026-11-25，本地候选数据包解压产物）：** 两套切分的 `idx` 与作者 SMILES 一一对应（各 1000 组），每个分子的记录在文件中连续；QM9 首个分子有 176 条参考，各组最少/中位/最多为 50/91/500 条，Drugs 首个分子有 50 条，各组为 50/70/100 条。套件按作者 SMILES 分组，与作者按 `idx` 分组等价。

参考组内行间 RDKit 分子不一致的组：QM9 2 组、Drugs 28 组；其中 Drugs 有 24 组行间只有立体标注不同、有序拓扑相同。按最终口径（有序拓扑或来源归属冲突）被拒绝的是 QM9 2 组、Drugs 4 组。例如 Drugs 的 `O=C(Cc1ccccc1)N[C@H](c1ccccc1)P(=O)(O)CO`（第 1113–1170 行，同属 `idx 15`）：58 行都是 40 个原子，RDKit 分子却有三种拓扑（29、15、14 行），作者的 SMILES 标签与三者都不同。作者的打包代码以首行分子图为模板、把全组坐标都当作参考；套件不把这类组混算。套件解析完整测试切分分别约 23 秒和 33 秒（CPU 单进程，最终代码）。

## Comments

**2026-11-25 解决（分支 `feat/geodiff` 工作区，未提交）**

公开命令 `python -m structure_generation.evaluate_geodiff covmat` 与未修改的作者 `CovMatEvaluator` 读同一份 SDF，四组输入的逐分子数组最大绝对差均为 0：2 条参考对 5 条候选的夹具（ratio=2 取前 4 条）；QM9 首个分子 176 条参考对 352 条生成（接口验证用 2 步扩散）；此前的 QM9（176 对 352）和 Drugs（50 对 50，ratio=1）输入。作者端 Torch 2.11.0+cu128 加隔离的 NumPy 1.23.5，套件端 Paddle 3.4.0 加 NumPy 1.26.4，RDKit 同为 2024.09.1。数值只说明算法一致，不是模型得分。

评审代理（`MetricContributionReview-2`、`MetricSemanticsReview-2`）意见的处理：参考组冲突只看有序拓扑与来源归属；立体差异只比较条件分子图指定的 R/S 中心和 E/Z 双键；命令先校验 `--output`，截断文件按用法错误报出，没有生成构象与不足预算分开计数；README 的评估示例改为采样首个 QM9 测试分子的 176 条参考，并写明默认步数下一条记录的采样耗时。评审代理建议的 `--skip-conflicting-references` 开关实现后，按用户 2026-11-25 补充的复现原则移除：规格只写显式拒绝评分，它不是通过性必需。

最终代码上，参考行拓扑或归属冲突的组为 QM9 测试 2/1000、Drugs 测试 4/1000，验证切分各 11/5000；被采样时命令退出码为 2。7 个测试文件 35 个用例通过，black 22.3.0、isort、ruff 无需改动。证据在 `runs/ppmat_covmat.Wm7zBp/final-verify/`。

待用户裁定（见根目录实验记录同日条目）：冲突参考组是否改为与作者一致照常计分；套件按参考行逐行采样，作者按分子只用首行分子图采样，这一差异是否对齐。

**2026-11-25 对齐作者实现（用户裁定）**

用户裁定采样与评估逻辑都与作者原实现一致。评估侧去掉了对参考组的全部拒绝路径：`_reference_groups` 以首行参考记录的分子图为模板、组内全部参考记录的坐标为参考，只从参考文件读取 `smiles`、`pos`、`node_ptr`、`rdmol_ptr`、`rdmol_data`，不再校验作者评估器不读的原子和边数组。仍然报错的只有配对和回读问题：缺来源标记、未知来源、同一来源是两个作者分组的首行、生成记录与来源的有序拓扑不符。

对照用的是作者自己的类：`PackedConformationDataset` 读作者原始 pickle 构造参考条目，`CovMatEvaluator` 计分；套件读发布的 NPZ。采样用正式权重和预设默认的 `2x`，扩散步数设为 2（只验证接口与算法，不代表生成质量）；这批采样实际落在 CPU 上（Paddle 3.4.0，工具 shell 里有一个空的 `CUDA_VISIBLE_DEVICES`）。QM9 取首分子和测试集里全部 2 个行间拓扑不一致的分子（74 条参考含 3 种拓扑、234 条参考含 2 种），共 968 条构象；Drugs 取首分子、全部 4 个拓扑不一致的分子和 1 个仅立体标注不同的分子，共 810 条构象。两套分别有 3 个和 6 个分子入选，逐分子的 60 个阈值覆盖率和 MAT、计数与作者结果的最大绝对差均为 0；每个分子的构象数都是参考数的 2 倍，来源标记只有首行分子图一个。RDKit 同为 2024.09.1；作者端 Torch 2.11.0+cu128 加隔离的 NumPy 1.23.5。证据在 `runs/ppmat_covmat.Wm7zBp/align-verify/`。

随后在显卡上按文档命令重跑首个 QM9 分子（`Sampler.sample_batch_iters=1`，预设默认 `2x`，扩散步数设为 2）：RTX 4060 Ti、Paddle 3.4.0、FP32，日志确认使用显卡，写出 352 条构象、1 个来源标记，套件与作者结果的最大绝对差为 0。两套测试集里参考最多的分子（QM9 第 979 个，500 条参考，单批 1000 个构象；Drugs 第 162 个，100 条参考，单批 200 个构象）用预设结构、2 步扩散测得 Paddle 峰值已分配显存均为 0.41 GiB；QM9 这一批用 `nvidia-smi` 复核，总占用从 5842 MiB 升到 6590 MiB。显存测量期间显卡与用户的 Windows 宿主共用。

参考文件按最终代码加载：测试切分里没有首行分子图相同的两个作者分组，验证切分 QM9 有 1 个、Drugs 0 个，训练切分 QM9 15 个、Drugs 20 个；六个切分都能完整加载。