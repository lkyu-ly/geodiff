# GeoDiff QM9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把已对齐的 QM9 模型接入 PaddleMaterials 的标准训练和分子构象采样，并交付可验证的权重包与数据包。

**Architecture:** 用已验收的数值主体实现一个套件模型，复用公共数据、训练、采样和权重加载接口。多组训练是显式启用的公共能力；GeoDiff 的数值复用各走独立回归。数据与权重在本地定版，发布地址与归档解耦。

**Tech Stack:** PaddlePaddle、PaddleMaterials、OmegaConf、NumPy NPZ、RDKit SDF、pytest。

**Spec:** [GeoDiff QM9 接入 PaddleMaterials](spec.md)。执行状态以[本地工单](issues/)为准；现只等用户提供正式链接。

## Global Constraints

- 首版只交付作者的 GEOM-QM9 配置和单卡 FP32 动态图训推；不加入 Torch/PyG 运行时依赖、CINN、测速、Drugs、测试集生成指标或历史 Torch 优化器恢复。
- 公共训练扩展须显式启用，旧模型和配置保持原行为；公共框架改动验证后可单独做一次本地提交，仅此范围。
- 公共归约和扩散建表只在证明等价后换用，且只改 GeoDiff 的调用；保留原构图、加噪、反向更新及发布权重的参数和缓冲区布局。其他核心数值改动须用户确认。
- 在现有 `structure_generation/sample.py` 添加显式的有键分子图模式，读 QM9 NPZ、输出保留原子和键的 SDF；保留现有晶体模式，不新增 SMILES 等输入格式。
- 权重包和数据包在内容与接口验收定版后分别放在 GeoDiff 根目录，由用户手动交给维护者；正式链接只写到包外的套件登记。
- 未经用户明确授权，PaddleMaterials 其余改动留在工作树；不推送至 fork 或官方仓库，PR 由用户手动创建。

## Review Focus

- 两路验证趋势相反，且各组梯度单独未超阈值、合并后超阈值。**公共双路更新**分别验证 LR 派发与联合裁剪。
- 变长分子有空边、孤立节点或超过邻居上限。**套件构建与固定去噪**验证样本和有序边不串图、类型 `0` 的边不消失。
- 验证批的原子数不同，恢复时待降 LR 的状态不同。**标准训练与原生恢复**验证原子均值与两组状态确实生效。
- 分子采样不需要训练集，SDF 重读后仍有显式氢和原键。**分子图采样命令**验证独立采样和既有晶体入口。
- 权重只警告缺键或归档带错误配置仍能解压。**定版并交包**必须用实际归档验证状态、配置、数据可读和真实输出，而不是只查路径存在。

## 开工准备

- [ ] 获得实施指令后，核对 PaddleMaterials `upstream/develop` 与官方 `develop`。从该引用建独立 `feat/geodiff` 工作树（建议 `../PaddleMaterials-geodiff`），不切换或改动 `feat/chemeleon`。路径或分支已被占用时先识别归属，不覆盖用户工作。
- [ ] 复用本地 `geodiff_paddle/`、`runs/paddle_qm9/qm9_default.pdparams`、`train_2.npz`、`val_2.npz`、`runs/torch_qm9_smoke/forward.npz` 和 `runs/qm9-end-to-end/train_inputs.npz`。如单图采样输入不在已有小样本内，只离线导出一个带真实化学键的样本。记录 Paddle、设备与精度；不重验已通过的 Torch 对齐作为开工门槛。

## 任务：公共双路更新

**依赖：** 开工准备。**文件：** PaddleMaterials `ppmat/optimizer/{__init__,optimizer,lr_scheduler}.py`、`ppmat/trainer/base_trainer.py`、必要的 `structure_generation/train.py` 指标接线、`test/test_grouped_training.py`。

- [ ] 先写行为测试：同一 Trainer 下让两路验证指标走势相反（测试用 `patience=0`），只降该降的一路；用范数分别为 3、4 的梯度在阈值 4 做联合裁剪，检查进入更新的梯度或 Adam 矩表现出共同缩放，而不是检查一次 Adam 更新后的参数碰巧一致。另用既有单 Adam/单 Plateau 配置覆盖原裁剪和调度。运行 `python -m pytest test/test_grouped_training.py -q`，确认红在缺少多组能力。
- [ ] 在现有 `build_optimizer` 增加独立的 `Optimizer.groups` 配置，每个命名组用 `model_attr` 选子层，其余仍是套件的 Adam/Plateau 配置；`clip_norm_global` 在组外启用联合裁剪。复用 `OptimizerList` 管理两套状态，保留其旧调用行为。多组训练的每个更新步，在第一个 Adam 更新前裁剪去重后的全体参数一次，保留无梯度语义；日志增加组名对应的 LR 与裁剪前范数，单组日志不变。
- [ ] 用现有 `SchedulerList` 表示多组调度，在同一次验证后按每个 Plateau 的 `indicator/indicator_name` 分别取指标。复用已有流式指标注册/附挂方式供标准训练入口使用；公共代码不识别 GeoDiff 类名。单组和既有调度配置、原生断点格式及默认更新顺序保持不变。
- [ ] 重跑上述测试，验证变化后的组状态能用正常保存/恢复路径还原。通过后用 `commit` 工具在独立 PaddleMaterials 工作树提交**一次仅含公共训练改动及其测试**的本地 commit，不推送。若必须重写公共训练循环、改变断点格式或扩大兼容范围，先说明影响，再征求确认。

## 任务：套件构建与固定去噪

**依赖：** 开工准备。**文件：** PaddleMaterials `ppmat/models/geodiff/`、`ppmat/models/__init__.py`、`ppmat/datasets/geom_qm9_dataset.py`、`ppmat/datasets/__init__.py`、`structure_generation/configs/geodiff/geodiff_qm9_default.yaml`、`test/test_geodiff.py`。

- [ ] 先写 `test_geodiff` 的数据与构图用例：两种原子数、空边和孤立节点，确认拼批偏移与原子顺序；构图用边界距离、同坐标跨分子和 >32 候选，核对半径截断顺序、去自环、键类型与零类型半径边。先跑 `python -m pytest test/test_geodiff.py -q` 验证缺少注册或正确输出时失败。
- [ ] 通过套件工厂构建 `GeomQM9Dataset` 和 `GeoDiff`。`GeomQM9Dataset` 以 `path` 读取显式 NPZ，以 `split` 保留原切分；数字字段用 `allow_pickle=False`，不在训练时还原全部 RDKit 分子。传 `Data`/`Batch` 及 `atom_types`、`cart_coords`、双向 `edge_index` 和 `edge_type`。若实际 reader 不能传送批对象，只在传输处把已由公共 `Batch` 拼好的字段转字典，不另写 DataLoader。
- [ ] 保持一个模型持有状态：根级 `betas`、累计 `alphas`、共享参数路径不动。公共 `forward(batch)` 返回 `loss_dict`，内部 `denoise_step` 沿用镜像的输入和返回语义；`load_pretrain` 走套件，验证另查缺键、多余键、形状、类型、共享关系及两张表。
- [ ] 先用原数值路径以同权重、同卡、同输入和 FP32 对拍；有序边、类型、掩码逐项相同，边长和 Global/Local 分数用 `atol=1e-5, rtol=1e-4`。再分别验证首轴浮点归约和 DDPM 建表候选：空边/显式输出大小以及逐原子 loss、梯度（`atol=1e-4, rtol=1e-3`），两张表逐项相同。建表精度不符时可通过已有 `trained_betas` 传入原表并重验；未通过的候选不替换，不放宽容差。临时对拍和既有本地资产不进 PaddleMaterials 测试目录。
- [ ] 跑 `python -m pytest test/test_geodiff.py -q` 验证新能力和原语义。记录真正换用的公共件与实测条件；本任务不提交 PaddleMaterials。

## 任务：标准训练与原生恢复

**依赖：** [公共双路更新](#任务公共双路更新)、[套件构建与固定去噪](#任务套件构建与固定去噪)。**文件：** `ppmat/models/geodiff/`、`ppmat/metrics/` 中 GeoDiff 验证归约、`geodiff_qm9_default.yaml`、GeoDiff 训练测试。

- [ ] 先写含 1 个和 3 个原子的两批验证夹具：某路批均值为 1 和 2，正确全局均值应为 `1.75`，而非 `1.5`。分别保留 Global 和 Local，验证重复验证不串状态，训练和 eval-only 均走现有指标装配。跑相关用例确认旧路不能满足。
- [ ] 将 Global/Local 的按原子均值绑定到各自的 Plateau，`loss_dict["loss"]` 仍是一次反传的总损失。用作者 `pretrain` QM9 配方：seed `2021`、训练与验证 batch size `64`、`model_global` 和 `model_local` 各自使用 Adam（`lr=1e-3`、`betas=(0.95, 0.999)`、无权重衰减）、联合裁剪 `10000`、每 5,000 次更新验证、Plateau `factor=0.6` 和 `patience=10`。`Trainer.max_iter=10000000`，`max_epochs` 给足按步停止的上限。区分本地镜像的 `3,000,000` 步；小跑用命令行覆盖，不改发布默认。
- [ ] 从本地发布权重以 `structure_generation/train.py` 运行一个小批的更新和验证（`Trainer.max_iter=1`、`Trainer.eval_freq_steps=1`、`Trainer.log_freq=1`、`Trainer.use_tensorboard=false`）。核对日志的总/分项损失、裁剪前范数及两路 LR。用验证后保存的 `latest` 或 `step_N` 复查原生断点的两组 Adam、Plateau 和下一步 LR；对固定下一输入/噪声验证恢复后的更新，不把作者 `.pdparams` 当训练断点。相关测试通过后留工作树。

## 任务：分子图采样命令

**依赖：** [套件构建与固定去噪](#任务套件构建与固定去噪)。**文件：** `structure_generation/sample.py`、`GeoDiff.sample()`、`geodiff_qm9_default.yaml` 的 `Sampler` 段、`test/test_geodiff_sampling.py`。

- [ ] 先写分子图采样测试：一条显式带氢的 QM9 图请求两个构象，核对两个结果的顺序、原子序数、键端点和键类型；无训练数据也能走采样，不调用晶体/CIF 路径。新增分子模式前跑 `python -m pytest test/test_geodiff_sampling.py -q`，确认缺的是公开行为。用已有晶体模式保留一个请求回归。
- [ ] `GeoDiff.sample()` 用有键图及指定构象数，返回 `MolecularSampler` 可聚合的 `pred`，每条图对应同一拓扑的多组坐标。保留高阶边、每步半径构图和反向更新。`structure_generation/sample.py` 新增显式 `by_molecular_graph` 模式，用公共 `MolecularSampler` 装配权重和批次；采样覆盖 `Dataset.train=null`、`Metric=null`，沿用 `Sampler.data`，不改统计或晶体调用的默认值。
- [ ] 用已知键和原子写 `conformers.sdf`；RDKit 元数据缺失时从输入的原子序数和明确键类型还原分子，不根据坐标猜键。用固定初始坐标和噪声与镜像比少量反向步，再跑实际命令生成短步结果。用 RDKit 保留显式氢回读 SDF，验证键、顺序和坐标（SDF 序列化允许 `atol=1e-4`）。上述测试通过后留工作树。

## 任务：定版并交包

**依赖：** [标准训练与原生恢复](#任务标准训练与原生恢复)、[分子图采样命令](#任务分子图采样命令)。**文件与产物：** `structure_generation/configs/geodiff/README.md`、任务域和发布索引的说明；GeoDiff 根目录 `geodiff_qm9_default.zip`、`geodiff_qm9_data.zip`（不入库）。

- [ ] 先锁定训练与采样共用的模型配置、NPZ 字段和所有源切分。模型 README 只写模型、数据获取、命令、模型图和引用；英文简洁，保留必要的版权/许可。模型数值证据记录条件，供用户写 PR；不把训练回放或 COV/MAT 当成果。
- [ ] 在独立环境用现有导出器从可信官方 `.pkl` 导出完整 train/val/test：保留 `train_data_40k.npz`、`val_data_5k.npz`、`test_data_1k.npz`、原有节点/边偏移和分子元数据。数据 ZIP 顶层 `geodiff_qm9_data/`，记录真实 MD5 供用户返回地址后登记；不夹带本机小样本、个人路径或日志。
- [ ] 用同一份已验原生权重和最终 YAML 组装 ZIP `geodiff_qm9_default/`，内为 `geodiff_qm9_default.yaml`、`checkpoints/best.pdparams`。权重不再转置，不打包训练断点。BCE 链接不写进任一包。
- [ ] 从两个实际 ZIP 解包验证命名、状态项与内容、各 split 的节点/边偏移和读取；用解包后的本地路径分别跑一个小批训练/短步采样。只跑能证明最终字节可用的规模。通过相关 pytest 和覆盖改动文件的 pre-commit；若还有待定项会改变权重、配置或数据格式，先解决再交包。把两包放 GeoDiff 根目录由用户交维护者，不替用户上传。

## 任务：回填并验证正式链接

**依赖：** [定版并交包](#任务定版并交包)，并收到用户给出的真实权重和数据链接。**文件：** `ppmat/models/__init__.py` 的 `MODEL_REGISTRY`、套件外部数据下载配置和相应文档/模型索引。

- [ ] 权重链接登记为 `geodiff_qm9_default`；数据按同一受验版本用套件现有下载及 MD5 校验。只改包外地址，不重新打包，不填假 URL。
- [ ] 用真实地址验证从未预置的缓存下载、解包、按名构建及一小批采样，同时验证训练数据路径解析。结果归档到 GeoDiff 实验记录；未收到链接时只交本地已验包，不声称注册或下载通过。
- [ ] PaddleMaterials 其余改动均留工作树。用户决定何时提交、推送并手动提 PR；本计划不授权这些操作。
