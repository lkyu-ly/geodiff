# GeoDiff：QM9 与 Drugs 双预设调研

**建议保留一个 `GeoDiff` 类，分别配置、打包和加载 GEOM-QM9 与 GEOM-Drugs。** 两个预设各用自己的权重和 train/val/test 切分，不互相依赖。以下区分作者 `pretrain` 发布分支、PaddleMaterials `upstream/develop` 的惯例和本地 `feat/geodiff` 的现状。

## 作者发布的是两套权重

[作者发布说明](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/README.md#L24-L56)只列出 `qm9_default` 和 `drugs_default` 两个预训练权重，要求权重和对应 YAML 配对；[作者主分支说明](https://github.com/MinkaiXu/GeoDiff/blob/main/README.md#L52-L56)也明确要求用 `pretrain` 复现发布权重。两套配置虽使用同名 split 文件，路径分别在 `data/GEOM/QM9/` 和 `data/GEOM/Drugs/`。`drugs_1k_default` 是同一 Drugs 切分的 [1,000 步消融训练配置](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/configs/drugs_1k_default.yml)，`beta_end` 为 `9e-3`；作者没有把它列为第三套发布权重。

| 作者 `pretrain` 配置 | 建议 PaddleMaterials YAML 文件名和未来注册键 | `smooth_conv` | train/val batch | 全模型裁剪阈值 |
| --- | --- | --- | --- | --- |
| [`qm9_default.yml`](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/configs/qm9_default.yml) | `geodiff_qm9_default.yaml` / `geodiff_qm9_default` | `false` | 64 | 10,000 |
| [`drugs_default.yml`](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/configs/drugs_default.yml) | `geodiff_drugs_default.yaml` / `geodiff_drugs_default` | `true` | 32 | 30,000 |

两个默认配置均用 5,000 个扩散步、sigmoid β（`1e-7` 至 `2e-3`）、seed 2021、训练上限 10,000,000 步、每 5,000 步验证。保留两组 Adam（`lr=1e-3`、β `0.95/0.999`、无权重衰减）和各看本组验证损失的 Plateau（`factor=0.6`、`patience=10`）；梯度在两组更新前按全模型统一裁剪。训练步数以 `pretrain` 为准，不用本地作者镜像中写成 3,000,000 步的配置。[作者训练循环](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/train.py#L49-L164)让训练和验证共用同一 batch 配置，采样 batch 则不是这项训练参数。

[GEOM 原始库](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF)、[GeoDiff 作者发布的预处理切分和权重](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing)、转换出的 Paddle ZIP 是三类资源。QM9 构象生成不使用作者另发的 [`qm9_property.pkl`](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/README.md#L83-L91)，也不等于原始 GEOM 全库。公开可下载不代表可以重新托管；本轮未核定数据或权重的再分发权限。

## PaddleMaterials 怎么组织同一模型的多个预设

- **InfGCN 是同类实例。** [`infgcn_qm9`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/electronic_structure/configs/infgcn/infgcn_qm9.yaml#L76-L129)、[`infgcn_mp`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/electronic_structure/configs/infgcn/infgcn_mp.yaml#L76-L129)、[`infgcn_md17_ethanol`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/electronic_structure/configs/infgcn/infgcn_md17_ethanol.yaml#L77-L127) 共用 `InfGCN`，但各自指定数据集、YAML 和[权重 ZIP 键](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/__init__.py#L170-L178)。这里的 `QM9DensityDataset` 是该模型的电子密度输入，不是 GeoDiff 的构象数据。
- **DiffCSP 是同一任务目录下的打包实例。** [`diffcsp_mp20` 和 `diffcsp_mp20_ab_initio`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/structure_generation/configs/diffcsp/README.md#L164-L221) 各有 YAML 和[权重 ZIP 键](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/__init__.py#L105-L111)。准确地说，其 ab initio 配置用 [`DiffCSPWithType(DiffCSP)`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/diffcsp/diffcsp.py#L621)，有原子类型生成差异；GeoDiff 两套默认配置不需要仿照它新增子类。
- **类名与发布键分开。** [`build_model`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/__init__.py#L202-L267)根据 YAML 的 `Model.__class_name__` 和 `__init_params__` 构建模型；[`build_model_from_name`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/__init__.py#L271-L294)则用 `MODEL_REGISTRY` 的键查 ZIP URL，再解析配置、构建并加载权重。[模型包约定](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/docs/model_package.md)要求 `<键>.zip` → `<键>/<键>.yaml` 与 `<键>/checkpoints/best.pdparams`。解析器兼容平铺或一层目录以及 `.yml`，正式包按文档用 `.yaml`；数据另外打 `geodiff_qm9_data.zip` / `geodiff_drugs_data.zip`，不放进权重包。

本次在 `feat/geodiff` 并列 [`geodiff_qm9_default.yaml`](../../PaddleMaterials-geodiff/structure_generation/configs/geodiff/geodiff_qm9_default.yaml) 与 [`geodiff_drugs_default.yaml`](../../PaddleMaterials-geodiff/structure_generation/configs/geodiff/geodiff_drugs_default.yaml)。两份配置用同一个 `GeoDiff`、[`GeomConformerDataset`](../../PaddleMaterials-geodiff/ppmat/datasets/geom_conformer_dataset.py) 和 collator，各自固定数据根、输出目录与训练参数。训练入口用 `-c` 和 `Global.data_dir`；分子采样用 `--mode by_molecular_graph`，本地配置与权重或注册名入口保持不变（[采样入口](../../PaddleMaterials-geodiff/structure_generation/sample.py#L29-L119)）。`smooth_conv` [传给 SchNet](../../PaddleMaterials-geodiff/ppmat/models/geodiff/geodiff.py#L59-L92)，在[硬截断与余弦权重之间切换](../../PaddleMaterials-geodiff/ppmat/models/geodiff/_encoders.py#L118-L125)。两份发布权重的键、形状和 dtype 相同，178 个状态值不同，不能凭加载成功判断是否配对。

## 正式地址未定：用本地实包验证，不先注册

1. **验证完整构象切分。** 原 QM9 数据 ZIP 的三个 NPZ 逐字节等于上轮不设 `--limit` 导出的作者三切分；训练、验证、测试分别有 200,000、25,000、125,216 条构象记录（[既有导出](../GeoDiff迁移实验记录.md#L178-L184)、[导出代码](../tools/export_paddle_assets.py#L89-L134)）。这是完整发布的构象三切分，不是小样本，也不包含单独的 `qm9_property.pkl` 或原始 GEOM 全库。
2. **用实际 ZIP 经套件下载与解压。** 旧 QM9 权重包经本地 HTTP 和默认下载后，解出文件但返回的模型根是不存在的 `<缓存>/geodiff_qm9_default/best`；旧数据 ZIP 的首成员同样是文件而非目录。[套件 ZIP helper](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/utils/download.py#L226-L263)按首成员推断返回路径。重新打包时先写显式 `<顶层目录>/`，不改权重/NPZ 内容；返回外层目录后用[现有模型包解析器](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/utils/model_package.py#L18-L64)定位模型 YAML，用适配的数据目录内三份 NPZ 驱动套件入口。不修改下载函数。
3. **正式地址延后。** 回环 HTTP 只在验证进程供真实包使用，模型名到地址的映射只存该进程内，数据走 `get_datasets_path_from_url(url, md5)`。源树不写回环地址或占位链接；用户定版取得地址后再填 [`MODEL_REGISTRY`](https://github.com/PaddlePaddle/PaddleMaterials/blob/develop/ppmat/models/__init__.py#L271-L294) 和相应数据下载配置。

## 权重体积实测

| QM9 文件或张量数据 | 字节 |
| --- | ---: |
| 发布 Torch 训练断点 | 9,441,045 |
| Torch 模型的去重张量存储（100 份） | 3,215,448 |
| Torch 双 Adam 的张量存储 | 6,114,320 |
| Paddle `.pdparams`（198 项） | 6,410,616 |
| 原 QM9 ZIP 包 | 5,978,771 |

实际 `.pt` 还含配置、双调度器、训练步和验证损失。Paddle 转换只取模型字典，序列化时为每个状态键写入其张量；共享层的别名键不会被省掉（[转换器](../tools/convert_released_weights.py#L22-L61)）。QM9 的 198 个状态项逐项比较，Linear 只转置，不改 dtype 或值，其余张量和源一致；原 ZIP 中 `.pdparams` 与本地文件逐字节相同。Drugs 以同一转换器导出 198 项，文件同为 6,410,616 字节。

## 双预设与本地整包验证

| 数据 ZIP | 训练构象记录 | 验证构象记录 | 测试构象记录 | 字节 |
| --- | ---: | ---: | ---: | ---: |
| [GEOM-QM9](../geodiff_qm9_data.zip) | 200,000 | 25,000 | 125,216 | 172,182,908 |
| [GEOM-Drugs](../geodiff_drugs_data.zip) | 199,950 | 25,000 | 71,785 | 336,255,134 |

两个完整数据 ZIP 及对应的[QM9 权重包](../geodiff_qm9_default.zip)、[Drugs 权重包](../geodiff_drugs_default.zip)已经从空缓存经回环 HTTP、套件下载和解包加载，每个预设取一张图跑了两步采样；另用完整 Drugs 包的一张训练图完成一次更新。此前的小样本验证为 train/val/test 各一张 Drugs 图（34 / 51 / 48 原子），完成一次更新、一次验证、两个反向采样步。运行侧为 Paddle 3.4.0、RTX 4060 Ti、FP32、无 Torch；未对 Drugs 做精度对齐。数据包使用作者完整构象三切分，不把小样本包、性质评估另份资源或原始 GEOM 库混入。更多数值见[实验记录](../GeoDiff迁移实验记录.md)。
