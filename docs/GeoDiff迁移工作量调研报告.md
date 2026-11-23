# GeoDiff 迁移工作量调研报告

**本项目只迁移 GeoDiff。** 作者源码齐全。迁移难点是旧 PyG 样本、按带噪坐标重建的非周期分子图，以及它们如何影响两路损失。历史 prompt 只提供工作规范，不扩大模型范围。

## 源码、权重与数据分开核对

2026-11-23，以作者 [GeoDiff 仓库](https://github.com/MinkaiXu/GeoDiff)的 [main 源码归档](https://codeload.github.com/MinkaiXu/GeoDiff/tar.gz/refs/heads/main)为参照，按相对路径枚举普通文件并直接比较字节。本地 `geodiff/` 与归档均为 **30 个文件，缺失 0、多出 0、内容不同 0**；没有子模块项或 LFS 指针。21 个 Python 文件按物理行（含注释和空行，末行无换行也计）共 3,680 行：模型 10 文件/1,469 行，工具和数据 7/1,652，训练、生成和两种评估入口 4/559。其余是三份配置、两张图片、环境文件、README、MIT 许可和忽略规则。核对前本地是干净的浅克隆，只有 `main` 检出；完整性结论指源码快照，不是完整提交历史。随后只移除了内层 Git 元数据。来源：[模型](../geodiff/models/)、[工具](../geodiff/utils/)、[配置](../geodiff/configs/)、[许可](../geodiff/LICENSE)。

作者在 [README 的 Generation / Known issues](../geodiff/README.md) 指定，已发布权重以 `pretrain` 为复现基准，并提示新 PyG 与现有代码及预处理数据不兼容。本轮另与官方 [pretrain 源码归档](https://codeload.github.com/MinkaiXu/GeoDiff/tar.gz/refs/heads/pretrain)逐文件比较：两支路径相同，**21 个 Python 文件逐字节相同**。不同的是 README 和三份 YAML；三份 YAML 都只改了 `max_iters`（`pretrain` 一千万，`main` 三百万）。可据此免去移植两套网络的估算。权重配哪份配置仍以作者说明、随权重配置和实际参数为准；分支比较没有验证 checkpoint 加载。

作者 [Drive 目录](https://drive.google.com/drive/folders/1b0kNBtck9VNrLRZxg6mckyVUpJA5rBHh?usp=sharing)可见三个压缩包：`checkpoints.tar.gz`（14.9 MB）、`QM9.tar.gz`（305 MB）、`Drugs.tar.gz`（500.6 MB），均为目录显示值，未下载、未解包。作者列出的发布模型是 `qm9_default` 和 `drugs_default`，未列第三份 1k 权重。[GEOM 原始数据库](https://doi.org/10.7910/DVN/JNGTDF)与 GeoDiff 作者的预处理分包是两种输入；[GEOM 作者说明](https://github.com/learningmatter-mit/geom#accessing-the-data)还提供按分子组织的 RDKit 记录。直接用原始库重建样本并不自动复现原 split。另有两处易误判为缺文件：README 示例用 `config/`，实际是 `configs/`；构象评估入口实际名为 [eval_covmat.py.py](../geodiff/eval_covmat.py.py)，且生成脚本写 `samples_all.pkl`。上游归档也是如此。

用户解压到 [`data/`](../data/) 和 [`checkpoints/`](../checkpoints/) 的两份 `.pt` 都保存模型、两个 Adam、两个调度器及迭代号：QM9 为 2,865,000，Drugs 为 2,415,000；恢复的学习率约为 `2.19e-8`，配置中新建优化器的学习率为 `1e-3`。包内配置指向 `train_data_50k.pkl` / `val_data_50k.pkl`，实际发布数据是 `train_data_40k.pkl` / `val_data_5k.pkl`。QM9 在当前 Torch 2.11.0 中用原版 GIN 完成固定输入前向，用 GIN 反传兼容补丁完成 5 次续训更新。[实验记录](../GeoDiff迁移实验记录.md)保留逐步数值和版本差异；这些数值不是作者原环境的无补丁训练基线，也不是作者的随机训练轨迹。Drugs 仅核对了文件。原版恢复入口见 [train.py](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/train.py#L29-L86)。

## 真正要迁的计算

[论文](https://arxiv.org/abs/2203.02923)的任务是**给定同一分子的原子和化学键，生成可能的三维构象**。输入决定了拓扑；变化的是笛卡尔坐标。不预测元素或晶格，没有晶体周期边界。训练时参考构象给出干净坐标，生成时用随机坐标起步。[数据转换](../geodiff/utils/datasets.py)保留原子序数、每条键的两个方向、键类型和坐标；作者按 `src × N + dst` 排边。训练取 `atom_type[N]`、`pos[N,3]`、`edge_index[2,E]`、`edge_type[E]` 及节点所属图；同一批内的索引要加各分子的节点偏移。采样还要保留同一分子的多份参考构象及其原子对应关系。RDKit 分子和 SMILES 对评估有用，不是去噪层的数值输入。[训练入口](../geodiff/train.py)；[采样入口](../geodiff/test.py)。

| 作者配置 | 扩散时间步 | β 终值 | 全局卷积截断 | 原始训练 batch / 梯度裁剪上限 |
|---|---:|---:|---|---:|
| [QM9](../geodiff/configs/qm9_default.yml) | 5,000 | 0.002 | 硬截断 | 64 / 10,000 |
| [Drugs](../geodiff/configs/drugs_default.yml) | 5,000 | 0.002 | 余弦平滑 | 32 / 30,000 |
| [Drugs 1k](../geodiff/configs/drugs_1k_default.yml) | 1,000 | 0.009 | 余弦平滑 | 32 / 30,000 |

三者均为 `diffusion + dualenc + mlp`，隐藏宽度 128、全局 6 层、局部 4 层、三阶键图、距离截断 10。两份 Drugs 配置的 train/val/test **路径完全相同**；“1k”减少的是扩散时间步，还改变终端噪声，不是更小的数据集。三份配置末尾的 `40k/5k/1k` 是文件名，不能当作本轮数过的样本量。

**一次去噪。** [构图函数](../geodiff/models/common.py)先按化学键补上两跳、三跳的局部边，再按当前坐标补半径边。两者重叠时保留拓扑类型；只由半径产生的边类型为 0。[全局 SchNet](../geodiff/models/encoder/schnet.py)看合并图，[局部 GIN](../geodiff/models/encoder/gin.py)只看类型大于 0 的边。`edge_length` 经 [MLP 边编码器](../geodiff/models/encoder/edge.py)与键类型嵌入逐元素相乘；两端节点特征的乘积再与边特征拼接。两个输出头给每条边一个标量。[等变变换](../geodiff/models/geometry.py)沿两原子连线方向，将同一贡献一正一负累加回两个端点，得到每个原子的三维更新。边长出现在分母；自环、重合坐标、错误的边方向会直接改变这一步。初调列出的 `cross/acos` 属于角度、二面角辅助函数；默认去噪不调用它们，也不调用对称边噪声生成器。[主干 forward 与损失](../geodiff/models/epsnet/dualenc.py)。

**权重工作量为中等，权重归属比数量重要。** 按上述配置展开构造函数，可数出 48 个不同的 Linear（2 个边编码器共 4，SchNet 30，GIN 8，两头共 6）和 4 个 Embedding；发布 checkpoint 的模型字典各有 198 个条目，包含重复引用的模块路径。Torch Linear 权重是 [out × in](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/nn/modules/linear.py)，Paddle 是 [in × out](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/Linear_en.html)，方阵也需要转置；不能把同样规则套在 Embedding 上。还要区分 GINE 的 `eps` buffer、无梯度的 `betas/alphas`，以及 SchNet 的 `Embedding(max_norm=10)`。[双编码器](../geodiff/models/epsnet/dualenc.py)创建两套边编码器、又通过两个 ModuleList 引用它们；但其局部分支实际再次调用 **`edge_encoder_global`**。按变量名改用 local 会改变前向和梯度归属。

**训练不是坐标 MSE。** [损失实现](../geodiff/models/epsnet/dualenc.py)在每图上抽时间步并拼入互补步，再按节点加入高斯噪声；在**带噪坐标**上构图，用同一组边比较干净距离和带噪距离。默认 diffusion 在编码器内的 `sigma_edge` 为 1，没有启用注释中的时间嵌入；时间表仍控制扰动和目标换算。半径边和局部边分别投回三维，按节点累计平方误差，全局乘 2、局部乘 5，然后对节点取均值。若换为逐图等权，大小不同的分子会改变梯度。[原训练器](../geodiff/train.py)对两组参数分设 Adam，合并梯度裁剪后分别更新；验证时用各自的平均损失驱动两个 ReduceLROnPlateau。一次反向正确不等于长期双调度或原样续训已等价。三份配置都没启用 DSM、蛋白侧链、粗粒度或 Gaussian 边编码；这些不计入发布配置的主干工作量。

**生成与评价是两种额外工作。** [生成脚本](../geodiff/test.py)先按分子打包、离线补高阶键，默认生成参考构象数的两倍；[扩散循环](../geodiff/models/epsnet/dualenc.py)每步仍重建半径边，按噪声阈值打开全局分支，更新后按分子减去坐标均值。入口默认 `ld`，另有 `generalized`、`ddpm_noisy`；缩短 `n_steps` 在该实现中截取末段时间步，不是均匀重采样整条轨迹。源码即使只保存最终结果，也会逐步复制坐标轨迹到 CPU；据此不能把短程检查的成本外推成完整采样的时间或内存。衡量单步数值可用同一噪声；衡量生成质量则要多构象。[COV/MAT](../geodiff/utils/evaluation/covmat.py)对每个分子的参考/生成构象做成对 RMSD，以两个方向的最小 RMSD 算覆盖率和平均匹配距离；[RDKit 对齐](../geodiff/utils/chem.py)先去氢。另一个 [量子性质评估](../geodiff/eval_prop.py)使用单独的 QM9 split 和 Psikit 计算能量、轨道等，和前向或训练损失对齐是不同的交付口径。

## 哪些依赖会把 Torch 带回 Paddle

**默认训练和采样不能只把 `torch` 换成 `paddle`，也不必移植整个 PyG。** 两个图卷积只用一种消息流；真正难替代的是随坐标变化的半径邻居选择，以及官方预处理数据里的旧 PyG 对象。前者会改变网络收到哪些边，后者会让程序在读到第一个样本之前就需要 Torch。以下“默认”指仓库的 [QM9](../geodiff/configs/qm9_default.yml) 和 [Drugs](../geodiff/configs/drugs_default.yml) 双编码器训练、采样入口；工作量为静态相对判断，不是实测工期。

### 安装清单不等于运行清单

[原环境](../geodiff/env.yml#L177-L185)写的是 Python 3.7.10、PyTorch 1.8.1、CUDA 10.2 和 cuDNN 7.6.5；[README](../geodiff/README.md#L13-L22)要求另装 PyG **1.7.2**，构建名为 `py37_torch_1.8.0_cu102`。[该 Linux 构建的发布元数据](https://api.anaconda.org/dist/rusty1s/pytorch-geometric/1.7.2/linux-64/pytorch-geometric-1.7.2-py37_torch_1.8.0_cu102.tar.bz2)实际要求 `pytorch 1.8.*`、`cudatoolkit 10.2.*` 及 cluster、scatter、sparse、spline-conv 四个扩展，**没锁四个扩展的小版本**。例如 [cluster 加载器](https://github.com/rusty1s/pytorch_cluster/blob/1.5.9/torch_cluster/__init__.py)用 `torch.ops.load_library` 加载对应 CPU/CUDA 的共享库，再由 Torch 调度算子。原二进制不能作为 Paddle 算子直接复用。下文引用的扩展版本来自[这一代 Torch/CUDA 的官方兼容包索引](https://data.pyg.org/whl/torch-1.8.1%2Bcu102.html)，用于解释实现，不代表已经确认作者机器的实际安装版本。

按[仓库 Python 源码](https://github.com/MinkaiXu/GeoDiff)的 21 个 `.py` 文件统计，18 个直接导入 `torch` 或其子模块。下表按包、按文件去重；同一文件可计入多行。调用点按代码中实际写出的调用计数，不把注释、类型注解、第三方包内的调用或循环次数算进去。星号导入的间接用法另列。

| 包 | 直接导入的文件 | 执行到哪里 |
|---|---:|---|
| `torch_geometric` | 12：`train`、`eval_prop`、`models/common`、`encoder/{edge,gin,schnet}`、`epsnet/dualenc`、`utils/{common,datasets,misc,transforms,evaluation/covmat}` | 默认用图数据、拼批、两个 `MessagePassing` 子类、稠密/稀疏边转换、半径构图。[采样](../geodiff/test.py#L65-L74)没有直接导入 PyG，却通过 `utils.transforms` 取得 `Compose`。`to_networkx` 只见于[可选分簇预处理](../geodiff/utils/datasets.py#L345-L359)；类型注解不是数值算子。 |
| `torch_cluster` | 0 | 经 [PyG 1.7.2 的包装函数](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/nn/pool/__init__.py)调用。GeoDiff 写了 1 处 `radius_graph`（默认）、1 处 `radius`（侧链分支）；两处都没传邻居上限。[调用点](../geodiff/models/common.py#L183-L221) |
| `torch_scatter` | 5：`models/{common,geometry}`、`encoder/coarse`、`epsnet/dualenc`、`utils/datasets` | 显式 `scatter_add` 5 处、`scatter_mean` 3 处、`scatter_max` 1 处、`scatter(..., reduce='sum')` 4 处。默认关键点是[两端等变回写](../geodiff/models/geometry.py#L9-L14)及[采样去中心](../geodiff/models/epsnet/dualenc.py#L637-L639)；`max` 只在[粗粒化助手](../geodiff/models/common.py#L257-L266)，后四处通用 `scatter` 在[数据准备或侧链分支](../geodiff/utils/datasets.py)。此外 PyG 消息聚合和 `to_dense_adj` 内部也调用 scatter。 |
| `torch_sparse` | 5：`models/common`、`encoder/{edge,gin,schnet}`、`utils/transforms` | 默认显式调用 `coalesce` 3 处：[训练构图 1 处](../geodiff/models/common.py#L155-L170)、[采样变换 2 处](../geodiff/utils/transforms.py#L39-L56)。`edge.py`、`schnet.py` 导入它却不调用；[GIN](../geodiff/models/encoder/gin.py#L3-L47)仅在非默认输入分支检查 `SparseTensor`，`matmul` 未用。半径边并集另用一次 **Torch 自身** 的稀疏张量 `.coalesce()`，不在这 3 处中。 |
| `torch_spline_conv`、`torchaudio` | 各 0 | 前者是上述 PyG 构建依赖；[PyG 的 SplineConv](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/nn/conv/spline_conv.py)会尝试导入扩展，GeoDiff 不实例化该层。后者只列在[环境](../geodiff/env.yml#L200-L210)中。Paddle 模型无需实现样条卷积或音频接口。 |
| `torchvision`、`torch.utils.tensorboard` | 原版分别 1、2 | `torchvision` 在[化学模块顶层](../geodiff/utils/chem.py#L1-L33)导入，即使只取 `BOND_TYPES` 也会遇到；`to_tensor` 只在可选画图分支。原版训练[无条件创建 `SummaryWriter`](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/train.py#L42-L59)，写 6 个训练和 3 个验证标量，另有[未调用的 hparams 助手](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/utils/misc.py#L67-L73)。本地 Torch 基线已移除两处 TensorBoard 记录代码。 |

### 需要保留的图语义与替代工作量

**消息传递：中；分组求和、求均值：小。** 默认全局 [CFConv](../geodiff/models/encoder/schnet.py#L70-L100) 从源节点取 `lin1(x)`，逐边乘滤波权重 `W`；局部 [GINEConv](../geodiff/models/encoder/gin.py#L33-L59) 对 `x_j + edge_attr` 施加激活，聚合后加回 `(1+eps)·x_i`。两者均为 `aggr='add'`。[PyG 1.7.2](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/nn/conv/message_passing.py) 对普通 `[2,E]` 整数索引默认从第 0 行取源节点，按第 1 行目标节点求和。Paddle 中只需沿该方向取行、计算消息，再累加到有 **N 行**的输出；无需复刻 PyG 的参数检查器、稀疏融合或整套 GNN 层。两个端点的等变回写须用相反符号，没收到边的节点仍占一行；采样按分子求均值，不能混批。已有 [ppmat 的 `scatter_sum` / `scatter_mean`](../../PaddleMaterials/ppmat/utils/scatter.py#L244-L276) 或 [Paddle `scatter_nd_add`](https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/scatter_nd_add_en.html) 可承担求和。ppmat 这个接口没有 `max`；默认路径用不到粗粒化的 `scatter_max`，不必为此补齐整个 torch-scatter。

**高阶键与稀疏并集：中。** 默认键阶数为 3。训练在[模型内部](../geodiff/models/common.py#L143-L170)从键连通关系求最短可达阶数；采样在[单分子变换](../geodiff/utils/transforms.py#L39-L58)中先算好，随后[关闭重复的高阶扩展](../geodiff/test.py#L103-L125)，半径边仍按当前位置重建。两个主路径各有 2 处 `to_dense_adj`、2 处 `dense_to_sparse`；可选角/二面角变换另有 2 处前者。只需复现[同坐标属性相加](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/utils/to_dense_adj.py)、[非零坐标与取值](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/utils/sparse.py)，以及 [COO 按行排序、重复键求和](https://github.com/rusty1s/pytorch_sparse/blob/0.6.10/torch_sparse/coalesce.py)。训练代码没给 `to_dense_adj` 传 `batch`，原写法会把整批节点当一张块对角稠密图；按分子处理可以避免为不同分子之间的零块分配空间。

这里有两种“零”不能混用：高阶矩阵中零表示没有边；[半径并集](../geodiff/models/common.py#L185-L228)中零却是**存在的半径边的类型**。[Torch 的 COO 合并](https://github.com/pytorch/pytorch/blob/v1.8.1/docs/source/sparse.rst#L218-L258)按坐标排序并累加重复项，因此只有半径的边保留类型 0，重合的键/高阶边保留原类型。[网络](../geodiff/models/epsnet/dualenc.py#L615-L620)正靠这个区别分出全局与局部边。Paddle 可按整数键 `src*N+dst` 排序、去重并对类型求和；若在并集后对类型矩阵直接取 `nonzero`，会删光纯半径边。

**半径邻居选择：高，是默认图路径最大的差异。** [PyG 1.7.2](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/nn/pool/__init__.py)默认 `loop=False`、`flow='source_to_target'`、`max_num_neighbors=32`，只在同一分子内找邻居。不能把它理解成“最近的 32 个”，也不能一律先去自环再截断。以兼容的 [cluster 1.5.9 包装层](https://github.com/rusty1s/pytorch_cluster/blob/1.5.9/torch_cluster/radius.py)为例，它先为每个目标取 **33 个含自身的候选名额**，再去自环，不二次截断。[CUDA 核](https://github.com/rusty1s/pytorch_cluster/blob/1.5.9/csrc/cuda/radius_cuda.cu)按分子内源节点原索引递增扫描，距离严格 `< r`，满额即停；如果自身不在先入选的 33 个里，就可能留下 33 个非自环邻居。[CPU 核](https://github.com/rusty1s/pytorch_cluster/blob/1.5.9/csrc/cpu/radius_cpu.cpp)取 KD 树查询的前若干项，显式关闭距离排序。最终 COO 排序只会改变边的排列，挽回不了截断时选错的邻居。侧链分支则直接调用 `radius(x=全部原子, y=侧链原子)`，默认取 32 个候选，再映回原编号、补反向边并去自环；默认 GEOM 入口不传侧链掩码。

[ppmat 的非周期 SphereNet 半径图](../../PaddleMaterials/ppmat/datasets/graph_utils/spherenet_graph_utils.py#L19-L63)已有分子隔离、严格半径和去自环，但返回半径内**全部**边，没有候选截断。它可以借用，不能直接宣称等价；带周期镜像的晶体构图也不适用。**推断：**以特定旧 CUDA 行为为对比目标时，难点在保留原子顺序、边界比较、截断与去自环次序，而不在欧氏距离公式；官方文件没提供的扩展小版本不应写成确定事实。

### 数据、评估和日志：旧环境该留在哪里

**拼批与旧数据：中至高，难点在文件边界。** [训练](../geodiff/train.py#L100-L133)只需要节点属性与坐标、带类型的有向键、原子到分子的 `batch`、每图节点数及图数。[PyG 1.7.2 的 DataLoader](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/data/dataloader.py)把 `Data` 列表交给 [`Batch.from_data_list`](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/data/batch.py)；普通 [`Data.__inc__`](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/data/data.py)对索引字段按此前图的节点数递增。Paddle 的 `Dataset`、`DataLoader` 加一个只处理所需字段的拼批函数即可表达，不是把大小不同的分子直接 `stack`。[采样](../geodiff/utils/datasets.py#L956-L993)还按分子归集参考构象，`pos_ref` 按原子顺序拼成 `[构象数×原子数,3]`；[复制输入并拼批](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/utils/misc.py#L84-L86)时须保持构象顺序和边偏移。只有可选的 `MolClusterData` 为 `subgraph_index` [另定义分簇偏移](../geodiff/utils/datasets.py#L345-L351)。

[官方已说明](../geodiff/README.md#L153-L156)代码和预处理数据均不兼容更新版 PyG。[数据读取器](../geodiff/utils/datasets.py#L777-L792)对 `.pkl` 直接 `pickle.load` 后调用 `.clone()`；[构造端](../geodiff/utils/datasets.py#L294-L342)放入 PyG `Data`、Torch 张量、RDKit `Mol` 及 SMILES，[采样输出和续采](../geodiff/test.py#L86-L104)也沿用此类记录。Python pickle [按模块和类名恢复实例](https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled)，而 [Torch 1.8.1 张量](https://github.com/pytorch/pytorch/blob/v1.8.1/torch/tensor.py#L90-L165)另有自己的重建函数。**面向 Paddle 的处置判断：**不能靠换类名或扩展名让 Paddle 直接读旧文件。若复用官方文件，可在隔离的 Torch 环境中把可信数据导出为数组、偏移和分子元数据；Paddle 运行时只读这些字段。保留键类型编号、显式氢、原子和构象的排列，以及评估时可按原子编号还原的分子拓扑，不能只留一串 SMILES 和坐标。

**Torch 基线直读官方数据。** 本地发布数据的 pickle 头部可见旧 `Data` 的平铺字段，和 [PyG 1.7.2 的类定义](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/data/data.py#L73-L135)相符，无法由此确定文件的确切生成版本。[PyG 2.8 的 `Data`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/data/data.py#L514-L555)则要求 `_store`，并在缺少它时报告旧对象。隔离安装 PyG 1.7.2 与[当前 Torch 匹配的扩展 wheel](https://data.pyg.org/whl/torch-2.11.0%2Bcu128.html)后，官方 QM9 验证集可直读，首条样本在 GPU 拼批和构图通过。旧版半径图走 [`torch-cluster`](https://github.com/pyg-team/pytorch_geometric/blob/1.7.2/torch_geometric/nn/pool/__init__.py#L205-L210)，不换成新版的 [`pyg-lib`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/__init__.py#L308-L322)，也不改写官方 `.pkl`。这不是作者的完整历史环境。

**化学工具保留，绘图只换接口。** [RDKit 2020.09.1 的包依赖](https://api.anaconda.org/dist/conda-forge/rdkit/2020.09.1/linux-64/rdkit-2020.09.1-py37h0c252aa_0.tar.bz2)不含 Torch。仓库 4 个文件直接导入 RDKit；[`BOND_TYPES`](../geodiff/utils/chem.py#L16-L18)由 `BondType` 枚举顺序生成，长度还决定高阶边编号，不能脱离旧数据任意重排。[COV/MAT](../geodiff/utils/evaluation/covmat.py#L15-L44)用 RDKit 分子回填坐标、[去氢并做最佳 RMS 对齐](../geodiff/utils/chem.py#L133-L137)，可选 MMFF 默认关闭；保留原化学计算比重写合理。NumPy、Pandas（汇总）、NetworkX（可选分簇）、[Psikit/Psi4（独立性质评估）](../geodiff/eval_prop.py#L15-L45)和 [py3Dmol（可视化）](../geodiff/utils/visualize.py)不需要做框架移植；默认导入 `utils.datasets` 时仍会遇到它顶层导入的 NetworkX。可选画图用 PIL/NumPy 转 Paddle 图像即可去掉 torchvision。

## 本项目沿用的迁移流程

1. **先跑通 Torch 并留下基线。** 优先保留 `paddletorch` 的 Torch 等大依赖；缺依赖时用 `uv pip`。训练以作者代码、发布权重和官方数据为起点，使用可调步长的原生日志保留总 loss、两路 loss、梯度范数和两路学习率。对齐时逐 step 打印，不写 TensorBoard 事件。前向用固定 NumPy 输入和同一发布权重，保存输入和输出两端。这里的前向是单次网络调用，不是多步构象采样。
2. **再做镜像转换。** 用 PaConvert 生成与 `geodiff/` 并列的 `geodiff_paddle/`，转换日志留项目根目录。手工处理失败 API 时用 Paddle 原生操作，保持逻辑和计算等价。
3. **处理没有随 API 转换消失的 Torch 依赖。** 优先核对 ppmat 是否已有语义相同的公共组件；依赖面很小则按函数裁剪源码后转换；代价过大的依赖单独评估。不为用到一个算子移植整包。
4. **Paddle 侧改完再跑通和验收。** 以共同的输入、初始权重与受控训练基线检查精度。GeoDiff 的实际验收配置和 Paddle 改法在 Torch 跑通后再定；不做国产显卡适配。

这一顺序与[实习总结](../../工作总结/实习总结.md)及实际记录相符：[MPP](../../工作总结/MPP迁移实验记录.md)先在小 SWE 上跑 Torch，再转换并处理 `dadaptation`，之后记录前向和训练对齐；[G-FNO](../../工作总结/G-FNO迁移实验记录.md)先跑 Torch、再改 Paddle 并处理绑定 Torch 的变体，共用权重、输入比对前向，[损失调试](../../工作总结/G-FNO损失对齐调试.md)显示还需统一批次顺序；[PROSE-FD](../../工作总结/prose-fd迁移实验记录.md)有转换失败后重写、复用小型已转换依赖的实例。没有作者权重的历史案例采用统一初始化；GeoDiff 已有发布权重，不套用随机权重的做法。

## 面向 PaddleMaterials，哪些经验适用

[贡献与 CINN 报告](../../工作总结/PaddleMaterials贡献流程与CINN接入调研报告_2026-09-07.md)是项目约束的入口；当前 [Trainer 成文契约](../../PaddleMaterials/docs/trainer_docs.html)把生成模型的训练和采样分开。GeoDiff 的两路损失可作为可观察的分项，总标量进入 `loss_dict["loss"]`；生成走模型的 `sample()`。可以复用训练编排及图归约原语，不等于可以照搬晶体样本或直接复用一份训练配置。尤其是原版两组学习率调度各看不同指标，若改成统一调度，应区分“固定学习率的短程对齐”和“复现完整训练策略”。有注册权重时，[模型包契约](../../PaddleMaterials/docs/model_package.md)规定注册名、zip 名、顶层目录名、YAML 文件名一致。

**任务入口仍有一个真实差别。** [StructureSampler](../../PaddleMaterials/ppmat/sampler/structure_sampler.py)的按化学式入口只给元素和数量，批处理会构建晶体并写 CIF。GeoDiff 还需要同分异构体各自的化学键，输出同一分子的多组笛卡尔坐标；仅给化学式辨不出键拓扑。另一方面，[MolecularSampler](../../PaddleMaterials/ppmat/sampler/molecular_sampler.py)的单批入口确实可转发模型专属输入，但 dataloader 路径预期含 `pred` 的映射，不能直接接原版 `(pos, pos_traj)`。这里记的是接口差额，没有预选任务域，也未展开 PR 或 CINN 实施调查。

GeoDiff 中，同 seed 不等于同实验：对齐时要确认源样本、批顺序、时间步和实际噪声张量能对应。逐步检查边集合、边类型、两路边分数、节点向量和两项 loss，可以区分构图差异与网络差异。原训练代码按整批总节点建稠密邻接，存储随总节点数的平方增长；采样又会逐步保留轨迹。第一轮验证不必先跑全量训练、5,000 步采样或论文指标。依据：[训练](../geodiff/train.py)、[图构造](../geodiff/models/common.py)、[损失和采样](../geodiff/models/epsnet/dualenc.py)。

[转换对齐 prompt](../../test/转换对齐prompt.md)、[CINN prompt](../../test/CINN-prompt.md)和[交付措辞规范](../../工作总结/ppmat贡献交付注释与措辞规范.md)中沿用的是最小努力和证据口径，不是其他项目的步数、验收阈值或硬件任务。对外注释用简短英文陈述行为，保留必要版权和来源；配置忠于选定权重，不写个人路径或迁移备忘。模型文档交代任务、输入与使用条件，实测数据附口径，论文成绩标原出处；本地调试写实验记录。

## 工作量如何随验收目标变化

对 **GeoDiff**，从张量前向扩大到作者的完整结果，具体多出的是这些行为，而非简单增加行数。下列为上文源码给出的相对工作量判断，不是排期：

| 希望回答的问题 | 新增的实质工作 | 这一步能说明什么 |
|---|---|---|
| 同一图、同一权重，网络是否算出相同结果？ | 中：对齐权重布局、两个消息公式、两路逐边标量及等变回写 | 固定边的单步前向；还没证明 Paddle 会选出相同的边 |
| Paddle 从官方样本开始能否保持同样的图与训练更新？ | 高：旧样本脱离 PyG、拼批偏移、键与半径边合并及截断；固定 t/噪声，分别比较两项 loss 与一次更新 | 小样本端到端等价；不等于长程训练、双调度或发布指标 |
| 是否需要生成可用构象，或复现论文的生成质量？ | 前者增加反向扩散、分子去中心和输出；后者还增加参考构象分组、成对 RMSD、覆盖率统计。量化性质评估要另算 | 固定噪声的短程一致性与整个分布的质量是不同结论 |
| 是否对接 ppmat 公共入口？ | 在已对齐数值主体之外适配 loss、批数据及分子采样输入输出。是否发布权重决定是否需要模型包 | 代码可被套件使用；CINN 或硬件性能仍须独立实测 |

本报告记录静态证据。权重与数据的加载、训练和前向结果以[实验记录](../GeoDiff迁移实验记录.md)为准。
