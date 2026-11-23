# GeoDiff 迁移实验记录

## 2026-11-23

### 源码核验与迁移工作量静态调研

- 官方 `main` 源码归档与本地 30 个文件逐字节一致；保留 `geodiff/` 原件，仅移除内层 `.git`。数据和权重由作者单独发布。
- `main` 与 `pretrain` 的 21 个 Python 文件相同，三份 YAML 的训练迭代上限不同。发布权重仍以作者指定的 `pretrain` 为参照，未验证加载。
- 对照既往记录和 ppmat 契约，梳理旧 PyG 数据、动态图构造、权重与两路损失的迁移范围。[详细报告](docs/GeoDiff迁移工作量调研报告.md)。
- 本轮未下载权重或数据集，未运行模型、测试或 CINN，未创建 Paddle 镜像或 Git 仓库。

### Torch 基线：范围与流程确认

- 项目仅做 GeoDiff。核对本地 QM9、Drugs 数据及两份发布权重；对照 MPP、G-FNO、PROSE-FD 的实验记录，把用户指定的通用迁移流程补入[调研报告](docs/GeoDiff迁移工作量调研报告.md)。
- 静态核实两份 checkpoint 含模型、两个 Adam 和两个调度器；QM9 / Drugs 保存步分别为 2,865,000 / 2,415,000，恢复的学习率约 `2.19e-8`。本地为 40k / 5k 发布文件，checkpoint 内嵌路径为 50k / 50k；原版控制台将 loss 四舍五入到两位，TensorBoard 保留逐 step 标量。
- `paddletorch` 现有 Torch 2.11.0+cu128 / PyG 2.8.0.post1，缺 `easydict`、`torch-sparse`、`pyg-lib`。发布数据是旧 PyG `Data` 布局，与权重加载是两个兼容问题。等待确认首次冒烟范围和基线留存口径。
- 本轮讨论到 Torch 冒烟和续训/前向基线，不开始 PaConvert、Paddle 迁移或精度验收；未改作者源码、未运行模型。

### Torch 基线：旧 PyG 加载与逐步日志

- 确认首轮只做 QM9：恢复模型、两个 Adam 和两个调度器，续训 5 次更新，验证用固定的官方样本批；同一权重用固定 NumPy 输入做直接前向。
- 作者指定 PyG `1.7.2`。建议在不替换 `paddletorch` 的 Torch 时先隔离试旧 PyG 直读，现代 PyG 的读入容器适配留作备选，不改写官方 `.pkl`。依据见[调研报告](docs/GeoDiff迁移工作量调研报告.md#数据评估和日志旧环境该留在哪里)；旧版组合尚待确认和实测。
- 用户允许直接改 Torch 训练日志。保留总 loss、两路 loss、梯度范数、两路学习率和原有验证指标；训练打印步长可调，对齐用逐 step，不写 TensorBoard 事件。
- 本轮仅核对源码和发布包，未安装依赖、未改模型代码、未运行训推。

### Torch：QM9 直读、续训与固定输入前向

- RTX 4060 Ti 上保留 `paddletorch` 的 Torch `2.11.0+cu128`；用 `uv pip` 隔离装 PyG `1.7.2`、`torch-cluster 1.6.3` 和 `torch-sparse 0.6.18`（扩展匹配当前 Torch/CUDA），目录 `~/.cache/geodiff/pyg-1.7.2-pt211-cu128`。官方 QM9 验证集直读出 25,000 条构象；首样本 18 个原子、36 条有向键，构图后 306 条边，其中 140 条仅由半径加入。未改写官方 `.pkl`。
- 原版 GIN 的残差 `+=` 覆盖 ReLU 反传所需的输出，首次反传报错。只改为非原地相加；两节点算例验证前向 `[13, 14]`、梯度 `[9, 9]`。同一官方权重的固定输入复查中，图和局部输出相同，全局输出最大绝对差 `7.15e-7`。
- 从 `qm9_default.pt` 恢复模型、两个 Adam 和两个调度器；官方 `train_data_40k.pkl`，batch 64，seed 2021，原版日志编号 2,865,000–2,865,004。最终两个 Adam 计数均为 2,865,005；验证在首、末步各取验证集第一批，两路调度器各推进两次，学习率保持 `2.1936950640377836e-8`。
- [本轮配置](runs/torch_qm9_smoke/resume/qm9_smoke.yml) 从 `geodiff/` 运行。入口参数见 [log.txt](runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/log.txt)：`--trust_checkpoint` 仅用于可信官方权重的 `EasyDict` 反序列化；`--log_interval 1 --val_batches 1 --record_inputs`。另用小图验证 `--log_interval 2` 跳过中间步，终端和文件输出一致，无 TensorBoard 事件。
- 产物：[逐步 loss CSV](runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/loss.csv)、[逐步批次、时间步与实际噪声](runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/inputs/)、[续训检查点](runs/torch_qm9_smoke/qm9_smoke_2026_11_23__20_27_55_resume/checkpoints/)。[前向脚本](runs/torch_qm9_smoke/forward.py)用 NumPy seed 2021 构造两图共 9 个原子；[前向 NPZ](runs/torch_qm9_smoke/forward.npz)含输入、扩展边及全局 `(32, 1)`、局部 `(30, 1)` 输出。未跑 Drugs、扩散采样或 PaConvert。

### Torch：关键数值与 GIN 补丁口径

以下是 **Torch 2.11.0 + PyG 1.7.2、带 GIN 反传兼容补丁**的 QM9 续训记录，不是作者原环境的无补丁基线。官方模型、Adam 和调度器都从 `qm9_default.pt` 恢复。训练 batch 64、seed 2021，两路学习率均为 `2.1936950640377836e-8`；删除 `runs/` 后仍可查到这五步的完整精度指标。

| 脚本 step | 总 loss | Global loss | Local loss | 裁剪前梯度范数 |
|---:|---:|---:|---:|---:|
| 2865000 | 64.587593078613281 | 26.278047561645508 | 38.309547424316406 | 1133.7117919921875 |
| 2865001 | 72.63543701171875 | 34.469028472900391 | 38.166408538818359 | 1616.042724609375 |
| 2865002 | 60.485034942626953 | 28.350147247314453 | 32.1348876953125 | 1011.3009643554688 |
| 2865003 | 76.866744995117188 | 31.766197204589844 | 45.100547790527344 | 877.3553466796875 |
| 2865004 | 79.479804992675781 | 24.333763122558594 | 55.146041870117188 | 2114.942138671875 |

- 验证集第一批 64 条构象：step `2865000` 的总 / Global / Local loss 为 `75.469567307692301 / 35.893882211538461 / 39.575682091346152`；step `2865004` 为 `65.432307692307688 / 30.522043269230767 / 34.910261418269229`。两次使用同一批样本，不固定成同一份扩散噪声。
- 固定前向：原版 GIN 在 `eval` 下用 QM9 发布权重通过；NumPy seed `2021`，两个图、9 个原子，时间步 `[3248, 572]`。全局 32 个分数，范围 `[-0.51270473, 2.03277159]`；局部 30 个分数，范围 `[-4.36293697, 3.30903912]`。这些摘要不替代逐元素对齐所需的输入和输出数组。
- [上游 shortcut](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/models/encoder/gin.py#L99-L113)在非原地 ReLU 后原地相加。[作者环境的 Torch 1.8.1](https://github.com/MinkaiXu/GeoDiff/blob/pretrain/env.yml#L177-L180) 中，[非原地 ReLU 的梯度](https://github.com/pytorch/pytorch/blob/v1.8.1/tools/autograd/derivatives.yaml#L1323-L1328)使用激活前的 `self`；[Torch 2.11.0](https://github.com/pytorch/pytorch/blob/v2.11.0/tools/autograd/derivatives.yaml#L2104-L2105) 使用激活后的 `result`。原版 `+=` 不覆盖前者，却覆盖后者，解释了当前环境的报错；未在 Torch 1.8.1 实测整条训练。
- `hidden = hidden + conv_input` 保持前向加法公式和损失定义，只是不再覆盖反传保存的激活输出。两节点实测前向结果不变，补丁后梯度为手算的 `[9, 9]`；当前环境的无补丁运行没有产生可比的梯度或续训数值。本轮只补记录和核对源码，未改 Torch 代码、未重跑实验。

### PaConvert 镜像与不兼容项核对

- 用 PaConvert `3.3.1` 将 `geodiff/` 转到 `geodiff_paddle/`，参数 `--show_unsupport_api --no_format`，根目录保留 `paconvert.log`。工具计数 560 次调用，558 次成功，2 次不支持（99.64%），都为 `torch.sparse.LongTensor`。静态解析 22 个 Python 文件，1 个因两处转换标记有语法错误；转换器另生成 `paddle_utils.py`，省略 `.gitignore`。
- 镜像仍有 32 条 Torch 生态导入、涉及 14 个文件：PyG 21 条、scatter 6 条、sparse 5 条；torch-cluster 经 PyG。另发现 GINE 的 MLP 参数被换成 `paddle.nn`、3 处 `F` 未绑定，NumPy RMSD 被误转为 Paddle Tensor 专用方法。
- [兼容性调研](docs/GeoDiff_PaConvert转换与兼容性调研.md)记录 Paddle 3.4 API、CUDA 半径截断、ppmat 可复用件、离线导出和分层验收候选。历史五步录了训练噪声，没录两次验证噪声；不能用它直接对齐整条含验证轨迹。
- 本轮未手改两侧模型，未装依赖，未运行 Paddle 训推。修改方案、冒烟范围和对齐顺序待确认。

### Paddle 迁移：Q2 的依赖替换口径

- Q1、Q3、Q4 已确认：原生 Paddle 层、离线数据和权重、单卡 QM9 先前向再冒烟。Q2 待确认；Q5 的训练对齐顺序尚未答复。
- Q2 只替换 Paddle 镜像里实际用到的 PyG 拼批、消息传递、稀疏坐标合并、分组归约和半径选择，不改 Torch 对照或模型公式。半径目标是已跑过的 torch-cluster `1.6.3` CUDA：同分子、严格半径、先取至多 33 个候选再去自环；COO 合并须保留类型 `0`。
- ppmat 已有 Paddle `Data`、`Batch`、`scatter_sum/mean`；[包入口](../PaddleMaterials/ppmat/__init__.py#L14-L25)会连带加载模型和 Trainer。建议独立镜像只用所需 Paddle 原语，接入 ppmat 时按相同小图和拼批结果验证后直接复用共享实现。不套用语义不同的现成半径图。
- 建议按拼批、COO、半径选择、消息归约逐段比对，边索引、类型和局部掩码要求完全一致；浮点按 Q4 已确认容差。本轮仅澄清方案，未改镜像、未跑训推。

### Paddle 迁移：第三方源码裁剪与实施入口

- 依赖迁移按用户指定的顺序：取 Torch 基线所用版本的上游源码，按函数和实际被调用的下层依赖裁剪，保留来源与许可；把留下的 Python 实现用 PaConvert 转一次，再手修不支持或误转的地方。Paddle 镜像切换到转换好的本地包，不保留 Torch 运行时兜底。
- PaConvert 不转换 Torch C++/CUDA 扩展。遇到裁剪后仍调用扩展的函数，按原生 Paddle 重写该功能并以 Torch 对照验证，尤其是 CUDA 半径截断和稀疏合并。Torch 仅用于对照和可信资产的离线导出。
- 已核对本地无 issue tracker、实施工单及 Git 工作树，未创建分支或开始改写；`implement-spec` 需先完成 tracker 设置并提供 spec/工单。Q5 尚未确认。

### 工程技能配置

- 已选本地 Markdown tracker，spec 与带依赖的工单存放在 `.scratch/<feature-slug>/`；沿用五个默认 triage 标签，领域文档采用单上下文。`AGENTS.md` 已链接 [配置说明](docs/agents/issue-tracker.md)。
- 本轮仅配置技能，未建仓、未发布 spec 或工单、未改模型或运行测试。

### Paddle QM9：建仓、依赖迁移与单卡验收

- 新建项目仓库，`init` 已推到 `origin/main`，含 68 个代码和文档文件。将误写的 `run/*` 改为 `/runs/`，权重、数据和运行产物留在本机。Q1–Q4 的[规格及工单](.scratch/paddle-qm9/spec.md)在集成分支 `paddle/geodiff-qm9`。
- 按函数裁剪 PyG、torch-scatter、torch-sparse、torch-cluster，分别对图张量子集、PyG 数据子集运行 PaConvert，识别调用数为 `225/230`、`59/59`；手修或重写后端后换成本地包。[图算子来源](geodiff_paddle/graph_ops/UPSTREAM.md) · [数据类来源](geodiff_paddle/graph_data/UPSTREAM)。
- 修正半径基准：实际 `torch-cluster 1.6.3+pt211cu128` wheel 先排除自身再取 32 个，不是同名标签先取 33 个再去自环。35 原子互邻算例产生 1,120 条有向边，目标 0、34 分别选原索引 `1–32`、`0–31`。四组半径参考边数组逐项相同；QM9 首个验证样本的 306 条边与类型逐项相同。
- RTX 4060 Ti、Paddle `3.4.0` / CUDA `13.0` 隔离环境，无 Torch 或 PyG 库。官方权重恢复 198 个状态项（含 98 个共享别名）。固定 NumPy 前向的边索引、类型、局部掩码和边长完全一致；Global `(32,1)` 的最大 / 平均绝对误差为 `1.3113022e-6 / 3.8184226e-7`，Local `(30,1)` 为 `4.7683716e-6 / 1.4722348e-6`，均通过 `atol=1e-5, rtol=1e-4`。
- 各取官方训练、验证集前 2 条构象，batch 2、seed 2021。从发布权重和新建双 Adam 开始，执行一次训练和验证，保存原生 Paddle checkpoint，恢复后执行第二次训练和验证。两组 Adam 矩和 beta 幂均推进，两路调度器按各自验证量推进；两路 LR 均为 `0.001`。

| 环节 | step | 总 loss | Global loss | Local loss | 裁剪前梯度范数 |
|---|---:|---:|---:|---:|---:|
| 同权重训练 | 1 | 38.153839111328125 | 18.169012069702148 | 19.984828948974609 | 3267.328369140625 |
| 验证 | 1 | 69.901319715711807 | 64.095404730902771 | 5.8059158325195312 | — |
| Paddle 断点恢复后训练 | 2 | 161.32717895507812 | 140.12895202636719 | 21.198226928710938 | 6979.6708984375 |
| 验证 | 2 | 35.048936631944443 | 25.10076904296875 | 9.9481684366861973 | — |

- 采样和评估只验证入口、原生资产加载、图准备和 RMSD 归约方向；没有执行完整扩散采样或性质评估。上述训练是 Paddle 冒烟，不作为历史五步 Torch 续训对齐。复现命令见 [Paddle README](geodiff_paddle/README.md)。
- 集成评审无未解决的规格问题。修正了默认测试发现、分环境资产测试和冒烟步数；重新生成参考的脚本与命令已入库，原始 NPZ 未覆盖。五个本地工单已关闭，六个实施与评审修复 worktree 已清理。

## 2026-11-24

### 原生 .pdparams 权重与 L1 梯度对齐

- 一次性脚本 `tools/convert_released_weights.py` 在 Torch 进程读取官方 `qm9_default.pt`，按 Paddle 骨架转置且仅转置 Linear 权重（含方阵与共享别名），用 `paddle.save` 写出 `runs/paddle_qm9/qm9_default.pdparams`（198 项，仅模型权重）。Paddle 全部入口改为 `paddle.load`，NPZ 权重路径删除。固定输入前向（RTX 4060 Ti、Paddle 3.4.0、无 Torch 环境）：边索引、类型、掩码、边长完全一致；Global `(32,1)` 最大 / 平均绝对误差 `1.5497208e-6 / 3.9115548e-7`，Local `(30,1)` 为 `5.2452087e-6 / 1.6232332e-6`，通过 `atol=1e-5, rtol=1e-4`。
- L1 回放：Torch 侧脚本 `tools/replay_torch_step.py` 加载官方 `.pt`，注入已录制的 step 2865000 批次（64 条构象、1243 原子、2496 有向键）、`time_step` 和 `pos_noise`，一次前向反向、不更新参数。回放 loss `64.587601` 与当时训练日志 `64.587593`、梯度范数 `1133.709` 与日志 `1133.712` 在 float32 逐次运行噪声内一致，确认回放路径正确。保存 89 个去重参数的梯度；`edge_encoder_local.*` 5 个张量两侧均无梯度，`no_grad` 直接比对两侧记录。录制的 64 条构象与官方 `train_data_40k.pkl` 中对应样本逐位一致（原子、键、类型、坐标）。
- Paddle 侧脚本 `tools/replay_paddle_step.py` 加载 `.pdparams`，用同一 `time_step`/`pos_noise` 走同一扰动与损失公式，边由损失前向的 hook 捕获，导出的梯度未被裁剪。比较脚本 `tools/compare_l1_alignment.py`（CPU、NumPy，容差 `atol=1e-4, rtol=1e-3`，边要求完全一致）全部通过。下表为评审修复后的 v2 重跑：

| 数组 | 形状 | 最大绝对误差 | 平均绝对误差 | 最大相对误差 | 范数比 Paddle/Torch |
|---|---|---:|---:|---:|---:|
| `edge_index` / `edge_type` / `local_edge_mask` | `(2, 20964)` / `(20964,)` | 完全一致 | — | — | — |
| `edge_length` | `(20964, 1)` | 0 | 0 | 0 | 1.000000 |
| `loss` | `(1243, 1)` | 1.22e-3 | — | 5.03e-5 | 1.000000 |
| `loss_global` | `(1243, 1)` | 1.22e-3 | — | 6.08e-4 | 1.000000 |
| `loss_local` | `(1243, 1)` | 1.04e-3 | 3.70e-5 | 3.55e-5 | 1.000000 |
| `loss_mean`（标量） | `()` | 7.63e-6 | — | 1.18e-7 | 1.000000 |
| `grad_norm`（标量，裁剪前） | `()` | 2.20e-3 | — | 1.94e-6 | 1.000000 |

89 个梯度：最大绝对误差 `1.36e-3`（`edge_encoder_global.mlp` 的一个 bias 元素），范数比 `0.99999751–1.00000533`。标量：loss 均值 Torch `64.58760071` / Paddle `64.58760834`；Global `26.27804976` / `26.27805436`；Local `38.30954752` / `38.30954635`；裁剪前梯度范数 `1133.708984` / `1133.711182`。半径独有边两侧均为 8942。Paddle 侧 Linear 权重梯度按参数类型转回 `[out, in]` 布局后比较，这是权重布局的必然结果。
- Torch 自身两次回放（历史与 v2）的梯度最大差 `1.74e-3`、loss 最大差 `4.27e-3`，与跨框架差异同量级：跨框架差异已在 Torch 的 float32 逐次运行噪声内。
- 结论：模型前向与反向传播在固定输入下等价；本层不涉及优化器。完整比较输出保存在 `runs/l1_alignment/compare_report_v2.txt`。单步 Adam 更新与五步续训对齐（L2/L3）另行评估。
