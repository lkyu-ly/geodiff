# 03: 去掉模型 Torch 依赖并对齐固定前向

Status: resolved
Blocked by: 01, 02

替换镜像的图算子、修正 PaConvert 无法识别或误转的位置，让 Paddle 原生 GeoDiff 加载官方权重并匹配已记录的直接前向。

- [x] 模型不导入 Torch 生态；GINE、SchNet 用本地算子、正确的消息和原生 Linear，不改变短接、全局/局部编码、激活、扩散或损失公式。两处稀疏构造改为保留类型 0 的坐标并集。
- [x] 提供严格发布权重加载，证实所有 state 键、shape、dtype 和共享模块别名。用既有固定输入 NPZ、GPU、同一权重逐项核对图与两路分数，按已定阈值写对照输出摘要。
- [x] 保持 beta/alpha 与 eps 的非训练语义。验证真实模型能够前向与反传，不以删除或短路可选计算代替迁移。

## Comments

已合入 `paddle/geodiff-qm9`。独立 Paddle 3.4 CUDA 恢复全部 198 个状态项，固定图完全一致，两路分数最大绝对误差 `1.3113022e-6`、`4.7683716e-6`，通过约定容差。官方首样本 306 条边及其类型逐项一致；两条训练构象通过 loss 与反传。复现见 [Paddle README](../../../geodiff_paddle/README.md)。
