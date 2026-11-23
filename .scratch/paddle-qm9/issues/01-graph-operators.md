# 01: 转换最小图算子依赖

Status: resolved
Blocked by: None

把模型、数据变换和可选构图用到的第三方张量算子裁剪转换为镜像内本地包，保持 CUDA 参考构图和归约行为。

- [x] 保留上游 release、被调用函数与下层依赖、许可和 PaConvert 证据。只替换 Python 子集无法表达的后端，保留可检查的来源与工作记录。
- [x] 提供 `graph_ops` 中现有调用需要的 `scatter`, `scatter_add`, `scatter_mean`, `scatter_max`, `coalesce`, `to_dense_adj`, `dense_to_sparse`, `radius`, `radius_graph`；只实现实际使用的形态，不改变调用者对输出大小、类型或顺序的依赖。`to_dense_adj` 支持显式节点大小。用原生 Paddle 张量运算，不用 Torch 或全局 API 猴补。
- [x] 小图证明重复加和、带类型 0 的坐标合并、输出顺序、归约反传、孤立和空边。与实际 `1.6.3+pt211cu128` CUDA wheel 比对批隔离、严格半径与按原索引截断；35 点全互邻时，两个端点都恰有 32 个非自环邻居。

## Comments

已合入 `paddle/geodiff-qm9`。六个算子用例通过，四组 CUDA 边数组与 Torch 完全相同。依赖来源、裁剪范围与 PaConvert 统计见 [graph_ops 记录](../../../geodiff_paddle/graph_ops/UPSTREAM.md)。
