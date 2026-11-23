# 02: 导出与读取 Torch-free 官方资产

Status: resolved
Blocked by: None

离线导出发布权重和最小官方数据，在 Paddle 端以本地图数据结构读取并拼批，不再反序列化原 Torch/PyG 对象。

- [x] 对 PyG 1.7.2 数据类实际使用的字段、偏移和辅助方法执行源码裁剪与 PaConvert，保留许可与证据；转换后的本地 `graph_data` 供模型与工具读取 `Data`, `Batch`, `Compose`, `to_networkx`。Paddle 原生 `Dataset`/`DataLoader` 承载数据集和拼批。
- [x] 提供可信 Torch 资产导出命令及本地 reader，遵守 spec 中 NPZ 键，转置且仅转置实际 Linear 权重。训练和验证可导出少量样本；保留键的方向和顺序、显式氢及化学元数据，不用 Torch 读取生成的 NPZ。
- [x] 用不同大小的分子及尾部孤立点证明拼批/拆批和边偏移。Paddle 加载导出的图与原 PyG 字段逐项一致。保留原有 packed 数据接入点和压缩数组读写用途。

## Comments

已合入 `paddle/geodiff-qm9`。导出 198 个发布状态项，按实际 Linear 层处理布局；首个官方验证样本的五组拼批数组逐项匹配。Paddle 独立读取用例和单独 Torch 解释器导出往返均通过。见 [graph_data 来源](../../../geodiff_paddle/graph_data/UPSTREAM)。
