# 05: 保留采样评估入口并清理残余导入

Status: resolved
Blocked by: 01, 02, 03

采样、数据变换和评估继续保留，切换到本地依赖与 Paddle 资产；不以可选分支扩大 QM9 首轮验收。

- [x] 采样和评估的图导入使用本地包，COV/MAT 恢复 NumPy 归约而非 Tensor 猴补；删除无调用者的生成 helper。不再要求 Paddle 安装旧 Torch 环境。
- [x] 入口消费导出的官方样本和 Paddle 状态，保留 RDKit 分子、样本重复和指标接口。用极小 NumPy RMSD 用例验证两个归约轴，不预跑完整采样或论文指标。
- [x] 文档给出离线导出、固定前向和训练使用命令，区分 CLI 冒烟与未测的扩散采样/评估路径。

## Comments

已合入 `paddle/geodiff-qm9`。四个入口用例通过，三条可选 CLI 可在无 Torch 环境启动；采样的两种权重来源都通过零步加载。30 个运行模块静态解析和实际入口导入均无 Torch 依赖。复现见 [Paddle README](../../../geodiff_paddle/README.md)。
