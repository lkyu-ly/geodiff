# 分子图采样命令

Status: resolved

## What to build

在已有结构生成命令增加显式分子图模式。输入带原子和共价键的 QM9 NPZ 以及选择的权重，输出固定原子顺序和拓扑的 SDF 构象。

## Acceptance criteria

- [x] 复用公共分子采样接口，没配置训练集也能生成；同一图的短步数值与镜像对拍。
- [x] 按请求的数量输出，SDF 回读保留键、显式氢和坐标；现有晶体模式不变。

Blocked by: 02

参考：[接入规格](../spec.md)、[对应任务](../plan.md#任务分子图采样命令)。

## Comments

`feat/geodiff`：命令的数值图、RDKit 元数据两种定向用例通过。正式权重在 RTX 4060 Ti、Paddle 3.4.0、FP32 下对一个作者测试图做两步 LD，写出两个 SDF 构象；回读均为 16 原子、15 条原键和 8 个显式氢。固定 seed 的公共采样与独立镜像坐标最大绝对差为 `1.91e-6`。仅显式分子模式改接 `MolecularSampler`，原晶体分支保留。

**2026-11-25 对齐作者采样逻辑（用户决定）**

作者的 `test.py` 把测试集按分子打包（`PackedConformationDataset`）：图取首行，`num_confs(参考数)` 个构象同批采样，默认 `2x`。套件改为一致：`GeomConformerDataset` 新增 `pack_by_molecule`，一个分子一项并带 `num_reference`；`GeoDiff.sample` 的 `num_confs` 接受整数或 `"Nx"`；两份预设的 `Sampler.data` 单独定义为按分子打包，`num_confs` 默认 `2x`。训练结束后的测试损失仍用逐行的 `Dataset.test`，没有改。六个切分里 `idx` 与作者 SMILES 一一对应，按作者 SMILES 分组与作者按 `idx` 打包的划分相同。
