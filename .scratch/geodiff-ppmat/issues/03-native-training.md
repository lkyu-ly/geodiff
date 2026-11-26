# 标准训练与原生恢复

Status: resolved

## What to build

用 PaddleMaterials 标准训练命令从原生 QM9 权重起步完成小批量训练和验证，并从套件自己的断点恢复。

## Acceptance criteria

- [x] 比较不等原子数的验证批次，两路 Plateau 使用分别按原子数汇总的验证损失。
- [x] 真实小批量可更新、验证并恢复两组 Adam 及调度；记录总/分项损失、裁剪前范数和两路 LR，不写 TensorBoard。

Blocked by: 01, 02

参考：[接入规格](../spec.md)、[对应任务](../plan.md#任务标准训练与原生恢复)。

## Comments

`feat/geodiff`：变长验证批的套件测试通过。Paddle 3.4.0（无 Torch）、RTX 4060 Ti、FP32、训练/验证各两图：标准命令从发布权重完成 `step_1`，再启动独立进程加载它并完成 `step_2`。两个断点均含两组 Adam 与两份学习率调度状态，LR 均为 `0.001`。独立的合成验证测试覆盖按各路指标降 LR 和用相同输入恢复后续参数更新；真实训练没有写 TensorBoard 事件。
