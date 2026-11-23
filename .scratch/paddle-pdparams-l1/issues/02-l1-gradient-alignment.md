# 02: L1 单步 loss 与梯度对齐

Status: resolved
Blocked by: 01

Torch 补录首步逐参数梯度，Paddle 用同一批次、时间步、噪声回放，在 CPU 上逐项比较。

- [x] Torch 侧一次性回放脚本：加载官方 `.pt` 和 `runs/torch_qm9_smoke/.../inputs/2865000.npz`，注入录制的 `time_step` 与 `pos_noise`，一次前向和反向，不更新参数；保存逐原子两路 loss、总 loss、裁剪前梯度范数、每个命名参数的梯度和无梯度参数名单。
- [x] 首步 64 条训练构象导出成数据 NPZ，原子、键、类型顺序与 Torch 录制一致；Paddle 回放脚本加载 `.pdparams`，注入同一 `time_step` 与 `pos_noise`，输出同结构结果。
- [x] CPU 比较脚本：边索引、类型、掩码完全相同；loss 与梯度按 `atol=1e-4, rtol=1e-3` 比较，输出逐参数最大/平均绝对误差、相对误差、逐层范数比和无梯度集合；结果摘要写入实验记录。

## Comments

已合入 `paddle/geodiff-qm9`。`tools/replay_torch_step.py` 与 `tools/replay_paddle_step.py` 均入库，`tools/compare_l1_alignment.py` 直接比对两侧 `no_grad`。89 个梯度全部在容差内，范数比 `0.9999975–1.0000053`；64 条构象与官方 split 逐位一致。完整表见 [实验记录](../../../GeoDiff迁移实验记录.md)。
