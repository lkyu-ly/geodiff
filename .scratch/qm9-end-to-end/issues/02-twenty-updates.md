# 02: 同批次同随机输入 20 步训练

Status: resolved
Blocked by: None

用各端原生权重和新双 Adam 做 20 次训练更新，按步骤比较损失。

- [x] 从 L1 的 flat batch NPZ 生成一份共享输入包，首步保持原样，后续 19 步固定 NumPy 乱数；Torch 和 Paddle 分进程复用该输入包。保留实验工具之外的模型和训练入口。
- [x] 复用现有 loss replay、原生双 Adam 与全模型裁剪，每步打印并保存更新前总 loss / 子 loss、裁剪前梯度范数和两路 LR；无验证与历史状态恢复。
- [x] 比较器保存 20 行对照，统计三个 loss 的绝对/相对误差和有符号范围，严格用总 loss 跨零给出结果。小例验证输入包第一步逐位保持，以及跨零、单侧和缺步/非有限异常。保留固定证据，记录复现命令。

## Comments

已在 `paddle/qm9-end-to-end` 完成。首次和评审修复后复跑均完成 20 次更新并通过跨零；总 loss 平均相对误差分别为 `0.00143%` 和 `0.15166%`。复跑两端输入的九个数组逐位相同。两次独立结果与同框架复跑偏差见 [实验记录](../../../GeoDiff迁移实验记录.md#qm9-固定前向与-20-步训练实测)；[命令](../../../geodiff_paddle/README.md#compare-fixed-input-forward-and-20-training-updates)。
