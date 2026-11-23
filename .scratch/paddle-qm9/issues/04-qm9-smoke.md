# 04: 单卡 QM9 训练、验证与原生重载

Status: ready-for-agent
Blocked by: 02, 03

使用已导出官方权重和最小训练/验证数据，在 Paddle 入口完成单卡同权重冒烟，并验证从 Paddle checkpoint 继续。

- [ ] 原生 loader、自定义图拼批和设备选择；训练走两路损失、一次全模型梯度裁剪、两个独立 Adam 和验证后的两路调度器。日志间隔可调，不写 TensorBoard。
- [ ] 正常保存并重载 Paddle 模型、优化器和调度器，下一次更新可执行；不把 Torch checkpoint 传给 Paddle 或把新 Adam 的结果冒称历史续训。
- [ ] 在不装 Torch 生态库的 Paddle 3.4 单卡环境运行约定的冒烟，确认权重实际更新并记录分支 loss、梯度范数、LR 和加载结果。
