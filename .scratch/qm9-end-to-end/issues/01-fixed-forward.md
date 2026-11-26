# 01: 固定 NumPy 输入前向误差

Status: resolved
Blocked by: None

用两侧原生发布权重重新运行一次固定去噪前向，报告两路分数的平均绝对/相对误差。

- [x] 沿用 `tools/export_qm9_references.py` 的固定输入生成和 `tools/check_paddle_forward.py` 的模型检查；两工具可只执行固定前向，不必附带验证读数或反传，原有显式附加检查仍能调用。
- [x] Paddle 输出恢复前向的精确图检查和既有逐元素容差，并报告两路 max/mean abs 与 mean rel；相对误差口径按 [spec](../spec.md)。用小数组证明带符号或零元素的误差统计，再跑一次同输入实际模型。

## Comments

已在 `paddle/qm9-end-to-end` 验证固定前向。历史输入逐项相同，图结构和边长一致，两路浮点输出通过容差。零值、异号和非有限值检查均有针对性用例。实测表见 [实验记录](../../../GeoDiff迁移实验记录.md#qm9-固定前向与-20-步训练实测)。
