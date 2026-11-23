# 01: 原生 .pdparams 发布权重

Status: resolved
Blocked by: None

用一次性脚本把作者 Torch 权重转换为原生 `.pdparams`，Paddle 所有入口统一用 `paddle.load` 加载，删除 NPZ 权重路径。

- [x] 新增 Torch 侧一次性转换脚本：读取可信 `.pt`，按 Paddle 骨架逐键校验并转置且仅转置 Linear 权重，`paddle.save` 写出仅含模型权重的 `.pdparams`；转换后用 NumPy 逐项核对 198 个条目。
- [x] `models/weights.py` 改为 `paddle.load` 后走现有严格校验；删除 NPZ 权重读取、导出器 `weights` 子命令；`train.py`、`test.py`、`check_paddle_forward.py` 与 README 改用 `.pdparams`。
- [x] 固定输入前向保持上一轮结果：边完全一致，两路分数在 `atol=1e-5, rtol=1e-4` 内；训练冒烟和采样加载用例通过。

## Comments

已合入 `paddle/geodiff-qm9`。`tools/convert_released_weights.py` 产出 198 项 `.pdparams`，固定前向最大绝对误差 Global `1.55e-6` / Local `5.25e-6`，边完全一致；训练冒烟一次更新加一次原生恢复通过。数值见 [实验记录](../../../GeoDiff迁移实验记录.md)。
