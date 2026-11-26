# GeoDiff：从独立 Paddle 镜像交接到 PaddleMaterials

**交接结论：QM9 独立镜像的 eager 前向、反传和短程训练精度已验收。下一步从 PaddleMaterials 接入开始，不再补测试集 metrics。** GeoDiff 是给定分子图生成三维坐标的条件扩散模型；这里验收的端到端前向是一次完整去噪网络调用，不是跑完整条扩散链。COV/MAT、性质评估和历史 Torch 优化器续训都不是下一阶段的门槛。

## 从哪里接手

- **迁移源**：从 GeoDiff 仓库的本地 `main` 接手；它已完整合入验收分支 `paddle/qm9-end-to-end`。`geodiff/` 留作 Torch 对照，接入用 [`geodiff_paddle/`](../../geodiff_paddle/)。此次合入未推送；换机器前先确认远端 `main` 已同步。
- **已完成的路径**：先用旧 PyG 直读官方数据，跑通带 GIN 反传兼容补丁的 Torch 基线；再用 PaConvert 生成并修复模型镜像。第三方依赖按调用函数裁剪、转换为本地 [图算子](../../geodiff_paddle/graph_ops/UPSTREAM.md) 和 [图数据类](../../geodiff_paddle/graph_data/UPSTREAM)，不让 Paddle 运行时加载 Torch。最后一次性转换发布权重，分别检查固定前向、单步梯度、20 步训练和 Paddle 原生断点恢复。
- **可信对照**：以[实验记录的最终结果](../../GeoDiff迁移实验记录.md#qm9-固定前向与-20-步训练实测)为准。RTX 4060 Ti、FP32 下，Global / Local 分数的平均绝对误差为 `3.44e-7 / 1.70e-6`，边索引、类型、掩码和边长完全一致；[同批单步反传](../../GeoDiff迁移实验记录.md#原生-pdparams-权重与-l1-梯度对齐)的 89 个有效参数梯度均在约定容差内。新建双 Adam 的 20 步总 loss 平均相对误差为 `0.15166%`，差值跨零。这是训练轨迹的验收结果，不是测试集生成质量，也不是 CINN 或速度成绩。
- **资产与复现**：发布源 `checkpoints/qm9_default.pt`；本机 Paddle 模型权重 `runs/paddle_qm9/qm9_default.pdparams`（只含模型状态，不是训练断点）。数据、权重和 `runs/` 不入库；新 worktree 可能没有这些文件。复用已有资产，缺失时按 [Paddle README](../../geodiff_paddle/README.md) 在独立环境导出，勿让 ppmat 安装 Torch 来读 `.pt` 或旧 PyG `.pkl`。

## 接下来的三项工作（均未实施）

1. **分支与接入。** 目标是同级的 `PaddleMaterials/`，交接时它检出的是 `feat/chemeleon`。先核对 `upstream/develop`，从目标基线另开 GeoDiff 分支或 worktree；不要在 Chemeleon 分支上叠加开发。先打通模型注册、QM9 数据与拼批、配置、原生权重加载和公开训推接口。按 [Trainer 契约](../../../PaddleMaterials/docs/trainer_docs.html)，生成模型训练返回 `loss_dict["loss"]`，采样走独立 `sample()`；保留两路 loss、双 Adam 和各自的调度指标。注册权重另按[模型包契约](../../../PaddleMaterials/docs/model_package.md)组织 `best.pdparams`。Chemeleon 是接口先例，不是迁移目标；不要套用晶体的周期构图。验收以接入前后的同权重、同输入 eager 去噪输出为对照。
2. **CINN 支持。** 先读 [CINN 接入规范](../../../PaddleMaterials/docs/cinn_ch.md)和公共 [runtime](../../../PaddleMaterials/ppmat/models/common/runtime.py)。优先评估 `denoise_step` 作为可复用的数值边界；不把文件读取、外层采样循环或评估包进边界。按规范验证 eager 回归、缓存、公开工作流，以及同权重同 GPU 输入的 eager/CINN 数值一致性。记录实际捕获方式、动态边数的处理和可验证的执行路径，不能只凭开关成功就标为支持。
3. **速度测试。** 数值通过后再按 [benchmark 入口](../../../PaddleMaterials/benchmark/README.md)和[性能矩阵口径](../../../PaddleMaterials/docs/cinn_performance_ch.md)测 **GeoDiff 自己的** eager/CINN。先跑一个 QM9 权重，同卡、同输入、同精度；明确计时的是一次去噪还是若干采样步。独立进程记录首调用（含编译）和 GPU 同步后的 warm 中位数，按 `eager warm / CINN warm` 报加速比及实测条件。不跑全模型矩阵，不做国产卡适配。

## 阅读导航

- **本项目约束与设计**：[项目记忆](../../AGENTS.md)、[迁移工作量报告](../GeoDiff迁移工作量调研报告.md)、[PaConvert 兼容性调研](../GeoDiff_PaConvert转换与兼容性调研.md)。报告说明为什么要保留半径选边顺序、类型为零的边和共享权重；实测与早期调研有出入时，以最终实验记录和运行代码为准。
- **贡献前必读**：[PaddleMaterials 贡献流程与 CINN 调研](../../../工作总结/PaddleMaterials贡献流程与CINN接入调研报告_2026-09-07.md)（分支、接入清单及 §7.9 更新）、[注释与交付措辞规范](../../../工作总结/ppmat贡献交付注释与措辞规范.md)、[套件架构](../../../PaddleMaterials/docs/ARCHITECTURE_ch.md)。套件当前 `docs/` 和源码优先于历史总结；本交接及本地实验日志不作为上游 README 搬入。
- **按需看先例**：[HIENet 实验记录](../../../工作总结/HIENet迁移实验记录.md)看贡献流程，[Chemeleon 配置说明](../../../PaddleMaterials/structure_generation/configs/chemeleon/README.md)和[去噪边界测试](../../../PaddleMaterials/test/test_chemeleon_cinn_workflows.py)看生成模型接口，[CINN 动转静训练踩坑手册](../../../工作总结/CINN动转静训练踩坑手册.md)用于排障。只借通用做法，不把其他模型的任务或结果写成 GeoDiff 的。

## Suggested skills（建议调用）

下一位 agent 按阶段用 Skill 工具调用：接入前用 `research` 核对目标分支的规范，只有会改变交付范围的分歧再用 `grilling`；按[本地工单约定](../agents/issue-tracker.md)以 `to-spec`、`to-tickets`、`implement-spec` 组织实施，`tdd` 验证公开接口。CINN 断图或数值异常用 `systematic-debugging`，提交前用 `code-review`。每轮继续追加根目录单文件实验记录；核心计算若需补丁，先与用户确认。
