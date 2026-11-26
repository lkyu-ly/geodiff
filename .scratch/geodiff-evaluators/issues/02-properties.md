# 量化性质：独立 Torch 基准先于 SDF 适配

Status: wontfix
决定：用户明确不将性质评估纳入本次 GeoDiff 贡献。保留下列已有诊断，不实现、不引入 Psikit/Psi4、不重打数据包。

1. 对照 `geodiff/eval_prop.py`、作者 `env.yml` 与 `runs/property-probe.yw7DeF/`。已知独立 Paddle + 临时读取偶极矩适配能跑，不能替代未修补 Torch 端。先对同一参考与生成坐标调用原 Torch 性质评估（无冷启动缓存复用、无猴补丁）。使用现有可选后端；遇第三方 API/ABI 阻塞、没有可用的未经修改 Torch 基准，需额外安装新的重型历史环境才有机会时，立即停止本项并报证据。不得强行跑通。
2. 只有基准成立，再为属性函数写可检验的红测试，限定新增文件 `ppmat/metrics/geodiff_properties.py` 与对应测试。父任务负责公开薄命令和共用读入。输入 `(key, rdkit_mol, ref_positions, generated_positions)`，只在显式调用时导入可选后端；按作者五项差值和分子维聚合。不要模仿裸 except：计算失败须报分子/构象与原因，无可比构象不得报有效成绩。
3. 用相同 SDF 回读坐标和 NPZ 参考，核对成功样本、五值顺序、换算和绝对误差；记录版本、线程/内存和最大差。文档分别写参考成功、临时诊断与公开验收结果；仅前两项不能算端到端完成。

通过公开命令对齐才 resolved；触及停线则 needs-info。不得修改或新增套件默认依赖、在库中给第三方打补丁或重打四个包。

## 冷缓存实测（2026-11-25）

同一条 5,000 步生成记录的来源已知，但公开输入不能配对：参考 NPZ 的 SMILES 是 `C#CC#C[C@@H](C)CO`，生成 SDF 回读为 `C#CC#C[C@H](C)CO`。16 个原子的顺序及键一致，手性相反。仅为定位后端故障，将这两组坐标分别放到来源分子的拓扑上，交给原版 Torch `PropertyCalculator`。这不是公开 SDF 评估。

Python 3.10.19、Torch 2.11.0+cu128、Psikit 0.2.0、Psi4 1.9.1（libint 2.9.0），SCF/6-31G**，CPU 单线程、2 GB、seed 2021，每侧 1 个构象。未读取任何性质缓存。参考和生成的单点能为 `-344.47750721905396 / -344.4724687815756 Ha`。两次都在 [`eval_prop.py:40`](../../../geodiff/eval_prop.py#L40) 读取偶极矩时失败：Psikit 调用 `psi4.get_variable('SCF DIPOLE X')`，Psi4 抛出 `UpgradeHelper`，作者的 `except` 丢弃整条记录。两侧成功数均为 `0/1`，性质矩阵均为 `(4, 0)`；没有五项 eV 误差，也没有可用于套件对齐的基准。[诊断脚本](../../../runs/evaluation-parity.pISHWq/properties/author_oracle.py)及[两份完整回溯](../../../runs/evaluation-parity.pISHWq/properties/author-oracle.log)保留在本地。

未改套件、第三方或默认依赖。此前独立 Paddle 试算临时替换了偶极矩读取，不算作者基准。恢复这项评估需要可信的独立兼容基准（作者固定 Psi4 1.3.2），以及不靠生成坐标反推来源的配对契约；不能通过挑掉手性不符的生成样本来获得好分数。本次未重建环境、改写标签或放宽配对规则。