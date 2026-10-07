# PGD 一致性长程入口：工程验收

2026-10-07。功能分支 `codex/defense-pgd-long-run` 从 `c827a8a` 创建。正式训练配置与冻结规则见 [预登记协议](DEFENSE_PGD_LONG_RUN_PROTOCOL.md)，可核查的紧凑记录见 [JSON](DEFENSE_PGD_LONG_RUN_INTEGRATION_RESULTS.json)。本页报告长程入口的工程验收，不报告防御有效性。

## 完成范围

已提交五份 Protocol A 训练配置：run seed 0–4，各 400 episodes、最多 200 步；checkpoint 固定 100/200/300/400，正式比较预先选择 episode 400。沿用已有 PGD 一致性损失、独立辅助随机流、原 SUMO 场景和 Clean 匹配初始化，无 Gate，不调参。

生产入口流式保存全部真实转移和 PGD 更新，独立 auditor 重算回报、轨迹、观测链、更新节奏、碰撞集合、扰动盒、KL 与损失，核对实际网络/optimizer 成本及四个完整 checkpoint。五个长程 run 全部通过后才输出 ignored 冻结登记候选；正式登记仍需下一项独立功能核查并提交。评估入口可扩展旧 registry，分别对实际冻结策略重新生成攻击。

原根目录算法、Clean 训练实现、道路配置和旧 `configs/frozen_victims.json` 相对分支起点未改变；OARL Robust 继续作为 SUMO 换道领域强基线。新基础防御没有新增在线网络，训练成本和推理延迟仍应分别实测。

## 测试与真实 SUMO 验证

路径修复前，全套 **235 项测试通过**。修复后相关 **24 项测试通过**：训练协议 10 项、训练支撑 14 项；其中一项新增回归验证首次写入不存在的相对模型目录后，可跨工作目录重载。这里没有把两轮重复测试加成独立测试数量，也没有声称修复后重跑了整个套件。

成功执行源提交：`16aa8a555c1e556474447bac4c321a4f39e19402`。使用冻结 Windows / Python 3.7.16 / Torch 1.3.1+cpu 单线程 / NumPy 1.21.6 / SUMO 1.22.0 环境；精确版本、pip freeze、角色 seed、源码/配置哈希和命令保存在运行 manifest。

短程生产入口为 seed 0、16 episodes、每集最多 16 步，实际 **199 个交互步、1,314 个 reset warmup 步、35 次更新**。checkpoint 13/16 的完整状态、actor 文件和探针重载全部通过独立审计。训练更新 actor forward 为 315 次 / 40,320 行，PGD 输入梯度 175 次，候选观测 26,880 行；该成本包含搜索与一致性 outer forward，不能再重复相加。

与上一阶段 `3ae20db` 的同配置连续路径比较，16 集全部原 episode 字段和轨迹 digest 相同；完整 agent（含 replay、optimizer）及全局/策略 RNG 数值状态相同。二进制文件包含不同 provenance，文件哈希不要求相等，数值状态与权重需相等：

| 核对项 | SHA256 |
| --- | --- |
| 最终完整数值状态 | `0af3872bf84ff5a01f65b4eb22f916a6a7eb2998133c50b4df4d1228b4b215e9` |
| episode 13 完整数值状态 | `43922a79930e0e3daca2860a9d0cc33b2ab59eca5ada408c3ad9469aa97a9874` |
| episode 16 actor 权重 | `c1c87a448ac83c5a47753409bb838208a05efb39ea2329d3f4635bdb7a6dcb67` |

冻结模型接入后，None/PGD 各执行一个已有工程交通 episode、最多 16 步，合计 **2 episodes / 32 实际步**。只读复核实际 actor 动作、盒边界、逐步成本、轨迹/回报、TTC/DRAC 公式与配对前缀；checkpoint 全程未改变。PGD 为 160 次输入梯度、192 次攻击策略 forward、176 次 objective；评估器另有每个条件 32 次策略 forward。

该短程样本中 None 回报 3.2267，PGD 回报 6.8280，两者均未观察到 ego 碰撞。攻击提高回报的结果完整保留，不能据此宣称防御有效或攻击无效。所有短程模型及 registry 均为 `engineering_only=true`、`benchmark_eligible=false`；没有使用旧最终 split 30。

## 失败记录与原始数据

第一次生产短程执行 `5c1e3b6` 完成训练，但首次 actor 记录了相对路径，跨进程审计无法找到 `model/policy13.pkl`。该批次已标记 failed，没有生成冻结登记。修复 `16aa8a5` 将产物引用固定为绝对路径，并补回归测试；用新的目录重新验证，保留失败数据。

- 失败批次：`.local/runs/20261007T105530Z-defense-pgd-long-integration`
- 成功训练/冻结候选：`.local/runs/20261007T110022Z-defense-pgd-long-integration`
- 数值等价核对：成功批次内 `numeric_equivalence.json`
- 评估原始数据：`.local/runs/20261007-defense-pgd-production-eval-smoke`
- 逐步只读评估复核：成功批次内 `integration_check.py` / `integration_audit.json`

成功训练 batch 命令：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/run_experiment_batch.py --configs configs/experiments/pgd_consistency_training_smoke.json --jobs 1 --name defense-pgd-long-integration
```

冻结工程评估命令：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/evaluate_attacks.py --config configs/evaluation/pgd_consistency_integration_smoke.json --victim-registry .local/runs/20261007T110022Z-defense-pgd-long-integration/frozen_defense_victims.json --allow-engineering-victims --output .local/runs/20261007-defense-pgd-production-eval-smoke
```

完成验收后将功能以 `--no-ff` 合并本地 main，再按已提交协议后台启动五 seed 长程。启动状态、执行 SHA、PID 和日志指针保存在 ignored launch JSON；运行完成状态以 batch.json、各 manifest 和独立 audit 为准。本页不提前将尚未完成的长程模型称为已冻结或已收敛。

## 下一阶段边界

长程批次期间固定 Git HEAD 与 tracked tree。完成后先核查五份审计及 episode-400 候选，独立提交新模型 registry；再实现防御新交通 namespace 和正式攻击矩阵。ACoE 的即时误差来源、belief 与 off-policy 目标仍待明确，自己的创新机制尚未启动。短程工程验证不取代长程学习、干净收益取舍、跨攻击防御效果或论文结论。
