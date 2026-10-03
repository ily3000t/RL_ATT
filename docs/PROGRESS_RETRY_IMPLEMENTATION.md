# 进展驱动的重试分配：开发版本

本版本从 `feat/proposed-attack` 的完整开发集证据提交 `2217ca2` 派生，开发分支为 `feat/proposed-progress-retry`。依据见 `OURS_V0_DEVELOPMENT_RESULTS.md`：v0 有大量重试，却极少发现并选用新动作分支。这里检验一个更保守的重试预算规则，不声明算法新颖性或安全效果提升。

## 单一修改

注册名称为 `ours_progress_return`、`ours_progress_safety`，显式参数 `retry_rule="strict_margin_progress"`。

1. 首次 PGD 与 v0 一致。
2. 尚未找到的目标仍允许第一次独立随机起点重试。
3. 对目标 t，设重试前最好最终 margin 为 m，当前重试最终 margin 为 m'。只有 m' > m 才允许后续重试；相等或下降则停止后续重启。
4. 仍受原来的每节点/目标最多三次尝试、规划块多维预算和完整候选数上限限制。

只使用已计费 PGD 的最终 margin，不新增模型 forward、梯度或 oracle 查询来决定是否继续。规则没有新数值阈值，不改变攻击时机、扰动范围、目标函数、PGD 步数、策略、交通或环境语义。停止只是分配预算的启发式，绝不标记目标不可达；也可能错过“先停滞、再成功”的随机重启。

`ProgressRetrySearch` 扩展现有搜索器，`ProgressRetryAttack` 扩展计划执行器。原 `ours_return`/`ours_safety` 默认路径保留；停止的目标不会被删除，已有实际动作 witness 仍可继续探索。日志的 `retry_stop_trace` 记录完整实际历史、目标、已执行尝试数、上次最好 margin、本次 margin 与停止原因；相同规划块内的相同停止决定只计一次。

这是一项由开发诊断驱动的版本修改，不是用未见测试调参，也不是已有方法有效性尚未确认时的模块消融。

## 验证与运行

单元测试覆盖：第一次重试保留、无进展停止、进展继续但不突破原上限、缓存调用不伪造进展、已知最优行为保持、冻结权重及精确 forward 计费。结果审计器逐条将停止决定与 `inner_attempt_trace` 对齐，并核对原始 episode 记录、计划执行、所有资源预算和 oracle 计数。

`configs/evaluation/proposed_progress_smoke_seed0..4.json` 对五个模型各运行两个开发 episode，包含 Clean、v0 Return/Safety、新版 Return/Safety，总计 50 episode，独立 attack seed 为 0。原 Zero-One 对照结果保留在之前冻结的实验中；本批次主要验证版本修改及成本权衡，不替代完整基线比较。

```powershell
& 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs configs/evaluation/proposed_progress_smoke_seed0.json configs/evaluation/proposed_progress_smoke_seed1.json configs/evaluation/proposed_progress_smoke_seed2.json configs/evaluation/proposed_progress_smoke_seed3.json configs/evaluation/proposed_progress_smoke_seed4.json --jobs 5
```

在 legacy Python 下运行 `scripts/analyze_progress_retry.py --batch <batch.json> --v0-reference <原 smoke batch.json> --output <summary.json>`，还会核对五个模型的 Clean/v0 三条件与原冻结轨迹，只排除 wall time。实际结果与边界见 `PROGRESS_RETRY_SMOKE_RESULTS.md`。验证集和最终测试集继续保留。

完整开发协议使用 `proposed_progress_development_seed0..4.json`：每个 checkpoint 取既有 development split 的全部十个交通 episode，attack seed 仍为 0；五条件、预算与 smoke 完全一致。总计 250 episode，其中前两个 episode 与 smoke 重叠，不能当作新的独立样本。配置与分析入口在实验前提交，算法没有再次修改。

```powershell
& 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs configs/evaluation/proposed_progress_development_seed0.json configs/evaluation/proposed_progress_development_seed1.json configs/evaluation/proposed_progress_development_seed2.json configs/evaluation/proposed_progress_development_seed3.json configs/evaluation/proposed_progress_development_seed4.json --jobs 5
```

完整批次审计使用 legacy Python 执行 `scripts/analyze_progress_retry.py --episodes 10 --batch <batch.json> --v0-reference .local/runs/20260921T085944130990Z-attack-benchmark/batch.json --output <summary.json>`。此处参考的是完整 v0 开发批次；除逐 episode 配对、停止决策和预算审计外，还要求保留的 Clean/v0 三条件与历史冻结记录逐步一致。回报和碰撞分别报告；实际轨迹一致、碰撞一致和资源减少是不同结论，不能互相替代。并行 wall time 不作为可靠的速度提升依据。
