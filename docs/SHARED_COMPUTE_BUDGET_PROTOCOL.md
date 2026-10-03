# 共享模型计算预算：预先固定的敏感性协议

完成状态（2026-09-24）：七批共 1,650 个新增 episode 全部通过，原协议未变更。完整网格、成本、退化和关键案例见 [SHARED_COMPUTE_BUDGET_RESULTS.md](SHARED_COMPUTE_BUDGET_RESULTS.md)。以下保留运行前登记内容。

2026-09-23。从 `10cd154` 建立分支 `feat/shared-compute-budget-sweep`。检验问题：降低模型计算上限后，进展重试搜索能否保留相对预算版 Zero-One 的额外碰撞案例，并缩小实际成本差距？本轮不修改攻击算法。

## 固定设计

- 梯度 / policy forward 三档上限：400/800（已有高档）、200/400、100/200。两项一起变化，只能解释为联合模型计算预算敏感性，不能区分各自独立作用。
- 所有档位使用五个冻结 Clean Victim、attack seeds 0/1/2、development split 10 的同一十个交通 episode，各 episode 最多 200 步。
- 两个低档均包含 Clean、预算版 Zero-One Return/Safety、进展版本 Return/Safety，合计 2×3×5×10×5 = 1,500 个新增 episode。不只选择此前出现额外碰撞的种子或 checkpoint。
- 保持观测扰动包络、horizon 20、最多 10 个搜索候选、PGD 两步、步长、最多尝试次数、目标和停止规则不变；每块新 shadow transition 上限仍为 200，物理 shadow step 上限仍为 4,000。尾块继续沿用原实现缩放梯度和新转移上限，forward 上限不另行改动。
- 高档复用已审计记录；seed 0 的 Zero-One 高档补跑 Clean + 两目标共 150 episode，使它也可与当前进展版本及低档逐文件校验完全相同的运行源码快照。历史 seed 0 结果保留，不覆盖。总计新增 1,650 episode。
- 这是预算敏感性实验，不改变攻击时机，不添加 Gate，也不是对新模块作效果归因的消融。

生成入口 `scripts/prepare_budget_sweep.py`；冻结清单 `configs/research/shared_compute_budget_sweep.json`。清单包含七批，每批五个隔离进程，依次运行 seed 0 的高档补录、200 档 seeds 0/1/2、100 档 seeds 0/1/2。全部配置和分析代码在启动前提交；七批来自同一提交，期间保持工作区和 HEAD 不变。

```powershell
$budgetProtocol = Get-Content configs/research/shared_compute_budget_sweep.json -Raw | ConvertFrom-Json
foreach ($budgetGroup in $budgetProtocol.groups) {
    & 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs $budgetGroup.configs --jobs 5
    if ($LASTEXITCODE -ne 0) { throw "Budget group failed: $($budgetGroup.gradient_cap), $($budgetGroup.attack_seed)" }
}
```

## 审计与解释

高档 seed 0 补录使用 `scripts/analyze_budgeted_zero_one_replications.py --attack-seed 0`，参考既有 seed 0 进展审计 `20260922T140115279891Z-attack-benchmark/verified-seed0-replication-audit.json`。seeds 1/2 的高档 Zero-One 参考分别为 `20260923T084816714509Z-attack-benchmark`、`20260923T085851662457Z-attack-benchmark` 下的 `verified-baseline-summary.json`；进展参考为先前相应 `verified-progress-replication-summary.json`。

每个低档批次在 legacy Python 下执行：

```text
scripts/analyze_budget_sweep.py --batch <batch.json> --attack-seed <0,1,2> --gradient-cap <100,200> --zero-one-reference <matched upper audit.json> --progress-reference <matched progress audit.json> --output <verified-budget-summary.json>
```

除原有全部预算、扰动、原始回报、终止、轨迹、计划执行、oracle、停止重试决定检查外，还要求：低档与同种子同方法的高档使用相同源码、模型、运行环境、有效种子及评价配置，唯一允许改变的是两个模型计算上限。每批 Clean 逐字段核对，只排除 wall time。ZOOpt 仍使用已固定哈希的可选 wheel。

额外统计：完整搜索候选不足的块数、不完整候选数、选择完整 Clean fallback 的块数、选择搜索候选的块数、没有完整计划时的 unplanned fallback。选择搜索候选不一定实际施加非零扰动；完整的 Clean fallback 与未验证 live fallback 不能混为一谈。

按种子 / 模型 / 目标 / 预算分别报告碰撞、Clean 无碰撞中的转换、回报、实际梯度/forward/仿真成本、回退与候选覆盖；配对比较同档方法和各方法相对高档的变化。特别追踪 seed 1/checkpoint 2/episode 5、seed 2/checkpoint 2/episode 9 两个已有额外案例，同时保留全部其他 episode 的收益与退化。

预算缩减可能导致非单调结果、更多完整 fallback 或不同终止时刻；不得把不完整轨迹当低回报候选，也不得把提前终止的成本下降全归为计算效率。上限相同不意味着实际消耗相同；共享交通、模型、种子和两个目标不是独立样本。验证集、最终测试集保持未使用，不根据结果临时移动预算档位或停止规则。
