# 进展重试：独立攻击随机种子复核

2026-09-23。分支 `feat/proposed-attack-seed-replication` 从完整开发证据提交 `8381878` 派生。停止规则保持 `strict_margin_progress`，本轮仅增加预先安排的 attack seeds 1/2，不改变攻击算法。

## 固定条件与检查

使用五个冻结 Clean Victim、development split 10 的同一十个 SUMO 种子及原预算。每个 attack seed 运行 Clean、v0 Return/Safety、新版 Return/Safety，共 250 episode；两批合计 500。配置为 `configs/evaluation/proposed_progress_development_attack{1,2}_seed{0..4}.json`，与已完成 attack seed 0 配置只有 `research_seeds.attack_seed` 不同。

两批顺序启动，每批五个隔离进程，避免同时启动十个 SUMO 实验。所有配置、分析器与本文在实验前提交；待启动的批次使用同一提交，执行期间保持工作区干净。每次运行继续记录源码、配置、命令、checkpoint 哈希、种子和完整运行环境。

```powershell
foreach ($replicateSeed in @(1, 2)) {
    $replicateConfigs = 0..4 | ForEach-Object { "configs/evaluation/proposed_progress_development_attack${replicateSeed}_seed$_.json" }
    & 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs $replicateConfigs --jobs 5
    if ($LASTEXITCODE -ne 0) { throw "Attack replicate $replicateSeed failed" }
}
```

审计命令在 legacy Python 运行，分别指定 `--attack-seed 1` / `--attack-seed 2`：

```text
scripts/analyze_progress_retry.py --episodes 10 --attack-seed <1 or 2> --batch <batch.json> --clean-reference .local/runs/20260922T140115279891Z-attack-benchmark/batch.json --output <verified-summary.json>
```

审计器要求配置和运行时 attack seed 都与请求值相同，并拒绝重复 checkpoint/attack 条件。继续逐步检查回报、轨迹、停止重试决定、扰动、执行计划、多维预算与 oracle 成本。跨攻击种子的 Clean 回归只允许 `effective_seeds.attack_seed` 不同，其他种子、模型和运行环境必须一致；实际 Clean 步骤逐字段相同，仅排除计时。不能要求不同 attack seed 的攻击轨迹与 seed 0 相同，也不能将这种有意变化误报为 v0 回归失败。

## 报告与解释边界

每个 seed、每个 checkpoint、每个目标分别比较新旧版本：实际轨迹相同数、配对新增/丢失碰撞、回报差、梯度/forward/仿真成本、停止次数和重试发现分支。包括退化案例，不按结果筛选种子，也不在本轮调整停止规则。

已有 seed 0 为开发证据，这两批使用相同开发交通，不能充当新场景或独立测试集。报告 seed 0/1/2 的描述性复核；共享交通、同一 checkpoint 和两个目标带来的相关性必须保留，不能把 3×5×10×2 条比较当成独立重复来作显著性结论。

本轮仅检验相对 v0 的成本/效果稳定性，不将 seed 1/2 结果与只跑了 seed 0 的 Zero-One 做不配对优劣结论。验证集、最终测试集保持未使用。完成后再根据完整证据决定是否进入验证集，或仍需解决相对基线的成本问题。
