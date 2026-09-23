# 预算版 Zero-One：缺失攻击种子的配对对照

2026-09-23。分支 `feat/budgeted-zero-one-seed-replication` 从进展重试复核结果 `6333518` 派生。目的为补齐 attack seeds 1/2 的同协议基线，判断此前相对 v0 的计算节省是否构成相对基线的优势。本轮不修改攻击算法。

配置 `configs/evaluation/zero_one_budgeted_development_attack{1,2}_seed{0..4}.json` 来自原完整开发配置，仅保留 Clean、`zero_one_budgeted_return`、`zero_one_budgeted_safety`，并将 attack seed 设为 1/2。每批五个冻结模型 × 十个既有开发交通 episode × 三条件，共 150 episode；两批顺序运行，共新增 300。其他随机种子、观测包络、规划时域、内层梯度步数、多维预算和候选数上限均不改变。

```powershell
foreach ($replicateSeed in @(1, 2)) {
    $replicateConfigs = 0..4 | ForEach-Object { "configs/evaluation/zero_one_budgeted_development_attack${replicateSeed}_seed$_.json" }
    & 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs $replicateConfigs --jobs 5
    if ($LASTEXITCODE -ne 0) { throw "Zero-One replicate $replicateSeed failed" }
}
```

实验前提交全部配置、分析器和本协议，保持工作区干净及提交不变直至第二批全部启动。继续记录配置、启动命令、源码/模型哈希、有效随机种子和运行环境。ZOOpt 使用现有哈希固定的 0.4.2 wheel zipimport，不安装或改动冻结环境。

已完成的 Ours-v0 / 进展重试版本不重跑。对应参考为：

- attack seed 1：`.local/runs/20260923T022110600839Z-attack-benchmark/verified-progress-replication-summary.json`。
- attack seed 2：`.local/runs/20260923T024352276526Z-attack-benchmark/verified-progress-replication-summary.json`。

legacy Python 审计入口：

```text
scripts/analyze_budgeted_zero_one_replications.py --attack-seed <1 or 2> --batch <new batch.json> --reference-report <matching reference summary.json> --output <verified-baseline-summary.json>
```

除逐步回报、轨迹、计划执行、扰动与全部预算审计外，跨批次比较要求模型、有效随机种子、实际源码快照、Python/pip/SUMO 环境、评价配置及两种目标各自的预算一致；进展版本只额外允许已声明的 `retry_rule` 参数。检查原审计输入哈希、哈希固定的 ZOOpt 依赖，以及 Clean 逐字段回归（仅忽略 wall time）。参考批次与新增批次 attack seed 必须相同。

按 attack seed / checkpoint / objective 报告回报、配对碰撞转换和三类成本，对 Ours-v0 与进展版本分别比较 Zero-One。梯度、forward、物理仿真成本分开报告，显式指出提前终止带来的累计成本变化。不挑选有利 seed，不把共享交通/模型/目标当成独立样本。seed 0 已有历史对照保留，可用于描述性并列，不以新结果修改旧批次。

本轮使用开发数据，验证集与最终测试集继续保留。结果可能支持停止扩大当前候选，也可能提示后续预算敏感性实验；不预先假定 Ours 应获胜，不因结果临时增添模块。
