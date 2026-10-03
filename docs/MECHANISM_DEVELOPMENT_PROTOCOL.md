# 完整 Return 机制对照协议

执行状态（2026-10-03）：九组共 2250 episode 已全部完成，369008 步原始记录审计通过，实验源码固定于 `4fda0927acb9e4c21d6455b3a4d29ade343e6ff7`。结果和方法局限见 [MECHANISM_DEVELOPMENT_RESULTS.md](MECHANISM_DEVELOPMENT_RESULTS.md)；下文保留运行前登记协议，未按结果改动预算或攻击。

开发分支 `codex/mechanism-development-study` 从 `main` 的 `5501e966279eca62344692d436cde17b24b5fe16` 创建。只增加实验配置、启动/审计和分析工具，保持 `rl_att/`、原 OARL、SUMO 环境、checkpoint 及既有搜索算法不变。

完整网格为梯度/forward 上限 100/200、200/400、400/800 × attack seed 0/1/2 × 五个冻结 Clean 模型 × split 10 的十个开发交通 episode × Clean 和四个 Return 搜索条件，共 2250 个 episode。45 个配置、九个批次，在运行前提交。450 个 Clean episode 是 50 个 model/traffic 单元的九次重复；1800 个攻击 episode 包含同一交通上的相关随机重启，不当作独立交通样本。

四个嵌套比较保持原 smoke 协议：

| first | second | 要检验的条件差异 |
| --- | --- | --- |
| 单次尝试行为搜索 | 预算版 Zero-One | 无重试时的外层搜索组织 |
| 固定重试行为搜索 | 单次尝试行为搜索 | 行为搜索中允许最多三次尝试 |
| 进展重试行为搜索 | 固定重试行为搜索 | 后续重试按 margin 进展停止 |
| 进展重试行为搜索 | 预算版 Zero-One | 当前完整方法相对匹配基线 |

主指标为同一预算下配对 Return 的 `first - second`，小于零表示 first 的回报攻击更强。安全指标报告配对 Clean 未碰撞单元中的新增碰撞 ASR、双方独有转换和原碰撞率；不预设独立重试或进展规则一定有益。记录总量和每 episode 实际梯度、forward、新 shadow transition、包含 replay/reset 预热的 shadow 步数，以及 episode 初始 oracle setup。保留候选重复、预算耗尽、失败目标、重试新分支等诊断量。扰动、规划、PGD、无 Gate、SUMO 碰撞定义等沿用 [MECHANISM_CONTROLS_PROTOCOL.md](MECHANISM_CONTROLS_PROTOCOL.md)。

先按 attack seed 分别统计，再按模型及交通单元汇总。每个模型对三个 attack seed 和十个交通平均；每个交通对五个模型和三个 attack seed 平均。报告 Return 胜/平/负、碰撞转换差、成本差，以及去掉每个交通后的描述性差值范围。挑选差异最大的正反案例用于动作分歧分析；不将总体均值、重复候选计数或尝试中的新分支归因当作单模块因果证明。不采用把 150 条相关记录当作独立样本的显著性检验。

每组 250 个 episode 完成即审计全部 raw steps、预算、完整候选、计划执行、有效 seed、权重/环境/来源哈希和逐 episode 聚合。所有组 Clean 与既有完整十 episode Clean 数据逐字段回归（排除 wall time 和不使用的 attack seed）。400/800、attack seed 0 的前两个 episode 另与本轮开发前五条件 smoke 逐字段回归，确认执行源码、首次尝试和已有结果未变。

执行最多两个批次并行，每批五个独立评估进程，策略计算线程数保持 1。实际资源成本不受墙钟时间推断影响；并行运行不用于声称速度提升。失败时记录失败组及异常，停止派发后续组，保留已经完成的原始数据。实验期间不改动 commit 或 tracked worktree。

```powershell
E:/Programs/anaconda3/python.exe scripts/run_mechanism_development.py --output .local/runs/<new-directory>
.local/envs/oarl-legacy/python.exe scripts/summarize_mechanism_development.py --pipeline .local/runs/<new-directory>/mechanism-development.json --output .local/runs/<new-directory>/verified-summary.json
```

首条命令自动调用既有批量评估启动器，按冻结训练环境运行本机 SUMO，逐组自动审计；manifest 记录 commit、分支、配置、角色 seed、命令、版本和 checkpoint 哈希。原始记录仍在 `.local/`，提交必要小型结果及分析文档。完成测试、实验和审计后合入 `main`；开发负结果不是阻止工程集成的理由。完整网格结束前不调参、不运行 split 20/30，也不启动防御开发。
