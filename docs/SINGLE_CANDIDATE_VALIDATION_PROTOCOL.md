# Single 简化候选：既有 validation 复核协议

分支 `codex/single-candidate-validation` 从 `main` 的 `aac4716fd7d535920f667eabde8af749b2360245` 创建。本轮候选仍是既有 `SingleAttemptAttack` 包装器，只冻结 `max_attempts=1`；不修改训练策略、搜索实现、SUMO、checkpoint 或参考 Zero-One 源码。

上一轮完整 Return 机制研究显示，独立重试没有新增碰撞，中、高档回报收益有限但梯度成本明显上升。因此冻结 Single 为待复核简化候选，保留现有 Progress 和预算版 Zero-One Return。**split 20 已用于方法判断，本轮是验证驱动的候选复核，不是未见测试。** 不因结果临时调参或删去预算档位，final split 30 继续保留。

完整网格为五个冻结 Clean checkpoint × split 20 的二十个交通 episode × attack seed 0/1/2 × 梯度/forward 100/200、200/400、400/800。新增 Clean 和 Single 各 900 episode，共 1800。既有已审计的 Progress Return 和 Zero-One Return 各 900 episode 作为对照，比较表共 3600 条条件记录；旧 Clean 用于回归核对，不另加入统计分母。

既有九组对照的 4500 episode / 707039 步完整摘要已重新核对原输入哈希并重建。原执行 SHA 为 `782ceb9bc8ee12f9b05b18d250f7dc792ff9a10d`，本轮注册/配置校验的新增 Single 分支在协议中记录文件及 Git diff 哈希；其余 49 个既有运行源码文件字节一致，另新增的包装器只调用原行为搜索。每批再次验证旧审计和所有源数据哈希，检查参数、版本、checkpoint、角色 seed 与 SUMO 交通一致。每批新 Clean 与该预算/attack seed 对应旧 Clean 的完整二十 episode 逐字段回归，只排除 wall time 与 Clean 未使用的 attack seed。共同 `(episode, full action history, target, attempt 0)` 的扰动原语 seed、实际动作和 margin 在三种攻击间核对一致。

所有条件保持 Return 目标、greedy 策略、原 `|delta_i| <= 0.2|obs_i|+0.05`、horizon 20、十个搜索候选、PGD 两步、步长 1，以及相同的仿真预算。当前 Progress 的 margin 停止、ZO 首次尝试缓存规则原样保留。无 Gate，不补做 Safety 新方法，不把原 `zero_one` 资源协议混入。

主要比较为每档 Single−ZO、Single−Progress 的配对 Return 差。保留 Progress−ZO 原比较核验复用结果。次指标为 Clean 未碰撞单元中的 collision-conversion ASR、双方独有转换、总及每 episode 梯度/forward/新 transition/shadow 成本、计划完整性和候选重复。Shadow 包含 replay/reset 预热，episode setup 单列；并行 wall time 不用于速度主张。安全指标沿用 SUMO minGap 事件及原纵向 TTC/DRAC 的测量范围，不新增不可测指标。

每档/条件 300 条记录仅来自 **100 个 model/traffic 单元和 20 个交通 seed** 的重复攻击。按五个模型、二十个交通分别汇总，报告胜/平/负和逐交通移除后的均值/净转换范围、关键正反动作分歧。没有预设所有模型均胜、全面成本优势或统计显著性；不把相关记录当作独立 N=300 样本。方法选择在全网格完成后据完整效果–成本取舍讨论，不自动覆盖现有验证结果。

最多两组并发，每组五个独立评估进程、策略线程 1。代码、45 个配置、来源和审计工具全部提交后才运行，期间固定 commit 和 tracked worktree。失败时停止后续派发、记录原因并保留原始结果。每批完成即审计，九组全部通过后汇总。checkpoint、raw、日志仍在 `.local/`；只提交必要小型摘要、图表和文档。测试/审计完成后 `--no-ff` 合入 `main`，保留功能分支，不公开 push，不为本次候选复核单独创建有效性 tag。

```powershell
E:/Programs/anaconda3/python.exe scripts/run_mechanism_development.py --phase candidate-validation --output .local/runs/<new-directory>
.local/envs/oarl-legacy/python.exe scripts/summarize_single_validation.py --pipeline .local/runs/<new-directory>/candidate-validation.json --output .local/runs/<new-directory>/verified-summary.json
```

既有结果：[MECHANISM_DEVELOPMENT_RESULTS.md](MECHANISM_DEVELOPMENT_RESULTS.md)、[SHARED_COMPUTE_VALIDATION_RESULTS.md](SHARED_COMPUTE_VALIDATION_RESULTS.md)。
