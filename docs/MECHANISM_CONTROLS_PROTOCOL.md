# Return 搜索机制对照协议

本阶段只增加实验对照，不修改候选方法、原 OARL、SUMO 环境或冻结 checkpoint。开发分支为 `codex/attack-mechanism-controls`，从合入既有七个功能分支的 `main` 创建。

| 条件 | 外层搜索 | 每个历史/目标的独立 PGD 尝试 |
| --- | --- | --- |
| Clean | 无 | 无 |
| `zero_one_budgeted_return` | 目标序列 ZOOpt 搜索 | 首次尝试；缓存复用，不独立重试 |
| `ours_single_return` | 现有实际行为历史搜索 | 最多 1 次 |
| `ours_return` | 同一行为历史搜索 | 最多 3 次 |
| `ours_progress_return` | 同一行为历史搜索 | 最多 3 次；首次重试不进展则停止后续重试 |

预算版 Zero-One 的配置仍保留原 `max_attempts=3` 字段；其目标序列路径始终查询 attempt 0，所以实际没有独立重试。本阶段不修改该语义来构造对照。

预先确定的比较依次回答：无重试时外层搜索组织的差异、行为搜索内允许固定重试的影响、行为搜索内进展停止的影响，以及当前完整方法相对预算版 Zero-One 的效果/成本。它们是嵌套的条件比较；没有目标序列搜索加重试的两个变体，不能声称完整因子实验，也不能将外层搜索组织差异全部归因于历史缓存。

各条件共享 Return 目标、16D 输入、greedy 策略、扰动包络 `|delta_i| <= 0.2|obs_i| + 0.05`、20 步规划、10 次候选搜索、2 步 PGD、step size 1、无 Gate。三档梯度/forward 上限为 100/200、200/400、400/800；new shadow transition 上限 200，物理 shadow step 上限 4000。记录实际消耗及 reset 预热、replay、失败目标、实际序列重复、首次尝试与重试贡献。所有变体沿用由 attack seed、episode、完整实际历史、目标及尝试序号导出的 witness seed。

先冻结工程 smoke：五个原 Clean checkpoint、split 10 的前两个开发交通 seed（1472154258、1940738230）、attack seed 0、400/800 上限、每 episode 最多 200 步，共 50 个完整 episode。该样本已用于开发，只验证接口、预算、执行与旧版本回归；不作为泛化或消融有效性的证据。逐字段回归只排除 wall time。

完整机制研究预先计划全部三档预算、三个 attack seed、五个 checkpoint、十个已有开发交通 episode、五条件，共 2250 个 episode。后续单独分支冻结并运行该完整网格，报告每模型和每交通单元的配对差异；三个 attack seed 是同一交通的重复，不当作独立交通样本。当前不运行验证/最终测试 split，也不调整候选方法去适配 smoke 结果。

工程启动命令（在干净、已提交分支运行）：

```powershell
E:/Programs/anaconda3/python.exe scripts/evaluate_attack_batch.py --configs configs/evaluation/mechanism_smoke_seed0.json configs/evaluation/mechanism_smoke_seed1.json configs/evaluation/mechanism_smoke_seed2.json configs/evaluation/mechanism_smoke_seed3.json configs/evaluation/mechanism_smoke_seed4.json --jobs 5
```

随后使用冻结 Python 3.7.16 执行 `scripts/analyze_mechanism_controls.py --batch <batch.json> --output <audit.json>`。启动器保留 commit SHA、分支、配置、角色 seed、命令、运行时版本与 checkpoint 哈希。完整原始记录仍在 `.local/`；只提交协议、工具、测试和必要小型摘要。工程验收通过后合入 `main`，不以论文结果正负作为功能分支合并条件。
