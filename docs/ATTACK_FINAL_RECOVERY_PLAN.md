# 最终攻击测试中断后的恢复计划

日期：2026-10-04。该功能只审计已存在的数据并生成恢复清单，**没有启动新 SUMO episode，没有重跑，没有修改原输出，也没有放宽原 final auditor**。算法、checkpoint、split 30、搜索预算、seed 和统计协议继续冻结。

## 1. 固定来源和中断状态

- 原实验执行 SHA：`85f981d5ac603439a6d5e7195d7fa092f8e0d311`。
- 本次恢复规划工具 SHA：`710abbe4d62a000d7e11ccce987e6daae52c213b`；工具 SHA 与原实验 SHA 分开记录，不能拿新工具 SHA 替代原实验来源。
- 原计划仍为 60 配置 / 12 组 / 14500 原始 episode / 50 个交通聚类。
- 原 dispatcher 与本项目子进程已退出。原 `final-attack.json` 的 running / 1250 与 stdout 和 batch/audit 不一致，不能用它判断进程存活或完整完成。
- 现有四个中断条件最后轨迹写入约为北京时间 11:30:32。尚不能证明应用更新就是原因，也不能解释总状态文件为何滞后；原控制文件保持不变。

原状态、日志、manifests、episode / summary 和审计副本在 `.local/runs/20261004-final-interruption-audit/evidence/`。旧轨迹保留原目录，48 份已有搜索轨迹已记录 SHA-256；没有删除尾部、覆盖旧记录或挑选不利结果重试。

## 2. 完整性审计与恢复范围

`scripts/plan_attack_final_recovery.py` 检查已保存证据哈希、原工具/测试 Git blob、固定训练 manifests、五模型文件、60 个登记配置和原协议。21 个完整配置的来源、角色 seed、方法列表与执行 snapshot 完整核验。

三个基础组原完整审计输入仍逐字节一致，执行 snapshot 再核验。已有独立复核共 3250 episode / 561116 步，其三份完整复核 JSON 与原审计 JSON 哈希相同。

另对六个完整搜索配置逐 episode / step 复用原审计 primitives，核验真实 safety、扰动、目标/实际动作、witness seed、retry、共享 witness、每块预算、fallback、live/oracle 及含 setup 的 physical shadow 账本，逐轨迹比较 canonical Clean。通过 **1500 episode / 255584 步**，其中 Clean 回归 57656 步。这是完整配置的恢复完整性证明，尚不是两个缺模型组的完整通过或科学效果结论。

| 恢复类别 | 配置数 | episode 数 | 处理方式 |
| --- | ---: | ---: | --- |
| 完整且可复用 | 21 | 4750 | 保留已有结果，不再仿真 |
| 中断配置 | 4 | 计划每配置 250，共 1000 | 暂不执行；如果批准并满足基础设施失败规则，各在新目录完整重试一次 |
| 从未派发 | 35 | 8750 | 后续按原 SHA / 配置首次运行 |

可复用 4750 episode 覆盖 816700 步；此计数不增加独立交通样本量。三组基础配置全部复用；六个搜索完整配置为 `g100 × attack_seed 0/1 × checkpoint 0/3/4`。

若恢复获准且工程接入通过，新仿真总量为 **9750 episode = 1000 中断配置重试 + 8750 首次运行**。原完整 4750 个不重跑，最终分析仍恰好 14500 个登记 episode。

四个中断配置已有 **820 个 episode 行**：

| attack_seed | checkpoint | Clean | ZO Return | ZO Safety | Single Return | Progress Return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 50 | 50 | 50 | 33 | 0 |
| 0 | 2 | 50 | 50 | 50 | 50 | 27 |
| 1 | 1 | 50 | 50 | 50 | 30 | 0 |
| 1 | 2 | 50 | 50 | 50 | 50 | 30 |

这些配置没有完整 launcher / oracle 结束记录。原 evaluator 没有持久化可用于精确恢复的全部进程状态和 oracle 最终计数；单有 episode 行不能证明整条执行链完整。不能手工拼接、根据 trace 推造 oracle 关闭记录、忽略残留步骤后标成 passed。本工具将它们明确标为 `interrupted_retry_requires_authorization`，saved row 即使有 50 条和 summary 也不会自动升级为可复用。

完整配置重试将重复上述 820 条已保存 episode 行；它们及残留半个 episode 保留为中断尝试证据，不混入最终比较或独立样本分母。失败尝试的已记录成本和未记录尾部的不确定性需单独披露，不能隐藏恢复开销或声称整次执行的总成本精确已知。

## 3. 测试和可复核输出

现有冻结 Python 3.7.16 / Torch 1.3.1+cpu 环境，Matplotlib 配置留在 `.local/`。新增六项测试包含：完整 50-episode mock 原始审计、拒绝未关闭配置/混合 SHA/模型/Gate/方法、拒绝错误 saved prefix、Clean 只忽略 wall time 与攻击 seed 且不修改输入、拒绝缺失/重复/错误计数/非法恢复类别。全部 **157 项单元测试通过**；mock 与本次已有轨迹审计均不启动真实 SUMO episode。

原始证据和工具输出均 ignored：

- `.local/runs/20261004-final-recovery-plan/plan.json`：60 条恢复条件、六配置完整性证明、逐输入 SHA-256、源 SHA、命令、`simulations_run=0` 和 `ready_to_execute=false`。
- plan SHA-256：`5825fc33f49a36ddf7ed27a6d0dd8981a78d05932b60d4c1b820ef66afdf0c3f`。
- `tests.log` SHA-256：`ae148467d2304f716f9ec0193e88fc563964232857e861a073d1108c4382fd65`。
- `audit.log` SHA-256：`67aff5833c372d748be91a601e424975e654b9297a2bfd8cf25c84627d2b8525`。

只读规划命令（完整工具 SHA、干净 tree、新输出文件；合入 main 后使用当时 HEAD 的完整 SHA重新规划，不用原始短 SHA）：

```powershell
$env:PATH="$PWD\.local\envs\oarl-legacy;$PWD\.local\envs\oarl-legacy\Library\bin;$env:PATH"
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR="$PWD\.local\matplotlib-tests"
$env:PYTHONWARNINGS='ignore'
$recoveryToolSha=(git rev-parse HEAD).Trim()
& .local/envs/oarl-legacy/python.exe scripts/plan_attack_final_recovery.py `
  --inventory .local/runs/20261004-final-interruption-audit/inventory.json `
  --receipt .local/runs/20261004-final-interruption-audit/receipt.json `
  --pipeline .local/runs/20261004-final-attack-test/final-attack.json `
  --expected-tool-commit $recoveryToolSha `
  --output .local/runs/20261004-final-recovery-plan/plan-next.json
```

## 4. 执行前仍须完成

本功能不是恢复执行器。原新运行预检仍拒绝已暴露的 split 30，不接受任意 resume；原完整最终审计器仍要求 12 完整组、14500 episode 和固定执行 SHA，没有放宽。

先明确用户“不重新运行”是否允许四个中断配置各一次完整重试。随后需要独立功能接入受控恢复 dispatch 和跨原批/重试批的完整审计：

1. 保存具体进程中断证据，满足原协议有证据的基础设施失败条件；应用更新归因不作既定事实。重试不能由 Return、碰撞或成本结果触发。
2. 确认原实验进程不存在，复核全部输入指纹、环境、模型和唯一 attempt 记录；若已有额外未知尝试，拒绝自动派发。
3. 使用原执行 SHA `85f981d...`、原配置/角色 seed/runtime/model，新目录，不改算法、不复制新的 `rl_att/` 进入原实验 snapshot。恢复控制工具另记录其已提交 SHA，不能伪造原启动链。
4. 每个中断配置最多一次完整重试；35 个未派发配置只首次运行；21 个完整配置由已有哈希锁定后复用。跨批组装必须显式记录真实结果路径、旧/新 attempts 和退出状态，不能改写旧 batch。
5. 先提交并在 mock 或 split 10/20 完成恢复调度、错误输入/重复 attempt/审计失败等测试；正式恢复时 source/config 固定，仍最多两组、每组五个模型工作进程。
6. 所有 12 组完整通过后才导出原 17 行表、三档主对照和原统计；保存失败成本、负例和局限。此前不发布部分最终矩阵，不开展防御。

若用户要求任何已保存 episode 都不重复，则当前四个配置不能按原严格协议无损续接，最终矩阵继续不完整。不能以降低审计要求、换 seed 或省去模型/方法来绕过这一约束。
