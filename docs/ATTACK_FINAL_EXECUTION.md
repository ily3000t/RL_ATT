# 最终攻击矩阵执行与审计

本功能接入 [最终预登记协议](ATTACK_FINAL_TEST_PROTOCOL.md)，不改变算法、五模型、50 个交通、预算、统计或失败规则。原机器协议的 `final_execution_ready=false` 保留为登记时状态，当前运行能力另由工程证书记录。原 OARL、Environment/Data、`rl_att/` 算法和冻结模型不变；旧 split-10/20 审计限制不放宽。

## 执行链

`check_attack_final_execution.py` → 干净 SHA 输入/模型/版本预检 → 全部测试 → 已有 split-20 轨迹复核 → ignored readiness 证书。

`run_attack_final.py` → 当前 SHA 再预检 → 先完成 basic_attack0 → 原 batch/evaluation launcher → 严格 final-50 审计 → 其他组最多两批并发 → 全部 14,500 episode 通过 → 17 行 JSON/Markdown 比较。

审计检查源码 snapshot、模型/版本/配置/角色 seed、完整 episode/step 和 trajectory digest、扰动/norm、简单攻击成本和 BO winner、SUMO safety 公式/有效数量、episode 和 pooled 原 summary 重算、首次动作分歧前轨迹、完整计划/预算/fallback、witness seed/实际动作/retry/shared witness、独立 oracle source/live 核验、含 setup 的 physical shadow 账本、canonical Clean 全轨迹回归。

输出以 episode 安全指标及有效分母呈现，不把 episode 百分位平均称为 pooled 百分位。三次攻击重复先在模型/交通内平均，Bootstrap 保持 50 个交通 block；FGSM 实际 250 个 episode，成本不广播。全部三档、逐模型/交通、leave-one-traffic-out 和真实成本留在 JSON。

## 工程验证与命令

新增测试包括 50-episode mock rollout、错误计数/traffic/norm/成本、Safety witness、oracle setup、缺失/重复矩阵、FGSM 真实计数、anchor/失败调度、完整 dispatcher–audit–summary 契约，不启动 SUMO 交通。历史 fixture 复核九方法的 1,100 个已有 split-20 episode、182,593 步，包含原始记录、已验证输入哈希、oracle snapshot 和成本；不修改旧轨迹。

Windows 状态文件可能短暂被读取占用，执行器只对元数据原子写入最多重试 20 次、间隔 10ms；不重试实验、攻击或不利结果。

先在干净已提交 tree 生成证书（输出目录/文件未存在）：

```powershell
$env:PATH="$PWD\.local\envs\oarl-legacy;$PWD\.local\envs\oarl-legacy\Library\bin;$env:PATH"
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR="$PWD\.local\matplotlib-tests"
$env:PYTHONWARNINGS='ignore'
$finalExecutionSha=(git rev-parse HEAD).Trim()
& .local/envs/oarl-legacy/python.exe scripts/check_attack_final_execution.py `
  --expected-commit $finalExecutionSha `
  --output .local/runs/20261004-final-execution-ready/readiness.json
```

证书冻结执行/审计/统计及引用工具、全部测试的 Git blob 哈希，记录 preflight、测试日志和旧 fixture 证明。功能分支合入 main 后若全部工具和测试字节相同，可以使用证书；正式运行仍在 main 的当前 SHA 重做输入预检。

```powershell
& .local/envs/oarl-legacy/python.exe scripts/run_attack_final.py `
  --expected-commit $finalExecutionSha `
  --readiness .local/runs/20261004-final-execution-ready/readiness.json `
  --output .local/runs/20261004-final-attack-test
```

长任务使用隐藏后台进程并重定向 stdout/stderr，另存启动命令和 PID。整个矩阵的 SHA 和 tracked tree 保持固定。`final-attack.json` 是实时状态，记录各组 batch/audit 路径、哈希、命令和步数；首次 launcher 派发前保守记录暴露时间。只有 `status=passed` 且 `verified_episodes=14500` 才是完整通过。

## 失败与证据边界

任一批或审计失败停止新派发，保留已经运行的批次及所有输出，不发布部分矩阵。不自动判定基础设施失败、不自动重试、不接受任意 resume。协议允许的至多一次基础设施重试须保留具体证据并复用同一 SHA/config/seeds/runtime/model，不能重跑正确完成但结果差的条件。

已有元数据包含 split 30 时，新启动预检拒绝，防止再次当作未见试验。发生问题保留日志，在 split 10/20 或 mock 调试；不能改算法后悄悄接续。新执行代码没有直接改动预登记状态或方法。

通过后 `.local/runs/.../final-comparison.json` 和 `.md` 保留全部结果。论文整理后续独立分支只提交紧凑 summary/图表和解释，原始轨迹、日志、模型及 snapshot 留在 `.local/`。不预设稳定安全优势、全面计算优势或新颖性结论，防御仍在完整最终比较之后。
