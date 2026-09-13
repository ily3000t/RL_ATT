# OARL Protocol A 长程复现结果

本轮完成 5 个 run_seed（0–4），每组 400 episodes、每集最多 200 steps；采用现有 SUMO 1.22.0。全部 20 个 actor checkpoint 已通过保存后校验、独立进程重新加载和冻结 SUMO 推理验证。

这是采用明确种子协议及已记录兼容修复的训练流程复现，不等同于原论文数值或攻击鲁棒性结论的完整复现。未实现 Gate，未接入 Zero-One 或 Ours。

## 结果

± 表示 5 个 run 层面数值的样本标准差。训练指标取最后 100 个 episode；评估使用 episode 400 的 checkpoint，每个 run 固定 20 个留出 episode、最多 200 steps、greedy argmax、无外部观测攻击。

| run_seed | 训练末 100 平均 return | 冻结评估 return | 碰撞 episode 比例 |
| --- | ---: | ---: | ---: |
| 0 | 133.099 | 141.339 | 0% |
| 1 | 135.412 | 131.235 | 10% |
| 2 | 135.666 | 129.037 | 20% |
| 3 | 138.816 | 145.609 | 5% |
| 4 | 131.048 | 126.083 | 10% |

- 训练末 100 return：134.808 ± 2.925。
- 最终 checkpoint 评估 return：134.661 ± 8.388。
- 评估碰撞 episode 比例：9/100；run 间标准差约 7.42 个百分点。
- 共 344,495 个交互 step，170,625 次训练更新，853,125 次 BO objective 评估。
- 63 次重复 BO 建议保留其 objective/梯度评估并复用 GP 观测；235,260 次极小概率 JS 评估使用 float64。

碰撞指标来自 SUMO 每个交互 step 报告的 Auto 碰撞 ID；不统计 reset 内的事件，也不使用上游 cn_epi 距离代理冒充碰撞。当前未生成 TTC/DRAC 或外部攻击成功率。不同 run 的训练轨迹和实际更新次数受策略、提前终止影响，配置上限和更新规则相同。

## 来源与验证

- 训练 commit：`97fdb5767cae6bfbfccda5d2e2dac3913ad5f467` 的完整 SHA 见 JSON；全部 run 使用同一训练 commit。
- 独立评估 commit：`555ef49`；训练与评估的 oarl.py、环境源码哈希一致。
- 环境：Windows 11、Python 3.7.16、PyTorch 1.3.1 CPU、NumPy 1.21.6、SciPy 1.7.3、scikit-learn 0.24.2、bayesian-optimization 1.2.0、SUMO 1.22.0。每个训练进程一个数值库线程。
- 15 项单元测试通过，涵盖种子隔离、终止兼容、上游完整更新等价、重复 BO 建议、极小/零概率稳定性、checkpoint 哈希与输出校验。
- 每个保存点记录 64 个真实观测探针，重载后权重和概率输出一致；独立评估前后权重与 checkpoint 文件哈希不变。
- 所有训练/评估种子、命令、配置、版本及源码哈希均在每次运行的 manifest/result 中记录。2,000 个训练 seed 与 100 个留出 seed 无重合。
- checkpoint 是 actor 推理文件，不含 replay、critic、optimizer、dual 和 RNG 全状态，不能作为精确训练续跑快照。

配置与协议见 [PROTOCOL_A.md](PROTOCOL_A.md)，机器可读汇总及 checkpoint 路径/哈希见 [OARL_PROTOCOL_A_RESULTS.json](OARL_PROTOCOL_A_RESULTS.json)。曲线可使用 `scripts/plot_reproduction.py` 重新导出；本地导出为 `.local/reports/oarl_protocol_a.png`。

## 保留的失败与兼容边界

此前两个批次完整保留，并排除在以上统计之外，详见 [REPRODUCTION_ATTEMPTS.json](REPRODUCTION_ATTEMPTS.json)：

1. 上游 BO 重复点注册导致 seed 0 在第 97 个 episode 失败，其余进程停止；参见 [BO_DUPLICATE_COMPATIBILITY.md](BO_DUPLICATE_COMPATIBILITY.md)。
2. 处理重复点后的批次中，seeds 2、3 分别在第 299、377 个 episode 内出现非有限参数；其余 3 个 run 完成，但未与最终批次混用。
3. 原 JS 表达式在旧 PyTorch 的极小概率下可复现 NaN 梯度；数值兼容说明及历史故障证据限制见 [JS_NUMERICAL_STABILITY.md](JS_NUMERICAL_STABILITY.md)。最终批次 5/5 完成，未发生参数非有限异常。

## Git 与下一步

开发在 `feat/controlled-seeds` 分支完成。配置、脚本、测试和必要的小型汇总进入 Git；checkpoint、raw results、logs、曲线及环境留在被忽略的 `.local/`。上游原始快照及 provenance tag 保留，不执行强制推送。

下一步在独立分支建立 OARL 派生的 Clean Victim：保持网络、环境、replay、训练预算与种子协议，移除 BO、actor/critic 鲁棒项及 dual 更新。它不会被误标为标准含熵 SAC，也不通过修改训练好的 OARL checkpoint 获得。

## 本功能原子提交

- `87da5d5` feat(repro): add explicit controlled seed streams with legacy defaults
- `f90a96f` test(repro): verify saved actor weights and inference outputs
- `2924066` feat(repro): record isolated training runs and checkpoint proofs
- `317dbd0` chore(experiments): define five controlled OARL training runs
- `4a0e0c7` fix(repro): preserve upstream class binding during agent initialization
- `11e1ccc` feat(repro): validate frozen checkpoints on held-out SUMO seeds
- `8a7bdb5` docs(repro): specify controlled seeds and frozen checkpoint validation
- `6173b97` test(repro): compare complete OARL updates with frozen upstream
- `2d54790` feat(repro): summarize verified five-seed experiment artifacts
- `cba313c` feat(repro): export learning curves with across-seed variability
- `bcf6684` fix(oarl): retain cached GP observations for duplicate BO proposals
- `9d0086c` feat(repro): record BO duplicate proposals and evaluation counts
- `a51fb85` docs(repro): record successful recovery past duplicate BO failure
- `7dc2aa5` feat(repro): include BO duplicate and evaluation totals in summaries
- `fc6be78` fix(oarl): stabilize JS gradients for extremely small probabilities
- `97fdb57` feat(repro): retain numerical failure diagnostics and precision counts
- `77a6030` docs(repro): retain failed cohorts and verified checkpoint subsets
- `555ef49` fix(repro): record active evaluation environment and unused RNG roles
