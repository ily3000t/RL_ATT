# Stage 2 完成记录

已采用适配器/包装器在现有 `rl_att/` 完成 Attack / Victim / Evaluation 解耦。保留原训练、原 BO、原环境、道路配置及旧 checkpoint 验证路径；没有重新训练或更换 victim，没有 Gate。实现说明见 [ATTACK_API.md](ATTACK_API.md)，指标口径见 [SAFETY_METRICS.md](SAFETY_METRICS.md)。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| 单元与回归测试 | 26 项通过；含原 upstream 回归、BO 完整更新/梯度/RNG 等价、冻结策略输入梯度、重复 BO 点、饱和概率、gap 修正、预算和配对分母 |
| NoAttack 与旧 checkpoint 评估 | Clean/OARL 各 5 seeds，全部 200 episodes、36,253 steps 通过 |
| 等价字段 | 逐 episode return、steps、termination、SUMO collision、action counts、SUMO seed、完整轨迹 SHA 均完全一致 |
| 冻结模型 | 10 个 episode-400 checkpoint 的文件/权重哈希及冻结模式验证通过 |
| BO 接线 | 两类 victim 的 seed 0，各 2×20 steps，合计 80 次攻击、400 次 objective；每次均 5 次，trace 获胜值及仿射边界通过复核 |
| 原文件保护 | 相对 v0.1.1-clean-victim，main.py、oarl.py、Environment/、Data/、experiment.py、checkpoint_evaluation.py 无修改 |

两次真实 SUMO 验证的代码 SHA 均为 `6a2a550fb5ee004ddfe22076788db511cdccd0ea`，使用训练时的 Python 3.7.16 / PyTorch 1.3.1+cpu / SUMO 1.22.0，环境和依赖与各训练 manifest 对照通过。

机器可读验证结果见 [STAGE2_VALIDATION.json](STAGE2_VALIDATION.json)。原始运行：

- NoAttack：`.local/runs/20260914T061154453044Z-attack-evaluation`
- BO smoke：`.local/runs/20260914T061153686995Z-attack-evaluation`

NoAttack 的 mean return 与上一阶段完全一致：Clean 五组均值 129.488，OARL 134.661；SUMO 报告碰撞分别 12/100、9/100。新增纵向安全采样未影响策略轨迹。各 seed 的 TTC/DRAC 和有效样本数已写入 JSON；这些条件统计不覆盖横向碰撞或 reset 内事件。

短程 BO 中两类 victim 的动作改变率均为 0、Return Drop 均为 0；实际 L∞ 最大值均为 0.25，L2 最大值分别为 Clean 0.632608、OARL 0.609586。这里仅证明接线、预算、日志和冻结性，不推断攻击是否有效；短程数据不足以评价安全退化。

## 关键解释

1. `search_batch` 直接调用原 Agent BO/JS，可保留训练等价。在线版本明确用 obs2=obs1，优化 2×单状态 JS，不能将它等同于原 replay 当前/下一状态联合鲁棒训练目标。
2. 原 BO 的预算是共享 u1/u2 的 affine box，不是 epsilon=0.2 的 L∞ 球；不加入投影/特征屏蔽改变原方法。后续统一 benchmark 需要显式确定各攻击预算如何匹配。
3. 原配置 minGap=25/30 m，本机默认 CLI collision.mingap-factor=−1（使用跟驰模型的默认阈值）。本轮碰撞指标是 SUMO 报告事件，包含可能的最小车距违规，不能直接称为物理接触碰撞。保留原参数以维护等价；物理接触指标应另行设计和验证。
4. TTC/DRAC 只计算实际同 lane 前/后跟驰 pair，补回 TraCI gap 排除的 follower minGap。无邻车/未闭合/undefined overlap 用状态和 null 明确区分；有效样本数始终与分位数一起报告。

## 复现命令

```powershell
python scripts/evaluate_attacks.py --config configs/evaluation/no_attack_equivalence.json
python scripts/evaluate_attacks.py --config configs/evaluation/oarl_bo_adapter_smoke.json
python scripts/summarize_stage2.py --equivalence .local/runs/20260914T061154453044Z-attack-evaluation --smoke .local/runs/20260914T061153686995Z-attack-evaluation --output docs/STAGE2_VALIDATION.json
```

前两项要求 Git 干净，输出新 timestamp 目录；重新导出摘要时使用实际新目录。配置、SHA、角色 seed、模型哈希、命令、运行时和源码前后哈希保存在各 manifest/evaluation 文件中。checkpoint、原始 JSONL、stdout/stderr 和运行快照仍不进入 Git。没有公开推送。

## 原子提交与后续

`refactor/attack-api`：

- `88879c3` refactor(attack): add frozen victim and observation attack interfaces
- `ac7aaed` refactor(attack): wrap original OARL BO with batch and online contracts
- `0b12947` test(attack): verify BO update equivalence and frozen policy isolation

`feat/eval-metrics`：

- `bff1305` feat(eval): record SUMO following safety and paired attack outcomes
- `cd385db` feat(eval): add budget-checked frozen victim evaluation loop
- `964f68d` feat(eval): launch reproducible attack evaluations from committed snapshots
- `8d599c8` test(eval): cover safety gaps budgets and paired outcome denominators
- `6a2a550` docs(eval): define adapter equivalence and longitudinal metric scope
- `cc3c966` feat(eval): export verified adapter and safety validation summaries

下一步为 Stage 3：在这些相同 frozen Clean checkpoints 上建立 Random/FGSM/PGD/OARL-BO 的统一预算 benchmark。本阶段没有提前实现这些新攻击，也没有接入 Zero-One 或开发 Ours。
