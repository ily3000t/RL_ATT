# Stage 3 基础攻击 benchmark 完成报告

已在相同五个 frozen Clean checkpoint 上完成 Clean、Random、FGSM-margin、PGD-margin、OARL-BO 的统一评估：5 seeds × 5 conditions × 20 episodes，共 **500 episodes、84,688 个实际交互 step**。全部五组 NoAttack 与旧 checkpoint 验证逐轨迹一致；全部模型哈希和逐 step 预算/成本复核通过。

配置预先固定，见 [STAGE3_PROTOCOL.md](STAGE3_PROTOCOL.md)。33 项单元与回归测试通过，覆盖原 OARL 等价、冻结性、解析梯度、逐步投影、随机流隔离和安全指标。没有 Gate，没有重训或选择新 checkpoint。

## 回报与 SUMO 事件

± 是五个 run-level 数值的样本标准差，不是置信区间。每个 run 每个条件 20 个相同 SUMO episode seeds，最长 200 steps。Return Drop=配对 Clean Return−攻击 Return，负数表示该配置下平均回报提高。

| 条件 | Return | Return Drop | SUMO 碰撞率 (%) | ASR 五组均值 (%) | 转换成功 / 合格 episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 129.488 ± 8.404 | 0.000 ± 0.000 | 12.000 ± 7.583 | 0.000 ± 0.000 | 0/88 |
| Random | 128.236 ± 8.539 | 1.253 ± 2.032 | 15.000 ± 12.748 | 7.361 ± 13.569 | 6/88 |
| FGSM-margin | 103.010 ± 29.710 | 26.478 ± 29.625 | 26.000 ± 21.909 | 20.214 ± 27.425 | 17/88 |
| PGD-margin | 103.959 ± 27.545 | 25.529 ± 27.541 | 25.000 ± 19.685 | 18.964 ± 24.663 | 16/88 |
| OARL-BO | 131.930 ± 7.778 | -2.442 ± 3.539 | 10.000 ± 6.124 | 1.250 ± 2.795 | 1/88 |

ASR 分母只包括配对 Clean 未发生 SUMO 报告碰撞的 episode。不同 seed 的合格分母不同，所以五组率均值不等于合并计数之比；两种信息同时保留。碰撞率是 SUMO 原配置报告事件（可能包含 minGap 违规），不是经验证的物理车身接触率。

## 每个 seed 的 Return

| run_seed | Clean | Random | FGSM-margin | PGD-margin | OARL-BO |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 141.374 | 139.483 | 117.970 | 117.970 | 141.365 |
| 1 | 134.011 | 132.986 | 115.399 | 115.317 | 133.635 |
| 2 | 127.547 | 125.013 | 49.893 | 54.719 | 133.369 |
| 3 | 119.773 | 116.820 | 115.714 | 115.714 | 119.773 |
| 4 | 124.735 | 126.877 | 116.075 | 116.075 | 131.506 |

梯度攻击在 seed 2 上的影响远大于其他 seed，不能只看总体均值。PGD 在本轮没有比 FGSM 带来更低的平均 Return；这里使用固定动作 margin、10 steps、一个随机起点和最终 iterate，不构成最坏情况保证，也不意味着更多梯度步骤一定造成更大长期安全退化。

## 纵向安全与实际扰动

| 条件 | 每组 minimum TTC 的均值 (s) | TTC p05 (s) | DRAC p95 (m/s²) | 全部 step 最大 L∞ | 全部 step 最大 L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 9.287 ± 3.656 | 19.826 ± 2.028 | 0.212 ± 0.037 | 0.000000 | 0.000000 |
| Random | 9.382 ± 3.242 | 19.285 ± 1.764 | 0.217 ± 0.034 | 0.250000 | 0.561188 |
| FGSM-margin | 8.588 ± 2.514 | 17.998 ± 2.021 | 0.232 ± 0.038 | 0.250000 | 0.722638 |
| PGD-margin | 8.588 ± 2.514 | 18.130 ± 1.758 | 0.225 ± 0.025 | 0.250000 | 0.722638 |
| OARL-BO | 9.325 ± 3.693 | 19.853 ± 1.339 | 0.212 ± 0.026 | 0.250000 | 0.746153 |

安全分位数先在各 run 的原始纵向 pair 样本上计算，再跨五个 run 汇总；有效样本数、未闭合/缺失状态均保存在机器可读结果。只覆盖交互 step 后的同 lane 前后 pair，不能解释全部换道、reset 内事件或连续时间风险。详见 [SAFETY_METRICS.md](SAFETY_METRICS.md)。

共同约束为 `max_i |delta_i| / (0.2|s_i|+0.05) <= 1`，容差为归一化 observation 单位的 2e-7。四种攻击在所有实际交互 step 实施，Attack Rate 均为 100%，Clean 为 0%。原 OARL-BO 仍只搜索共享 u1/u2 的仿射子空间，其余三种可独立改变 16 个特征；没有物理裁剪。不能把这次比较称为相同搜索空间或等计算预算的优化器比较。

## 查询与梯度成本

| 条件 | 实际 step 数 | objective 次数 | 攻击内 forward 次数 | 输入梯度次数 |
| --- | ---: | ---: | ---: | ---: |
| Clean | 17,873 | 0 | 0 | 0 |
| Random | 17,618 | 0 | 0 | 0 |
| FGSM-margin | 15,469 | 30,938 | 30,938 | 15,469 |
| PGD-margin | 15,623 | 171,853 | 187,476 | 156,230 |
| OARL-BO | 18,105 | 90,525 | 199,155 | 0 |

不同策略提前终止造成实际 step 总数不同，单次攻击预算保持不变。wall time 也记录在 JSON 中；五进程并发运行存在资源竞争，不把累计耗时直接解释为严格的算法速度排名。

## 可支持的结论与边界

- 在本次固定 envelope 和 held-out seeds 上，FGSM-margin/PGD-margin 的平均回报下降和 SUMO 事件率上升比 Random 更明显，但五组方差较大，不宣称统计显著性或跨场景泛化。
- OARL-BO 本轮平均 Return Drop 为负，未产生平均回报退化。它优化单状态策略 JS，且受二维仿射搜索限制；这些是待验证因素，不能直接断言原 OARL 鲁棒训练或 BO 算法无效。
- 不能把 logit-margin FGSM/PGD 标为 cross-entropy 版本，也不能把 observation-scaled envelope 标为普通固定 epsilon 的 L∞ 球。当前参数未根据结果重新调优。
- 本轮结果提供后续 Zero-One adapter 的固定对照。没有提前开发 Ours 或接入 Zero-One，也没有更换环境、PPO victim 或安全判定参数。

## 产物与复现

完整小型 summary：[STAGE3_BENCHMARK_RESULTS.json](STAGE3_BENCHMARK_RESULTS.json)。对照图位于 `.local/reports/stage3_attack_benchmark.png`，同时提供 SVG；已进行可视检查。

- 实验 Git SHA：`b2e665bc4584ddd4a01b15f10141c316137a7f00`。
- 汇总代码 SHA：`3cfea2b1012a0f1ec1092abb00c98b808157fb5d`。
- 完整批次：`.local/runs/20260914T064729768497Z-attack-benchmark`。
- 短程检查：`.local/runs/20260914T064613549968Z-attack-evaluation`，五条件各 2×20 steps 均通过。
- 运行时沿用 Python 3.7.16 / PyTorch 1.3.1+cpu / SUMO 1.22.0；各 manifest 已记录命令、配置、角色 seed、依赖版本和源码前后哈希。
- `main.py`、`oarl.py`、`Environment/`、`Data/`、冻结模型清单及原训练/checkpoint 验证入口相对 v0.2.0-attack-api 保持不变。

```powershell
python scripts/summarize_benchmark.py --batch .local/runs/20260914T064729768497Z-attack-benchmark --output docs/STAGE3_BENCHMARK_RESULTS.json
python scripts/plot_attack_benchmark.py --summary docs/STAGE3_BENCHMARK_RESULTS.json --output .local/reports/stage3_attack_benchmark.png
```

完整重跑命令见协议文档；每次输出新的本地运行目录。checkpoint、raw results、log、环境、图像和源快照均未提交 Git。由于上游许可证尚不明确，本轮未公开推送。

## 功能分支原子提交

- `a1c8179` feat(victim): expose frozen actor logits for input-gradient attacks
- `acdd88e` feat(attack): define observation-scaled budgets and isolated state RNG
- `6a64344` feat(attack): implement reproducible uniform observation noise
- `0aeb4d6` feat(attack): implement frozen-policy logit-margin FGSM
- `9570b2d` feat(attack): implement random-start projected logit-margin PGD
- `2598a8c` feat(eval): integrate basic attacks with common bounds and gradient costs
- `64da740` test(attack): verify analytical gradients projections and seed isolation
- `16b38a3` feat(eval): dispatch isolated attack benchmark trials at one commit
- `b2e665b` chore(experiments): declare five-seed Stage 3 attack benchmark protocol
- `f0a8f93` feat(eval): audit raw attack budgets and aggregate paired five-seed results
- `ea7c42f` feat(eval): visualize benchmark trials with run-level uncertainty
- `3cfea2b` docs(attack): describe baseline registry and floating-point budget tolerance

下一阶段按计划进入 Stage 4，分析 Zero-One 的适配要求，在相同环境、frozen Clean checkpoint、observation 和 seeds 上接入，先验证接口与预算公平性，再运行扩展 benchmark。
