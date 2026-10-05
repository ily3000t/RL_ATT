# OARL Robust 防御对照：开发交通 pilot

2026-10-05。350 个实际 episode、57,195 个真实交互步骤通过完整原始数据审计；173 项测试通过。这是五个开发交通种子的探索性对照，不是防御独立最终测试，也没有训练自己的防御模型。

OARL Robust 在 PGD 下有平均回报改善，但在 Random 与预算版 Zero-One 下更差；Single 下两组平均回报几乎相同。局部鲁棒训练的收益随攻击、模型和交通变化，当前不能宣称普遍防御或稳定安全收益。

## 固定协议与样本

- 实验执行提交：`db4dcbc3d9b0dbcfd454d5dbcfa109ee9cb51e7f`；源码与 tracked tree 在整批运行期间固定。
- 审计提交：`8f8099a9c1d21ce1de5477b24d8ac9f81292e2c6`；显式指定原执行提交，原始执行审计与历史审计的 comparisons 和 raw source hashes 完全一致。
- feature branch：`codex/defense-baseline`；十个 episode-400 checkpoint 保留原文件与注册哈希。
- 五个训练 seed 对应五对固定 Clean/OARL 模型；每对在相同五个 split 10 交通种子上评估。每种攻击每组 25 个 episode，但只有五个交通聚类。一次 attack seed（0）；没有将重复模型或攻击次数当成新的独立交通样本。
- 交通种子：`1472154258, 1940738230, 1610385403, 258385557, 2018177452`。角色种子派生规则沿用现有 `research_split_v1`，完整 effective seeds 在每个 run manifest/evaluation 中。
- Normal / Pn=0.14，最多 200 步，greedy argmax，无 Gate。扰动盒 `|delta_i| <= 0.2|o_i|+0.05`。
- FGSM 一步；PGD 十步、归一化 alpha=0.2；BO 五次 objective evaluation。搜索保留冻结高档每块 400 梯度、800 攻击前向、200 新仿真转移、4000 物理步；horizon 20、10 候选、两步内部 PGD。简单攻击保持固定配置，实际成本照实报告。只有搜索高档，未完成三档防御比较。
- 针对正在部署的实际策略重新生成每个攻击；没有复用 Clean-targeted 扰动作为 Robust 的适应性证据。无攻击策略的轨迹自然不同；策略/攻击导致动作分歧后不强求状态仍一致。
- 未使用攻击最终 split 30 进行防御开发。防御最终名单将在方法冻结后独立预登记。

环境沿用 Python 3.7.16、Torch 1.3.1 CPU、NumPy 1.21.6、SUMO 1.22.0。运行器版本、pip freeze、配置、模型哈希、命令与角色种子都已记录，并与冻结训练记录核对。

## 干净性能与攻击后绝对表现

Return 越高越好；Drop 为各自无攻击回报减当前攻击回报，负值表示这次扰动后反而改善。碰撞列是 25 个 episode 中的 SUMO ego collision 计数，包含对应策略无攻击时已有的碰撞。

| 条件 | Clean Return | Robust Return | Clean Drop | Robust Drop | Clean 碰撞 | Robust 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 无攻击 | 123.88 | 122.15 | 0.00 | 0.00 | 1/25 | 2/25 |
| Random | 125.61 | 115.38 | -1.73 | 6.76 | 1/25 | 3/25 |
| FGSM | 87.97 | 89.20 | 35.92 | 32.95 | 7/25 | 11/25 |
| PGD | 88.61 | 93.16 | 35.27 | 28.99 | 7/25 | 7/25 |
| OARL-BO | 123.94 | 122.91 | -0.05 | -0.76 | 0/25 | 2/25 |
| 预算版 Zero-One Return（高档） | 79.49 | 71.19 | 44.39 | 50.96 | 9/25 | 13/25 |
| Single Return（高档） | 77.36 | 77.39 | 46.52 | 44.76 | 9/25 | 10/25 |

无攻击时 Robust 平均回报比 Clean 低 1.74，碰撞由 1/25 变为 2/25，不能忽略这一基线差异。PGD 下 Robust Return 高 4.55，扣除自身无攻击表现后的平均损失少 6.29，但总碰撞均为 7/25。预算版 Zero-One 下 Robust Return 低 8.30，自己的损失反而多 6.57，碰撞多四次。Single 下绝对回报只高 0.03，自己的损失少 1.76，碰撞多一次，不能表述为有明显防御提升。

## ASR 使用策略自己的无攻击分母

Clean 有 24 个无攻击未碰撞样本，Robust 有 23 个；共同未碰撞样本有 23 对。ASR successes 只统计这些 eligible 样本受攻击后出现的 ego collision，不能与所有 episode 的碰撞率混用。

| 攻击 | Clean successes/eligible | Clean ASR | Robust successes/eligible | Robust ASR |
| --- | ---: | ---: | ---: | ---: |
| Random | 1/24 | 4.17% | 2/23 | 8.70% |
| FGSM | 6/24 | 25.00% | 9/23 | 39.13% |
| PGD | 6/24 | 25.00% | 5/23 | 21.74% |
| OARL-BO | 0/24 | 0.00% | 0/23 | 0.00% |
| 预算版 Zero-One Return（高档） | 8/24 | 33.33% | 11/23 | 47.83% |
| Single Return（高档） | 8/24 | 33.33% | 8/23 | 34.78% |

## 收益集中性与失败案例

下面按固定训练 seed，在五个交通上平均；delta Return 为 Robust−Clean，delta 碰撞为 Robust−Clean 次数。

| 模型训练 seed | PGD delta Return | PGD delta 碰撞 | Zero-One delta Return | Zero-One delta 碰撞 | Single delta Return | Single delta 碰撞 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -7.78 | +0 | -0.04 | +0 | -0.06 | +0 |
| 1 | 5.10 | +0 | 1.37 | +0 | 4.34 | +0 |
| 2 | 39.29 | -2 | 21.54 | +0 | 40.93 | -2 |
| 3 | -14.62 | +2 | -66.63 | +4 | -46.42 | +3 |
| 4 | 0.76 | +0 | 2.25 | +0 | 1.34 | +0 |

Robust seed 2 对 PGD 和 Single 改善较大，seed 3 的搜索攻击结果则明显恶化。五个交通聚类上的成对方向也不统一，完整 per-traffic 结果在小型 JSON 中。这提示应检验行为分支覆盖与偏离后恢复，但尚不能用这些汇总确定缺失机制，更不能称新防御有效。后续应先与标准 PGD 一致性训练比较，再用独立模块实验检验这个解释。

## 实际计算成本

下表全部为每个实际 episode 的平均数。Grad 是攻击梯度评估，Attack Fwd 是攻击网络前向，Shadow 是包含 warmup 与前缀 replay 的独立 SUMO 物理步。碰撞提前终止会改变 episode 长度，因此 JSON 同时保留总成本与每真实交互步成本。Wall 是攻击调用计时，不是整个实验 wall time，也不是防御延迟。

| 条件 | Clean Grad | Robust Grad | Clean Attack Fwd | Robust Attack Fwd | Clean Shadow | Robust Shadow | Clean Wall(s) | Robust Wall(s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无攻击 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 |
| Random | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.038 | 0.033 |
| FGSM | 149.32 | 145.72 | 298.64 | 291.44 | 0.00 | 0.00 | 0.142 | 0.135 |
| PGD | 1508.00 | 1568.40 | 1809.60 | 1882.08 | 0.00 | 0.00 | 1.064 | 1.084 |
| OARL-BO | 0.00 | 0.00 | 2200.00 | 2150.28 | 0.00 | 0.00 | 11.092 | 11.232 |
| 预算版 Zero-One Return（高档） | 961.36 | 821.92 | 2707.80 | 2322.72 | 3233.56 | 2513.84 | 17.325 | 8.726 |
| Single Return（高档） | 934.72 | 1047.44 | 2710.12 | 3131.56 | 3327.72 | 4384.80 | 15.138 | 17.672 |

OARL 对照是已训练好的同构 actor，没有增加在线防御网络调用；在线仍做原策略推理。评估器每真实步两个前向（实际动作和无扰动对照动作）单独计费。这不等于鲁棒训练免费：本次没有重新测训练代价，也没有单独 benchmark 防御推理延迟。五个并行任务存在争用，Wall 只能描述这次执行，不能作为串行部署速度排名。

## 条件安全指标

下表为每个 episode 的 TTC minimum、TTC p5、DRAC p95，再对有定义的 episode 平均；每个值旁边给出有效 episode 数。不是合并所有步后重新计算的全局 percentile。

| 条件 | 策略 | TTC min(s) [有效n] | TTC p5(s) [有效n] | DRAC p95(m/s²) [有效n] |
| --- | --- | ---: | ---: | ---: |
| 无攻击 | Clean | 14.610 [25] | 20.975 [25] | 0.235 [25] |
| 无攻击 | Robust | 14.375 [25] | 19.412 [25] | 0.250 [25] |
| Random | Clean | 14.545 [25] | 20.968 [25] | 0.239 [25] |
| Random | Robust | 17.284 [25] | 22.565 [25] | 0.250 [25] |
| FGSM | Clean | 19.140 [21] | 24.788 [21] | 0.222 [21] |
| FGSM | Robust | 14.777 [22] | 20.933 [22] | 0.262 [22] |
| PGD | Clean | 18.852 [21] | 24.661 [21] | 0.215 [21] |
| PGD | Robust | 14.696 [22] | 21.484 [22] | 0.231 [22] |
| OARL-BO | Clean | 14.425 [25] | 20.387 [25] | 0.243 [25] |
| OARL-BO | Robust | 14.091 [25] | 19.452 [25] | 0.253 [25] |
| 预算版 Zero-One Return（高档） | Clean | 18.754 [19] | 24.745 [19] | 0.229 [20] |
| 预算版 Zero-One Return（高档） | Robust | 15.723 [18] | 21.539 [18] | 0.265 [20] |
| Single Return（高档） | Clean | 21.636 [19] | 26.684 [19] | 0.186 [20] |
| Single Return（高档） | Robust | 15.210 [18] | 21.018 [18] | 0.224 [20] |

邻车条件不成立、无 closing 或缺少有效 leader 时保留 null；提前碰撞终止及不同轨迹长度会改变有效样本。这些条件均值不能抵消或替代碰撞结果。SUMO 的 minGap/碰撞检测也不能直接改写为确认发生物理接触。

## 审计与复现记录

逐步审计覆盖回报、观测扰动边界、真实模型动作、冻结权重、角色种子、交通前缀、碰撞/TTC/DRAC、搜索 witnesses、oracle 重置/回放/warmup 的完整账本以及 source snapshot 字节哈希。Windows 的 `git archive` 会应用 CRLF 导出；初次 smoke 的 Git blob 哈希对比失败属于审计器问题，已改为比较 committed export 的实际字节，没有因此重新运行模拟或更改算法。

- smoke：执行 `8dbd492652709e90783f967c7cdba7540ff199ae`，30 episodes、5,431 real steps，通过；仅工程证据。
- pilot：执行 `db4dcbc3d9b0dbcfd454d5dbcfa109ee9cb51e7f`，350 episodes、57,195 real steps；原执行审计与显式历史审计均通过。
- tests：173 项通过；日志在隔离 worktree 的 `.local/defense-test-historical-audit.log`。日志中 mock failure 是失败路径测试，不能当作本轮实际实验失败。
- 完整原始数据：`C:\Users\asus\.codex\worktrees\defense-baseline\OARL-master\.local\runs\20261005T010509049212Z-attack-benchmark`。为保留 ignored 原始数据，当前不归档该 worktree。
- 完整历史审计 SHA256：`c4736f8b1d691fbde64297bfe47d53aa00dfba98ae0f0f2bbb77de5f12cbee41`。
- 原执行审计 SHA256：`496a7908309e17ae989466f1c11c076674bd649e606be08cd124041c5372536d`。
- 精确数值、实际成本及 manifest/source 指纹见同目录 `DEFENSE_BASELINE_RESULTS.json`；raw steps、trajectories、logs 和 checkpoint 不进 Git。

在记录的执行提交及已就绪的冻结 `.local` 资产上，启动命令如下：

```powershell
$env:PATH='E:/Att/OARL-master/.local/envs/oarl-legacy;E:/Att/OARL-master/.local/envs/oarl-legacy/Library/bin;'+$env:PATH
$env:MPLCONFIGDIR=Join-Path $PWD '.local/matplotlib'
& 'E:/Att/OARL-master/.local/envs/oarl-legacy/python.exe' scripts/evaluate_attack_batch.py --configs configs/evaluation/defense_oarl_development_seed0.json configs/evaluation/defense_oarl_development_seed1.json configs/evaluation/defense_oarl_development_seed2.json configs/evaluation/defense_oarl_development_seed3.json configs/evaluation/defense_oarl_development_seed4.json --jobs 5
```

上面的启动命令会生成一批新的实验记录，当前已完成的批次无需重跑。合并后的代码也可对旧批次只做审计：

```powershell
$env:MPLCONFIGDIR=Join-Path $PWD '.local/matplotlib'
# cwd: C:/Users/asus/.codex/worktrees/defense-baseline/OARL-master；输出必须使用新的 .local/runs 路径
& 'E:/Att/OARL-master/.local/envs/oarl-legacy/python.exe' scripts/analyze_defense_baseline.py --batch .local/runs/20261005T010509049212Z-attack-benchmark/batch.json --expected-execution-commit db4dcbc3d9b0dbcfd454d5dbcfa109ee9cb51e7f --output .local/runs/20261005T010509049212Z-attack-benchmark/new-audit.json
```

该命令核对真实历史 source commit 与完整注册协议，分别记录 execution 和 audit SHA；不允许把旧运行冒充当前源码产生的新实验。

## 下一项开发

优先在独立 branch 实现标准 PGD 一致性训练，保持现有 actor–critic 与训练交互上限，记录额外攻击/更新计算并生成独立模型。随后才加入自己的行为分支覆盖、恢复片段采样模块；每个模块独立消融，并针对每个最终防御策略重新生成攻击。随机平滑另做 EOT，有记忆过滤器必须将记忆克隆/重置纳入搜索，不能直接套用当前确定性回放。具体文献、部署边界与候选训练目标见 `DEFENSE_STAGE_PLAN.md`。

本轮没有修改任何原攻击、OARL 算法、SUMO 场景或已冻结模型，没有引入新依赖，也没有 public push 或发布。工程对照已完成；新方法的效果、创新性和正式防御结论仍待验证。
