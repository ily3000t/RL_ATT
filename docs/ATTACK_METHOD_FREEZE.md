# 攻击定稿：Single Return 与可检验的论文主张

冻结日期：2026-10-04（Asia/Shanghai）。运行算法冻结于 `main` 的 `d218309627830be924c48ef7fd9d984be11da247`。正式候选名称为 **Single Return：基于实际动作证据的行为分支搜索，单次目标诱导**，实现名保持 `ours_single_return`，不新增名称相近但行为不同的攻击。

本文件冻结方法、研究问题和证据边界，最终效果结论由未见交通检验。既有 Progress Return 保留为重试机制对照；预算版 Zero-One Return 是主要匹配目标的算法对照，Zero-One Safety 是补充目标对照。当前不加入 Single Safety、Gate、场景扩展、新 victim 或防御。

## 方法与威胁模型

同一 SUMO 原场景、Normal 流量、五个 episode-400 frozen Clean victim、16D 观测、greedy 三离散动作。攻击者白盒访问 actor，并拥有独立的 SUMO rollout oracle；只修改当前送入 victim 的观测，不直接更改真实环境动作、奖励、终止、时序或未来真实观测。

扰动满足 `|delta_i| <= 0.2|observation_i|+0.05`，每步允许攻击，无 Gate。该观测空间约束不保证离散特征、缺失标记或跨时间物理一致性。TTC/DRAC 沿用已有纵向 SUMO state 定义，碰撞事件沿用原 SUMO/minGap 语义。

每个规划块最多 20 个真实交互步、10 条搜索候选，另有一条计费的完整干净保底 rollout。Return 目标最小化原始 reward 的 horizon 内累计值，真实提前终止时停止累计，没有新增碰撞奖励。PGD 使用随机初始点、两次 targeted logit-margin 梯度和归一化步长 1；搜索与执行均受已有独立资源账本约束。

Single 的执行规则完全沿用现有源码：

1. 节点以 episode 起点以来的**完整实际动作历史**为键，不以 16D 观测合并状态。缓存 oracle transition 和诱导结果是共用机制。
2. 节点保存干净动作的零扰动证据。对尚未尝试的非干净目标，按当前 logit 从高到低、动作编号打破平局，每次节点访问至多执行一次目标诱导。
3. 一次 PGD 得到目标 `a` 的扰动后，策略实际输出 `b`。保存能诱导 `b` 的观测证据，继续以 `b` 推进仿真，即使 `a != b`。同一节点的同一实际动作只保留一个分支证据，优先保留最大绝对扰动较小的证据。
4. 在已有实际分支中按完成 rollout 的访问次数、已观测到的最佳完整 Return、动作编号排序。每个节点/目标最多一次诱导尝试；失败目标不重启，但不被判为不可达。
5. 只有达到 horizon 或真实终止的完整候选参与选择。每块选择累计 Return 最小的完整候选；预算耗尽保留已有完整计划，无完整计划时按已有显式干净回退规则记录。
6. 执行每一步验证干净观测、扰动后的策略动作和 SUMO transition 与计划一致。原 observation 不能被后续攻击原地改写，victim 权重始终冻结。

独立扰动 seed 由 `(attack_seed, episode, full_actual_history, target, attempt)` 的现有 SHA-256 规则生成；Single 只使用 attempt 0。不根据遍历顺序共享环境随机流。源码锚点：[single_attempt.py](../rl_att/attacks/single_attempt.py)、[behavior_search.py](../rl_att/attacks/behavior_search.py)、[proposed.py](../rl_att/attacks/proposed.py)、[search_budget.py](../rl_att/attacks/search_budget.py)。

## 与 Zero-One 和 Progress 的准确区别

Zero-One 原论文已经组合外层无梯度动作优化与内层梯度观测诱导。本项目比较的是 SUMO 三离散动作、state-only 的预算适配版，不是其连续控制和 timing-jitter 实验的数值复现。[Zero-One 论文](https://stanleybak.com/papers/bak2024iccps.pdf)，[作者单位的发表记录](https://researchconnect.stonybrook.edu/en/publications/zero-one-attack-degrading-closed-loop-neural-network-control-syst/)。

| 机制 | 预算版 Zero-One Return | Single Return | Progress Return |
| --- | --- | --- | --- |
| 外层组织 | ZOOpt 搜索 categorical 目标动作序列 | 按已发现实际分支的访问次数与完整 rollout 分数探索 | 与 Single 相同的行为分支规则 |
| 目标未命中 | 仿真使用策略实际输出动作 | 保存实际输出动作的有效证据并参与分支选择 | 同 Single，允许后续独立起点重试 |
| 内层诱导 | 共用两步 PGD，目标序列调用 attempt 0 | 同节点/目标仅 attempt 0 | 每目标最多 3 次，严格 margin-progress 停止规则 |
| 共用机制 | 完整实际历史缓存、扰动原语、oracle、完整保底、执行核验、账本 | 全部共用 | 全部共用 |
| 目标与权限 | 原始 Return，白盒策略＋独立仿真 | 相同 | 相同 |

预算版 Zero-One 参数保留 `max_attempts=3` 的既有配置，但其 target-sequence 路径始终调用 attempt 0，并未因此实际执行三次重试。不能将参数值当作实际成本；源码为 [zero_one_controls.py](../rl_att/attacks/zero_one_controls.py)。也不能声称 Zero-One 忽略实际动作、直接控制环境或没有实际历史缓存。

## 已有实验支持和不支持什么

| 证据 | 支持的有限判断 | 不支持的判断 |
| --- | --- | --- |
| 完整开发集机制研究，2250 episode | 单次行为搜索已获得三档相对 ZO 的 Return 改善；固定重试新增 Return 收益较小、没有新增碰撞且显著增加梯度 | 重试永远无用；缓存单独构成创新；完整因果或全因子结论 |
| split 20 候选复核，新增 1800 episode | Single−ZO 三档平均 Return 差 −6.910/−4.652/−3.014；Single 与 Progress 的 900 对碰撞结果一致，中/高档攻击梯度减少 38.03%/37.86% | 未见交通泛化；全模型全指标胜出；整体运行速度提高 38% |
| 仿真与成本账本 | 模型调用和 physical shadow 可以分别核算；中/高档相对 Progress 的 shadow 增加 4.60%/1.67% | 全面计算优势；任意加权后的综合效率优势 |
| 失败轨迹与交通分组 | 存在未命中、错过有效后续重试和收益集中；需保留反例 | 动作不可达证明；稳定碰撞优势；物理可实现攻击 |

来源：[MECHANISM_DEVELOPMENT_RESULTS.md](MECHANISM_DEVELOPMENT_RESULTS.md)、[SINGLE_CANDIDATE_VALIDATION_RESULTS.md](SINGLE_CANDIDATE_VALIDATION_RESULTS.md)、[VALIDATION_COMPARISON_RESULTS.md](VALIDATION_COMPARISON_RESULTS.md)。split 20 已参与开发判断，不能重新标作独立测试。

## 冻结的研究问题

**RQ1：** 同一扰动与多维计算上限下，Single Return 相对预算版 Zero-One Return，能否在未见交通上造成更大的原始累计回报损害？逐预算报告配对 Return 差和交通分组的不确定性，不选择最有利预算作为唯一结果。

**RQ2：** Single 相对 Progress 省去重试后，模型计算减少多少，执行效果损失或收益多大，仿真成本如何变化？分别报告 Return 差、碰撞转换得失、梯度、攻击/评估 forward 和含 setup 的 shadow，不预设非劣效、等价或所有资源均更优。

**RQ3（补充）：** 相对固定 Random、FGSM、PGD、OARL-BO，轨迹搜索带来的额外 Return 损害和碰撞转换是否值得其实际计算开销？简单攻击配置固定，不强迫其消耗相同梯度预算；明确额外仿真权限。

候选论文主张限定为**有限预算下的回报损害与模型计算分配取舍**。若最终结果只支持部分预算或仅支持成本侧，则收窄结论；若不支持，完整报告负结果，工程完成不等于有效性成立。碰撞、TTC/DRAC、跨模型统计是补充，不提前升级为稳定安全或跨场景主张。

## 已有工作的边界

动作序列规划再诱导策略执行并非新概念，Lin 等的 enchanting attack 已使用这一思路。[IJCAI 2017 原论文记录](https://www.ijcai.org/Proceedings/2017/525)。Zero-One 的两层优化也已明确存在；离散动作轨迹树及鲁棒性分析还有 [Provable observation noise robustness](https://www.cambridge.org/core/journals/research-directions-cyber-physical-systems/article/provable-observation-noise-robustness-for-neural-network-control-systems/33C1C135187FF661586451DED1C2A19C) 等既有研究。

本项目待论证的增量是**有限预算下按实际动作证据组织启发式分支探索，并用完整公平控制检验其效果–成本取舍**。这不是“第一个动作树/第一个 PGD 轨迹攻击”的主张，也不是形式化可达性验证。当前文献核对只确定这些边界，不替代完整投稿前的新颖性审查。上述一手来源核对于 2026-10-04；没有复制其项目源码。

## 冻结与最终测试的关系

[ATTACK_FINAL_TEST_PROTOCOL.md](ATTACK_FINAL_TEST_PROTOCOL.md) 和机器协议将冻结所有模型、完整 50 个 split-30 seed、方法/预算/统计和失败处理。预登记阶段只做输入、版本和分析工具测试，不读取最终 SUMO 结果。最终执行器和逐批审计需在运行前就绪并再次固定代码 SHA；任何最终结果不得用于重选方法、预算、checkpoint 或调参。防御研究在正式攻击比较之后单独开展。
