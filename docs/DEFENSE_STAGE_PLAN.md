# 防御阶段：基线、适应性攻击与方法设计

2026-10-05。状态：防御对照工具、协议与 350 episode 开发交通 pilot 已完成并通过原始数据审计，173 项测试通过。实测结果与限制见 [DEFENSE_BASELINE_RESULTS.md](DEFENSE_BASELINE_RESULTS.md)；新方法为待检验设计，不宣称已有效或已证明创新性。

## 先建立 OARL Robust 对照

使用 `configs/frozen_victims.json` 中 episode-400 的五个 Clean 和五个 OARL 模型。已有训练控制审计表明它们采用匹配的架构、交通、种子与超参数；Clean 去掉 BO 与鲁棒双变量约束，OARL 保留原鲁棒训练。两组不是同一个 checkpoint 加/不加过滤器，而是两个不同训练方案的固定产物。

直接复用 `VictimAdapter`、攻击注册表、SUMO evaluator 与独立 simulator oracle，不修改 `main.py`、`oarl.py`、环境、路网、攻击实现或任何 checkpoint。第一项防御就是 OARL 训练所得策略本身，在线仍只做原 actor 推理，没有额外过滤、Gate 或动作覆盖。

`configs/research/defense_baseline.json` 记录具体模型哈希、攻击参数和配置哈希。两个入口均先提交再运行：

| 阶段 | 模型 | 交通 | 攻击 | 实际 episode | 用途 |
| --- | --- | --- | --- | ---: | --- |
| engineering_smoke | 两组各五个 | split 10 首个 seed | None、PGD、Single 高档 | 30 | 工程接入与逐步审计，不能推断防御效果 |
| development_pilot | 两组各五个 | split 10 前五个 seed | None、Random、FGSM、PGD、OARL-BO、预算版 Zero-One Return 高档、Single Return 高档 | 350 | 探索性防御对照，五个交通聚类、一次随机攻击重复 |

沿用 Normal / Pn=0.14、16D、三动作、greedy argmax、最多 200 步和逐特征扰动盒 `0.2|o_i|+0.05`。搜索保留冻结高档 400 梯度 / 800 攻击前向 / 200 新转移 / 4000 物理步每块、horizon 20、10 候选、两步 PGD。简单攻击配置固定，不补足梯度预算。pilot 只有高档，不代表完整三档防御 benchmark 已完成。

每个攻击拿到当前正在评估的实际策略，分别针对 Clean 和 OARL 重新生成扰动。两组 SUMO reset 使用同一交通名单，attack seed 为 0。策略间首次动作分歧后轨迹自然不同，不能要求全过程 state 完全相同。审计逐步重算观测盒、回报、碰撞、TTC/DRAC、资源账本与实际策略动作，并验证动作分歧前的配对交通一致。

## 同时回答四个问题

1. **干净性能**：分别报告无攻击回报、碰撞和条件安全指标。不能把防御带来的干净性能下降藏在相对百分比中。
2. **攻击后性能**：报告各攻击的绝对回报和碰撞率，以及相对各自无攻击策略的回报下降。`OARL攻击回报−Clean攻击回报` 与 `OARL回报下降−Clean回报下降` 分开报告，避免只因 OARL 原本回报更高就称其抗攻击。
3. **成本**：分别报告攻击梯度、攻击/评估前向、独立仿真步（含初始化、回放、warmup）、实际 wall time；保留总量、每 episode、每真实步。OARL 没有额外在线网络调用，训练代价不能据此当成零，也不能从这次评估估计。
4. **适应性攻击**：重新攻击真正部署的防御策略。针对 Clean 生成后转移到防御的扰动只能叫 transfer 对照。随机化防御使用 EOT / smoothed attack 并计费底层采样前向；非可微过滤需 BPDA 或无梯度攻击；有记忆防御的搜索分支必须克隆、重置和回放防御状态，不能让真实执行和仿真各自使用不同记忆。

ASR 只在相应策略无攻击未碰撞的样本内计算，两策略分母可能不同。报告各自 eligible 与 successes，同时报告原始碰撞率及共同未碰撞样本数。TTC/DRAC 保留有效分母和 null；SUMO/minGap 碰撞不等于必然发生物理接触。五个训练 seed 是五个固定模型，不代表五个独立交通集合；对同一交通的多个 attack seed 先平均，再按交通聚类统计。

攻击最终 split 30 不用于防御开发或调参。正式防御方法与预算冻结后，另登记全新的、与已有交通及训练 reset seed 不相交的防御保留测试名单，再扩展受控 seed 协议；当前工具没有启用这份新测试名单。pilot 不生成显著性结论。

## 有没有类似 FGSM 的通用防御基线

有通用方法家族，但没有与 FGSM 完全对应、所有 RL 策略都能直接套用的单一算法。建议最小防御矩阵为：Clean、OARL Robust、PGD 一致性训练、推理随机平滑。先保持同一 actor–critic、场景、训练交互上限；训练类防御允许产生新的模型，原冻结模型继续保留为对照。

| 基线 | 改什么 | 本项目中的合理适配 | 必须检查 |
| --- | --- | --- | --- |
| FGSM / PGD 对抗训练 | 训练数据或损失 | 在同一扰动盒内生成观测攻击，混合真实 on-policy 交互，加入明示的策略一致性损失 | RL 没有分类真标签；不能把监督分类 CE 直接当成等价 RL 算法；攻击需随被训练策略更新 |
| 状态对抗一致性正则化 | actor loss | 在干净观测与扰动观测之间约束策略分布，保持原 RL 目标 | 正则权重、干净性能、强攻击效果；应标注为 SA 思想的适配而非完整 SA-PPO 复现 |
| 随机平滑 / 噪声增强 | 推理或训练 | 独立 defense seed；明示 Gaussian 尺度、样本数和聚合规则 | EOT、干净性能下降与底层网络调用；不能自动继承论文的认证结论 |
| 时间过滤 / 特征合法域投影 | 推理前处理 | 可作廉价补充对照；连续特征与车道、缺车标记分别处理 | 持续攻击污染历史、换道/邻车切换、BPDA；不要对 16D 一律平均或裁剪到 [0,1] |

PGD 对抗训练的鲁棒优化源头参见 [Madry 等，ICLR 2018](https://arxiv.org/abs/1706.06083)。完整 RL 防御参考如下：

| 论文与源码 | 核心用途 | 接入优先级与限制 |
| --- | --- | --- |
| [Robust Deep RL against Adversarial Perturbations on State Observations，NeurIPS 2020](https://arxiv.org/abs/2003.08938)；[StateAdvDRL](https://github.com/chenhongge/StateAdvDRL)、[SA-DQN](https://github.com/chenhongge/SA_DQN)、[SA-PPO](https://github.com/huanzhang12/SA_PPO) | SA-MDP 与策略鲁棒正则化 | 优先借鉴一致性目标；原仓库分别对应 DQN/PPO/DDPG，不能把本项目 actor–critic 更名为 SA-PPO |
| [Robust RL on State Observations with Learned Optimal Adversary，ICLR 2021](https://arxiv.org/abs/2101.08452)；[ATLA](https://github.com/huanzhang12/ATLA_robust_RL) | 交替训练策略与学习型观测攻击者 | 后续强训练对照；不直接迁入 MuJoCo/PPO 全栈；训练预算单列 |
| [Robust Deep RL through Adversarial Loss，NeurIPS 2021](https://arxiv.org/abs/2008.01976)；[RADIAL-RL](https://github.com/tuomaso/radial_rl_v2) | 对抗损失与鲁棒 RL 训练 | 研究鲁棒损失及其干净性能取舍；不宣称目前已适配 |
| [Reward Certification for Policy Smoothed RL，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30139)；[ReCePS](https://github.com/TrustAI/ReCePS) | 平滑策略与回报认证 | 可核查训练、攻击和认证代码；认证前提与本项目观测相关盒不同，先做经验对照 |
| [Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents，ICML 2024](https://proceedings.mlr.press/v235/sun24b.html)；[S-DQN/S-PPO](https://github.com/Trustworthy-ML-Lab/Robust_HighUtil_Smoothed_DRL) | 平滑策略训练与 Smoothed Attack | 优先参考平滑攻击和干净性能评估；不直接更换 victim 或 Torch 环境 |
| [On Minimizing Adversarial Counterfactual Error in Adversarial RL，ICLR 2025](https://arxiv.org/abs/2406.04724)；[ACoE](https://github.com/romanbelaire/acoe-robust-rl) | 对观测不确定性的 belief 与反事实误差目标 | 新方法的重要近邻：已讨论一致性约束、保守性及攻击成功后的性能取舍，包含 Highway 实验；不能把“考虑部分可观测性”或“降低保守性”单独宣称为本项目新概念 |

上述为 2026-10-05 核查的论文/作者仓库，当前仅调研，没有下载、复制或混入其源代码。在线平滑的理论来源同时参见 [Policy Smoothing for Provably Robust RL，ICLR 2022](https://arxiv.org/abs/2106.11420)。关于梯度掩盖及适应性测试参见 [Athalye 等，ICML 2018](https://arxiv.org/abs/1802.00420)。

## 自己方法的第一版研究设计

暂定方向：**行为分支覆盖与恢复导向的鲁棒训练**。这是待检验假设，不是完成的创新证明。

攻击结果说明，局部诱导得到的实际动作会改变后续状态分布。因此防御不只应让一次观测的 logits 更稳定，还应学习在行为已偏离后恢复。保持原 16→128→3 actor 和原双 critic，不先更换为 PPO、记忆网络或新场景。

拟议训练流程：

1. 从匹配初始化或明确记录的训练起点建立**新的**可训练策略，原十个冻结 checkpoint 不动。Clean/OARL/PGD 一致性训练采用匹配初始权重、训练交互上限和外层更新数；额外攻击计算分别记录。
2. 在训练交通的当前真实状态，针对当前策略寻找观测扰动。对扰动产生的**实际行为分支**建立样本集合，同动作的重复目标诱导不无限加入；未知动作不能当成已覆盖动作。
3. 真实 attacked rollout 与部分 clean rollout 混合进入单独的训练 replay。只有真实执行过的转移作为 critic 学习样本；独立仿真的候选若用于训练，必须作为另一项明确的模型辅助实验计费与标注，不能悄悄当成免费真实数据。
4. 初版在原 RL actor loss 上加入分支平衡的一致性项：`L_actor = L_RL + lambda * mean_state(mean_actual_branch KL(stopgrad(pi(o)), pi(o+delta_branch)))`。分支只有零扰动或未发现有效扰动时明确记为空/退化。不要把此 KL 本身称为新创新，它与已有状态对抗正则化相近。
5. 对实际动作分歧后的有限恢复片段保留独立采样标记，用固定总采样量比较普通均匀采样与恢复片段重采样；不把“恢复”定义为强制复制原 Clean 模型的动作，而由原 RL 回报目标检验是否改善后续控制。

潜在研究增量是**训练样本按实际行为分支覆盖组织**与**偏离后的恢复片段分配**；它们是否比普通 PGD 一致性训练或 ATLA 式 rollout 训练更有价值，需要实验以及更深入近邻文献核查。仅给 PGD、KL 和 replay 换名字不能成立论文创新。

先做标准 PGD 一致性基线，再在独立 branch 上加入分支覆盖，最后加入恢复采样。新增模块各自消融，同时保持训练交互/更新次数匹配。主要检验“在相同训练交互和明确额外计算下，适应性轨迹攻击中的回报损害是否减少，干净性能损失是否可接受”；安全优势单独检验，不能从回报推出。

部署时新策略只接收真实传入的 16D 观测，不得获取未扰动观测、真实 SUMO state、attack flag 或扰动值。训练阶段可用的干净状态、分支搜索和片段标记全部留在训练侧。评估仍必须针对最终冻结的新策略重新运行 PGD、预算版 Zero-One 与 Single；不只测试训练中使用的攻击。

未来 `DefendedPolicy` 需要区分不提交状态的策略查询与一次真实转移后的状态提交；评估器查询同状态的无扰动动作，不能悄悄推进防御记忆。有状态防御与随机化防御不能直接塞进现有严格确定性规划回放：必须明确 cloned memory、随机流可见性、EOT 搜索样本与真实执行随机流的隔离，以及随机动作分歧后的计划处置。原始攻击比较保持原协议，新的防御适应性协议另行提交、测试和预登记。

## 后续功能分支

- `codex/defense-baseline`：当前协议、原始结果审计、Robust 对照与调研设计。
- `codex/defense-pgd-consistency`：标准训练基线，独立 checkpoint 登记与训练成本。
- `codex/defense-smoothing-eot`：平滑 wrapper、独立随机流、真实执行与搜索一致的 EOT 评估。
- `codex/defense-behavior-coverage`：自己的第一项候选模块。
- `codex/defense-recovery-sampling`：通过前一步后，再加第二模块及独立消融。

每项完成测试与对应审计后再 `--no-ff` 合并 main；只在稳定、完整可复现实验里程碑创建 tag。新方法效果不足不妨碍合并验证正确的工程，但结果和限制必须完整保留。
