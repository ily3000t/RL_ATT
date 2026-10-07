# 防御阶段：基线、适应性攻击与方法设计

2026-10-05。状态：防御对照工具、协议与 350 episode 开发交通 pilot 已完成并通过原始数据审计，此前全套 173 项测试通过。实测结果与限制见 [DEFENSE_BASELINE_RESULTS.md](DEFENSE_BASELINE_RESULTS.md)。用户已审阅完整方案并同意推进，补充将 OARL Robust 列为与当前 SUMO 换道任务高度匹配的强基线之一。本次登记确认路线 B 优先考察 ACoE，ATLA 备选；论文目标适配和自己的最终公式仍须通过诊断与映射检查，不因方案批准就视为已经确定。通用防御、OARL 强基线与论文母方法承担不同的比较职责。此前“从 PGD 一致性基线逐步加模块”的开发路线已撤回，行为覆盖与恢复仅保留为待诊断问题。已有实验结果和执行协议不变，本次登记不启动训练。

已审阅原稿位于仓库外 `E:/Att/DEFENSE_RESEARCH_PLAN_REVIEW_20261005.md`，SHA256 为 `8c116c1d297422b11cebcc885547400e5cadfebfcf80eb8b9312e845b29f2c94`，审阅源码为 `bfc3b13fa01cca3eda094ce9e9d6c17d8ac0388b`。原稿保留不变，本文件登记批准范围及 OARL 强基线补充；原稿和攻击论文稿不提交 Git。

D1 进度（2026-10-07）：完整历史轨迹诊断已通过，见 [诊断结果](DEFENSE_D1_RESULTS.md) 与 [复现协议](DEFENSE_DIAGNOSTIC_PROTOCOL.md)；ACoE 目标核对、逆可行集合和梯度检查见 [离散映射](ACOE_DISCRETE_MAPPING.md)。现已完成 [控制性 reset/分支验证](DEFENSE_CONTROLLED_REPLAY_RESULTS.md)：350 次 rollout、150 个配对，无攻击和持续攻击历史精确复现，warmup 未观测到 ego 碰撞。OARL 两种搜索的单次转换均为集中交通的 5 次早失败，持续后增至 11/8 次；其他交通单次输入恢复有时提高回报，不能预设所有行为偏离有害或恢复模块已被支持。候选奖励来源、belief 和 off-policy 目标尚未选定，D1 没有据此关闭全部验收，也没有开始母方法或自己的防御训练。

D2 进度（2026-10-07）：[基础训练支撑](DEFENSE_TRAINING_SUPPORT_RESULTS.md) 已通过 22 项相关测试和 96 次真实 SUMO 短程 episode 的独立审计。关闭新防御时，Clean/OARL 原循环、新连续路径及独立进程恢复的逐 episode 轨迹和完整数值训练状态一致。新增完整 snapshot、隔离辅助随机流、实际成本计量及工程 actor 产物记录，旧入口与冻结模型不变；恢复仅限 episode 边界，工程产物不进入 benchmark。正式防御冻结登记和新交通 namespace 仍未完成。

D3 进度（2026-10-07）：[独立 PGD 一致性基线](DEFENSE_PGD_CONSISTENCY_RESULTS.md) 已实现，35 项相关测试、144 次 SUMO 短程 episode 及独立原始更新审计通过。系数/预算为零时精确退化到 Clean；enabled 方法完成独立重复和 episode 边界恢复。只在 replay batch 上搜索并正则化 actor，正常 SUMO 交互和 critic 目标保持；不称为完整 SA-PPO，不作为自己的母方法。尚未长程训练或测得防御收益，PGD 对抗交互训练、随机平滑和 ACoE 适配均未完成。下一项登记五 seed 长程训练与正式模型冻结/评估接入，同时补新的防御交通协议和 ACoE 数学目标。

## 先建立 OARL Robust 对照

使用 `configs/frozen_victims.json` 中 episode-400 的五个 Clean 和五个 OARL 模型。已有训练控制审计表明它们采用匹配的架构、交通、种子与超参数；Clean 去掉 BO 与鲁棒双变量约束，OARL 保留原鲁棒训练。两组不是同一个 checkpoint 加/不加过滤器，而是两个不同训练方案的固定产物。

直接复用 `VictimAdapter`、攻击注册表、SUMO evaluator 与独立 simulator oracle，不修改 `main.py`、`oarl.py`、环境、路网、攻击实现或任何 checkpoint。第一项防御就是 OARL 训练所得策略本身，在线仍只做原 actor 推理，没有额外过滤、Gate 或动作覆盖。

**OARL Robust 是主比较中的领域强基线。** 其定位依据是与当前 SUMO 换道场景、观测、动作和原训练流程的高度匹配，以及已实现的 BO/JS/dual 鲁棒训练。它保留五个冻结 episode-400 模型，参加完整攻击矩阵、全部搜索预算及独立防御测试，逐项报告新方法相对它的收益、负例和成本。该定位不改变 pilot 中收益不一致的事实，也不预设它在每个攻击条件下排名最优。若研究路线最终采用其他论文机制，OARL 的强基线角色与该论文适配版的母方法角色同时保留。

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

有通用方法家族，可承担攻击比较中 FGSM、Random、PGD 那样的基础对照角色；各方法仍需按 RL 的观测与训练目标适配。这里的“作为基线”指分别实现、固定配置和独立评估，用于回答自己的防御相对常用方法改善多少。基础防御的实现分支与自己的研究方法分支分开，完成这些基线不代表选定它们作为新方法的算法母体。

比较分三层：

| 层次 | 方法 | 比较目的 |
| --- | --- | --- |
| 无防御 | Clean | 量化防御的干净收益取舍和攻击后提升 |
| 领域强基线 | 原 OARL Robust | 在高度匹配的 SUMO 换道任务上检验新方法相对已有鲁棒训练的表现 |
| 通用基础防御 | FGSM/PGD 对抗训练、PGD 一致性训练、推理随机平滑；廉价过滤可选 | 检验研究方法是否比常用防御更有效，以及额外计算是否值得 |
| 相关论文方法与自己的方法 | 选定论文机制的独立适配版本、自己的版本 | 检验新增机制相对研究出发点的增量；自己的模块再做消融 |

基础比较先选 PGD 类训练对照和推理随机平滑；FGSM 对抗训练可以作为较弱补充。PGD 对抗交互训练与 PGD 一致性正则化属于不同训练目标，应明示损失、交互方式和配置；若最终只实现一个混合版本，就只算一个方法，不能拆成两个独立基线。训练类防御产生新的模型，旧模型保留；每个防御模型一旦冻结，就用于所有攻击条件，不针对测试攻击再更新权重。

| 基线 | 改什么 | 本项目中的合理适配 | 必须检查 |
| --- | --- | --- | --- |
| FGSM / PGD 对抗训练 | 训练交互与数据 | 在同一扰动盒内生成观测攻击，混合真实 attacked/clean rollout，由原 RL 目标学习 | RL 没有分类真标签；不能把监督分类 CE 直接当成等价 RL 算法；攻击需随被训练策略更新，实际交互与更新数计费 |
| PGD 一致性训练 / 状态对抗正则化 | actor loss | 在干净观测与扰动观测之间约束策略分布，保持原 RL 目标 | 独立比较基线；标注为 SA 思想的适配而非完整 SA-PPO 复现，记录是否同时加入 attacked rollout |
| 随机平滑 / 噪声增强 | 推理或训练 | 独立 defense seed；明示 Gaussian 尺度、样本数和聚合规则 | EOT、干净性能下降与底层网络调用；不能自动继承论文的认证结论 |
| 时间过滤 / 特征合法域投影 | 推理前处理 | 可作廉价补充对照；连续特征与车道、缺车标记分别处理 | 持续攻击污染历史、换道/邻车切换、BPDA；不要对 16D 一律平均或裁剪到 [0,1] |

PGD 对抗训练的鲁棒优化源头参见 [Madry 等，ICLR 2018](https://arxiv.org/abs/1706.06083)。PGD 也可在论文方法中充当扰动求解工具，但它不决定研究路线，使用它本身不构成贡献。下面的论文用于寻找与当前失败现象相关的研究机制，并确定需要加入的强比较方法。

## 相关论文与作者代码

| 论文与源码 | 核心用途 | 接入优先级与限制 |
| --- | --- | --- |
| [Robust Deep RL against Adversarial Perturbations on State Observations，NeurIPS 2020](https://arxiv.org/abs/2003.08938)；[StateAdvDRL](https://github.com/chenhongge/StateAdvDRL)、[SA-DQN](https://github.com/chenhongge/SA_DQN)、[SA-PPO](https://github.com/huanzhang12/SA_PPO) | SA-MDP 与策略鲁棒正则化 | 基础正则化适配及相关方法对照；原仓库分别对应 DQN/PPO/DDPG，不能把本项目 actor–critic 更名为 SA-PPO |
| [Robust RL on State Observations with Learned Optimal Adversary，ICLR 2021](https://arxiv.org/abs/2101.08452)；[ATLA](https://github.com/huanzhang12/ATLA_robust_RL) | 交替训练策略与学习型观测攻击者 | 候选研究出发点及强比较方法；需分清其 learned adversary 与现有 simulator search 的区别，记录攻击者训练成本 |
| [Robust Deep RL through Adversarial Loss，NeurIPS 2021](https://arxiv.org/abs/2008.01976)；[RADIAL-RL](https://github.com/tuomaso/radial_rl_v2) | 对抗损失与鲁棒 RL 训练 | 研究鲁棒损失及其干净性能取舍；不宣称目前已适配 |
| [Reward Certification for Policy Smoothed RL，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/30139)；[ReCePS](https://github.com/TrustAI/ReCePS) | 平滑策略与回报认证 | 可核查训练、攻击和认证代码；认证前提与本项目观测相关盒不同，先做经验对照 |
| [Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents，ICML 2024](https://proceedings.mlr.press/v235/sun24b.html)；[S-DQN/S-PPO](https://github.com/Trustworthy-ML-Lab/Robust_HighUtil_Smoothed_DRL) | 平滑策略训练与 Smoothed Attack | 优先参考平滑攻击和干净性能评估；不直接更换 victim 或 Torch 环境 |
| [On Minimizing Adversarial Counterfactual Error in Adversarial RL，ICLR 2025](https://arxiv.org/abs/2406.04724)；[ACoE](https://github.com/romanbelaire/acoe-robust-rl) | 对观测不确定性的 belief 与反事实误差目标 | 新方法的重要近邻：已讨论一致性约束、保守性及攻击成功后的性能取舍，包含 Highway 实验；不能把“考虑部分可观测性”或“降低保守性”单独宣称为本项目新概念 |

上述为 2026-10-05 核查的论文/作者仓库；后续审阅已将 ACoE 和 ATLA 的固定版本关键文件保存至 ignored `.local/defense-review-20261005/`，仅用于只读参考，未导入、执行或混入本项目，也未完成本地复现。ACoE 固定提交为 `a6897fe504650f94f2fef0d013ff925f0a838762`，根目录为 MIT 许可；ATLA 固定提交为 `dfdf8f56a025da95e0a3d652e5e7010b30904a29`，该 tree 未查得项目根许可证，嵌套组件许可不能覆盖整个项目。在线平滑的理论来源同时参见 [Policy Smoothing for Provably Robust RL，ICLR 2022](https://arxiv.org/abs/2106.11420)。关于梯度掩盖及适应性测试参见 [Athalye 等，ICML 2018](https://arxiv.org/abs/1802.00420)。

优先精读两个方向：

1. **ATLA：强对手下的交替学习。** 它与“针对当前策略生成攻击，再学习受攻击轨迹”的需求相关；论文方法包含在线训练的攻击者，作者代码以 SA-PPO 为基础。若用现有搜索器替换 learned adversary，应作为明确改变机制的适配或新版本，保留对应母方法对照。把自己的攻击器接进训练循环只能形成起点，是否有新的防御贡献还需证明。[论文](https://arxiv.org/abs/2101.08452)、[作者实现](https://github.com/huanzhang12/ATLA_robust_RL)。
2. **ACoE：观测不确定性下的长期误差与干净收益取舍。** 论文包含 highway-env 的 Highway/Merge 实验，其部分可观测性、belief 和累计误差机制与当前问题接近。该环境与现有 SUMO 不同；论文的 PPO/DQN 实现也与当前 actor–critic 不同，不能直接移入后称为同一环境、同一 victim 比较。[论文最新版](https://arxiv.org/html/2406.04724v4)、[作者实现](https://github.com/romanbelaire/acoe-robust-rl)。

ACoE 的 [steps.py](https://github.com/romanbelaire/acoe-robust-rl/blob/main/policy_gradients/steps.py) 已提供 belief 采样、累计误差学习与 PPO 更新相关入口。适配前须核查论文与代码的对应关系、额外网络和采样成本、随机数来源，以及 Torch 1.3.1 兼容性。现有观测盒依赖原观测绝对值；若从已扰动观测 y 估计候选原观测 x，应核对 `|y_i-x_i| <= 0.2|x_i|+0.05` 的逆可行域，不能直接将以 y 为中心的同形盒当成相同 threat model。作者代码中的邻域采样也不能未经核查就继承到本项目。

## 开发出发点：OARL 鲁棒算法还是论文机制

需要分别确定工程框架、母算法与权重初始化：复用 OARL 的 SUMO、16D 观测、三动作、actor–critic 结构、记录器和评估器，不等于继续优化 OARL 的鲁棒损失；借鉴一篇论文的机制也不等于整体迁移其项目。已有 Robust checkpoint 是比较对象，若未来选择从它微调，必须明确标注，并设置相同起点和相同额外训练预算的对照。

| 路线 | 具体含义 | 优势 | 主要代价与必要对照 |
| --- | --- | --- | --- |
| A：直接扩展 OARL 鲁棒算法 | 在独立 Agent 适配实现中研究 BO 搜索、JS 鲁棒约束、双变量更新或 critic 目标的变化 | 与现有问题和实现最接近，容易隔离改动 | 要说明原局部约束为何不足；原 OARL、仅改变内层攻击搜索的版本及自己的新增机制必须区分。换攻击器或调约束权重本身不保证创新 |
| B：以相关论文机制为出发点，在现有工程中适配 | 选择 ATLA 的交替学习或 ACoE 的累计误差机制，建立独立适配版本，再研究明确的机制增量 | 有可核查的目标与作者代码，能围绕长期后果提出研究问题 | PPO/DQN、采样、额外网络与当前 off-policy actor–critic 的差异需解释。对应论文的适配版本必须单独比较，不能把移植收益全部归因于新模块 |
| C：直接迁移论文完整框架 | 采用作者原训练器、网络及依赖，再连接 SUMO | 更接近论文实现，可用于核查原机制 | 可能同时改变 victim、优化算法及运行环境；若采用，须新增匹配的无防御和 OARL 思想对照，单独解释跨框架结果 |

**已批准的方向：优先路线 B，继续复用 OARL 的工程与控制任务，原 OARL Robust 保留为独立强基线。** ACoE 为第一候选，ATLA 为备选；先完成数学映射与机制诊断，不自动叠加两套目标。若论文核心机制无法在现有框架中被清楚适配、计费和验证，回到路线讨论，路线 A 仍是合理选择。直接扩展 OARL 也可能有研究价值，选用较新的论文并不会自动获得创新性。

最终选择依据是：能否解释已观测的失败现象，新增机制是否超出已有工作，作者代码与许可证是否允许所需复用，能否在固定 threat model 下实现可审计的母方法对照，以及训练/推理成本是否可承受。代码适配到 SUMO、改成三动作或换一个名字属于工程工作，应与科研贡献分开报告。

## 候选研究问题与决策顺序

暂保留“实际行为分支覆盖”和“动作偏离后恢复”两项问题，不预先固定 KL 损失、replay 方案或模块叠加顺序。现有 pilot 显示模型与攻击间差异较大，尚未确定这些差异由哪种机制导致。ACoE 已讨论长期误差与攻击成功后的表现，ATLA 已研究受攻击轨迹学习；提出自己的方法时必须说明与这些工作的具体差别。

1. 精读 ATLA/ACoE 等论文及关键代码，记录目标、训练数据、belief/攻击者假设、额外计算、许可证和版本兼容性。
2. 对照当前失败轨迹，明确要解决的问题与已有方法仍未覆盖的部分，形成可证伪假设。
3. 讨论选定路线与一个主要母方法，固定需要保留的机制、适配差异及对应控制组，再编写独立实现计划。
4. 通用基线作为单独比较任务实现；被选论文机制建立独立适配基线。基础基线代码可复用共有工具，但不作为自己的方法逐层添加模块的默认起点。
5. 自己的新增机制在独立 branch 实现，先验证工程语义，再做匹配预算实验与模块消融。主张取决于干净性能、攻击后性能和实际成本；安全优势单独检验。

最终比较分别回答“相对通用基础防御提升多少”“相对 OARL 领域强基线表现如何”和“相对母方法新增机制有何价值”。只超过基础防御不足以定位新增机制的贡献；也不要求每个指标全面超过所有方法，保留负例、成本取舍和样本限制。

训练模型采用独立登记与冻结，研究方法和相关基线在可比的架构、初始化、交互上限与更新预算下运行，额外网络和攻击计算照实记录。只有真实执行的转移作为通常的训练数据；若使用独立仿真候选训练，单列为模型辅助数据与成本，不能当作免费真实交互。

部署时新策略只接收真实传入的 16D 观测，不得获取未扰动观测、真实 SUMO state、attack flag 或扰动值。训练阶段可用的干净状态、分支搜索和片段标记全部留在训练侧。评估仍必须针对最终冻结的新策略重新运行 PGD、预算版 Zero-One 与 Single；不只测试训练中使用的攻击。

未来 `DefendedPolicy` 需要区分不提交状态的策略查询与一次真实转移后的状态提交；评估器查询同状态的无扰动动作，不能悄悄推进防御记忆。有状态防御与随机化防御不能直接塞进现有严格确定性规划回放：必须明确 cloned memory、随机流可见性、EOT 搜索样本与真实执行随机流的隔离，以及随机动作分歧后的计划处置。原始攻击比较保持原协议，新的防御适应性协议另行提交、测试和预登记。

## 已批准的执行顺序与验收门槛

以下是开发设计，尚未生成可执行的新防御最终协议或模型；具体名单、损失和配置在对应分支实现、测试并提交后冻结。

| 阶段 | 任务 | 验收与证据 |
| --- | --- | --- |
| D1 诊断与目标映射 | 检查全部五模型 × 五开发交通；核对即时/后续失败与 ACoE 目标 | 不只选 seed 3；控制性分支核对 reset warmup、真实注入时刻与碰撞定义；明确 belief、奖励/递推来源、离散输出和梯度路径 |
| D2 训练与登记支撑 | 基础 snapshot、独立随机流、恢复和成本已验证；生产审计与扩展登记入口验收完成 | 原 22 项测试、96 次 SUMO episode；新增短程完整状态/冻结评估接入通过；正式五 seed registry 须等长程全部审计通过后独立提交，新交通 namespace 尚未实现 |
| D3 独立比较方法 | PGD 一致性工程实现及五 seed 长程入口验收完成，其他方法待补 | 原 35 项测试、144 次 SUMO episode；生产 16 集路径与原实现完整数值等价，None/PGD 两次评估通过；400 集实际状态以 ignored batch 记录为准，尚无防御效果结论 |
| D4 单一新增机制 | 诊断支持后才确定一个主要增量 | 与母方法、等预算普通辅助数据/仿真控制比较；核对 WocaR、RADIAL 等近邻；不预设创新成立 |
| D5 验证与冻结 | 建议 30 个新验证交通、五模型、三个随机攻击重复 | 全部候选及负例保留；冻结超参数、episode-400 选择规则与科学主张 |
| D6 最终测试 | 建议 50 个新的保留交通、五模型、搜索三档预算 | 所有方法预先冻结，攻击逐策略重新生成；结果不反向用于选版本或调参 |

初版研究问题同时覆盖高后果的即时动作偏离和后续回报损害，控制干净收益与训练成本。既有轨迹显示 Robust 的 Zero-One 碰撞 13 次中 7 次、Single 碰撞 10 次中 7 次出现在前三个真实交互步；这包含策略自身无攻击可能碰撞的样本，是描述统计而非 ASR 或因果解释。部署初版仍建议只使用确定性 `16→128→3` actor，训练侧真实 state、原观测和分支标签不进入部署输入。

ACoE 适配前检查对称邻域采样、作者未显式播种的 `default_rng()`、部分路径固定扰动尺度及旧 Torch API。从 `y` 推断原观测 `x` 必须满足 `|y_i-x_i| <= 0.2|x_i|+0.05` 的逆集合；不能直接沿用以 `y` 为中心的同形盒。16D 候选观测不是完整 SUMO state，不能据此伪造候选奖励。相同修复适用于母方法与自己的版本。[固定作者实现](https://github.com/romanbelaire/acoe-robust-rl/blob/a6897fe504650f94f2fef0d013ff925f0a838762/policy_gradients/steps.py)

新训练默认从五个匹配初始化开始，沿用 400 episodes、每集最多 200 步、batch 128、每两步一次主更新及原 RL 超参数；额外网络用独立辅助随机流。80,000 是真实交互上限，提前终止会导致各方法实际步数不同，记录真实步数、更新与全部辅助成本。若以后从 Robust 权重微调，另设相同起点、相同额外训练预算的继续训练对照。

## 主比较、统计与新交通协议

确定性主比较的六个核心配置为 Clean、OARL Robust、PGD 对抗训练、PGD 一致性、选定论文的独立适配母方法、自己的版本。随机平滑是另一个通用对照，完成 EOT 与随机搜索执行验收后进入完整比较；此前仅列清楚标注的局部攻击补充结果。

攻击矩阵包含无攻击、Random、FGSM、PGD、OARL-BO，以及预算版 Zero-One Return 和 Single Return 的低/中/高档，共 11 个条件。三档继续使用每块 100/200/400 梯度、200/400/800 攻击前向、200 新仿真转移、4,000 物理步，保留两种搜索各自冻结的重试机制。简单攻击固定配置并记录实际成本。OARL 与其他确定性策略使用相同外部评估协议，不重新调 OARL 来适配测试攻击。

正式报告把新方法相对母方法、OARL Robust 和 PGD 一致性的回报差作为三项预先声明的核心比较；主要端点建议为 PGD、Zero-One Return 高档、Single Return 高档的等权平均攻击后回报，完整逐攻击结果同时展示。若对三项作共同主要统计判断，冻结时采用三比较家庭校正，例如每项约 98.333% 的区间；同时给出描述性 95% 区间。这扩展了已审阅原稿中两项主要比较的设计，原因是用户补充 OARL 为强基线。最终配置需明确区间算法和多重比较规则，不以所有比较均显著胜出作为工程合并条件。

同一交通先平均攻击重复，再等权平均五个固定模型，按交通块配对 bootstrap；建议 10,000 次重采样、analysis seed `20261005`。区间条件于五个固定模型，不将三个攻击 seed 当成三个独立交通。干净回报、各自 Return Drop、原始碰撞率、eligible ASR、共同未碰撞样本及 TTC/DRAC 有效分母分别报告；干净收益下降不超过 5% 可作开发目标，尚不是已通过非劣检验。

新的防御验证和最终名单采用独立 `defense_split_v1`，建议 root `20261005`、split `110/120/130`，逐个 SUMO seed 检查与所有已有训练、开发、验证和旧 split 30 不相交。当前旧校验器仍只接受原协议，新命名不能视为已经实现。方法选择结束前禁用保留测试执行，冻结配置和名单哈希后再运行。

按五模型 × 50 交通 × 三随机攻击重复，每个确定性配置预计 7,250 个实际 episode，六个配置 43,500 个；随机部署防御另计无攻击/FGSM 的部署噪声重复与 EOT 成本。先估算已审计吞吐再安排执行，不在协议未完整登记前启动这批测试。历史原始数据保留于 existing worktree 的 ignored 目录，不能归档后丢失。

## 后续功能分支

- `codex/defense-baseline`：已完成的 Robust 对照、原始结果审计与执行协议。
- `codex/defense-route-discussion`：本次角色澄清与研究路线讨论，只修改方案文档。
- `codex/defense-protocol-design`：登记用户批准范围、OARL 强基线定位和 D1–D6 设计，统一已撤回路线；本轮不启动训练。
- `codex/defense-training-support`：基础完整状态、随机流隔离、恢复、成本和工程产物记录已完成，见 [审计结果](DEFENSE_TRAINING_SUPPORT_RESULTS.md)；正式防御模型冻结登记与新交通协议另行补齐。
- `codex/defense-pgd-consistency`：独立基础比较方法实现与短程原始更新/恢复审计已完成，见 [结果](DEFENSE_PGD_CONSISTENCY_RESULTS.md)；不代表自己的方法起点。
- `codex/defense-pgd-long-run`：五 seed 长程配置、生产原始记录/审计、冻结候选与评估扩展入口完成工程验收，见 [结果](DEFENSE_PGD_LONG_RUN_INTEGRATION_RESULTS.md) 和 [协议](DEFENSE_PGD_LONG_RUN_PROTOCOL.md)；长程模型只有实际运行与审计完成后才可正式登记。
- `codex/defense-pgd-training`：独立对抗交互训练，明示 clean/attacked 数据对应与实际成本。
- `codex/defense-smoothing-eot`：独立基础比较方法，包含真实执行与搜索一致的 EOT 评估。
- 论文机制适配分支：讨论选定具体论文后命名，建立对应母方法对照。
- 自己方法的分支与消融：路线确定后按实际新增机制命名。原拟行为覆盖、恢复采样分支暂不启用。

每项完成测试与对应审计后再 `--no-ff` 合并 main；只在稳定、完整可复现实验里程碑创建 tag。新方法效果不足不妨碍合并验证正确的工程，但结果和限制必须完整保留。
