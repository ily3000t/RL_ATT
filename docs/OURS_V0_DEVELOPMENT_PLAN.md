# Ours-v0 开发方案：基于实际诱导行为的预算分配

Single 候选验证更新（2026-10-03）：独立分支 `codex/single-candidate-validation` 的九批新增 1800 episode / 297467 步全部通过审计，复用 Return 对照 272619 步、Clean 回归 162243 步和原 45 个 Progress−ZO 配对条件核验一致，125 项测试通过。Single 与 Progress 的 900 对碰撞结果逐对相同，中/高档梯度减少 38.03%/37.86%，forward 减少 29.10%/29.47%，physical shadow 增加 4.60%/1.67%。Single−ZO 三档 Return 差为 −6.910/−4.652/−3.014，净转换仍为 +3/−1/+2；安全集中性与高档模型负例继续保留。建议冻结 Single Return 为下一轮正式候选，同时保留 Progress 机制参考，先独立预登记最终未见交通上的全对照协议。详见 [SINGLE_CANDIDATE_VALIDATION_RESULTS.md](SINGLE_CANDIDATE_VALIDATION_RESULTS.md)。本轮只复核已使用的 split 20，未启用 final split 30、Safety 简化候选或防御；不覆盖既有方法名称或比较结果。以下阶段记录保留。

完整机制研究更新（2026-10-03）：`codex/mechanism-development-study` 的九批 2250 episode、369008 步审计全部通过；九批 Clean 共 89748 步及高档五条件 smoke 前缀 8504 步回归一致。Single 已取得低/中/高档相对预算版 ZO 的平均 Return 差 −10.511/−1.087/−3.685；固定重试只再改善 0/0.093/0.322，没有新增碰撞，高档梯度约翻倍。进展规则高档比固定重试节省 18.20% 梯度，但保留错过有效后续尝试的真实反例。当前 Progress 相对 ZO 的净新增转换仍为 +10/−1/+2，安全收益集中性和成本局限没有消失。详见 [MECHANISM_DEVELOPMENT_RESULTS.md](MECHANISM_DEVELOPMENT_RESULTS.md)。建议下一功能独立冻结并验证简化候选，不自动替换现有方法；final split 30 继续保留，当前不启动防御。以下 smoke 和各阶段判断保留为历史记录。

机制对照工程更新（2026-10-03）：既有七个完成分支已依赖顺序合入 main，新分支 `codex/attack-mechanism-controls` 增加单次尝试行为搜索对照，与预算版 Zero-One、固定重试版、进展版形成四个 Return 条件。五模型、两个已有开发交通的 50 episode / 8504 步审计及 110 项测试通过，四个旧条件逐字段回归一致。当前 smoke 未显示独立重试带来额外效果，不能预设该模块有益。完整三预算、三 attack seed 的 2250 episode 机制研究尚未运行；最终 split 30 继续保留。详见 [MECHANISM_CONTROLS_SMOKE_RESULTS.md](MECHANISM_CONTROLS_SMOKE_RESULTS.md)。此工程结果不替代下述完整验证结论，后续分支/测试/合并规范见根目录 `AGENTS.md`。

完整对照更新（2026-10-01）：同validation交通上的Clean/Random/FGSM/PGD/OARL-BO已补齐；三批新增1,300episode、225,248步审计通过，形成保留全部三档搜索预算的17行效果–成本比较。FGSM的ASR16.09%，当前方法三档为16.48%/16.09%/18.39%；高档相对FGSM的+6条相关转换仅来自checkpoint2的两个交通。相对预算版Zero-One仍是回报更低、碰撞优势小且不稳定、计算无全面优势。详见 [VALIDATION_COMPARISON_RESULTS.md](VALIDATION_COMPARISON_RESULTS.md)。下一步诊断已有失败案例与收益集中性，先形成机制假设；尚不支持稳定安全改进，不启动final split30或以新增消融替代核心比较。以下保留各阶段历史判断。

验证阶段更新（2026-09-25）：`feat/shared-compute-validation` 的九批 4,500 episode 已全部完成，707,039 步审计通过。新交通低档进展/Zero-One 为 82/300 对 79/300 碰撞，但去掉 episode 1 后净差从 +3 变为 −1；中档进展为 81/300，对照 Return/Safety 为 82/300、84/300；高档 87/300 对 85/300 的全部额外碰撞也集中于 episode 1。现有证据不足以支持稳定的安全收益，详见 [SHARED_COMPUTE_VALIDATION_RESULTS.md](SHARED_COMPUTE_VALIDATION_RESULTS.md)。建议先分析已有 episode 11 漏检和 episode 1 三步碰撞，不急于最终测试或按结果临时调参。split 20 现已用于判断；后续方法若依此修改须明确验证驱动开发，final split 30 继续保留。下段保留开发阶段结论。

状态更新（2026-09-24）：P0/P1、v0 的 P2、进展重试三个攻击种子开发复核、预算版 Zero-One 配对对照及三档共享计算预算实验均已完成。本轮七批新增 1,650 episode 审计通过；每个目标合并三个相关攻击 seed 后，100/200 档进展版本与 Zero-One 分别为 41/150、31/150 碰撞，200/400 档为 38/150、39/150，400/800 档为 42/150、40/150。低档配对收益中 7 个重复同一交通 episode；总计算下降受提前终止影响，不能当作独立效率收益。完整成本、回退和反例见 [SHARED_COMPUTE_BUDGET_RESULTS.md](SHARED_COMPUTE_BUDGET_RESULTS.md)。当前分支 `feat/shared-compute-budget-sweep`。建议下一步冻结候选，在 validation split 20 的新交通预登记复核完整三档；验证和最终测试仍未执行。进展规则错过有效后续重启的已有反例、原开发分支和负结果全部保留。以下保留最初设计依据；名称仅为工程代号，不代表已确立论文创新或全面效果优势。基线冻结点为 `de9b10d`。不修改两个上游项目源码、已有攻击配置、冻结 checkpoint、奖励或终止语义。

## 1. 研究问题与可否定假设

研究问题：固定观测扰动与计算预算下，如何将计算分配给能够通过观测扰动实际诱导、且可能导致安全事件的行为序列？

- H1：按实际动作历史组织搜索，可以减少不同目标序列落到同一实际行为的重复探索，在相同预算下覆盖更多不同动作分支。
- H2：这种预算分配改善能够转化成更高的配对新增碰撞率，或以更少梯度/仿真成本达到相同攻击效果。
- H3：收益不只来自把回报目标换成碰撞目标，也不只出现在已高度敏感的 checkpoint 2。

H1 成立不代表 H2 成立。若仅减少重复，但安全效果与成本均没有改善，就不支持新攻击有效性的主张。未找到诱导扰动不等于动作不可达；本方法不提供可达性或最优性证明。

## 2. 证据与已有工作的边界

既有诊断见 `BASELINE_DIAGNOSTIC_RESULTS.md`。seed 2 的 FGSM 局部动作改变率约 90.6%，Zero-One 约 37.8%，但后者报告更多碰撞。Zero-One 的执行目标动作命中率跨 checkpoint 为 33.6%–59.6%。不能把这些访问不同状态的统计直接解释为因果机制。

2026-09-21 对已保存的 `candidate_trace` 作只读检查：在每个规划块内，以完整实际动作序列去重，跨块累计重复比例。checkpoint 0–4 分别为 88.35%、62.65%、37.32%、87.93%、86.45%；候选总数分别为 1820、1550、410、1640、1550。源为 Stage 4 正式批次 `20260914T072528603795Z-attack-benchmark`。完整序列重复会受提前终止影响，需同时记录前缀覆盖和非终止候选统计；现有 oracle/PGD 已有缓存，重复比例不能当作物理查询浪费比例。

创新性尚待核对，以下已有工作必须明确区分：

- [Zero-One](https://stanleybak.com/papers/bak2024iccps.pdf)：目标动作序列搜索、梯度诱导与闭环回报评价。
- [Lin et al., 2017](https://arxiv.org/abs/1703.06748)：攻击时机与目标行为规划。
- [Critical Point Attack](https://ojs.aaai.org/index.php/AAAI/article/view/6047)：预测未来损害并寻找攻击时刻。
- [Provable observation noise robustness](https://www.cambridge.org/core/journals/research-directions-cyber-physical-systems/article/provable-observation-noise-robustness-for-neural-network-control-systems/33C1C135187FF661586451DED1C2A19C)：离散动作轨迹树和最小噪声失败轨迹。
- [Illusory Attacks](https://arxiv.org/abs/2207.10170)：观测攻击的统计可检测性约束。

因此，“动作树 + PGD + 碰撞目标”不能直接声明为新贡献。拟验证的增量是有限预算下诱导尝试与行为分支探索的分配机制；若与已有方法等价或没有实证收益，应明确定位为适配/负结果，而非重命名为创新。

## 3. 第一版威胁模型

- 相同五个 frozen Clean victim、16D 输入、greedy 三离散动作、原 SUMO 场景与 Normal 流量。
- 白盒策略访问和独立 SUMO oracle，与 Zero-One 使用相同仿真权限；不额外访问未来真实评估轨迹。
- 保留共同约束 `abs(delta_i) <= 0.2*abs(observation_i)+0.05`。
- Horizon 20、block 执行，尾块由 episode 剩余步数截断。所有步骤均可攻击，不引入 Gate 或稀疏时机选择。
- PGD 原语沿用 targeted logit margin、每次尝试 2 个梯度步、归一化步长 1。允许后续独立起点重试，但必须计入总梯度/forward 预算。
- 不增加物理裁剪；本版仍是观测空间攻击。离散 lane、缺失标记和时间一致性约束属于后续独立协议，若引入则所有方法需在同一约束下重评。

## 4. 搜索状态与扰动证据

一个节点由 episode 标识和从 episode 起点开始的完整实际动作历史标识。相同 16D 观测不代表同一 SUMO 状态，禁止据此合并节点。

每个节点维护：原始观测、真实 transition、已发现动作分支、各目标的尝试次数、最佳目标 margin、对应扰动、分支访问次数及已完成轨迹得分。每个有效动作分支都须持有通过冻结策略实际确认的扰动证据（witness）。

- 干净动作具有 delta=0 的已知证据，但不自动计为已施加扰动。
- PGD 想诱导 a，实际输出 b 时，记录 b 的有效证据；a 记为“本次未成功”，不把它当作已执行动作，也不证明 a 不可达。
- 同一节点的同一实际动作只对应一个行为子分支。可保留成本更低的证据，但不能把仅扰动不同当作新行为。
- RNG 由独立 attack seed、episode、历史键、目标动作和尝试序号确定，不能因遍历顺序改变而偷偷共用环境随机流。

## 5. 可实现的首版调度规则

先采用明确、可审计的规则，不训练额外调度网络：

1. 每块先构造并计费一条干净策略延续到 horizon/termination 的完整保底轨迹。
2. 从块根节点开始构造候选。优先尝试未尝试过的非干净目标动作，目标之间以当前 logit margin 较大者优先、动作编号作为稳定 tie-break。
3. 若目标失败，保存实际输出动作的 witness，并保留未成功目标。只有后续访问该节点、且已对其其他目标至少尝试一次后，才允许独立起点重试；每个节点/目标最多 3 次尝试。
4. 在已发现的实际分支中，先访问尚未完成过 rollout 的分支；否则选择完成 rollout 次数最少的分支，同次数时用已观测到的最佳轨迹得分排序。这样把计算从反复等价的目标转到已找到的不同实际行为。
5. 执行 witness 推进一步，重复直到 horizon 或真实终止。相同完整动作序列仍会如实记为重复，不因为有缓存就从统计分母中消失。
6. 每块最多 10 条搜索候选完整轨迹，另加 1 条计费保底轨迹；同时受下面的多维计算预算约束。未完成候选不能伪装成更低回报的短完整轨迹。
7. 选择最好完整计划并按 block 执行，每步核对原观测、实际动作及下一 transition。预算不足时保留已完成计划；如果保底计划也无法构造，明确记录预算耗尽并退回干净动作，不能超预算补算或报告成功。

这是首版可否定的搜索规则，不是已经证明有效的最优调度算法。若规则无法充分探索延迟风险，再依据开发集证据修订；修订须独立提交并重新冻结协议。

## 6. 目标函数与公平控制

定义两个明确目标，不把环境 reward 改写成新 reward：

- Return：最小化 horizon 内原始累计 reward，沿用 Zero-One 目标，用于先隔离搜索机制。
- Safety：按字典序最大化 `(是否发生 ego SUMO 碰撞事件, -原始累计 reward)`。即碰撞优先，同类候选用原回报打破平局。不添加未定义的 TTC 缺失惩罚或声称能测量横向碰撞几何。

Safety 的碰撞事件仍包含 SUMO 原配置的 minGap 语义，不声称物理接触。它较稀疏，第一版可能效果有限；这是需测试的限制。增加横向连续风险信号必须作为后续有独立定义和测量验证的修改。

最低必要对照是 2×2：

| 搜索 | Return | Safety |
| --- | --- | --- |
| Zero-One 目标序列搜索 | 新预算控制版本 | 同安全目标控制版本 |
| Ours-v0 实际行为分支搜索 | 隔离搜索机制效果 | 候选新方法 |

四个条件共享预算执行器、保底策略、oracle、PGD 原语、缓存语义、模型和 seed 配对。原 Zero-One 及其已发布结果保持原样；新控制版本以新名称、新配置、新结果记录，不覆盖旧 baseline。Safety 目标或新增保底 rollout 的成本必须同样用于控制版本。

已有 Clean/Random/FGSM/PGD/OARL-BO 可提供整体参考；与无 oracle 方法的比较不宣称等信息权限。

## 7. 第一版预算与严格计数

以下为开发起点，不是从新测试结果调出的最优值。每个 20-step block 上限：

| 资源 | 上限 | 说明 |
| --- | ---: | --- |
| 输入梯度计算 | 400 | 对齐现有 10×20×2 的无缓存 PGD 数量级；每次重试计费 |
| 策略 forward | 800 | 所有规划、目标排序、验证与保底调用均计费 |
| 新 shadow transition | 200 | 缓存命中不计为新的仿真执行，但单独记录 |
| 全部 shadow step | 4000 | 包括重置后的历史 replay，不把 replay 隐藏为免费 |
| 搜索完整候选 | 10 | 保底轨迹额外 1 条，但计入全部计算上限 |

尾块的新 transition 和梯度预算按剩余 horizon 比例向下缩放，至少足够构造保底或明确退出；统一执行器实现相同规则。4000 的 replay 上限先保持每块固定，同时记录实际成本。reset 次数、IPC、缓存命中和 wall time 另报，不能把上述多维预算压成一种“等查询次数”。

预算检查必须在实际调用前进行。oracle 的 step 可能包含多个 replay step，所以应提供只读成本预估或原子预算预留，在 worker 内拒绝会超额的请求。完整计划的剩余执行确认成本也要预留。公共 evaluator 的诊断 forward 单独记录并保持四条件一致。

## 8. 数据与 seed 划分

现有已反复查看的交通结果只作为探索证据，不再充当新方法的未见测试集。

- 保留五个 frozen checkpoint，明确训练 seed 与评估交通 stream 分离。
- 新开发/验证/最终测试交通 stream 分别编号 10/20/30，每个 stream 初始设置 10/20/50 episodes，各模型共用同一组交通。
- 建议明确种子生成式：`SeedSequence([20260921, 1, split_id, episode]).generate_state(1)[0] % (2**31-1)`；生成清单并检查各 split 之间、与旧训练和评估 seed 的实际交集，发现碰撞即在运行前解决。
- 独立 attack seeds 0/1/2；开发 smoke 可先用 0，验证与最终比较全部覆盖。Clean 控制对同一模型/交通只需运行一次，不能因重复使用而扩大独立样本量。
- 最终测试配置与 seed 清单先冻结并保存 hash；参数选择完成前不读取其输出。测试后若修改方法，原测试转为探索证据，需要另设未见测试。

这些是新评估配置，不修改原 Protocol A 记录。前五个模型仍不是无限总体样本，不以 episode 数量代替模型间不确定性分析。

## 9. 验收与开发顺序

### P0：协议、预算与候选级审计

实现严格多维计数器；补齐实际分支数、不同目标数、重复轨迹、失败目标重试、新 transition、replay/缓存等统计。先验证重复来自何种情况，区分真实不可改变行为与当前内层优化未找到扰动。

### P1：最小 Ours-return

新增搜索器、witness 存储与回放适配，先使用原 Return 目标。验证解析小模型中已知动作可诱导、失败目标不误判不可达、相同观测不同历史不合并、输入不被修改、模型冻结和全预算不超额。再跑两个完整 SUMO smoke episodes，核对每个实际 transition。

### P2：Safety 与同目标控制

增加目标策略对象及四条件开发集比较，先验证碰撞事件、正常终止和时间截断区分正确。所有新控制记录独立名字与 commit。不要只报告 Ours-safety 对原 Zero-One-return 的结果。

### P3：验证与锁定

在验证集确定是否支持 H1/H2/H3。覆盖五个 checkpoint，尤其 checkpoint 0/1/3/4；checkpoint 2 原 Zero-One 碰撞率已达 100%，不把它作为唯一开发目标。参数冻结后一次性执行最终配对测试。

主指标为 collision-conversion ASR（配对 Clean 未碰撞的 episode 为分母），同时报告原始碰撞率、Return/Drop、每模型结果、已验证成本、首次分歧和终止。时间变化的攻击率同时报告 eligible/attempted/applied/changed，避免把干净 witness 或预算退出伪装成攻击。

只有主方法建立可重复收益后，才做其新增机制的模块消融（例如分支归并、重试分配），不提前堆叠新模块。收益若仅来自改目标、额外成本或单一 checkpoint，应收窄或否定方法主张。最终保留可复现实验与负结果，不为得到正结果修改测试协议。

## 10. 预期模块与 Git 交付

- `rl_att/attacks/proposed.py`：BaseAttack 适配与计划执行。
- `rl_att/attacks/behavior_search.py`：历史节点、witness、调度与候选选择。
- `rl_att/attacks/search_budget.py`：统一预算预留、结算、耗尽状态。
- `rl_att/attacks/rollout_objectives.py`：Return/Safety 的明确排序。
- `rl_att/attacks/zero_one_controls.py`：新协议下的对照入口，不覆盖原实现。
- oracle 扩展：计费预留、预算拒绝、干净保底 witness 的验证路径；旧接口默认行为保持一致。
- `configs/evaluation/proposed_*`、针对性测试与方法/结果文档。

以上模块规划现已按实现文档落地。每个单一职责改动独立 Conventional Commit，必要的兼容修改单独提交。保持功能分支，阶段功能与实验完成后再 merge main。达到实际稳定复现里程碑才创建 `v0.5.0-proposed`；设计文档或 smoke 不构成算法有效性里程碑。
