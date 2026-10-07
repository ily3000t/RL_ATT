# ACoE → SUMO 离散 actor–critic：D1 目标映射

2026-10-05。第一候选仍为 ACoE，原 OARL Robust 仍是独立领域强基线。本文核对研究目标与实现边界，没有新增防御训练器或确定自己的最终损失。已完成逆可行集合实现与数学测试；候选奖励的来源、off-policy 误差递推和母方法版本仍待验证，尚不能启动 ACoE 全量训练。

## 来源与实际核查范围

论文固定为 [ACoE v4](https://arxiv.org/html/2406.04724v4)，作者源码固定为 [a6897fe504650f94f2fef0d013ff925f0a838762](https://github.com/romanbelaire/acoe-robust-rl/tree/a6897fe504650f94f2fef0d013ff925f0a838762)。已读取 `steps.py`、`agent.py`、`pgd_act.py`、依赖与 MIT 许可，参考副本留在 ignored `.local/defense-review-20261005/`。未导入或执行作者工程；Atari DQN 的完整作者实现尚未审计。

论文定义观测不确定性的 belief、即时奖励差及其累计量，联合正常收益优化；PPO 将累计误差信号用于采样动作的优势，附录 DQN 使用动作相关误差值。A2B 依策略 KL 构造权重，A3B 再用近似最坏扰动的 KL 作参照。这里只保留机制定位，不继承其理论结论；本任务的 16D 观测、扰动盒与训练器均有适配差异。[正文第 3–4 节及附录 B](https://arxiv.org/html/2406.04724v4)

作者当前训练代码还存在邻域采样、批次权重、固定 eps、离散 shape 与旧 Torch API 适配问题；它们是共同工程修复，不是自己的贡献。[采样与误差更新](https://github.com/romanbelaire/acoe-robust-rl/blob/a6897fe504650f94f2fef0d013ff925f0a838762/policy_gradients/steps.py)、[网络创建及策略更新](https://github.com/romanbelaire/acoe-robust-rl/blob/a6897fe504650f94f2fef0d013ff925f0a838762/policy_gradients/agent.py)

## 1. 本项目已有的量与信息边界

区分 `z`（完整 SUMO state）、`o=h(z)`（真实但不完整的 16D 观测）、`y`（传给 actor 的观测）和 `x`（防御推断的候选原观测）。无攻击时 `y=o`。候选 `x` 不是一个完整 `z`，且数值合法不保证可还原成物理交通状态。

当前 Clean 训练目标在 `rl_att/agents/clean_victim.py` 中为：

```text
Qmin(o,a) = min(Q1(o,a), Q2(o,a))
reward_target = r + gamma*(1-done)*sum_a pi(o_next,a)*Qmin_target(o_next,a)
actor_loss = -mean_o sum_a pi(o,a)*stopgrad(Qmin(o,a))
```

这是带双 Q 和目标网络的离散 actor–critic，没有 SAC 熵项、PPO ratio/GAE 或 DQN 的纯 Q argmax 控制器。actor、Q1、Q2 的形状均为 `[B,16] → [B,3]`。保持该任务与结构，不自动意味着作者 PPO/DQN 更新等价。

原 OARL 的额外项是 BO 寻找当前与 next 观测的 JS 差、双变量更新及对应 actor/critic 约束。OARL 对照保持原样。论文适配母方法默认从 Clean RL 目标出发；若保留 OARL dual/BO，则另命名混合方法，并保留独立控制，不能默默把母方法改成 OARL 加模块。

| 量 | 已有来源 | 对适配的限制 |
| --- | --- | --- |
| 实际 `r_t` | 环境执行真实动作之后返回 | 与实际访问状态和动作对应，不是任意候选 `x` 的奖励 |
| `pi(y)` | 原 actor | 部署只接收 `y`，不接收原观测、真实 state 或攻击标记 |
| reward Q | 实际 replay 的累计收益估计 | 不是直接可调用的即刻奖励，也不是任意候选观测的真实模型 |
| replay | 当前仅存原观测、动作、奖励、next、done | 没有行为概率、完整攻击输入、训练策略版本或片段标记 |
| checkpoint | 部署 actor-only | 不能恢复原 critic、replay 或优化器，也不能读取“原训练的误差 Q” |
| SUMO oracle | 已登记真实动作历史的独立仿真 | 可以研究同一真实 state 的动作后果；不能直接为任意 `x` 生成唯一 latent state |

## 2. 逆可行集合已明确且可独立验证

正向攻击满足 `|y-x| <= rho*|x|+beta`，当前 `rho=.2, beta=.05`。对 `0<=rho<1, beta>=0`，一维逆集合连续；两个带符号可行区间的并可写为：

```text
lo(y) = (y-beta)/(1+rho),  y>=beta
        (y-beta)/(1-rho),  y< beta
hi(y) = (y+beta)/(1+rho),  y<=-beta
        (y+beta)/(1-rho),  y> -beta
```

逐坐标笛卡尔积就是数值逆盒。`y=1` 给出 `[.7916667,1.3125]`，`y=0` 给出 `[-.0625,.0625]`。以 `y` 为中心的 `.2|y|+.05` 盒不是同一个集合。该推导针对本项目既有 threat model，是工程与协议正确性，不算新增防御机制。

实现为 `rl_att/utils/observation_uncertainty.py`：

- `inverse_observation_box()` 返回 float64 的逐特征上下界，支持正负、跨零和批次；拒绝非有限输入与 `rho>=1`。
- `sample_inverse_observations()` 需要调用者显式提供局部 NumPy RNG，返回 `[...,K,D]` 均匀候选；零值特征也可向正负两个方向采样，不消费全局 NumPy 随机流。
- 函数不生成 posterior、A2B/A3B 权重或物理 state，也没有接入原训练/攻击路径。不裁剪 lane/sentinel 或修改既有攻击盒。

性质测试用 2,000×16 个随机原观测、边界及内部正向扰动核对原观测必在逆盒中；另验证随机候选满足正向约束、边界紧致、形状及 RNG 隔离。这里的 RNG 是数学测试的局部流，不新增任何实验交通或防御模型。

## 3. actor 梯度路径是适配的必要条件

若冻结误差网络后，仅在现有 actor loss 上增加 `lambda*delta(y)`，且 `delta(y)` 与 actor 参数无关，这项对 actor 没有梯度。把同一个标量广播给三动作也无效：`sum_a pi(y,a)*delta(y)=delta(y)`。已用三动作 softmax 的反例测试验证这一点。

作者 PPO 的 sampled-action ratio 可以把轨迹误差信号转为策略梯度；现有全动作期望损失不具备同一机制。若选择保持当前训练器，一个待核对的动作相关代价设计为：

```text
D_psi(y) -> [B,3]，每个动作一项累计误差估计
D_target = c_t + gamma*(1-done)*sum_a pi(y_next,a)*D_target_net(y_next,a)
L_actor_candidate = L_clean + lambda*mean_y sum_a pi(y,a)*stopgrad(D_psi(y,a))
L_D_candidate = MSE(D_psi(y,a_executed), stopgrad(D_target))
```

这是**本项目待验证的策略评估式适配候选，不是作者原算法原样复制**，目前未实现。`c_t` 来源尚缺；该期望 bootstrap 也不同于分别最小化代价的控制 backup，需要在正式版本固定选择。若采用 cost-min 的独立 backup，而部署用 reward/cost 联合 actor，须解释其策略不一致误差。不能把 reward critic 的常量缩放或动作无关误差当成已经实现累计反事实机制。

另一选择是显式 sampled-action 梯度与行为概率修正，但这需要 replay 的行为策略概率、版本和序列目标；当前 buffer 不具备这些记录。更换完整 PPO 训练器则属于路线 C，不能作为路线 B 的隐含兼容修复。

对误差网络训练，母方法与自己的版本必须采用相同 belief、stop-gradient、目标网络及数值稳定规则。误差信号可能为负；不能无依据截为非负。训练 actor 保留梯度，不能交给会冻结参数的 `VictimAdapter` 原地训练。

## 4. 候选奖励与 belief 尚需解决

需要的即时量是“同一个动作在观测代理与 belief 候选上的奖励差”。当前仅有真实执行奖励。可行方案的证据权限不同：

| 候选来源 | 可以得到什么 | 必须验证 | 当前状态 |
| --- | --- | --- | --- |
| 额外条件奖励模型 `rhat(o,a)` | replay 分布下的即时奖励近似 | 留出真实转移误差、候选 OOD、缺失动作覆盖与辅助训练成本 | 未实现，不能称真实 `R(x,a)` |
| reward Q 的 TD 代理 | 当前 critic 条件下的近似量 | discount/terminal 一致性；任意 `x` 不能共用真实 next 而当成其转移 | 尚未选定，不直接照抄作者 shape 运算 |
| 合法 SUMO 控制分支 | 同一真实状态下不同实际动作的后果 | 同 seed/prefix、真实可行状态、分支回放、额外仿真权限与成本 | [控制性诊断](DEFENSE_CONTROLLED_REPLAY_RESULTS.md) 已完成；不提供任意数值 `x` 的 latent reward |

A2B 可以先作为低复杂度 sanity check；条件 belief 要逐观测归一化，作者批次权重单独记账。A3B 的 PGD 参照必须作用于实际三动作分布，在候选原观测的正向合法盒内搜索，固定 KL 方向、零分母、采样和梯度成本。数值修复与采样协议共同用于母方法和自己的版本，不只给新方法更有利的输入。

观测不完整使这些网络估计依赖训练访问分布；重放独立候选不等于知道真实 Bayesian belief。本项目 reward 还可能为负，episode 可提前结束。为满足论文分析前提而平移 reward，会因 episode 长度不同改变优化目标；本轮保持原 reward，最终只报告经验证的经验结论。

## 5. 通过、待验证与下一步

| 项目 | 判定 |
| --- | --- |
| SUMO/16D/三动作与原 actor 保留 | 明确，未修改原路径 |
| 原 OARL 强基线及论文母方法分开 | 明确，现有模型保持冻结 |
| 逆可行集合、显式候选 RNG、梯度反例 | 已实现并通过 7 项冻结环境测试 |
| paper/作者代码 → 当前 shape 与梯度差异 | 已形成映射；不是完成原论文复现 |
| 即时误差的来源和 off-policy 目标 | 待验证，不能伪造候选 reward 或默认代理正确 |
| 行为后果、正常输入恢复与交通集中性解释 | 控制性分支已审计；即时失败与持续损害分开，尚未确证恢复模块 |
| 基础训练 snapshot、恢复与成本支撑 | 22 项相关测试、96 次 SUMO episode 通过；仅 episode 边界，未训练母方法 |
| 独立 PGD 一致性基础比较 | 35 项相关测试、144 次工程 episode 通过；未完成长程或测得防御收益，不选为母算法 |
| 新防御有效性与创新 | 未训练、未测量，不作结论 |

2026-10-07 补充：D1 控制性分支已完成，见 [结果](DEFENSE_CONTROLLED_REPLAY_RESULTS.md)，原输入/权重/算法保持不变。D2 的基础策略快照、独立进程训练恢复和成本支撑已通过 [工程验证](DEFENSE_TRAINING_SUPPORT_RESULTS.md)，独立 [PGD 一致性基础对照](DEFENSE_PGD_CONSISTENCY_RESULTS.md) 的实现和短程审计也已通过；两者没有解决即时反事实误差来源、belief 和 off-policy 更新选择。PGD 对照长程、正式模型冻结登记和新交通 namespace 仍待补齐。ACoE 母方法仍需上述数学目标明确后再开始，基础支撑与比较方法不能替代数学核对。自己的创新模块继续后置，并在同样的工程适配与辅助数据预算下与母方法比较。
