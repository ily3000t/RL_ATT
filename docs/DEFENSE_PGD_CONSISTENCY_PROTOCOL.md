# 独立 PGD 一致性防御基线

2026-10-07。本功能在 `codex/defense-pgd-consistency` 开发，从 `777b870` 的本地 main 创建。它属于通用基础比较方法，不是自己的防御，也不是 ACoE 母方法。原 OARL Robust 保留为领域强基线，原训练入口和冻结权重不修改。

## 目标与适配边界

参考 [SA-MDP 论文](https://arxiv.org/abs/2003.08938) 的策略鲁棒正则化思想，[作者代码入口](https://github.com/chenhongge/StateAdvDRL) 供核查。这里独立实现当前三动作 off-policy actor–critic 的一致性对照，没有导入作者项目，不称为 SA-PPO/SA-DDPG 的完整复现，不继承认证或论文效果结论。

基础仍为 Clean：原双 critic、双 target、FIFO、Categorical batch 采样、actor/critic 学习率、折扣和 Polyak 更新保留。训练交互使用正常 observation；PGD 仅在 sampled replay 的当前观测 batch 上搜索，不向 SUMO 注入训练攻击，不生成特权仿真转移，也不修改 critic backup。

对 batch 中 `o`，使用三动作概率 `p=stopgrad(pi_theta(o))`，预先固定 KL 方向：

```text
r_i(o) = epsilon * (0.2 * |o_i| + 0.05)
u* ≈ argmax_{u in [-1,1]^16} KL(p || pi_theta(o + r(o) * u))
L_actor = mean[-sum_a pi_theta(a|o) * min(Q1,Q2)(o,a)]
          + lambda * mean[KL(p || pi_theta(o + r(o)*stopgrad(u*)))]
```

干净目标在一致性项内 stop-gradient，原 RL 项保留干净策略梯度。search 使用输入 `autograd.grad`，不向参数 `.grad` 累加；outer loss 正常更新 actor。没有 entropy、BO、dual 更新、额外网络、Gate 或部署动作覆盖。部署仍是原 `16→128→3` actor。

PGD 逐样本独立搜索，扰动盒始终锚定干净 `o`，不是随迭代位置重算半径。一个独立辅助 attack NumPy 流生成 uniform random start；在归一化 `u` 空间执行 5 步 sign ascent，每步 0.4，投影 `[-1,1]`。逐样本保留随机起点、迭代点与最终点中的最好结果，并包含零扰动候选；平局保留零候选。不是精确全局最大化，不保证每个 observation 都改变动作。

为避免概率下溢，p/q 先分别 clamp 到 `1e-8`、再按行归一化；浮点 KL 的微小负值在外层按零处理。λ 初始为 0.1、epsilon 为 1、乘法/加法界为 0.2/0.05；当前只是预登记的工程配置，没有在验证或保留交通上选参。λ=0、epsilon=0 或完全零半径时直接调用原 Clean 更新，不执行额外 forward、backward 或随机采样。

## 记录和成本

每个 enabled 主更新含原 actor 两次 batch forward、PGD 六次 forward/五次输入梯度、outer 扰动 actor 一次 forward，共 **九次 actor forward**。batch 128 对应 1,152 行输入，不能记成九个 observation。candidate_observations 记录实际搜索的六个 batch、共 768 行，包含最终候选；零候选复用干净概率，不增加 forward。

actor、两个 critic 各执行一次原 optimizer.step，dual 和额外网络更新为零。真实交互、warmup、训练 wall time 与全部实际 network hooks 继续使用 D2 账本。搜索计数是主更新的子项，不与包含搜索的 actor 总次数再次相加。辅助更新只指额外 optimizer.step，不把 PGD 输入梯度误记为辅助网络训练。

完整 snapshot 保存防御配置、累计搜索计数、最后更新摘要及辅助 attack RNG，连同已有全部网络/梯度/Adam/replay/RNG。episode 边界独立进程恢复；部署 actor 产物明示防御配置与训练计数。旧 frozen registry 不自动增补；工程模型不用于正式比较。

新增 ignored `defense_updates.jsonl` 逐主更新记录干净/扰动观测、对应更新前策略概率、KL/actor loss 和实际搜索计数。独立 NumPy auditor 重算扰动盒、KL、总损失、行数及最后摘要，核对真实 actor hooks、optimizer 更新和 snapshot 中搜索 RNG。它不声称验证 PGD 的全局最优解或物理可行性；这里仍沿用既有数值观测 threat model。

## 短程工程验收

配置为 `configs/research/defense_pgd_consistency_smoke.json`。三个训练方法 Clean/OARL/PGD 各用 run seed 0、16 episodes × 最多 16 步、原 batch 128 和 buffer 1,000,000；完成 episode 13 后 split。训练更新仍为原零基 episode >10、累计真实交互每两步一次。

四路径为 reference、continuous、prefix(13)、resumed(3)。Clean/OARL reference 执行原 main.py，PGD 的 reference **明确指另一个进程从头独立重复**，不冒充上游实现。PGD 验证 enabled repeat/continuous/resumed 全部数值状态与轨迹一致，prefix 与连续 boundary 一致。

总计 **144 个物理 episode、12 个 worker**。这是训练 seed 上的工程重复，不是独立交通统计、长程收敛或防御效果评估，不使用旧 split 30 或新保留测试。代码、配置和 auditor 先提交，批次期间 tracked tree/源码提交固定，沿用冻结 Python 3.7.16 / Torch 1.3.1+cpu / SUMO 1.22.0，所有大文件留在 ignored `.local/`。

```powershell
& .local/envs/oarl-legacy/python.exe scripts/validate_training_support.py --config configs/research/defense_pgd_consistency_smoke.json --output .local/runs/20261007-defense-pgd-consistency
```

已有 D2 默认配置和启动方式继续支持。运行后只读 `rl_att.training.support_audit.audit_run(manifest_path)` 可重新审计，不重跑 SUMO。通过相关测试、短程审计后 `--no-ff` 合并本地 main，不创建里程碑 tag，不启动未登记的 400-episode 训练。

后续正式长程训练须独立登记五训练 seed、完整模型产物和训练成本，并针对每个冻结策略重新生成所有测试攻击。ACoE 数学目标和新的防御交通协议仍独立推进；本基线接入不决定自己的母算法或创新模块。
