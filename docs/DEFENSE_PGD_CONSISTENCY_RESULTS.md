# PGD 一致性基础防御：工程验证结果

2026-10-07。独立 PGD 一致性基线实现完成，35 项相关测试与 144 次真实 SUMO 短程 episode 通过。启用正则化后的独立重复、连续训练和独立进程恢复，逐 episode 轨迹及完整数值训练状态一致。原 Clean/OARL 循环兼容性保持。

这说明实现和恢复正确；尚未完成长程收敛、冻结模型攻击评估或防御有效性验证。方法是通用基础对照，不是自己的防御，也不是完整 SA-PPO 复现。OARL Robust 保留为领域强基线。

目标、stop-gradient、扰动搜索和版本适配见 [协议](DEFENSE_PGD_CONSISTENCY_PROTOCOL.md)，完整计数和指纹见 [结果 JSON](DEFENSE_PGD_CONSISTENCY_RESULTS.json)。

## 实现与验证

独立 `PGDConsistencyAgent` 保留 Clean 的双 critic/target、replay、batch action sampling、RL 目标和更新节奏，在同一次 actor 更新加入 `0.1 * KL(stopgrad(clean policy) || perturbed policy)`。PGD 在正常 replay observation 的 `0.2|o|+0.05` 盒内执行五步、归一化 step size 0.4、一个隔离辅助随机起点，逐样本保留最佳合法候选（含零扰动）。部署只使用原 actor。

测试核对：零系数/零预算时完整更新精确退化到 Clean；单次 enabled 更新的 critic/target 和基础 RNG 与 Clean 相同；输入梯度不污染参数梯度；固定 KL 方向及干净目标 stop-gradient；满 FIFO/非空 Adam/辅助搜索 RNG 恢复；扰动、原始 KL/总损失、实际网络和 optimizer 成本。8 项新算法测试、5 项新审计测试、14 项 D2 测试、5 项 seed、1 项原 Clean 和 2 项 checkpoint 测试通过。

12 个 worker 全部成功。三个方法各完成 16-episode 独立 reference、16-episode 连续训练、13-episode prefix 和另进程 3-episode resumed，总计 **144 次物理 episode，13,617 次实际 SUMO step**，包含 reset warmup。PGD 的 reference 明确是独立从头重复；只有 Clean/OARL reference 使用原 main.py。

完整状态比较涵盖所有网络、梯度、Adam、FIFO、global/policy RNG、训练进度，以及 PGD 配置、搜索 RNG 和累计计数。PGD prefix/boundary 相同，最终重复/连续/恢复相同。Clean/OARL 最终状态摘要与先前 D2 验证完全相同，表明新方法接入没有改变原训练路径。

## 实际成本与原始记录

以下各列是一个连续 16-episode 逻辑路径，不能将它们当成已收敛模型的训练成本排名。

| 指标 | Clean | OARL Robust | PGD 一致性 |
| --- | ---: | ---: | ---: |
| 真实交互步 / warmup 步 | 199 / 1,314 | 199 / 1,314 | 199 / 1,314 |
| 主更新 | 35 | 35 | 35 |
| 训练 actor forward 次数 / observation 行数 | 70 / 8,960 | 420 / 53,760 | 315 / 40,320 |
| 每个 reward critic forward 次数 | 70 | 70 | 70 |
| 每个 target critic forward 次数 | 35 | 35 | 35 |
| actor、qf1、qf2 各 optimizer.step | 35 | 35 | 35 |
| dual optimizer.step | 0 | 35 | 0 |
| BO objective 调用 | 0 | 175 | 0 |
| PGD 输入梯度调用 | 0 | 0 | 175 |
| PGD 搜索 actor forward / 行数 | 0 | 0 | 210 / 26,880 |
| 一致性 outer 扰动 actor forward / 行数 | 0 | 0 | 35 / 4,480 |
| 被攻击真实交互、额外 optimizer、特权仿真转移 | 0 | 0 | 0 |

PGD 的两项额外 forward 已包含在训练 actor 总次数中，不能再次加到总量。这里没有被攻击 SUMO rollout；梯度搜索仅对 sampled observation 进行，不把数值候选当成新的仿真转移。原双 critic/target forward 的输入行数、所有 phase wall time 和 segment 计数见 JSON。

独立 auditor 从 `defense_updates.jsonl` 的观测和更新前概率用 NumPy 重算边界、KL、总 actor loss，核对实际 hooks 与完整 snapshot。35 次连续主更新的平均 batch KL 范围为 **0.001249–0.001682**；这是正则项输入审计，不是防御强度或效果指标。最大数值盒超量约 **4.77e-8**，在预登记 2e-6 浮点容差内，源于 float32 运算。该容差不是放宽 threat model 的实验参数。

独立进程恢复只执行 suffix 的 19 次更新，prefix 已执行 16 次；实际批次成本按 segment 加总。新增模型全部 `engineering_only=true`、`benchmark_eligible=false`。旧模型注册表运行前后哈希相同，没有替换旧 victim。

## 执行身份与限制

源提交为 `3ae20dbadac79d32597d3491c64a3c686adeb9d1`，实现提交为 `7c42db4`。协议、原始记录器及 auditor 已提交后才启动，批次期间源码提交和 tracked tree 固定。原始数据在 `.local/runs/20261007-defense-pgd-consistency`，运行时间为 UTC 07:23:40.528671–07:24:31.824804，父批次约 51.30 秒，包含进程、保存和审计。

配置 SHA256 为 `07052e51427b8628fab28be88bf7069574f2d58652f8578b87092c987f515306`，audit SHA256 为 `2894c9e9833aca73c7c226a9404090c68e32224d0fbc917a313005c0ccd6391e`；父 manifest、完整状态、actor 权重和文件指纹见 JSON。完成后已只读重新审计，结果与保存报告一致，没有再次运行 SUMO。

使用已有 Windows / Python 3.7.16 / Torch 1.3.1+cpu 单线程 / NumPy 1.21.6 / SUMO 1.22.0 冻结环境。基础 run seed 为 0，SUMO 沿用逐 episode 派生训练名单；PGD search 使用独立辅助 attack seed 1500372476，其角色种子、完整 `pip freeze --all` 和启动命令保留在 manifest。

当前仅一个训练 seed、每集最多 16 步。144 次执行是兼容性重复，不能当作 144 个独立交通样本，不能据此报告碰撞优势或论文显著性。也不保证跨版本或 episode 内恢复、物理可行性、PGD 全局最优或认证鲁棒性。

λ=0.1 是初始配置，尚未检验长程收益和干净性能取舍；工程审计得到非零 KL 不等于防御有效。下一项应独立登记五 seed 的 400-episode 训练、训练成本、正式新模型冻结与可复用评估入口，再针对实际冻结策略重新生成攻击。新的防御交通 namespace 继续独立实现，旧 split 30 不用于选择 λ 或版本。ACoE 母方法的误差来源、belief 和 off-policy 目标仍待明确，自己的防御创新模块保持后置。
