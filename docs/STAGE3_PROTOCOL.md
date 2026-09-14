# Stage 3：预先固定的基础攻击 benchmark

本轮仅评估 `configs/frozen_victims.json` 中 **Clean Victim 的 5 个 episode-400 checkpoint**。每个 run_seed 的所有攻击使用同一个文件/权重哈希，禁止训练和替换模型。保留 OARL Robust 作为已冻结模型，本轮不扩大到额外 victim，以免混淆基线比较。

配置在完整 benchmark 启动前提交。每个 seed 0–4 使用已有 Protocol A 的 20 个留出 SUMO episode，每集最多 200 steps，greedy argmax，Normal 交通，SUMO 1.22.0 和原训练依赖不变。总计划为 **5 seeds × 5 conditions × 20 episodes = 500 episodes**，每个攻击条件最多 20,000 个交互 step。无 Gate，无状态/时间稀疏选择，每个实际交互 step 都尝试攻击。

## 共同预算与比较范围

以原归一化 clean observation s 为基准，定义固定于该 step 的特征尺度：

`w_i(s) = 0.2 * abs(s_i) + 0.05`

统一约束为 `max_i abs(delta_i) / w_i(s) <= epsilon`，本轮 **epsilon=1**。日志中的 `scaled_linf_max` 为左侧数值，原单位的 L∞、L2 也分别保存。这是 **observation_scaled_linf**，不是常数 epsilon=0.2 或 0.25 的普通 L∞ 球。

原 BO 的 float32 计算可能带来舍入误差；实现的逐特征边界检查容差为 2e-7（原归一化 observation 单位），因此 scaled L∞ 可能略高于 1。该容差只用于验证已有浮点运算，不用于扩大搜索参数或添加扰动。

选择这一 envelope 是因为原 OARL-BO 的 `u1∈[0.8,1.2], u2∈[-0.05,0.05]` 必然满足它，因而无需裁剪、投影或更改原 BO。BO 搜索的是这一区域中的共享 u1/u2 仿射子集；Random/FGSM/PGD 可独立改变 16 个分量，搜索空间更大。这是共同上界下的算法基线比较，不能解释为相同搜索空间或相同计算量的优化器消融。

所有方法均不添加物理裁剪、合法 lane 修复、缺失邻车特征屏蔽；这些约束会改变威胁模型，应作为后续独立实验。本轮同时记录该限制，不将扰动称为物理可行。

## 攻击定义（启动前固定）

| 条件 | 目标 / 更新 | 步数与成本（每个实施攻击 step） |
| --- | --- | --- |
| Clean (`none`) | 完整保留 observation | 0 次攻击查询 |
| Random (`random`) | 每维独立 uniform(-epsilon,epsilon)，再乘 w_i | 0 次 policy query |
| FGSM-margin (`fgsm`) | 固定原干净动作 y，最大化 `max_{j!=y} z_j(s_adv)-z_y(s_adv)`；单次 sign gradient | 1 梯度，2 forward，2 objective（含最终诊断） |
| PGD-margin (`pgd`) | 同上；一次均匀随机起点，固定 y，在同一个 clean-state envelope 内逐步投影，返回最终 iterate | 10 梯度，12 forward，11 objective；归一化 step_size=0.2，1 restart |
| OARL-BO (`oarl_bo`) | 沿用 Stage 2 在线适配：obs2=obs1，最大化 2×JS | 5 objective，11 forward，0 输入梯度；原 UCB、参数范围、重复点与 float64 JS 处理不变 |

z 是原 Actor 的 softmax 前 logits，经临时 forward hook 读取现有 pi 层，不修改 Actor.forward、参数或 checkpoint。FGSM/PGD 使用 **logit-margin**，不是未注明的 cross-entropy 默认实现，也不是 C&W 优化器。在策略 softmax 已饱和的情况下，该目标可避免仅由 softmax 饱和造成的微小/零梯度；真正为零的输入梯度仍如实记录。解析测试验证梯度方向、动作翻转和饱和情形。本轮不以观察到的 benchmark 成绩调换目标或调参。

方法参考：[FGSM](https://arxiv.org/abs/1412.6572)、[PGD](https://arxiv.org/abs/1706.06083)、[logit-based adversarial objectives](https://arxiv.org/abs/1608.04644)。这是这些一阶攻击在已声明加权 box 和 policy-action 目标上的实现，不声称复现论文中的图像分类设置。

## 随机流

run_seed、policy_seed、python_seed、numpy_seed、torch_seed、sumo_seed、attack_seed 的基值沿用 Protocol A。SUMO 训练/评估分相派生规则不变；greedy policy 不消费采样 RNG。

Random 和 PGD 的每 step RNG 为 `RandomState(SeedSequence([attack_seed,3,episode_index,step_index]).generate_state(1)[0])`，索引从 0 开始。两种方法使用相同均匀起点序列，但输出由其方法决定；局部 RNG 不推进全局 NumPy/Torch RNG，也不因之前 episode 提前终止而移位。FGSM 无随机过程。BO 保留每次攻击用 attack_seed 重建优化器及隔离被丢弃的 Torch 采样。全部有效角色 seed 与每 step 的派生 seed 在日志中记录。

## 验证、输出与限制

先运行 seed 0 的 2×20-step smoke，再运行全部五组；完整每组首先运行 NoAttack，并与旧 checkpoint 验证的 20 episodes 逐轨迹核对。每个攻击前后检查 frozen checkpoint 哈希。逐 step 验证实际扰动上界；PGD 的每一步投影另有单元测试。全局 RNG 隔离、频率、零预算、日志成本与输入梯度也有测试。

记录 Episode Return、配对 Return Drop、SUMO Collision Rate、Minimum TTC、TTC p05、DRAC p95、ASR、Attack Rate、实际 L∞/L2、scaled L∞、objective/forward/gradient 次数和 wall time。统一评估中的两个动作诊断 forward 不计入 attack_cost；攻击返回数据的轻量校验也不计入计时。跨 seed 使用五个 run-level 指标的均值与样本标准差，并保留每个 seed，不把 step 当独立 trial。

沿用 [安全指标口径](SAFETY_METRICS.md)：collision 是原 SUMO 配置报告的事件，可能包含 minGap 违规，不能直接当物理车身接触；ASR 是配对无攻击未碰撞 episode 中攻击后出现该事件的比例。纵向 TTC/DRAC 仅涵盖 post-step 同 lane 前后 pair；无有效样本为 null，分位数伴随有效样本数。它们不能覆盖横向或 reset 内全部安全风险。

```powershell
python scripts/evaluate_attacks.py --config configs/evaluation/benchmark_stage3_smoke.json
python scripts/evaluate_attack_batch.py --configs configs/evaluation/benchmark_stage3_seed0.json configs/evaluation/benchmark_stage3_seed1.json configs/evaluation/benchmark_stage3_seed2.json configs/evaluation/benchmark_stage3_seed3.json configs/evaluation/benchmark_stage3_seed4.json --jobs 5
```

每组独立进程和 SUMO 文件快照；所有源码、配置和 SHA 在启动前固定。原始数据、checkpoint、环境和日志均留在 `.local/`；Git 仅收配置、脚本和小型 summary。本轮没有 Zero-One、Ours、PPO 或场景变更。
