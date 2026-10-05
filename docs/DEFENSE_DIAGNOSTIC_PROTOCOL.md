# D1：既有防御 pilot 的可复现轨迹诊断

本工具仅分析已完成的 Clean/OARL Robust 开发对照，保留 OARL 的领域强基线角色。它不启动 SUMO，不调用 actor，不更新策略，也不使用防御最终交通；输出是失败机制诊断线索，不是新的防御有效性实验或模块消融。

## 输入与 provenance

登记配置为 `configs/research/defense_diagnostics.json`。输入是 `docs/DEFENSE_BASELINE_RESULTS.json` 所引用的完整历史 batch 与历史审计，执行提交为 `db4dcbc3d9b0dbcfd454d5dbcfa109ee9cb51e7f`。固定 summary、batch 和历史审计 SHA256；每个读取的 manifest、evaluation、episode 和 step 文件必须与历史审计的字节指纹一致。模型身份与角色随机种子复用已有注册和校验函数。

覆盖两个策略 cohort、五个训练 seed、五个共同开发交通和七种攻击，共 350 个实际 episode、57,195 个真实交互步。三个强攻击是重点讨论对象，其余方法保留完整结果，不只选择 seed 3。独立交通样本仍为五个；多个模型和攻击条件不会扩充这个样本数。

源码、配置和测试先提交，再从干净且固定的 tracked tree 执行。分析 manifest 记录分析提交、原实验提交、命令、配置和输入哈希、Python/Torch/NumPy/SUMO/系统版本、起止时间和输出指纹。原始逐 episode 案例和完整来源映射留在 ignored `.local/runs/`；必要的小型摘要进入文档。

## 验证与指标

逐条验证整个 episode，包括动作分歧之后的步编号、观测连续性、有限值、扰动盒、奖励累加、终止边界、轨迹 digest 和动作计数。原始策略动作正确性继承固定历史审计，本次不会伪称重新推理证明。

每个攻击 episode 只与同一策略、同一 checkpoint 和同一 SUMO seed 的无攻击 episode 配对。复用 `paired_trace()` 验证第一次动作分歧之前的共同状态和转移；分歧之后不逐步比较两个不同状态的动作。局部动作改变使用日志中的 `clean_action_at_visited_state`，表示同一实际访问状态的无扰动策略动作。

- 首次碰撞按 SUMO ego colliding IDs 的 post-step 记录计算，不使用环境 reward 的启发式碰撞量，也不认证物理接触。
- 步编号从 0 开始；前三个真实交互步为 index 0–2，第 21 步及以后为 index >=20。其余有碰撞样本单列为第 4–20 步。
- 原始碰撞率与各自 eligible ASR 分开：无攻击已碰撞样本不计为攻击造成的碰撞转换。早/晚碰撞转换另列。
- 动作改变率使用累计改变步数除累计真实步数；同时保留 3×3 的同状态动作转换表、首次分歧、条件碰撞延迟、车道特征实际变化、目标动作命中与已有资源计数。
- 请求目标 `target_action` 与可执行 witness 的 `planned_action` 分开。请求未命中但实际动作等于 witness 可属于合法行为分支；实际动作不等于已登记 witness 则拒绝分析，不能把二者都算作回放错误。
- 逐 checkpoint、逐交通结果完整保留，避免 pooled 均值掩盖集中性。首次分歧至碰撞延迟仅统计存在先行分歧的碰撞 episode；不是 ASR 分母或因果效应。

## 证据边界与下一步

旧 evaluator 没有记录 reset 结束时的碰撞标志；reset warmup 不在真实交互轨迹内。第一步碰撞的攻击归因、单次攻击之后的恢复能力及持续攻击的额外效果不能由本工具证明。D1 后续需要独立登记控制性 replay：同一真实状态与已执行动作前缀，比较无攻击、关键步攻击后正常输入、持续攻击，并记录 reset 状态及完整资源成本。

轨迹时序不能独自决定新增损失。ACoE 目标映射另在独立分支核对；模型登记、完整 resume 和新训练留在 D2。随机平滑也不进入本次确定性历史诊断。

## 启动

在冻结环境与已提交的干净工作区运行，使用尚不存在的新输出目录：

```powershell
$env:PATH='E:/Att/OARL-master/.local/envs/oarl-legacy;E:/Att/OARL-master/.local/envs/oarl-legacy/Library/bin;'+$env:PATH
$env:MPLCONFIGDIR=Join-Path $PWD '.local/matplotlib'
& .local/envs/oarl-legacy/python.exe scripts/diagnose_defense_pilot.py --config configs/research/defense_diagnostics.json --output .local/runs/20261005-defense-d1-traces
```

输出 `manifest.json` 和 `diagnostics.json`。失败会保留 failed manifest；修复后的再次分析使用新目录，不覆盖历史分析。
