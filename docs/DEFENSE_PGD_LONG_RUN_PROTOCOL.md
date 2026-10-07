# PGD 一致性：五 seed 长程训练与冻结协议

2026-10-07。本功能分支为 `codex/defense-pgd-long-run`，从 `c827a8a` main 创建。沿用已审计的 PGDConsistencyAgent，不改损失或搜索算法；补充正式训练输出、逐更新/转移流式审计与冻结策略评估入口。

## 固定训练

五份 `configs/experiments/pgd_consistency_protocol_a_seed0.json` 至 `seed4.json` 分别记录 run seed 0–4，Protocol A，各 400 episodes、每集最多 200 步，Normal / Pn=0.14、16D / 三动作 / 128×1 网络。所有 run 从匹配的原初始化开始，不从 OARL Robust 微调。原 actor/critic 学习率、gamma、Polyak、batch 128、replay 1,000,000、原更新开始时间和间隔保持。

防御统一使用 λ=0.1、epsilon=1、乘法/加法尺度 0.2/0.05、五步 PGD、归一化 step size 0.4、一个随机起点、概率 floor 1e-8。参数沿用工程版本，此次没有扫描 λ 或依据攻击结果选版本。普通交互、critic backup 保持，PGD 仅用于 replay batch 的 actor 一致性正则化；无 Gate。

基础 Python/NumPy/Torch/policy seed 使用各自 run seed。SUMO 用原受控训练派生名单；辅助 search 使用 D2 隔离 attack NumPy 流。原环境与 replay 共享的 NumPy 流保留。manifest 记录完整角色 seed、Git SHA/配置 SHA、命令、原上游源文件哈希及精确 Python/Torch/SUMO/pip freeze 环境；与同 seed 已冻结 Clean 训练记录核对运行环境，不新装依赖。

每个 run 实际交互上限 80,000，但提前终止会改变实际步数/更新数。报告实测 episode 回报、观察到的 SUMO ego 碰撞、交互/warmup 步、全部网络调用/输入行数、输入梯度、optimizer 更新和训练 wall time。训练中的碰撞不是正式防御测试结果，不能据此宣称攻击安全优势。

## Checkpoint 与冻结资格

预先固定保存 episode **100/200/300/400**，每个同时保存完整训练 snapshot 和原格式 inference actor。正式候选固定 episode 400，不按干净或受攻击回报挑选。完整 snapshot 为 episode 边界恢复所需状态；本入口目前从头运行，恢复能力继承 D2 支撑，但没有自动拼接被中断长程批次的记录。

推理文件仍位于 run 的独立源码目录 `source/model/policy<N>.pkl`，保持已有评估器的训练 manifest 定位约定。报告和完整训练状态/原始数据在 run 根目录，均 ignored。每个新 actor 先标记 `benchmark_eligible=false`，只有独立完整训练审计通过才生成可冻结引用。

逐转移记录写入 `transitions.jsonl.gz`；每次 PGD 更新的 observation、扰动、概率、loss、cost 写入 `defense_updates.jsonl.gz`。按流写入和复核，避免长程原始数组一次加载内存。训练保存 gzip/JSON 与审计属于记录成本，actor/critic phase 成本另计，不混成纯算法训练加速。

独立 auditor 重算每集 return、轨迹 digest、碰撞集合、正常连续 observation、termination/truncation、原更新节奏；逐更新重算盒和 KL/总 loss，核对实际 hooks；检查四个 checkpoint 的完整状态、实际 episode 计数、文件/权重哈希、探针重载及 RNG。审计失败将 run 标记 failed，不能仅因训练进程 exit 0 标记成功。

五个 run 全部完成并通过审计后，batch 在 ignored 输出内生成 `frozen_defense_victims.json` 候选登记，必须包含五 seed、同一源提交/损失配置及四个已审计 checkpoint，选 episode 400。下一次独立登记分支再将核验后的小型文件提交为正式防御 registry；旧 `configs/frozen_victims.json` 和十个 Clean/OARL checkpoint 不修改。

## 评估接入与工程检查

`scripts/evaluate_attacks.py --victim-registry <committed defense registry>` 扩展旧模型名单，复用相同 VictimAdapter、attack registry、SUMO evaluator/oracle。正式 registry 必须已提交，检查训练 manifest/audit 哈希和 actual checkpoint，支持相同冻结 PGD 策略被所有既有攻击重新攻击。新的确定性 actor 不需要额外在线网络调用；这不等于训练成本或推理延迟为零。

短程生产入口检查用 `pgd_consistency_training_smoke.json`：seed 0、16×16，checkpoint 13/16，完整 raw/文件审计。独立评估接入配置 `pgd_consistency_integration_smoke.json` 使用一个既有 phase-separated 工程交通、None/PGD、最多 16 步，显式 `--allow-engineering-victims` 才可载入。短程产物的 registry/model 均标记 engineering，不进入 benchmark；不启用旧 split 30，不生成效果结论。

长程启用前，完成相关测试、短程训练流式审计、与已有 D3 同配置原始轨迹/权重/训练状态比对及两次真实 SUMO 评估接入。代码/配置/auditor 提交后才运行；功能通过工程验收以 `--no-ff` 合并 main，再以该 main SHA 在后台启动五个独立 run。源码与 tracked tree 在整个 batch 期间固定；启动器绑定 expected commit，不继续同时修改代码。

```powershell
& .local/envs/oarl-legacy/python.exe scripts/run_experiment_batch.py --jobs 5 --name defense-pgd-consistency-long --configs configs/experiments/pgd_consistency_protocol_a_seed0.json configs/experiments/pgd_consistency_protocol_a_seed1.json configs/experiments/pgd_consistency_protocol_a_seed2.json configs/experiments/pgd_consistency_protocol_a_seed3.json configs/experiments/pgd_consistency_protocol_a_seed4.json
```

后台 PID、原命令和 stdout/stderr 指针记录在 ignored launch JSON；batch.json 与各 progress.json 是状态依据。没有自动通知任务或自动 Git 提交；长程完成后先审计/登记，再安排新防御开发交通协议和正式攻击矩阵。ACoE 的即时误差来源、belief/off-policy 目标和自己的创新仍独立后置，OARL Robust 持续作为领域强基线。
