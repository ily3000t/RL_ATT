# D1 控制性回放协议

2026-10-07。本协议在仿真前提交，不改变原算法、环境、攻击、checkpoint 或已有结果。用途是解释开发交通中的失败；不属于新防御评估，也不是创新模块消融。

## 预登记范围与三个条件

复用完整已审计 pilot 的五个 split 10 交通，两个策略 cohort 各五个冻结 episode-400 模型。聚焦 PGD、预算版 Zero-One Return、Single Return，各 25 对/策略，共 150 配对。所有病例保留，包括无动作分歧和原本已经碰撞的病例。无攻击基准只仿真一次并由三攻击共用，不增加独立样本数。

| 条件 | 行为 | 证据用途 |
| --- | --- | --- |
| no_attack | 当前冻结 actor 每步读取正常观测 | 逐转移复现自身历史 None |
| single_then_clean | 正常执行共同前缀，仅在历史首次动作分歧步输入该步历史扰动观测，随后使用新访问状态的正常输入 | 隔离首次行为偏离的后果；不是实施主动恢复控制器 |
| sustained_reference | 原访问状态上的历史扰动观测序列，由实际 actor 再推理 | 严格复现历史攻击执行；不是在控制分支重新生成攻击 |

共 50 个共享无攻击 episode、150 个单次条件和 150 个持续条件，**350 次物理 rollout、150 个配对病例、五个独立交通簇**。engineering_smoke 固定为两种策略的 checkpoint seed 2、全部五交通和三攻击，共 70 rollout/30 配对；开发诊断覆盖全部模型。smoke 与完整诊断重复数据不能合并增加样本量。

动作分歧从同访问状态的正常 action 与受扰 action 首次不等处确定。正常前缀不沿用旧扰动，但其实际动作/转移必须与历史一致。无分歧病例标明未实施干预，不能作为“攻击后恢复”。单次组离开历史路径后禁止使用历史未来扰动；持续组一旦访问状态不等于历史即失败，不进行宽容回放。

## reset、物理状态与命令

独立进程在当前提交的 ignored 源码副本中使用原 `HighwayEnv`，按原五 seed 顺序 reset；不在根目录修改 SUMO 配置。继承原角色种子、冻结运行环境与 checkpoint 文件/权重哈希。攻击 RNG 本次不用，标记 `fixed_historical_inputs_no_new_attack_rng`；不假装重做了原攻击优化。

记录 reset 的每次 `simulationStep` 后 ego 存在/碰撞 ID、仿真时间，以及 reset 返回时的真实观测和所有车辆可查询运动学/控制模式。首次分歧之前要求实际动作/转移一致；分歧点三条件的全部车辆可查询状态指纹必须相同。此指纹不是 SUMO 隐藏 RNG/内部状态的序列化证明；同 seed/reset/动作前缀、可查询状态及严格历史复现共同支撑配对。

记录每步换道命令的起始车道、目标、持续秒数与时间。原环境只对 action 0/1 发出符合边界的 `changeLane(..., 100)`，action 2 不调用换道 API；本轮保留原语义，不用额外命令取消已有请求，也不从单次观测恢复推断立即恢复车辆行为。碰撞仅指 SUMO ego colliding ID，不认证几何接触；原 reward 的碰撞启发式不替代它。

主分析保持既有 step 后碰撞定义，同时报告 warmup/reset ego 碰撞及排除该边界的 eligible 分母。若 reset 已有 ego 碰撞，不能将后续事件完全归因于攻击；不修改历史 ASR。TTC/DRAC 保存真实 SUMO 同车道采样及有效分母，不增加未测量指标。

## 审计与资源

历史输入首先通过原完整审计绑定的 SHA256；大体积搜索 metadata 按行验证哈希后丢弃，仅保存诊断所需字段。源码、配置、launcher、独立 raw-file auditor 先提交，再运行。None 和持续条件必须逐步精确复现历史 observation、action、reward、termination、SUMO safety 与最终轨迹指纹；单次条件核对共同前缀和分歧转移，并逐步检查之后输入为正常观测。

每个条件记录实际 actor 前向（正常决策及执行核验各一次）、真实交互、reset warmup、完整条件 wall time。本次没有梯度/新增攻击搜索转移；历史攻击成本仍见原 pilot，不把低成本回放当成新攻击的低成本证据。源码副本仅允许 `Data/StraightRoad.sumocfg` 改 seed；checkpoint 保持冻结。原始轨迹/状态/日志/审计放 `.local/`，必要小型摘要进 Git。

按策略与攻击报告三条件回报、单次/持续损害、碰撞 paired 四格、各自正常未碰撞 eligible 转换、前三步和第 21 步以后转换，并按交通分解。可以判断“这一历史首偏离在正常输入后仍造成后果”或“持续条件更差”；不能从该固定序列推出所有适应性攻击的因果机制，不证明特定恢复损失有效，不产生显著性或新防御泛化结论。

启动（冻结 Python，输出须为尚不存在的目录）：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/replay_defense_controls.py --group engineering_smoke --output .local/runs/20261007-defense-controls-smoke
& .local/envs/oarl-legacy/python.exe scripts/replay_defense_controls.py --group development_diagnostics --output .local/runs/20261007-defense-controls-development
```

每次 launcher 生成 manifest，记录命令、提交、配置、环境、历史输入/模型哈希，完成后执行独立 `audit_run`。复核审计可从 `rl_att.evaluation.controlled_replay` 导入 `audit_run(manifest_path)`，不加载策略或重新仿真。
