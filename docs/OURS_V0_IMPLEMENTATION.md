# Ours-v0 实现与复现实验接口

本实现对应 `OURS_V0_DEVELOPMENT_PLAN.md` 的 P0/P1 和 P2 对照接口。工程代号 Ours-v0 尚不代表已验证的创新或性能优势。原始 OARL/Zero-One 基线、冻结模型和环境算法保持原样。

## 已实现的方法

| 注册名称 | 搜索 | 完整 rollout 排序 |
| --- | --- | --- |
| `ours_return` | 实际动作历史分支 | 原始累计 reward 最小 |
| `ours_safety` | 实际动作历史分支 | ego SUMO 碰撞优先，同类中原始回报最小 |
| `zero_one_budgeted_return` | ZOOpt 目标动作序列 | 原始累计 reward 最小 |
| `zero_one_budgeted_safety` | ZOOpt 目标动作序列 | 同上，碰撞优先 |

四条件共享 `BehaviorSearch` 中的节点、PGD、缓存、完整保底 rollout、预算与回放原语。新的 Zero-One 对照使用固定 0.4.2 wheel；原 `zero_one` 注册项仍调用原适配器，不混用旧结果。控制搜索保留每个节点/目标第一次 PGD 结果；Ours 在未找到该实际分支时可分配独立重试。这是待检验的搜索机制差异，不是已证明的收益。

## 数据流与 witness

```mermaid
flowchart LR
    A[冻结策略与当前观测] --> B[按完整实际动作历史定位节点]
    B --> C[计费 PGD 诱导并确认实际动作]
    C --> D[存储实际动作的扰动证据]
    D --> E[隔离 SUMO 预算预检和历史回放]
    E --> F[完整 rollout 得分]
    F --> G[选择并执行完整计划]
    G --> H[逐步核对观测、策略动作与 transition]
```

节点缓存限定在单个规划块内，但键包含从 episode 起点开始的整个实际动作历史。不同历史即使观测相同也不合并。目标失败时记录实际动作 witness、margin 与尝试次数，不把目标标记为不可达。干净动作的零扰动 witness 不计为 applied attack。

Ours 每次访问节点至多进行一次诱导：先按未扰动 logits 排序尝试尚未尝试的非干净目标；所有其他目标至少尝试一次后，才重试仍未发现的目标分支，每目标最多三次。已发现分支按完成 rollout 访问次数、最好完整得分、动作编号选择。未完成候选不更新完成访问次数，也不参加最佳计划选择。

PGD 的局部随机数种子取 SHA-256 对 `[attack_seed, episode, full_history, target, attempt]` 的前四字节、小端解码。控制与 Ours 对相同键使用相同扰动尝试；ZOOpt 外层单独隔离 Python/NumPy/Torch RNG。元数据记录每次尝试的种子、实际动作、margin、完整与未完成候选。

Safety 用原始字典序 `(-any(ego_collision), sum(original_rewards))` 选择最终计划。为满足 ZOOpt 标量接口，用 `2*collision_key + 2/pi*atan(return)` 映射到互不重叠的区间；最终选择仍按原始元组。碰撞沿用 SUMO minGap 检测语义，不声称物理接触；未引入新奖励、Gate、物理裁剪或风险指标。

## 成本和回退

每 20 步规划块上限为梯度 400、forward 800、新 transition 200、物理 shadow step 4000；完整搜索候选至多 10，另有一条同样计费的干净保底。尾块仅缩放梯度和新 transition 上限。

- 新节点包含干净动作和排序 logits 两次 forward。
- 一次 2-step PGD 包含两次带输入梯度的 forward、一次实际 greedy 确认、一次最终 margin forward，共 2 梯度、4 forward；最终 margin 审计额外成本同样用于四条件。
- 每块预留 horizon 次实时动作确认 forward，提前扣住额度；每块还计费一次应急干净动作 forward。
- oracle 在 worker 内原子预检新 transition、历史 replay 和 reset 预热，预算不足时不改变 cursor、缓存或仿真状态。
- `shadow_steps = new_shadow_transitions + replay_steps + warmup_steps`。规划期间 reset 的实际预热计入 4000 上限；每个 episode 初始化 oracle 的一次预热单列 `oracle_episode_setup_cost`，不归入某个规划块。
- cache hit、reset 次数、IPC 和 wall time 单独记录。evaluator 自身每步两次 forward 记在 `research_audit.evaluator_policy_forward_calls`，不藏在攻击成本中，也不从攻击预算中重复扣除。

预算耗尽时保留最佳已完成计划。若连完整保底都未完成，则只返回当前已计费的干净动作。`oracle_unplanned_fallback` 标记该回退路径：若实际 transition 已有缓存，仍严格核对；否则显式增加 `live_unverified_fallback_steps`，不能把它报告为已预演验证。后续若回放到它，必须与记录的真实 transition 一致。原 `observe` 接口继续拒绝未预演动作。

`eligible` 表示当前步骤属于可攻击时刻，`attempted` 表示选中 witness 来自 PGD，`attacked/applied` 与 `changed` 表示实际非零扰动。规划期间所有未选中尝试另记 `inner_attempt_trace`，不能将执行率误当搜索尝试率。

## 交通划分与运行

`configs/research_seed_splits.json` 冻结开发/验证/最终测试的 10/20/50 个 SUMO seed，保存规范 JSON payload SHA-256；与旧受控训练、旧评估、常量 0–4 及 upstream paired seed 检查无交集。各模型共享交通与环境 RNG；checkpoint training seed 与独立 attack seed 分开。新协议不修改旧 Protocol A。

已有两类配置：`proposed_smoke_seed0..4.json` 使用开发集前两个 episode；`proposed_development_seed0..4.json` 使用完整十个开发 episode。二者重叠，不能把 smoke 追加为独立样本。当前配置 attack seed 为 0，后续验证需覆盖 0/1/2。验证/最终测试 seed 清单已冻结，但未执行其评估，也未把这份清单等同于已经冻结最终算法参数。

PowerShell 中，从干净 commit 启动五模型 smoke：

```powershell
& 'E:\Programs\anaconda3\python.exe' scripts/evaluate_attack_batch.py --configs configs/evaluation/proposed_smoke_seed0.json configs/evaluation/proposed_smoke_seed1.json configs/evaluation/proposed_smoke_seed2.json configs/evaluation/proposed_smoke_seed3.json configs/evaluation/proposed_smoke_seed4.json --jobs 5
```

完整开发集将上述 `smoke` 替换为 `development`。启动器自动使用训练时的 legacy Python、依赖与 SUMO，校验模型哈希、环境版本与上游源码，并将 commit、命令、配置、角色 seed、源文件哈希记录在 `.local/runs/`。代码变更必须先提交；不覆盖旧实验目录。

`scripts/summarize_proposed_smoke.py --batch <batch.json> --output <summary.json>` 在 legacy Python 下核对四条件配对、完整计划选择与执行、逐块预算、原始步数及 oracle 账本，保存输入文件哈希。报告只用于 smoke，不给两 episode 结果计算方法显著性或泛化结论。下一步应先完成开发集比较，检查 H1/H2/H3，再决定是否推进验证及自身新增机制的消融。
