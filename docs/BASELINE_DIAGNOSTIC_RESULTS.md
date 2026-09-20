# 基线可信度与 seed 2 诊断报告

本轮完成基线诊断，没有开发 Ours，也没有做创新模块消融。研究顺序仍是：确认基线可信 → 分析失败案例与研究问题 → 提出自己的方法 → 验证效果 → 对新增模块做消融。

## 主要结论

交叉评估后，可以把原先“seed 2 组合更敏感”的判断收窄并加强为：**在当前 FGSM-margin 与固定扰动预算下，checkpoint 2 在所有五组评估交通中都表现出更大的回报下降和碰撞事件率增加，不能仅用交通组 2 解释。**这不是对所有攻击、其他训练重复或其他场景的普遍鲁棒性结论；本轮没有完成 Zero-One 的 checkpoint × traffic × attack seed 全因子评估。

已有六方法 benchmark 在本轮检查范围内可信：重新加载哈希验证的冻结模型核对 98,139 个 step、196,278 个干净/攻击动作输出，没有发现记录与策略不符；首次动作分歧之前所有配对环境轨迹相同；扰动边界与已定义指标复核通过。重新运行 Stage 4 汇总校验也通过，包含 Zero-One 候选选择、成本和 13,451 个实际步骤的 oracle 一致性证据。可信范围是当前实现及已声明的 SUMO/观测攻击协议，不是物理攻击可行性保证或原 MuJoCo 数值复现。

## 1. checkpoint 与交通因素

预先固定 [BASELINE_DIAGNOSTIC_PROTOCOL.md](BASELINE_DIAGNOSTIC_PROTOCOL.md)，执行 5 checkpoint × 5 traffic group × 2 conditions × 20 episodes = **1,000 episodes**。两种条件为 Clean 和原 FGSM-margin，最多 200 steps，参数、交通密度、奖励和 termination 均未改变。FGSM 不消耗攻击随机性，因此这里无需引入额外攻击 seed 因素。

五个原始对角线组合的 Clean/FGSM 共 **200 集**，除耗时外所有 episode 字段（含 trajectory SHA）与 Stage 3 完全一致。新旧配对受控；没有通过重新训练或换 checkpoint 适配结果。以下按每个 checkpoint 的五组交通等权汇总，每种条件共 100 集。

| Checkpoint 训练 seed | Clean Return | FGSM Return | Return Drop | Clean 碰撞率 | FGSM 碰撞率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 128.845 | 115.831 | 13.014 | 7.0% | 18.0% |
| 1 | 128.596 | 109.811 | 18.784 | 8.0% | 14.0% |
| 2 | 137.260 | 52.725 | 84.535 | 21.0% | 72.0% |
| 3 | 126.994 | 122.222 | 4.772 | 9.0% | 13.0% |
| 4 | 129.669 | 117.282 | 12.386 | 7.0% | 15.0% |

checkpoint 2 在各组交通的 Return Drop 为 77.655–97.231，FGSM 碰撞率为 65%–80%；其余 checkpoint 的 FGSM 碰撞率为 5%–25%。checkpoint 2 自身的 Clean 碰撞率也较高（五组平均 21%），但攻击后进一步升到 72%，平均增加 51 个百分点；其余 checkpoint 的平均增加为 4–11 个百分点。另一方面 checkpoint 2 的平均 Clean Return 最高，说明原奖励与碰撞风险并不形成简单的一致排名。

交通因素仍存在：例如 checkpoint 0 的 Return Drop 在 group 1 为 -1.874，在 group 0 为 23.405。不得因为 checkpoint 2 在所有列都敏感，就宣称交通不重要。该矩阵是受控描述性证据，不报告独立同分布样本下的显著性或训练 seed 本身的因果效应。

完整矩阵见机器可读 summary；对照图 `.local/reports/baseline-diagnostics-20260920-final/crossed_diagnostics.png`（同时有 SVG）。

## 2. 攻击究竟改变了什么

原 benchmark 的 seed 2（其自身交通组）中：

| 条件 | 首次动作有分歧的集数 | 第 0 步即分歧的集数 | 实际步骤 | 访问状态上的局部动作改变率 | 观测到的车道 index 转移次数 | 碰撞集数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 0/20 | 0/20 | 3428 | 0.00% | 210 | 4/20 |
| fgsm | 20/20 | 20/20 | 1879 | 90.58% | 483 | 13/20 |
| pgd | 20/20 | 20/20 | 2033 | 90.36% | 487 | 12/20 |
| zero_one | 19/20 | 11/20 | 621 | 37.84% | 157 | 20/20 |

局部动作改变率比较的是“攻击已访问到的同一干净观测上，不扰动与扰动的动作”；它不是与 Clean rollout 同一时间索引的动作差。首次分歧之后两条轨迹已经处于不同状态，不能据此直接作单特征因果归因。车道 index 转移数也不是换道请求数或经过验证的完整连续换道次数，且实际 episode 长度不同。

- **Episode 1：**Clean 第 0 步选择 action 2（不发新请求）；FGSM 与 Zero-One 均改为 action 1，实际 lane index 2→3。干净 logits 为 [-7.309, -1.018, 1.516]，攻击后为 [-14.277, 6.471, 0.326]，记录动作确实由冻结模型产生。此后 FGSM 本集运行至 200 steps、未报告碰撞；Zero-One 在第 33 步（0-based）报告碰撞并移除 Auto，回报从配对 Clean 的 122.249 降到 21.080。相同首次动作并不决定同样的长期结果。
- **Episode 2：**Zero-One 第 0 步将 action 2 改为 action 1，lane 2→3；第 19 步执行 action 0 后报告碰撞并终止，回报从 152.807 降到 8.725。本集 FGSM 第 0 步选择的是 action 0，最终未碰撞。不能仅统计是否发生动作翻转就解释长期危险性。
- **Episode 3：**Zero-One 与 Clean 没有任何动作分歧，两者第 0 步均碰撞终止、回报 0。该 episode 不是攻击新增碰撞。Zero-One 的 20/20 碰撞中，配对 Clean 原本有 4/20 碰撞，所以本组 collision-conversion ASR 为 16/16，而非把 20/20 都算成攻击成功。

案例详情包含原/攻击观测、delta、logits、概率、起始与终止附近动作事件。全部 600 集的逐集诊断在 `.local`，Git summary 保留六个事先按 episode 1–3 选取的 seed 2 FGSM/Zero-One 案例及 seed 2 全部 20 集 Zero-One 标量结果；这些案例不是用于估计总体效果的独立试验。

## 3. 决策与适配核验

在各模型自身原 Clean 轨迹上的 top-two logit margin：

| Checkpoint | margin p05 | margin median |
| --- | ---: | ---: |
| 0 | 6.727 | 15.914 |
| 1 | 4.111 | 11.906 |
| 2 | 0.524 | 5.144 |
| 3 | 10.088 | 20.529 |
| 4 | 9.490 | 21.132 |

checkpoint 2 的 margin 较小，是可继续检查的决策特征。但不同模型的 logit 尺度、访问状态分布不同，margin 不能直接当作输入鲁棒半径，也不能单独证明造成高敏感性的机制。本报告关于 checkpoint 2 的主要归因证据来自完整交叉对照，而非仅比较这些 margin。

Zero-One 原对角线组目标动作命中率（只统计最终执行的计划，不把未执行候选当成功）：

| Checkpoint | 命中 / 已执行目标 | 命中率 |
| --- | ---: | ---: |
| 0 | 1278/3605 | 35.45% |
| 1 | 1359/3010 | 45.15% |
| 2 | 370/621 | 59.58% |
| 3 | 1076/3205 | 33.57% |
| 4 | 1018/3010 | 33.82% |

目标命中不等于攻击成功。目标动作不一定在固定预算下可达，外层实际评价的是诱导后真实动作产生的奖励。现有适配器按实际动作推进仿真并逐步确认获胜计划，因此未命中目标不会被虚构成已执行目标。全部候选选择、投影、缓存成本以及 live/oracle 一致性校验重新通过；没有发现因此需要修正攻击算法的证据。

## 4. 观测物理含义与安全指标边界

当前 box 约束是归一化观测空间限制，不保证物理可实现性。seed 2 原 Zero-One 的 621 个实际步骤中：605 步扰动后 lane 特征不属于合法的 {0,0.1,0.2,0.3} 网格；256 步至少一个速度特征为负；547 步至少一个归一化距离不在 [0,1]。这些扰动仍满足已声明的共同 box，所以不是超预算实现错误；却意味着不能把本轮结果解释为已经验证的物理传感器攻击。其他方法同样存在此问题，完整计数保留在 summary。

本轮不通过事后裁剪或禁用特征修改既有基线。观测缺失标记、离散 lane 特征、角度量及合法物理范围之间的约束关系，可以形成后续研究问题，但尚未据此设计新算法或执行模块消融。

安全指标仍按既有定义：SUMO 原配置事件可包含 minGap 违规；TTC/DRAC 仅使用实际交互后同 lane 前后车辆对，缺失时保留缺失值。本轮复算现有 pair 的 TTC/DRAC 和全部已发表安全汇总通过。这证明了记录和公式的一致性，不补足未记录的碰撞参与车辆、横向接触几何或连续时间风险。碰撞移除后的 next observation 是旧环境保留的最后有效状态，诊断已将终止后的 lane 记为缺失。

## 5. 交付与研究阶段边界

本阶段完成：角色种子解耦、预先固定的交叉诊断配置、1,000 集真实 SUMO 对照、200 集对角线等价证明、六方法原始日志/模型动作/安全指标复核、案例与图表。44 项测试通过。原训练入口、Actor/Critic/BO、Clean 训练、Environment/Data、基线配置、冻结 checkpoint 均保持不变。Zero-One 参考目录仍只读。

下一步可以基于这些证据提出并讨论研究问题，例如：哪些可达动作序列真正造成风险，回报目标与安全事件为何不一致，如何定义可信的物理观测约束。这些是待论证的问题，不是已确定的 Ours 结构。只有提出并验证新方法后，才对新增模块做消融。

## 复现记录

- 交叉实验 commit：`57e431b24f4d4bb85383c9f26d738d7e16bab20a`。
- 诊断分析 commit：`32dbaa4d9557f062769cc2c8e9d9956c85d6fb2a`。
- 批次：`.local/runs/20260920T025043039879Z-attack-benchmark`。
- 完整诊断：`.local/reports/baseline-diagnostics-20260920-final`，manifest 记录命令、Python/Torch/NumPy、pip freeze、SUMO、输入 hashes 和 summary hash。
- 分析与仿真均沿用 Python 3.7.16 / PyTorch 1.3.1+cpu / SUMO 1.22.0；各交叉 trial manifest 保存配置、所有角色 seeds、源码及 checkpoint 哈希。
- 早期一次分析在交叉批次结束前到达完整性检查，被拒绝生成 summary；目录 `baseline-diagnostics-20260920` 已标记 incomplete，不纳入结果。随后完成的分析及增加观测域检查后的最终分析均 passed。
- 本轮是基线诊断，不新增算法里程碑 tag，不公开推送许可证未明确的上游源码。

```powershell
python scripts/evaluate_attack_batch.py --configs configs/evaluation/diagnostic_traffic0.json configs/evaluation/diagnostic_traffic1.json configs/evaluation/diagnostic_traffic2.json configs/evaluation/diagnostic_traffic3.json configs/evaluation/diagnostic_traffic4.json --jobs 5
# 在既有 legacy 环境下，批次完成后运行；替换批次路径和新的输出目录：
$env:PATH = "$PWD\.local\envs\oarl-legacy;$PWD\.local\envs\oarl-legacy\Library\bin;$env:PATH"
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:MPLCONFIGDIR = "$PWD\.local\matplotlib-tests"
.local/envs/oarl-legacy/python.exe scripts/diagnose_baselines.py --cross-batch .local/runs/20260920T025043039879Z-attack-benchmark --output .local/reports/diagnostic-repeat
python scripts/plot_baseline_diagnostics.py --summary docs/BASELINE_DIAGNOSTIC_RESULTS.json --output .local/reports/crossed_diagnostics.png
```

机器可读结果：[BASELINE_DIAGNOSTIC_RESULTS.json](BASELINE_DIAGNOSTIC_RESULTS.json)。
