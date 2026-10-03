# 共享计算预算：新交通验证预登记

完成状态（2026-09-25）：九批共 4,500 episode、707,039 个交互步全部通过审计。低档净收益对单一交通敏感，中档仍退化；完整结果和预登记的集中性检查见 [SHARED_COMPUTE_VALIDATION_RESULTS.md](SHARED_COMPUTE_VALIDATION_RESULTS.md)。以下保留运行前协议，不作结果驱动修改。

2026-09-25。在观察任何 split 20 结果前登记。从开发阶段冻结点 `57ae7e8` 创建 `feat/shared-compute-validation`，继续验证现有进展版本；不新增或调整攻击模块。开发集结果见 [SHARED_COMPUTE_BUDGET_RESULTS.md](SHARED_COMPUTE_BUDGET_RESULTS.md)。

## 问题与固定条件

检验开发集的低预算收益能否在新交通复现，以及中档退化、首步碰撞集中性是否仍存在。只改变交通 split 和 episode 数；方法、checkpoint、环境、奖励、终止、扰动与三个模型预算档位均沿用开发阶段。这里的泛化仅指同场景、同五个冻结模型下的新交通，不能解释为跨场景或跨策略泛化。

- 五个 episode-400 Clean Victim，checkpoint seeds 0–4，引用及 SHA 固定于 `configs/frozen_victims.json`。
- 已预先生成的 validation split 20，20 个 SUMO seed，顺序不变；attack seeds 0/1/2；每 episode 上限 200 步。`configs/research_seed_splits.json` 已登记与 development、final test、旧训练和评估流的种子无交集检查；本轮重新计算并要求整个文件内容一致。
- 模型梯度/forward 上限 400/800、200/400、100/200。固定 horizon 20、10 个搜索候选、内层 PGD 2 步、step size 1、max attempts 3、进展规则 `strict_margin_progress`。
- 扰动包络 `abs(delta_i) <= 0.2*abs(obs_i)+0.05`，epsilon 1；新 shadow transition 上限 200、物理 shadow steps 上限 4,000。尾块规则保持不变。
- 每批包含 Clean、Zero-One Return/Safety、进展版本 Return/Safety。无 Gate、不更换 victim 或场景、不修改两个上游项目。
- 九批按 400→200→100、各 attack seeds 0→1→2 顺序运行，每批五个隔离进程。

完整清单为 `configs/research/shared_compute_validation.json`，45 个配置在 `configs/evaluation/budget_validation_*.json`，确定性生成器为 `scripts/prepare_budget_validation.py`。合计 **4,500 episode**：3,600 个攻击 episode 和 900 个 Clean 控制。为了逐批审计与现有评估路径一致，Clean 在每批重复；它们仅对应 **100 个唯一模型–交通组合**，不能作为 900 个独立样本或扩大攻击成功率分母。每方法/目标/预算有 300 个相关运行记录（5×20×3），同时保留每 seed 的 100 个配对结果。

## 冻结、审计与失败处理

实验前提交配置、分析脚本和此协议；全部批次使用同一 clean HEAD，不在运行中切分支或修改文件。`run_budget_validation.py` 启动前检查运行源码与 `57ae7e8` 无差异，检查种子协议及模型注册表哈希，每批前再次确认 HEAD 与工作区。

每批完成后，使用 legacy Python 执行 `analyze_budget_validation.py`，通过后才进入下一批。逐批检查：

1. 固定 development manifest 的哈希；运行源码、Python/PyTorch/NumPy、pip freeze、SUMO 版本和模型引用与开发阶段完全一致；ZOOpt wheel SHA 不变。
2. 配置只能从既有开发配置改变 split、episode 数、相应交通列表、attack seed 与已登记梯度/forward 上限；攻击方法和参数不能漂移。
3. 从固定公式重算每个 effective seed，并分别验证 RNG 机制标签；逐 episode 的 SUMO seed 必须符合固定列表。
4. 原始步数、回报累加、动作计数、终止、轨迹哈希、碰撞标记与 episode 记录一致；重新计算汇总和 Clean 未碰撞分母。
5. 预算账本、扰动界限、完整候选评分/选择、计划执行、oracle 回放/预热费用、实际目标尝试和停止重试证据一致。完整 Clean fallback、未完成搜索候选、unplanned fallback 分开记录。
6. 首批 400/800、attack seed 0 建立新交通的 Clean 参考；其余八批逐步回归，仅排除 wall time 并允许 Clean 未使用的 attack seed 值不同。Clean 第一处分歧前的共同状态/动作前缀也逐条件检查。

任一运行失败、审计失败或工作区改变时，流水线记录错误并停止；不自动重试，不根据效果换 seed、选子集、改变终止或调参数。若发生工程错误，保留原始失败记录和修复提交，说明原因后再决定如何重跑。后台启动窗口隐藏，日志和运行源码快照均在 `.local/runs/`。

## 分析规则

主要指标为配对 Clean 未碰撞 episode 中的 collision conversion，分别报告每个预算、目标、checkpoint 和 attack seed 的分子与分母。不把目标、攻击种子或重复 Clean 当作独立场景。

完整报告六个预算/目标组合，不只报告开发集最好的 100/200 档。每个组合给出进展独有碰撞、Zero-One 独有碰撞、共同碰撞、共同未碰撞；实际回报/回报下降、攻击率、梯度/forward/规划仿真费用和完整候选/回退统计。进一步与相同方法的高档配对，保留所有收益与丢失案例，不只报告净差。

按交通 seed 聚合额外碰撞，列出首个交互步即碰撞的贡献及各 checkpoint 分布。报告逐一去掉一个交通 seed 后的配对净差范围，用于描述收益是否集中于单一交通；这属于结果集中性检查，不是修改攻击的模块消融，也不替代独立泛化试验或因果归因。

同时报告总费用和每实际交互步费用，明确提前终止带来的影响；不把同上限等同于同实际消耗，不根据并行 wall time 宣称速度优势。TTC/DRAC 保持 step 后同车道前后邻车的定义，碰撞可能包含 SUMO minGap 事件；缺失指标不补造。

本阶段属于验证，不是 final test。若低档收益没有复现、只依赖单一交通、或成本仍不利，直接保留负结果。若收益复现，冻结选择和假设后再规划 final split 30；本轮不读取该 split 的任何运行结果，也不启动 final test。本轮不归因于单独的进展停止模块，其消融仍需另行设计。

## 启动与进度

启动前 92 项测试通过。当前 legacy 环境为 Python 3.7.16、Torch 1.3.1+cpu、NumPy 1.21.6、SUMO 1.22.0，运行器逐次记录实际版本与训练环境比较。启动命令（输出目录必须不存在）：

```powershell
& 'E:\Programs\anaconda3\python.exe' scripts/run_budget_validation.py --output .local/runs/20260925-shared-compute-validation
```

顶层 `validation.json` 记录固定 SHA、启动命令、批次进度、已审计 episode 数及审计哈希。每批 stdout 位于 `group-N-batch.log`，首行包含具体 batch 路径；审计日志为 `group-N-audit.log`。各 batch 的 `verified-validation-summary.json` 保存原始输入哈希与完整 episode 配对记录。全部九批通过后再形成正式结果报告；中途日志不代表整个验证已通过。

准备提交：`31b1a96` 固定验证网格。方法尚未达到完整研究里程碑，当前不创建方法 tag，也不将开发阶段的有利结果当作验证成功。
