# Stage 4 Zero-One 适配与六方法 benchmark 完成报告

已完成 Zero-One 的离散动作、仅观测攻击适配，并在相同五个 frozen Clean checkpoint 上完成 5 × 20 集评估。新增 100 集 Clean 对照与 Stage 3 **全部逐 episode 字段及轨迹 SHA 完全一致**。六方法合并结果共 **600 集、98,139 个实际交互步骤**；额外重复的 100 集 Clean 对照不重复计入比较样本。

这是保留“两层优化”结构的 SUMO 适配版，不是原连续控制 MuJoCo 实验的数值复现。Zero-One 使用额外的仿真查询权限，不能宣称六方法拥有相同信息或相同计算预算。没有 Gate、Ours、时序攻击、新场景或 PPO；没有重新训练、筛选或更换 victim。

## 实现与验证

- `ZeroOneAttack`：外层 ZOOpt 0.4.2 搜索 20 步离散目标动作序列，每块最多 10 个候选；内层 targeted logit-margin PGD，2 steps、一个随机起点、归一化步长 1。按原奖励的候选轨迹总和选择，执行整个获胜 block。
- `SimulatorOracle`：独立进程和私有配置副本，恢复 reset 前环境属性与 NumPy 状态，按完整动作历史重放；不用不完整的 SUMO save/load 状态作为精确克隆。相同动作前缀和确定性 PGD 查询使用缓存，分别记录命中。
- 39 项单元及回归测试通过，包含原 OARL 训练更新等价、冻结性、梯度方向、投影、RNG 隔离、目标动作诱导、ZOOpt 固定预算/重复性、block 执行、缓存及失配拒绝。
- 真实 SUMO 两集 smoke：400 个交互 step 全部与独立 oracle 相符。正式 seed 0 前两集与 smoke 的全部非耗时 episode 字段再次完全一致。
- 正式 Zero-One 的 **13,451 个实际 step** 全部核对计划观测、动作及真实 transition，含换道碰撞和提前终止；全部模型哈希、源码完整性及逐步扰动/查询成本验证通过。
- 重新扫描 Stage 3 全部 84,688 条原始 step，结果与原已提交 summary 相同。新旧 Clean、checkpoint、Protocol A seeds、Python/PyTorch/SUMO/已安装依赖和原环境源码全部核对后才合并。
- 首次 smoke 的 SUMO 子进程 stdout 混入 IPC，已单独修复并重跑；失败结果保留但不纳入统计。离线核验显式按 Python 3.7 顺序累加浮点回报，避免 Python 3.12 `sum` 的微小差异，未放宽轨迹一致性标准。

## 回报与安全事件

± 为五个 run-level 数值的样本标准差，不是置信区间。Return Drop = 配对 Clean Return − 攻击 Return。碰撞率指原 SUMO 配置报告的事件，可能包含 minGap 违规，不等于已证实的物理接触率。

| 条件 | Return | Return Drop | SUMO 碰撞率 (%) | ASR 五组均值 (%) | 转换成功/合格 episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 129.488 ± 8.404 | 0.000 ± 0.000 | 12.000 ± 7.583 | 0.000 ± 0.000 | 0/88 |
| Random | 128.236 ± 8.539 | 1.253 ± 2.032 | 15.000 ± 12.748 | 7.361 ± 13.569 | 6/88 |
| FGSM-margin | 103.010 ± 29.710 | 26.478 ± 29.625 | 26.000 ± 21.909 | 20.214 ± 27.425 | 17/88 |
| PGD-margin | 103.959 ± 27.545 | 25.529 ± 27.541 | 25.000 ± 19.685 | 18.964 ± 24.663 | 16/88 |
| OARL-BO | 131.930 ± 7.778 | -2.442 ± 3.539 | 10.000 ± 6.124 | 1.250 ± 2.795 | 1/88 |
| Zero-One SUMO adapter | 96.241 ± 43.594 | 33.247 ± 42.446 | 36.000 ± 36.297 | 28.863 ± 39.955 | 24/88 |

ASR 只统计配对 Clean 未碰撞、攻击后发生 SUMO 碰撞的转换，合并计数与各组率均值的分母不同。Zero-One 为 24/88，不能把五组 ASR 均值直接当作 24÷88。

## 每个 seed 的结果

| run_seed | Clean Return | FGSM Return | PGD Return | Zero-One Return | Zero-One 碰撞率 (%) | Zero-One steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 141.374 | 117.970 | 117.970 | 128.476 | 10.0 | 3605 |
| 1 | 134.011 | 115.399 | 115.317 | 108.074 | 25.0 | 3010 |
| 2 | 127.547 | 49.893 | 54.719 | 19.553 | 100.0 | 621 |
| 3 | 119.773 | 115.714 | 115.714 | 114.721 | 20.0 | 3205 |
| 4 | 124.735 | 116.075 | 116.075 | 110.380 | 25.0 | 3010 |

Zero-One 对 seed 2 的影响最强：20/20 集发生 SUMO 碰撞事件，平均回报 19.553，20 集合计只运行 621 steps。该组对总体均值和方差影响很大，不能据此宣称所有 victim 上均显著优于梯度基线，也不构成最坏情况攻击保证。

## 纵向安全与扰动

| 条件 | 每组 minimum TTC 均值 (s) | TTC p05 (s) | DRAC p95 (m/s²) | 全部 step 最大 L∞ | 全部 step 最大 L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean | 9.287 ± 3.656 | 19.826 ± 2.028 | 0.212 ± 0.037 | 0.000000 | 0.000000 |
| Random | 9.382 ± 3.242 | 19.285 ± 1.764 | 0.217 ± 0.034 | 0.250000 | 0.561188 |
| FGSM-margin | 8.588 ± 2.514 | 17.998 ± 2.021 | 0.232 ± 0.038 | 0.250000 | 0.722638 |
| PGD-margin | 8.588 ± 2.514 | 18.130 ± 1.758 | 0.225 ± 0.025 | 0.250000 | 0.722638 |
| OARL-BO | 9.325 ± 3.693 | 19.853 ± 1.339 | 0.212 ± 0.026 | 0.250000 | 0.746153 |
| Zero-One SUMO adapter | 9.042 ± 3.111 | 18.612 ± 3.342 | 0.259 ± 0.110 | 0.250000 | 0.690401 |

TTC/DRAC 只覆盖每次实际交互后同 lane 的前后纵向 pair，缺失、不闭合和重叠状态保留有效样本计数；不能解释全部换道风险或 reset 内事件。定义沿用 [SAFETY_METRICS.md](SAFETY_METRICS.md)。

所有非 Clean 条件在每个实际 step 实施攻击，Attack Rate 100%。共同边界为 `abs(delta_i) <= 0.2*abs(s_i)+0.05`，不做物理裁剪。OARL-BO 保留共享两个参数的仿射子空间，其他方法可独立改变特征；因此也不属于完全相同搜索空间的优化器比较。

## Zero-One 成本

| run_seed | 候选 rollout | 输入梯度次数 | 攻击内 forward | 实际影子 steps | 其中重放 steps | transition 缓存命中 | PGD 缓存命中 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1820 | 19412 | 32723 | 4365 | 176 | 31982 | 26465 |
| 1 | 1550 | 16510 | 27775 | 49021 | 40404 | 21638 | 22000 |
| 2 | 410 | 7102 | 11274 | 8162 | 4592 | 3121 | 3140 |
| 3 | 1640 | 16814 | 28426 | 4662 | 1003 | 28450 | 23702 |
| 4 | 1550 | 16442 | 27673 | 6932 | 3154 | 26592 | 22149 |

总计 6970 个候选 rollout、76280 次输入梯度、127871 次攻击内 forward、73142 个影子 steps（含 49329 个历史重放 steps）。共 854 次搜索重放 reset，另有 100 次 episode 初始化 reset。

候选 objective 是仿真奖励求和；PGD 梯度是目标动作 margin 导数，不能将两者与其他方法的 objective 次数混为同一种查询。每个实际 plan 动作额外做一次 forward 确认，全部已计入。相同状态/目标/步号的 PGD 结果可缓存，重复候选 rollout 也可复用精确动作前缀。

Zero-One 累计 attack wall time 为 614.334 秒，含规划和动作确认，不含 episode 初始化、真实环境执行及之后的 IPC 核验。五个进程并发存在资源竞争，不能用累计耗时作严格速度排名。

## 已知边界与下一阶段

- 当前仅验证了固定 horizon 20、外层预算 10、内层 2 steps 的离散 SUMO 适配；没有根据结果调参，也没有声称复现原连续环境数值。原环境专用的终止奖励修正、timing attack、连续动作损失没有被带入 SUMO。
- 原奖励总和会受提前终止和规划 horizon 影响；本轮保留原奖励，没有额外碰撞奖励或 Gate。TTC/DRAC 分位数未必随碰撞率单调变化，需要结合逐轨迹事件解释。
- 下一阶段先确认基线可信，分析 seed 2 的动作分歧与碰撞轨迹，必要时用 checkpoint × 交通种子的诊断性对照区分因素。之后提出研究问题和自己的方法，验证效果后才对新增模块做消融。当前不预设或实现 Ours，也不更换场景或 victim。后续诊断结果见 [BASELINE_DIAGNOSTIC_RESULTS.md](BASELINE_DIAGNOSTIC_RESULTS.md)。

## Provenance 与复现

- Zero-One 正式实验 SHA：`169a8f4c0b276cf31d8a44219f8f97432e72fb64`。
- Stage 3 复用实验 SHA：`b2e665bc4584ddd4a01b15f10141c316137a7f00`。
- 汇总及绘图代码 SHA：`62e6db51bf91d909ef930369281fa91744fc337c`。
- 正式批次：`.local/runs/20260914T072528603795Z-attack-benchmark`。
- 通过的 smoke：`.local/runs/20260914T072338849068Z-attack-evaluation`。
- 失败的首次 smoke：`.local/runs/20260914T072206592569Z-attack-evaluation`，未计入比较。
- 环境继续为 Python 3.7.16 / PyTorch 1.3.1+cpu / SUMO 1.22.0。额外 ZOOpt 0.4.2 wheel 的版本、MIT license、SHA256 和实际路径写入 manifest；没有安装进既有训练环境。
- 原 `main.py`、`oarl.py`、`Environment/`、`Data/`、冻结模型清单、Clean 训练及 checkpoint 验证入口相对 `v0.3.0-baselines` 无改动。`Zero-OneAttack-main` 完整文件指纹再次核对一致。
- 上游 archive 没有 Git 元数据，远端 HEAD 连接失败，因此没有伪造 upstream commit；以 [ZERO_ONE_LOCAL_PROVENANCE.json](ZERO_ONE_LOCAL_PROVENANCE.json) 的内容哈希明确标识参考版本。未发现上游明确许可证，仍仅本地开发，没有公开推送。

完整协议及下载/启动命令见 [ZERO_ONE_ADAPTER_AUDIT.md](ZERO_ONE_ADAPTER_AUDIT.md)。机器可读结果：[STAGE4_BENCHMARK_RESULTS.json](STAGE4_BENCHMARK_RESULTS.json)。图保存在 `.local/reports/stage4_attack_benchmark.png` 和同名 SVG，已检查可读性。

```powershell
python scripts/summarize_zero_one_benchmark.py --stage3-batch .local/runs/20260914T064729768497Z-attack-benchmark --batch .local/runs/20260914T072528603795Z-attack-benchmark --output .local/reports/stage4_reverified.json
python scripts/plot_attack_benchmark.py --summary docs/STAGE4_BENCHMARK_RESULTS.json --output .local/reports/stage4_attack_benchmark.png
```
