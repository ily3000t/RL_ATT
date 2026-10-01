# 同一验证交通上的简单攻击与轨迹搜索对照

2026-10-01。按用户确定的两个问题补齐比较：轨迹搜索相对简单攻击的效果–成本取舍，以及当前方法相对预算版 Zero-One 的核心配对比较。后者保留 [SHARED_COMPUTE_VALIDATION_RESULTS.md](SHARED_COMPUTE_VALIDATION_RESULTS.md) 的完整三档结果和负结果，不用新增简单对照替代它。

## 固定实验

同五个 episode-400 Clean checkpoint、validation split 20 的同20个交通seed、200步上限、greedy策略及原终止语义。复用既有Stage 3简单攻击参数；只把交通接入既有research seed协议。不修改攻击源码或冻结checkpoint。

| 简单方法 | 固定参数 | 独立攻击seed | 实际新增episode |
| --- | --- | --- | ---: |
| Random | 独立均匀观测扰动盒，epsilon 1 | 0/1/2 | 300 |
| FGSM | 单步 untargeted logit margin | 确定性，运行一次 | 100 |
| PGD | untargeted logit margin，10步，归一化步长0.2，随机起点，1次restart | 0/1/2 | 300 |
| OARL-BO | 5次评估，原在线包装 `obs2=obs1`，不访问未来真实观测 | 0/1/2 | 300 |

均每步调用，无Gate，共同扰动上界 `abs(delta_i)<=0.2*abs(obs_i)+0.05`。OARL-BO的共享仿射参数限制为u1∈[0.8,1.2]、u2∈[-0.05,0.05]，其可行集合是共同逐特征扰动盒的子集，需要在比较表脚注标注。内层目标和访问权限不同，属于各方法既有协议。

Random不调用攻击梯度，FGSM每步1次梯度，PGD每步10次梯度，BO每步5次目标评估；全部记录实际调用量，不为匹配形式上的预算增加无效计算。攻击内部forward与评估器的共同2次策略调用分别记账。简单方法无shadow仿真，轨迹搜索另记仿真新转移、replay、预热、episode setup。

三批分别为attack seed0（Clean+四攻击）、seed1/2（Clean+Random/PGD/BO），每批五个隔离进程。共1,000个新增攻击episode、300个Clean回归控制，合计1,300；Clean仍只有100个唯一模型–交通组合。FGSM也只有100个唯一条件，不能将复制到三个搜索seed的配对参考当作三次独立实验。原 `zero_one` 适配版本轮不加入；若以后加入，单独列补充资源协议。

配置由 `scripts/prepare_basic_validation.py` 生成，清单 `configs/research/basic_validation_controls.json` 冻结既有搜索pipeline和摘要哈希。既有模型、交通、评估配置完全复用；新增配置、审计脚本和本协议在启动前提交。search摘要由已完成的4,500episode引用，搜索方法不重跑。

## 审计与解释

每批完成后自动执行 `scripts/analyze_basic_validation.py`，通过再进入下一批，失败停止并保留日志。检查固定源码、版本、权重和配置；逐步核对原始回报、轨迹、动作、终止、扰动、成本和随机seed；重新计算ASR分母；首次动作分歧之前要求与Clean输入/转移一致。BO额外检查候选搜索界、目标轨迹、获胜候选及在线适配语义。Clean与既有搜索验证的首批Clean逐字段回归，只排除wall time与无效attack seed。

使用既有legacy运行环境：Python3.7.16、Torch1.3.1+cpu、NumPy1.21.6、SUMO1.22.0。每次启动manifest记录实际commit、命令、环境版本、配置、有效seed和checkpoint哈希。最终split30继续保留。

最终形成三张主表：

1. 效果表17行：Clean及四简单方法；预算版Zero-One Return/Safety、当前方法Return/Safety各三档。报告实际episode数、Return、Return Drop、Collision、ASR分子/分母及访问协议。Clean/FGSM按100个真实条件报告，随机方法和每个搜索条件按300条相关记录报告。
2. 成本表与效果行对应：攻击梯度、forward、目标评估、仿真新转移与总费用；报告总量、每episode、每真实交互步，另注明通用评估器和仿真setup费用。并行wall time不作为速度优势证据。
3. 核心搜索配对表6行：三预算×两目标，完整保留双方独有碰撞、净差、回报/成本差及交通集中性。简单对照不会改变既有核心结论。

附录保留逐checkpoint和attack seed、攻击率/动作改变率、扰动范数及安全指标。不同采样范围的TTC/DRAC或分位数不随意混合；SUMO碰撞仍包含原minGap语义。可对搜索与简单方法做同交通配对，但不同权限和目标的结果不能单独归因于“增加仿真搜索”这一模块。若成本更高而效果没有改善，直接报告负结果。

## 启动

启动前100项测试通过。输出目录必须是新的忽略目录：

```powershell
& 'E:\Programs\anaconda3\python.exe' scripts/run_basic_validation.py --output .local/runs/20261001-basic-validation-controls
```

顶层 `basic-validation.json` 记录已审计episode及固定SHA；各批日志和 `verified-basic-validation.json` 保存完整输入哈希与结果。全部三批通过后再形成正式17行比较，不把中途结果当作完成。

## 完成记录（2026-10-01）

三批1,300episode、225,248步全部通过；300个Clean回归控制逐步一致。17行完整比较、资源计费与六组核心搜索配对已形成，见 [VALIDATION_COMPARISON_RESULTS.md](VALIDATION_COMPARISON_RESULTS.md) 和 [VALIDATION_COMPARISON_TABLES.md](VALIDATION_COMPARISON_TABLES.md)。配置与算法仍按预登记冻结。FGSM的ASR16.09%；当前方法三档16.48%/16.09%/18.39%，高档额外转换集中于checkpoint2的两种交通。相对预算版Zero-One的碰撞与成本限制仍成立。104项测试通过，final split30保留。
