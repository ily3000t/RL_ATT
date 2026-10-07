# PGD 一致性五 seed 长程训练结果

五个 400-episode run 已完成，全部 20 个 checkpoint 通过审计和重新加载。当前明确发现是：**该配置存在两个持续低收益、常动作的训练 run，训练碰撞更少不能单独作为防御成功的证据。** 全部五个预先固定的 episode-400 模型均已登记，不剔除 seed 2/4，不换成中间 checkpoint，也不据此调 λ。

训练源提交 `1e91acbba12f5ec868536fc33d820e171c89ffae`，只读分析源提交 `43b1238a514bf60da1cc38a497704dc776c30102`。Protocol A，run seed 0–4，Normal / Pn=0.14、每集最多 200 步，原 16→128→3 网络；λ=0.1、五步 PGD、一个独立随机起点，盒为 `0.2|o_i|+0.05`，无 Gate。沿用 Windows / Python 3.7.16 / Torch 1.3.1+cpu 单线程 / NumPy 1.21.6 / SUMO 1.22.0 环境。

## 完整性与冻结

- 实际完成 2,000 episodes、365,080 个交互步、164,234 个 reset warmup 步、180,918 次主更新。提前终止造成真实步数和更新数不同，80,000/run 是上限。
- 每个 seed 的 episode 100/200/300/400 都有完整训练状态和 actor-only 模型。全部五个 run、生产 auditor 和 batch 状态为 passed。
- 本轮重新计算所有已审计文件的 SHA256，绑定原逐更新盒/KL/loss 审计；重新流式检查全部转移的回报、轨迹、碰撞集合、观测连续性和更新节奏；20 个 actor 从文件重新加载，权重、64 个实际观测探针概率精确一致。没有重跑 SUMO，也没有再次扫描全部 PGD 更新数组重算 KL。
- 冻结清单为 [configs/frozen_defense_victims.json](../configs/frozen_defense_victims.json)，固定五 seed 的 episode 400。原 Clean/OARL 十个模型和旧 registry 保留。`benchmark_eligible=true` 表示工程上具备进入比较的资格，不代表防御有效、性能达标或策略全局收敛。
- 分析/统计与原登记边界共 16 项相关测试通过；配对交通、不完整窗口、碰撞分母、缺失模型以及常动作/高熵策略辨别均有检查。分析时 tracked tree 与 HEAD 固定。

## 训练收益与碰撞

下表全部是 **训练第 301–400 集** 的描述统计，采用训练中的随机策略，不能与历史冻结 greedy 评估或受攻击结果混用。已从原始训练记录核对三类策略在每个 run/episode 使用相同 SUMO seed；策略更新后的轨迹自然不同。± 为五个 run 的样本标准差，不是置信区间。

| 策略 | 末 100 集回报，五 run 平均 ± SD | 末 100 集观察到的 ego 碰撞 |
| --- | ---: | ---: |
| Clean | 134.65 ± 5.37 | 45/500（9.0%） |
| OARL Robust | 134.81 ± 2.93 | 55/500（11.0%） |
| PGD 一致性 | 117.61 ± 20.16 | 17/500（3.4%） |

PGD 一致性的平均回报比 Clean 低 17.05，训练参考相对下降约 **12.66%**。所有五个 run 的末 100 集回报均低于对应 Clean，但下降主要集中于 seed 2/4。训练中未实施观测攻击，因此该回报不是“攻击后回报”，碰撞比例也不是抗攻击成功率。

| run seed | Clean 回报 | OARL 回报 | PGD 一致性回报 | PGD 碰撞/100 | PGD 实际动作 2 比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 135.74 | 133.10 | 130.39 | 6 | 99.470% |
| 1 | 134.73 | 135.41 | 133.72 | 5 | 99.463% |
| 2 | 142.19 | 135.67 | 96.78 | 0 | 100% |
| 3 | 133.42 | 138.82 | 132.75 | 6 | 99.469% |
| 4 | 127.19 | 131.05 | 94.38 | 0 | 100% |

同一 run 中多个训练 episode 来自持续更新的策略，存在训练依赖；500 是记录的 episode 分母，不作为 500 个独立冻结策略测试样本。本轮不计算显著性或防御优势置信区间。

## 低收益 run 的行为证据

seed 2/4 从第 101 至第 400 集，**各自 60,000 个实际交互步全部选择动作 2**，没有选择动作 0/1。对应 300 集均跑满 200 步，没有观察到 ego 碰撞。末 100 集的动作前观测解码平均速度分别为 19.88 和 19.45 m/s；seed 0/1/3 为 26.23、26.49、26.62 m/s。这里用真实未攻击输入的 `observation[0]×35` 解码，只是训练访问状态上的描述性量，不额外声称 TTC、DRAC 或实际完成换道次数。

动作 2 在当前环境中表示“不发起新的换道请求”，不会显式取消已有请求或保证横向位置立即不变。常动作现象和低速、低回报共同支持“低收益保守策略退化”的诊断；但不能仅凭 action 2 比例高就判断退化：另外三个 run 的该比例也超过 99%，通常仍有每 episode 少量动作 0/1。动作次数不等于完成换道次数。

固定各自 run 最早采集的 64 个观测探针后，seed 2/4 的四个 checkpoint 都对全部探针 argmax 选择动作 2。episode 400 的平均最大动作概率为 1.0（float32 数值），平均熵分别约 `3.93e-9` 和 `5.94e-9` nats。seed 0/1/3 的最终探针仍有动作 1/2 区分，平均熵为 0.141、0.185、0.325 nats。探针属于各自 run，不能把它们当作所有模型共享的交通测试集合，也不能据此断言任意未见状态上都输出同一动作。

原独立 auditor 在 seed 2/4 的更新记录中观察到最小 KL 为零。高度饱和、少作状态区分的策略可能让一致性目标容易降低；**零 KL、少碰撞和高置信度均不能证明鲁棒性**。是否由当前正则项导致、受攻击时是否仍保持常动作、是否能够恢复必要决策，还需要冻结模型上的诊断控制与适应性攻击。工程审计已排除记录/加载不一致，但不能替代这类因果解释。

## 实际训练成本

五个 run 共 904,590 次 PGD 输入梯度、1,628,262 次训练 actor forward，actor/qf1/qf2 各完成 180,918 次 optimizer.step，dual 未更新。逐 run 完整 forward、输入行数、candidate 及 wall time 见 [JSON](DEFENSE_PGD_LONG_RUN_RESULTS.json)。实际交互没有被 PGD 扰动，没有额外仿真训练转移。

每次主更新实际为九次 actor forward：原干净 actor 路径两次、搜索六次、一致性 outer 一次；五次输入梯度。D3 的同框架工程对照中 Clean 为两次 actor forward，OARL 为十二次；这只是 actor 调用结构比较，不是总训练速度或内存优势排名。critic、BO optimizer、日志、SUMO 及保存/审计成本均不能忽略。

五个 CPU 单线程 run 并发，批次 UTC 11:07:56–11:24:13，约 **16 分 17 秒**，含启动、记录与审计。单 run 训练器 wall time 为 847.06–909.61 秒，包含 SUMO、gzip/JSON 和 checkpoint 保存。历史 Clean/OARL 长程没有同样的完整 phase 计费和记录负载，本轮不声称训练加速。部署仍为一个确定性 actor，没有新增在线网络；推理延迟尚未专门测量。

## 可复现入口与下一步

训练配置和命令见 [预登记协议](DEFENSE_PGD_LONG_RUN_PROTOCOL.md)。全部模型、完整 RNG、seed 名单、pip freeze、命令和原始记录仍在 `.local/runs/20261007T110756Z-defense-pgd-consistency-long`。

只读分析可复建到新的 ignored 目录：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/analyze_pgd_training.py --batch .local/runs/20261007T110756Z-defense-pgd-consistency-long --output .local/reports/defense-pgd-long-analysis-new
```

完整分析 JSON 和图像在 `.local/reports/defense-pgd-long-analysis`；本文对应紧凑 summary 保留完整报告哈希、各模型指纹、配对训练表和统计方法。Matplotlib 配置目录应继续置于 `.local/`。

目前 **可以把 PGD 一致性作为已实现并冻结的基础防御基线，但还不能判断其攻击下是否有效**。下一项应先独立实现和登记防御新交通 namespace，再在同一交通与全部五个冻结模型上比较 Clean、OARL Robust、PGD 一致性的 greedy 无攻击收益和重新生成的 PGD、预算版 Zero-One、Single；保留低收益模型，区分绝对攻击后回报、各自 Return Drop、碰撞和攻击成本。旧最终 split 30 不用于调 λ 或选版本。

如果后续探索不同 λ、恢复机制或论文目标，当前版本和本轮负例继续作为独立基线，不静默替换。ACoE/ATLA 的机制映射和自己的主要增量尚未确定；本轮不启动新训练、不开发创新模块，也不将当前策略退化自动认定为新方法的贡献。
