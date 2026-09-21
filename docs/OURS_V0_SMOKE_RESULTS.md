# Ours-v0 首轮开发与工程验证结果

日期：2026-09-21。结论：P0/P1 工程验证通过，P2 四条件接口已经实现；当前结果不支持声称 Ours 获得碰撞优势或计算效率优势。完整开发集、验证与最终测试尚未运行。

## 实验与验证范围

- 首轮 commit：`6edc06a1b9a39db34cdcbdc84efc1fff92b1e9c3`。
- 首轮批次：`.local/runs/20260921T071026843375Z-attack-benchmark`。
- 五个冻结 Clean checkpoint，每个共用两个新开发交通 episode，Clean 加四个搜索条件，共 50 个 episode、8,504 个实际步骤。这是十个 model/traffic 配对单元，不是五十个独立样本。
- 其中四个攻击条件共 6,532 个实际步骤，全部通过 oracle 的逐步精确验证；预算超额、未验证 live fallback 均为零。
- 重复批次：`.local/runs/20260921T071839513600Z-attack-benchmark`；commit `3b2dbb765f5b2cf7f19daf480bc7900ddc8ab3dd`。覆盖 checkpoint 0、2 的十个条件、20 个 episode、2,504 个实际步骤。只排除 `attack_cost.wall_seconds`，其余每个步骤字段均与首轮一致。
- 审计器 commit：`226e645da92dbcdbdde0ec31c0bb1cf34212aa14`；全部 63 项单元/回归测试通过，包括原始更新等价检查、预算原子拒绝、reset 预热、失败目标处理、冻结模型、计数与篡改检测。
- 实验运行时沿用 Python 3.7.16、PyTorch 1.3.1+cpu、SUMO 1.22.0；启动器已与冻结训练的 Python、pip freeze、SUMO 版本逐项核对。完整配置、启动命令、OS、角色 seed、模型及代码哈希保存在两个批次的 manifest。

重复实验期间仅修正注册接口的默认 objective 绑定并新增审计器；首轮配置始终显式传入 objective。真实逐步比较确认显式配置的运行结果未因此改变。

交通 seed 为 `1472154258`、`1940738230`；独立 attack seed 为 0。80 个开发/验证/最终测试交通 seed 清单规范 payload SHA-256 为 `564bd6606d8e0b3ed94b05d7270d6dd241ddbd24a0959cc46271be1de776e93f`。与记录的旧训练/评估及 upstream seed 无交集。本次两个 episode 属于开发集，不能再计入未见测试。

## 初步结果与限制

以下每个 checkpoint 的单元均是两个 episode 的平均 Return；攻击方通常希望更低。此表是工程 smoke 描述，不给出统计显著性或泛化结论。

| Checkpoint | Clean | Budgeted Zero-One Return/Safety | Ours Return/Safety |
| --- | ---: | ---: | ---: |
| 0 | 126.124425 | 127.793966 | 124.288982 |
| 1 | 126.436620 | 113.814787 | 102.074241 |
| 2 | 148.851081 | 10.445766 | 10.387719 |
| 3 | 126.436620 | 125.472805 | 125.472805 |
| 4 | 126.399925 | 124.043832 | 123.962394 |

这两个开发 episode 上，每种搜索的 Return 与 Safety 版本给出相同结果；不能据此主张 Safety 目标有效或无效。Clean 碰撞 1/10，四个攻击条件分别都是 2/10；按配对 Clean 无碰撞为分母的新增碰撞 ASR 均为 1/9，新增事件仍来自 checkpoint 2。Ours 在部分 checkpoint 降低回报，没有提高本样本的新增碰撞率。

每个搜索的 Return/Safety 两版本在本次运行中具有相同计数，下面各列仅计一个版本，避免把相同交通下的两个目标当作独立证据。

| 量 | Budgeted Zero-One | Ours |
| --- | ---: | ---: |
| 规划块 | 83 | 83 |
| 完整搜索候选（不含保底） | 815 | 746 |
| 分块去重的完整实际序列 | 223 | 216 |
| 重复完整实际序列 | 592 | 530 |
| 分块去重的实际动作前缀 | 3,602 | 3,712 |
| 输入梯度计算 | 11,548 | 20,856 |
| 策略 forward | 32,414 | 51,668 |
| 新 shadow transition | 3,964 | 4,279 |
| shadow step（含 replay 与规划 reset 预热） | 35,373 | 35,864 |
| replay step | 16,157 | 15,513 |
| 规划 reset 预热 step | 15,252 | 16,072 |
| 因预算耗尽结束搜索的块 | 8 | 24 |
| 失败目标尝试 / 重试尝试 | 3,180 / 0 | 8,611 / 4,996 |

四条件共享上限，但实际消耗并不相同。Ours 的重试使梯度和 forward 明显增加，完整候选数量更少；略多的前缀覆盖不能直接解释为效率提升。上述覆盖计数也包含不同方法访问不同状态、提前终止的影响，不是单一机制的因果证明。

物理 shadow 总计 142,474 步为规划期间成本；各 episode 初始 oracle 预热另外记账。预算耗尽均安全保留了已完成计划，没有超额补算或将部分低回报轨迹伪装为完整候选。碰撞仍为原 SUMO minGap 语义；本轮未引入 Gate、物理约束或新风险度量。

## 已提交的原子修改

| Commit | 内容 |
| --- | --- |
| `ea39167` | 多资源预算与 oracle 原子预检 |
| `bd65b22` | 完整历史节点、PGD witness 与行为搜索 |
| `f78c1f8` | 同预算 Return/Safety 对照 |
| `74fd516` | 将 reset 预热纳入新仿真预算 |
| `6edc06a` | 研究 seed 清单、配置及统一评估接入 |
| `c4619fb` | 注册名称与默认 objective 一致性 |
| `3b2dbb7` | 计划、预算、配对轨迹审计及接口文档 |
| `226e645` | 固定 seed 逐字段重复性验证 |

当前继续保留 `feat/proposed-attack`，不创建 `v0.5.0-proposed`。`main.py`、`oarl.py`、`Environment/`、`Data/`、原 `zero_one.py` 与 `configs/frozen_victims.json` 相对开发前 `de9b10d` 无修改；Zero-One 参考目录 118 个来源文件哈希全部一致。模型和原始实验结果均保留在忽略的 `.local`，没有公开 push。

下一步使用已冻结的十 episode 开发配置检查收益是否稳定，重点观察失败目标重试的梯度成本、checkpoint 0/1/3/4 的新增安全事件和同目标配对差异。现阶段不叠加新模块，也不把 smoke 回报下降写成论文结论；若收益继续只来自 checkpoint 2 或更高成本，应收窄或否定方法假设，再决定是否修订调度规则。

机器摘要见 `OURS_V0_SMOKE_RESULTS.json`；完整审计输入哈希及重复性明细位于首轮批次 `verified-smoke-summary.json`。运行与接口说明见 `OURS_V0_IMPLEMENTATION.md`。
