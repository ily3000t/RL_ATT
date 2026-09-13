# Clean Victim 完成与配对结果

Clean Victim 已在 OARL 长程复现完成后，从头训练并完成全部验证：5 个 run_seed、每组 400 episodes、每集最多 200 steps，20 个 actor checkpoint 全部通过保存后和独立进程验证。未使用 BO、actor/critic 鲁棒项或 dual 更新，未增加 Gate。

## 最终 checkpoint 对照

每个 run 均使用预先固定的 episode-400 checkpoint，在相同的 20 个留出 SUMO episode 上执行 greedy argmax；没有外部观测攻击。± 为 5 个 run 的样本标准差。

| Victim | 训练末 100 平均 return | 冻结评估 return | 碰撞 episode 比例 |
| --- | ---: | ---: | ---: |
| OARL Robust | 134.808 ± 2.925 | 134.661 ± 8.388 | 9/100 |
| Clean | 134.652 ± 5.367 | 129.488 ± 8.404 | 12/100 |

| Clean run_seed | 训练末 100 return | 冻结评估 return | 碰撞 episode 比例 |
| --- | ---: | ---: | ---: |
| 0 | 135.736 | 141.374 | 0% |
| 1 | 134.726 | 134.011 | 10% |
| 2 | 142.190 | 127.547 | 20% |
| 3 | 133.417 | 119.773 | 15% |
| 4 | 127.190 | 124.735 | 15% |

这些结果描述本机固定协议下的无攻击表现；不用于宣称统计显著性或对抗攻击下的优劣。碰撞来自 SUMO 在交互 step 报告的 Auto 碰撞，不包括 reset 内事件。TTC/DRAC 与攻击成功率尚未纳入本轮。

## 控制与验证

- Clean 共完成 352,395 个交互 step、174,576 次 actor/critic 更新；BO objective 次数为 0，全部 episode 的 JS 与 dual 字段为 null。
- 两类 victim 使用相同网络初始化路径、超参数、replay、更新与保存规则、SUMO 环境/交通、训练和留出种子、依赖版本。不同策略导致的轨迹、提前终止及实际更新次数差异均保留。
- 配对源码/配置/seed/运行时与评估审计见 [VICTIM_CONTROL_AUDIT.json](VICTIM_CONTROL_AUDIT.json)。
- 16 项单元测试通过，包括与零鲁棒项 OARL 的完整更新等价、禁止 Clean 调用 BO/dual，以及原 OARL 回归检查。
- 两类 victim 共 40 个 checkpoint 完成文件/权重哈希、64 个真实观测探针输出一致性验证，并各执行 20 个留出 episode；评估前后 checkpoint 与权重不变。
- Clean 是 OARL 派生的匹配 actor-critic，未额外加入标准 SAC 的熵项；实现与损失定义见 [CLEAN_VICTIM.md](CLEAN_VICTIM.md)。

## 冻结模型入口

最终 10 个 checkpoint（两类 victim 各 5 个 seed）已固定在 [configs/frozen_victims.json](../configs/frozen_victims.json)。所有路径相对仓库根目录；加载前验证文件哈希，使用已记录的 Python/PyTorch 环境，设定 eval 模式及 requires_grad_(False)，后续攻击不训练或替换这些 victim。清单只包含引用和哈希，模型文件仍留在被忽略的 .local/。

这些 actor-only checkpoint 用于冻结推理，不是精确训练续跑快照。每个模型的训练 SHA、配置、checkpoint SHA 和权重 SHA 均已记录；没有依据评估成绩重新选择训练 checkpoint。

机器可读结果见 [CLEAN_VICTIM_RESULTS.json](CLEAN_VICTIM_RESULTS.json) 和 [OARL_PROTOCOL_A_RESULTS.json](OARL_PROTOCOL_A_RESULTS.json)。对照曲线为 `.local/reports/victim_comparison.png`，可复建：

```powershell
python scripts/plot_reproduction.py --summaries docs/OARL_PROTOCOL_A_RESULTS.json docs/CLEAN_VICTIM_RESULTS.json --output .local/reports/victim_comparison.png
python scripts/export_frozen_victims.py --summaries docs/OARL_PROTOCOL_A_RESULTS.json docs/CLEAN_VICTIM_RESULTS.json --output configs/frozen_victims.json
```

## Git 与后续阶段

- OARL 训练 SHA：`97fdb5767cae6bfbfccda5d2e2dac3913ad5f467`；里程碑 `v0.1.0-oarl-reproduced`。
- Clean 训练 SHA：`4be94fad8ce43a516f63b6edffd7f9f474496598`；开发分支 `feat/clean-victim`。
- checkpoint、raw results、logs、环境及图像未进入 Git；本轮未执行公开推送。此前失败批次保留，详见 [REPRODUCTION_ATTEMPTS.json](REPRODUCTION_ATTEMPTS.json)。
- 下一步按 Stage 2 解耦 Attack / Victim / Evaluation，基于已冻结的 Clean checkpoint 建立统一预算与可靠安全指标，再开展基础攻击 benchmark。当前没有接入 Zero-One、Ours 或 PPO。

本功能原子提交：

- `d0fd0b8` feat(victim): add OARL-matched clean actor-critic without robust terms
- `3fb015c` feat(repro): select and record clean victim training explicitly
- `4be94fa` chore(experiments): define matched five-seed clean victim protocol
- `cc683f5` feat(repro): verify absence of robust activity in clean run summaries
- `7b266ab` feat(victim): export hash-pinned final checkpoint references
- `f00a817` docs(victim): verify matched OARL and clean training controls
- `5bbe19c` chore(victim): pin validated final OARL and clean checkpoints
