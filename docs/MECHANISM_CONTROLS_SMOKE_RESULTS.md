# Return 机制对照：工程验证结果

2026-10-03。四个 Return 搜索条件及 Clean 已在五个冻结模型上完成工程验证。新对照 `ours_single_return` 只是原行为搜索的单次尝试包装器，没有改变原攻击、OARL、SUMO、冻结权重或 Zero-One 参考源码。此轮数据不能用作论文有效性或泛化证据。

实验 commit 为 `d17e9202aa7f80bcc266d426b2b0935322b91994`；审计 commit 为 `3cce1a8316351964b49cc4f8099b46d929a49084`。批次为 `.local/runs/20261003T034414646128Z-attack-benchmark`。五 checkpoint、两个已使用过的开发交通 episode、attack seed 0、400/800 梯度/forward 上限、最多 200 步，共 50 个 episode 和 8504 个实际步骤。完整配置、命令、Python 3.7.16、Torch 1.3.1+cpu、SUMO 1.22.0、其他环境版本及权重哈希保存在每个 run manifest。

全部 110 项测试通过。25 个模型/条件记录的原始 episode Return、终止、动作、轨迹摘要及成本核对通过；20 个攻击条件的计划执行、扰动包络、预算和 oracle 校验通过。两个无重试搜索条件的 attempt 序号全部为 0；其余条件的重试次数与输入梯度成本一致。六组方法配对在每个模型上共有的首次尝试 seed、实际诱导动作及最终 margin 全部相同。

Clean、预算版 Zero-One Return、固定重试 Return 与原 smoke 共 5238 个实际步骤逐字段一致；进展版 Return 与历史进展 smoke 共 1633 步一致。两项回归只排除 `attack_cost.wall_seconds`，没有排除动作、扰动、候选或成本字段。

来源哈希审计单独修复了 Windows 导出换行：`git show` 的 blob 字节与本机 `git archive` 导出的 CRLF 字节不同。改为对启动器实际导出的字节计算哈希，冻结源码映射与五个已完成 manifest 一致。修复仅涉及配置生成器的哈希记录和协议来源映射，没有改动任何执行源码或实验参数，因此无需重跑实验；保留实验与审计的不同 commit。

## 描述性成本与结果

每行均为相同十个 model/traffic 单元的结果；只有两个交通 seed，不能当作十个独立交通样本。Shadow 步数为规划期间实际物理成本，包含 replay 和规划 reset 预热；episode 初始 oracle setup 另有记录。

| 条件 | 平均 Return | 碰撞 | 新增碰撞 ASR | 梯度 | Forward | Shadow 步数 | 独立重试 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 130.849734 | 1/10 | — | 0 | 0 | 0 | 0 |
| Budgeted Zero-One Return | 100.314231 | 2/10 | 1/9 | 11548 | 32414 | 35373 | 0 |
| 行为搜索单次尝试 | 97.170082 | 2/10 | 1/9 | 10962 | 31992 | 35738 | 0 |
| 行为搜索固定重试 | 97.237228 | 2/10 | 1/9 | 20856 | 51668 | 35864 | 4996 |
| 行为搜索进展重试 | 97.237228 | 2/10 | 1/9 | 17228 | 44442 | 35884 | 3155 |

| Checkpoint | Clean | Zero-One | 单次尝试 | 固定重试 | 进展重试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 126.124425 | 127.793966 | 124.288982 | 124.288982 | 124.288982 |
| 1 | 126.436620 | 113.814787 | 101.738509 | 102.074241 | 102.074241 |
| 2 | 148.851081 | 10.445766 | 10.387719 | 10.387719 | 10.387719 |
| 3 | 126.436620 | 125.472805 | 125.472805 | 125.472805 | 125.472805 |
| 4 | 126.399925 | 124.043832 | 123.962394 | 123.962394 | 123.962394 |

在本 smoke 中，无重试的行为搜索相对无重试 Zero-One 平均 Return 低 3.144149，梯度少 586、forward 少 422，shadow 多 365；这是外层搜索组织的条件比较，不能证明历史缓存单独造成收益。

固定重试相对单次尝试反而使平均 Return 高 0.067147，新增 9894 次梯度和 19676 次 forward，十个配对轨迹中九个相同。当前小样本没有显示重试提高攻击效果，不能预设该机制有益。进展规则相对固定重试保持十个实际轨迹相同，减少 3628 次梯度、7226 次 forward 和 1841 次重试，但 shadow 增加 20；减少计算也不能解释为新增安全优势。

这些结果记录了有可能削弱方法主张的情况，不据此改变方法或挑选预算。完整三预算、三 attack seed、五 checkpoint、十开发交通 episode 的 2250 episode 研究尚未执行。下一项功能在新的独立分支冻结并运行该网格，以检验上述条件效果是否稳定、额外重试是否值得成本。Safety 的既有完整比较继续保留；最终 split 30 未使用，尚不创建算法有效性里程碑 tag。

完整审计位于批次 `verified-mechanism-summary.json`；小型摘要为 `MECHANISM_CONTROLS_SMOKE_RESULTS.json`。可用以下命令重建摘要：

```powershell
.local/envs/oarl-legacy/python.exe scripts/analyze_mechanism_controls.py --batch .local/runs/20261003T034414646128Z-attack-benchmark/batch.json --output .local/runs/20261003T034414646128Z-attack-benchmark/verified-mechanism-summary.json
E:/Programs/anaconda3/python.exe scripts/summarize_mechanism_smoke.py --audit .local/runs/20261003T034414646128Z-attack-benchmark/verified-mechanism-summary.json --output docs/MECHANISM_CONTROLS_SMOKE_RESULTS.json
```

原子修改：`06cff6b` 单次尝试对照及测试；`276e007` 协议、来源和配置；`d17e920` 逐步机制审计及历史回归；`3cce1a8` 导出源码哈希兼容修复。本功能开发和验证完成后按 `AGENTS.md` 合入 `main`，历史分支保留。
