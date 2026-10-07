# D2：基础训练支撑与完整状态恢复结果

2026-10-07。新增训练适配支撑已通过固定 batch、真实 SUMO 原循环等价和独立进程恢复检查。Clean 与原 OARL 的逐 episode 轨迹、最终完整数值训练状态一致。该结论只说明工程兼容性；没有训练新的防御，没有增加有效性或创新性结论。

执行协议见 [DEFENSE_TRAINING_SUPPORT_PROTOCOL.md](DEFENSE_TRAINING_SUPPORT_PROTOCOL.md)，机器可读汇总见 [结果 JSON](DEFENSE_TRAINING_SUPPORT_RESULTS.json)。OARL Robust 的领域强基线定位、既有冻结模型与所有原实验入口保留。

## 完成的能力

- 在 `rl_att/training/` 用组合适配原 Agent/CleanVictimAgent，保留原采样、FIFO、更新时刻与 BO/dual 更新。根目录、SUMO 环境和既有 Clean 算法文件没有改动。
- 区分推理 actor-only 与完整训练 snapshot。后者保存 actor、双 critic/target、dual、梯度、Adam、replay/cursor、全局与策略/辅助 RNG、训练进度、环境 reset 属性、成本账本，以及注册的额外网络和优化器。
- 辅助 attack、belief、initialization、sampling 各有隔离 NumPy/Torch 流。旧环境与 replay 共享的基础 NumPy 流保留，避免为了整理 seed 改变原算法轨迹。辅助流的存在不表示已经定义 belief 模型。
- 成本统计实际网络调用、输入 observation 行数、优化器更新、BO objective、真实交互及 reset warmup；恢复进程另记录 segment 增量，避免重复累计 prefix。额外仿真/辅助数据留有显式账本，本轮没有启用这些路径。
- 导出原格式的新 actor，检查重载后的概率及哈希，记录 `engineering_only=true`、`benchmark_eligible=false`。不替换旧 checkpoint 或加入正式冻结模型注册表。

Torch 1.3 的 optimizer 序列化使用进程内参数对象 ID；审计按参数组顺序规范化 ID 后比较实际 Adam 数值，未改变更新运算。跨进程 pickle 文件字节不要求相同；完整数值状态与文件完整性分别审计。

## 真实 SUMO 验证

两算法各用 run seed 0，16 episodes、每集最多 16 步，batch 128、replay 容量 1,000,000，沿用基础超参数与训练 seed schedule。仍按原零基 episode >10、累计真实交互每两步一次更新。完成第 13 个 episode 后保存，确保恢复时已经存在非空 Adam 和 replay 状态。

| 路径 | 每算法物理 episode | 结果 |
| --- | ---: | --- |
| 原 wrapper 执行原 main.py | 16 | 与新连续路径逐 episode 轨迹和最终完整数值状态一致 |
| 新支撑连续执行 | 16 | episode-13 boundary 与独立 prefix 一致 |
| 独立进程执行 prefix | 13 | 完整 snapshot 作为恢复输入 |
| 另一个进程恢复并完成 suffix | 3 | prefix + suffix 的轨迹、最终完整数值状态和累计计数与连续路径一致 |

八个 worker 全部成功，总计 **96 次物理 episode、9,078 次实际 SUMO step**（包含 reset warmup）。父运行起止时间间隔约 42.11 秒；该时间包含进程启动、保存、记录和审计，不是新防御训练性能比较。

| 单个连续 16-episode 逻辑路径 | Clean | OARL |
| --- | ---: | ---: |
| 真实交互步 | 199 | 199 |
| reset warmup 步 | 1,314 | 1,314 |
| 主训练更新 | 35 | 35 |
| BO objective 实际调用 | 0 | 175 |
| 训练 actor forward：调用 / observation 行数 | 70 / 8,960 | 420 / 53,760 |
| 每个 reward critic forward：调用 / 行数 | 70 / 8,960 | 70 / 8,960 |
| 每个 target critic forward：调用 / 行数 | 35 / 4,480 | 35 / 4,480 |
| actor、qf1、qf2 各 optimizer.step | 35 | 35 |
| dual optimizer.step | 0 | 35 |
| 交互 actor forward：调用 / 行数 | 199 / 199 | 199 / 199 |
| 辅助更新、被攻击交互、特权分支/回放 | 0 | 0 |

提前终止使实际交互小于 256 步上限。96 次执行包含用于验证同一逻辑路径的重复，不能当作 96 个独立交通样本；它们也不是防御验证或最终保留交通。

独立 auditor 从绑定 SHA256 的原始 episode、snapshot 和 canonical 文件重新计算状态摘要，核对原循环/连续/恢复最终相等、prefix/boundary 相等、实际 BO 次数、optimizer 次数、segment 成本和 actor 产物资格。完成后的只读复核与原审计报告一致，没有重复运行 SUMO。

## 测试与执行身份

相关 **22 项测试通过**：训练支撑 9、原始记录审计 5、原 Clean 更新 1、checkpoint 验证 2、seed 协议 5。测试包含 fixed-batch 原更新等价、满 FIFO wrap、非空 Adam、额外组件恢复、随机流隔离、错误身份/哈希、episode 内保存拒绝、推理文件恢复拒绝，以及计费 hook 异常清理。

执行源码提交为 `0ea0071d10ab4b835c49c6efc5319fcffa3613a7`；配置和 auditor 已随该提交固定，整个批次期间 tracked tree 不变。基础实现提交为 `337f94e`。批次结束后，`29d0daa` 补充了显式额外网络的实际 forward/optimizer 计费，并通过针对性单元测试；96-episode SUMO 矩阵没有使用该辅助训练路径，不能把它记成新的 SUMO 实测能力。

运行目录为 `.local/runs/20261007-defense-training-support`。父 manifest SHA256 为 `c8a10cab68e3b1dfbb06cce0ea6e5c8a6afc96591474da720f81f1e0c22b6716`，audit SHA256 为 `138ea55f28953a9a9822e7f625bce3c8217dd8742e2562694314681fbbe0fc32`。冻结模型注册表运行前后均为 `abdfb1437842832134fca2ff7de5d8c9091c27014f4d7e18bfb0e9ec062f5af9`。完整网络状态、各文件及 actor 权重哈希见结果 JSON 和原始审计，不把大文件提交 Git。

环境：Windows 10 build 26100、Python 3.7.16、Torch 1.3.1+cpu（单线程）、NumPy 1.21.6、Gym 0.15.4、SUMO 1.22.0。manifest 保留完整 `pip freeze --all`、SUMO 版本输出、命令、输入指纹及全部角色 seed。基础 run/policy/python/numpy/torch/attack seed 均为 0；SUMO 使用已登记的逐 episode 派生名单。辅助四角色 seed 分别为 1500372476、1060443234、1886378803、797080042。

原执行命令如下；目录已存在，不应为查看结果再次启动。

```powershell
& .local/envs/oarl-legacy/python.exe scripts/validate_training_support.py --output .local/runs/20261007-defense-training-support
```

## 适用范围与下一步

恢复支持 episode 结束、下一 reset 之前的边界；不支持进行中的 SUMO 状态恢复，不承诺跨源码/配置/运行环境恢复。当前是一个训练 seed、短 horizon 的兼容性检查，不能替代 400-episode 多 seed 收敛或攻击评估。

D2 的基础训练状态、随机流、成本和工程产物登记已完成；正式防御模型冻结登记和新交通 namespace 仍需独立实现。下一项可以在独立分支实现 **PGD 一致性基础比较方法**，明确损失、采样和额外成本，先完成固定 batch 与短程工程验收，再登记长程训练。它不是自己的方法母体。

ACoE 仍须先明确即时反事实误差来源、belief、离散策略目标和 off-policy 更新，不能从任意数值 16D 候选伪造完整 SUMO state 或 reward。母方法和自己的机制训练继续后置；OARL Robust 保留为独立领域强基线。
