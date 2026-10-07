# D2：完整训练状态与恢复验证协议

2026-10-07。源码与本协议先提交，再开展短程 SUMO 工程验证。原 `main.py`、`oarl.py`、Clean 更新、环境与 frozen checkpoint 不修改，旧 `rl_att.experiment` 和训练入口保留。新增 `rl_att/training/` 支撑后续独立防御实现，不预先实现 ACoE 或自己的损失。

## 状态与随机流

actor-only 文件只用于推理；新增 full-training snapshot 保存 actor、双 critic、双 target、dual、四优化器的 Adam 状态、参数梯度/模式、FIFO 数组与 cursor、完整 Python/NumPy/Torch CPU RNG、策略流、辅助流、训练计数、探针、资源账本、注册的额外网络/优化器与环境 reset 属性。未填满 replay 保存实际使用前缀，恢复未使用零槽；满 buffer 保留实际数组顺序与 ptr。Torch 1.3 的优化器使用对象 ID，按参数组顺序归一化为稳定编号，仅用于序列化/审计，不改变 Adam 更新。

恢复仅支持 **episode 完成、下次 reset 前**。保存 `HighwayEnv.__dict__` 与 reset_times，在新进程构造并启动环境之后恢复，再执行下一原 seed 的 reset；不声称恢复某个正在进行的 SUMO episode。文件写入新路径并以 SHA256 验证，拒绝覆盖旧 snapshot、actor-only 恢复、源身份/配置/角色 seed/组件 schema 不一致和 mid-episode 保存。源身份默认严格绑定原提交；跨版本恢复需后续单独验证，不自动允许。

基础角色复用 Protocol A：run/policy/python/numpy/torch/SUMO/attack 记录完整。原环境与 replay 的 NumPy 全局流保留，以保证旧算法兼容；辅助流用 `SeedSequence([20261005, run_seed, role_id])`，role_id 为 attack=1、belief=2、initialization=3、sampling=4，各自有 NumPy 与隔离 Torch 流。组件创建与采样不得前进基础全局或 policy 流；独立辅助流不是 Bayesian posterior。

## 资源和信息权限

记录真实交互、reset warmup、主/辅助更新、被攻击交互、额外仿真分支/回放与候选观测。实际 forward hooks 分网络、分阶段记录**调用次数与输入 observation 行数**，批量 forward 不当成一条样本；实际 optimizer.step 计数，不把 OARL dual 或额外网络成本忽略。BO objective 数计实际调用。阶段 wall time 单独记录，不估计未测量 FLOPs、延迟或能耗。

恢复的账本是完整逻辑轨迹累计成本；每个新进程另记录相对 resume 起点的 segment delta。批次实际物理成本按进程 segment 加总，不能再次累计 prefix 成本。原始仿真状态/正常 observation 只用于训练和审计，工程配置未开启特权分支、辅助训练或观测攻击，这些成本为零；未来使用时必须显式计费并与母方法共享相同权限。

新 actor 使用原格式导出，检查重载概率与哈希，并生成模型产物记录。本轮记录标记 `engineering_only=true`、`benchmark_eligible=false`，不自动加入或覆盖既有 `configs/frozen_victims.json`。正式防御登记与验证/最终新交通 namespace 留待方法和训练协议冻结，工程 smoke 不产生新防御性能主张。

## 预登记短程比较

两种原算法各使用 run seed 0、原初始化/超参数和原训练 seed schedule，16 episodes × 最多 16 步、batch 128、buffer 1,000,000。仍按原零基 episode >10、累计交互每两步一次更新，split 固定在完成 episode 13 后，确保已经执行训练并产生 Adam 状态。

| 模式 | 执行 | 比较 |
| --- | --- | --- |
| reference | 旧 wrapper 执行未修改的 main.py，读取结束时状态 | 验证关闭防御的新训练支撑与原始循环等价 |
| continuous | 新 facade 连续 16 episodes，保留 episode-13 boundary | 与 reference 完整状态及逐 episode 轨迹一致 |
| prefix | 新进程从相同初始 seed 执行前 13 episodes | 与 continuous 的 boundary 一致 |
| resumed | 再启动新进程，加载 prefix full snapshot，完成最后 3 episodes | prefix + suffix 与 continuous/reference 轨迹一致，最终完整数值状态一致 |

总计两算法 × (16+16+13+3)=**96 次物理 episode**。seed 和短 horizon 用于工程检查，不构成 400-episode convergence 复现、独立交通统计或防御效果评估。逐状态 digest 覆盖所有网络/梯度/优化器/replay/RNG，不只比较最终 actor；wall time 不纳入数学一致性。独立审计核对原始 episode、资源 segment/累计、四个模式完整性、snapshot 和模型文件哈希。

先通过小 buffer 的固定 batch 与 wrapped FIFO 测试、非空 Adam 恢复、独立辅助流/额外组件恢复及错误边界测试。再通过记录器在 ignored Git 快照中运行上述矩阵，源码期间冻结，每个 child 只允许 SUMO 配置 seed 改动。环境沿用既有冻结 Python/Torch/SUMO，Matplotlib 配置位于 `.local/`。

```powershell
& .local/envs/oarl-legacy/python.exe scripts/validate_training_support.py --output .local/runs/20261007-defense-training-support
```

输出目录必须尚不存在。manifest 记录当前提交、配置、原始运行环境指纹、启动命令、所有模式的身份、恢复输入哈希与源快照。完成后 `rl_att.training.support_audit.audit_run(manifest_path)` 可只读复核，不再运行 SUMO。新 feature 完成相关测试与本协议审计后 `--no-ff` 合并本地 main，不因工程 smoke 创建 milestone tag。
