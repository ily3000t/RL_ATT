# 攻击定稿与最终未见交通测试预登记

登记日期：2026-10-04（Asia/Shanghai）。本阶段只冻结输入和分析方法，**尚未运行 split 30**。方法见 [ATTACK_METHOD_FREEZE.md](ATTACK_METHOD_FREEZE.md)，机器协议见 [attack_final_test.json](../configs/research/attack_final_test.json)。既有 split 20 已参与候选选择，不能作为独立最终测试。

算法冻结点为 `d218309627830be924c48ef7fd9d984be11da247`，正式候选为 `ours_single_return`。协议工具的实际执行 SHA 与算法冻结点分别记录；后续执行器和最终审计工具完成后，须先提交并固定新的**工具执行 SHA**再启动整个批次。算法、环境、checkpoint、预算和 seed 不因此改变。本次没有新攻击、防御、Gate、奖励或终止修改。

## 1. 方法、模型与完整矩阵

五个原始 Clean episode-400 checkpoint，训练 run_seed 0–4，引用、文件/权重哈希与训练来源沿用 [frozen_victims.json](../configs/frozen_victims.json)。不用最终结果选择 checkpoint。Normal 原 SUMO 场景、16D、greedy_argmax 三动作、每 episode 最多 200 步。全部方法使用相同真实交通与 `|delta_i| <= 0.2|observation_i|+0.05` 观测约束。

| 条件 | 固定配置 | 每方法实际攻击 episode |
| --- | --- | ---: |
| Clean | 无攻击；基本组 attack_seed 0 为唯一展示基准 | 250 个唯一模型/交通对 |
| Random | 固定扰动盒、每步 | 750 |
| FGSM | untargeted logit-margin，一步，无随机初始化 | **250** |
| PGD | 同 untargeted 目标，10 步，归一化步长 0.2 | 750 |
| OARL-BO | 原适配定义，5 次评估，每步 | 750 |
| 预算版 Zero-One Return | 目标动作序列搜索，原 Return | 每档 750 |
| 预算版 Zero-One Safety | 原碰撞优先目标；补充目标对照 | 每档 750 |
| **Single Return** | 实际行为分支搜索，每节点/目标一次诱导 | 每档 750 |
| Progress Return | 同实际行为搜索，最多三次、margin-progress 重试 | 每档 750 |

所有参数复制已审计的 `basic_validation_*`、`budget_validation_*`、`single_validation_*` 配置，除完整最终交通名单/episode 数及所选方法列表外不变。不会强迫简单攻击用满搜索梯度。预算版 Zero-One 的配置 `max_attempts=3` 不表示实际三次重试，目标序列路径实际使用 attempt 0；详见方法定稿。

| 搜索预算档 | 梯度上限/规划块 | 攻击 forward 上限/规划块 | 新 shadow transition | physical shadow step |
| --- | ---: | ---: | ---: | ---: |
| 低 | 100 | 200 | 200 | 4000 |
| 中 | 200 | 400 | 200 | 4000 |
| 高 | 400 | 800 | 200 | 4000 |

沿用 horizon 20、10 条搜索候选、2 步 targeted PGD、归一化步长 1 和既有尾块预算规则。完整干净保底、执行核验和回退均按原账本计费；不是每 episode 总额，不能仅以配置上限代替实际成本。

最终矩阵：基本组 3 个 attack seed × 5 模型 = 15 份配置；搜索 3 档 × 3 attack seed × 5 模型 = 45 份配置，共 **60 份 / 12 批**。

- 基本组：3,250 个原始 episode，其中攻击 2,500、Clean 750。FGSM 仅在 attack_seed 0 运行，另两组不包含 FGSM。
- 搜索组：11,250 个原始 episode，其中攻击 9,000、Clean 2,250。
- 总计 **14,500 个原始 episode = 11,500 个攻击 + 3,000 个 Clean**。Clean 中仅 250 个模型/交通对是唯一展示基准，其余 2,750 个是回归检查，不增加独立样本数。
- 比较表固定 17 行：Clean 和四个简单攻击，另加四个搜索条件各三档。五模型、50 交通是 **250 个模型/交通对，50 个交通聚类**，不是 14,500 个独立交通样本。

本轮不加入原资源协议 `zero_one`、Single Safety 或 Progress Safety。既有这些方法及历史结果保留；若另作原 Zero-One 补充实验，须单独声明资源协议，不能混进匹配预算的主比较。

## 2. Seed 与运行环境

保持 [research_seed_splits.json](../configs/research_seed_splits.json) 原登记名单和顺序，使用 **split 30 全部 50 个 seed**，不抽取有利子集。root_seed=20260921，attack_seed=0/1/2。既有开发/验证/训练 seed 的无交集检查继续有效；预登记时读取 seed 列表不运行交通。

- `checkpoint_training_seed` 标识原五模型。评估交通与模型 seed 解耦，不把 checkpoint 2 的差异直接归因于训练或交通之一。
- 每 episode 的 SUMO seed 为固定 split-30 列表；Python/NumPy/Torch/初始 SUMO 公共环境 seed 由 `SeedSequence([20260921,1,30,2**31-2])` 的既有规则确定，所有模型和攻击相同。
- inference policy 使用 greedy_argmax，无活跃的 policy 抽样 seed；不虚构 `policy_seed`。三个 attack seed 使用既有独立攻击随机流；NoAttack/FGSM 没有随机攻击抽样。
- Random/PGD 复用 episode/step 局部 NumPy seed；行为诱导按完整实际历史、目标、attempt 的 SHA-256 seed；ZO 外层 phase-5 seed；BO 每调用重置其既有优化器/丢弃 Torch 抽样规则。源实现不变，最终 manifest 记录完整角色值和 RNG 标签。

固定现有 Python 3.7.16、Torch 1.3.1+cpu、NumPy 1.21.6、SUMO 1.22.0 及训练 manifest 的完整 `pip_freeze`、版本 stdout、线程和 PATH。预检实际探测，不以本表代替环境记录。ZOOpt 0.4.2 wheel 校验 SHA-256 并继续原 zipimport 加载，不安装依赖或修改训练环境。

协议记录 52 个运行源码/场景文件的 Git archive 字节哈希、模板、五模型、训练 manifest、seed 注册表、方法文档与旧证据。新配置用明确的 canonical JSON SHA-256，避免 Git LF/CRLF 转换造成无语义的哈希差异；执行 manifest 仍保留实际配置/源码字节哈希。

## 3. 研究问题、统计与结论边界

**主要检验：** 每一档报告 `Single Return − 预算版 Zero-One Return` 的配对 episode Return 差；负值表示更强损害。报告全部三档，不以一个最有利档代替主结果。Safety 是不同目标的补充，不能取代匹配 Return 的比较。

**机制对照：** 每档报告 `Single − Progress` 的连续效果变化及分别核算的成本变化。没有预设非劣效/等价界限，不以碰撞计数相同证明 Return 等价，也不把模型计算下降写成总速度提升。

**简单攻击对照：** 各档 Single 与固定简单攻击配对，回答额外 trajectory search 的效果和实际成本；明确其额外仿真权限。FGSM 的确定性结果可在配对计算时与三个攻击重复对应，但真实 episode 数仍为 250、成本只计算其实际运行，不制造三个独立副本。

逐模型/交通先平均三个攻击重复，再对五个固定 checkpoint 等权平均，得到 50 个配对交通差。以**同一交通为 block**对方法、预算、模型共同重采样；checkpoint 保持固定。

预登记分析 seed=20261004，10,000 次 percentile traffic-block bootstrap，报告描述性 95% 区间；三个主预算另报告 Bonferroni 尾概率调整的 98.3333% 区间。它们是对固定五模型的近似条件区间，不是有限样本显著性的保证，不扩展为模型总体或跨场景结论。所有 contrast 使用相同 seed 和交通顺序以保留联合配对。

同时完整报告逐 checkpoint、逐 traffic、leave-one-traffic-out、模型/交通平均差的胜/平/负（容差 1e-9）。不删除集中获益/负例，不凭最终结果改平均方式、挑模型或重命名候选。[final_statistics.py](../scripts/final_statistics.py) 已用合成数据验证完整配对、聚类和 FGSM 实际计数；尚未接入最终原始结果。

碰撞率以实际 episode 为分母；ASR 使用原定义——配对 Clean 未碰撞 episode 中的攻击后碰撞转换。报告成功数/eligible 分母和 null，不把原本就碰撞的 Clean 算作成功。同一模型/交通的三次随机攻击按重复比例平均，交通聚类仍为 50。相对算法对照同时报告新增与丢失转换及净差，不用净差掩盖失败。

## 4. 指标与真实成本

完整报告原始 Episode Return、相对 Clean 的 Return Drop、Collision Rate、ASR、Attack Rate、动作/观测改变率、L_inf/L2/归一化 L_inf，以及真实计算记录。

TTC/DRAC 继续从真实 SUMO post-step 同车道前后车 state 计算：gap 加回 follower minGap；不闭合时 TTC 为 null、DRAC=0；重叠 TTC=0、DRAC 未定义。保留 episode minimum TTC、TTC p5、DRAC p95、有效 pair 数/缺失与 overlap 数，不伪造横向几何。仅将已有 episode 百分位作为逐 episode 指标报告，不能把其平均值称为 pooled 百分位；没有足够有效值时明确 null 和分母。既有 SUMO/minGap 碰撞不等于实际物理接触。

成本分别列出：梯度、攻击 forward、评估 forward、总 forward、新 transition、含初始化/setup 和 replay 的 physical shadow。评估循环本身每真实步两次策略调用，与攻击账本分开；wall time 是受并发/机器影响的辅助测量。每项都报告总量、每实际 episode 和每真实步，说明提前终止造成的总成本下降；不合成任意“梯度＋仿真”分数，不强行把简单攻击填满预算。

## 5. 执行顺序、失败与审计要求

首先完整运行 `basic_attack0` 五模型基准。其 Clean 250 个 episode 完成并通过审计后才允许其他组。后续最多两个组并行、每组五个模型进程，最多十个 live evaluation，模型线程固定 1。完整 12 批必须全部通过才出最终表。

运行前固定干净 Git tree 和同一个工具执行 SHA；所有子进程检查 expected commit。manifest、命令、角色 seed、环境、checkpoint 和原始结果按现有 launcher 保存到 `.local/`。禁止批次运行中编辑 tracked tree。未来 dispatcher 必须将每个执行/audit 的 SHA、退出状态、输入哈希与实际 episode/step 计数记录到总 manifest。

必须在首次最终仿真之前完成针对 50 episode 的全矩阵审计：逐 episode identity、Clean 完整轨迹回归、扰动边界、每块各预算、fallback、目标/实际动作、完整候选和 live/oracle transition、shared witness、Progress retry、原 SUMO collision 和真实成本账本。旧 20-episode validation auditor 有明确 split/count 限制，**不可更名后直接用于最终测试**。

失败规则提前固定：

1. 任意配置或 integrity audit 失败，停止派发新组，保留正在运行的输出与所有尝试；不把部分成功的矩阵发布为完整结果。
2. 只允许有证据的基础设施失败重试一次，原 SHA/config/seeds/runtime/model 全相同，使用新输出目录，保留失败目录和原因。不能因回报/碰撞不理想重试；正确完成但结果差不算失败。
3. seed、模型、环境、replay、预算或矩阵不一致使本批无效，不能删除不利交通、补换 seed 或修算法后沿用旧结果。工程调试回到 split 10/20。
4. 任一 split-30 仿真开始即记录 final exposure。若此后改变算法/超参数，split 30 已被使用，必须另行预登记新的独立保留集，不能重新称为未见交通。失败和偏离协议都进入论文限制。

## 6. 本阶段完成状态与下一步

本阶段完成方法文档、60 份配置、机器协议、统计函数和**不启动交通**的输入/环境预检。预检检查已有 `.local/runs` 的 manifest/evaluation/batch 元数据是否出现 split 30，不解析最终轨迹；这一检查不保证检测未通过 launcher 记录的运行。

旧算法和五模型不变。最终 dispatcher、50-episode 综合审计器及完整结果导出属于下一独立功能，当前 `final_execution_ready=false`。这是执行工具尚需接入的明确状态，不是要求改变协议或选择新方法。本轮没有新增最终 efficacy 证据，也没有开始防御。

只运行预检的 PowerShell 命令（在干净已提交的 repository root；输出使用未存在的文件）：

```powershell
$env:PATH="$PWD\.local\envs\oarl-legacy;$PWD\.local\envs\oarl-legacy\Library\bin;$env:PATH"
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:MPLCONFIGDIR="$PWD\.local\matplotlib-tests"
$env:PYTHONWARNINGS='ignore'
$finalSourceSha=(git rev-parse HEAD).Trim()
& .local/envs/oarl-legacy/python.exe scripts/check_attack_final_protocol.py `
  --expected-commit $finalSourceSha `
  --output .local/runs/20261004-attack-final-protocol/preflight.json
```

预检只探测 Python/pip/SUMO 版本、重载五模型并核对冻结输入，不运行 SUMO episode。后续执行功能合入 main 并通过此预检、完整审计测试后，再以固定 SHA 开展一次最终矩阵；最终结果用于检验和收窄主张，不用于调参。之后才整理攻击论文初稿并进入防御对照。
