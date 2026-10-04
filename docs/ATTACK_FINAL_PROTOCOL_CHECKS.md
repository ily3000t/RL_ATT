# 攻击定稿与最终协议工程检查

检查日期：2026-10-04（Asia/Shanghai）。功能分支 `codex/attack-final-protocol` 从 `main` 的 `d218309627830be924c48ef7fd9d984be11da247` 创建；本记录是预登记工程检查，**没有最终交通实验结果**。

| 原子提交 | 内容 |
| --- | --- |
| `f0ada17` | Single Return 方法、主张和已有工作边界冻结 |
| `412c68b` | 60 份最终配置、完整矩阵、统计/失败规则预登记 |
| `8e85a4d` | 配对交通聚类统计与确定性 FGSM 实际计数 |
| `78bd027` | 不启动交通的冻结输入、模型和环境预检 |

使用冻结 Python/Torch 环境运行 `python -m unittest discover -s tests`，**137 项测试通过**（此前 125 项，本功能新增 12 项）。覆盖缺失/重复/更换条件、改预算/目标/重试、交通配对、统计聚类、确定性实际样本数、运行版本漂移、脏 tree/错误 SHA 和已记录最终暴露保护。测试使用合成数据和既有 mock，不启动 split-30 SUMO episode。

在干净提交 `78bd027867522756809151d213eda3f040b69be8` 上执行完整预检：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/check_attack_final_protocol.py `
  --expected-commit 78bd027867522756809151d213eda3f040b69be8 `
  --output .local/runs/20261004-attack-final-protocol/preflight.json
```

环境变量/完整 PATH、命令、60 份配置的真实字节哈希和角色 seed、五模型重载记录、完整版本探测和运行元数据暴露检查均保存在该 ignored manifest。复查命令的环境前缀见 [ATTACK_FINAL_TEST_PROTOCOL.md](ATTACK_FINAL_TEST_PROTOCOL.md)。manifest SHA-256 为 `a73d93ba207867d01f147820389007cdcf401621e43d19a372d44cc6da600bb7`；协议文件真实 SHA-256 为 `58c7a9ea8c0a79b236021f17ff2e314cd3f43e700917b01c611ccc3260f9ea91`。

预检结果：

- 60 份配置及 12 组矩阵与登记内容完全一致。14,500 episode 是未来计划量，其中 250 个唯一模型/交通对、50 个交通聚类；**本轮新仿真 episode 为 0**。
- 五模型的 checkpoint 文件/权重哈希匹配冻结注册表，加载后架构/有限性检查通过。
- Python 3.7.16、Torch 1.3.1+cpu（线程 1）、NumPy 1.21.6、SciPy 1.7.3、scikit-learn 0.24.2、SUMO 1.22.0；完整 pip 和版本 stdout 与五个原 Clean 训练 manifest 一致。
- 52 个实际运行源码/场景哈希与已审计 Single 实验一致；root OARL、Environment/Data、`rl_att/`、requirements、冻结模型注册表和原 seed 注册表相对冻结 main 无差异。
- ZOOpt wheel 哈希一致；已有 `.local/runs` manifest/evaluation/batch 元数据无 split-30 匹配。这不保证检测未记录的手工运行。
- `final_execution_ready=false`：下一项独立功能需完成 50-episode 最终 dispatcher、全矩阵原始结果/账本审计和导出，并在新固定执行 SHA 上重做预检。旧 20-episode 审计未被绕过。

本轮没有 efficacy 或稳定安全优势的新证据。论文主张继续以未见交通的后续检验为准，保留原负例、实际成本与样本限制；未开发防御。按仓库规范完成本地 `--no-ff` 合并并保留功能分支，不创建 milestone tag 或执行公共 push。
