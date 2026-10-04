# 最终执行链工程验证记录

2026-10-04（Asia/Shanghai），独立分支 `codex/final-attack-execution` 从 main `6135d39` 创建。此记录为执行准备，不是最终攻击效果。

| 原子提交 | 内容 |
| --- | --- |
| `caef813` | 严格 final-50 原始记录、oracle 成本审计及旧轨迹兼容复核 |
| `87291e4` | 完整 17 行比较、配对交通区间、成本和集中性导出 |
| `00da379` | Clean anchor 顺序、受控并发、失败保护和工程证书 |

在干净 `00da3797333e15951f08a4faf0e8aff8d9b909fc` 上生成证书：

```powershell
& .local/envs/oarl-legacy/python.exe scripts/check_attack_final_execution.py `
  --expected-commit 00da3797333e15951f08a4faf0e8aff8d9b909fc `
  --output .local/runs/20261004-final-execution-ready/readiness.json
```

完整环境前缀及运行说明见 [ATTACK_FINAL_EXECUTION.md](ATTACK_FINAL_EXECUTION.md)。ignored readiness 文件及相邻 `protocol-preflight.json`、`check-0.log`、`check-1.log`、`fixtures.json` 保留完整命令、输入/工具/测试哈希、版本、角色 seed、模型重载和检查证明。

- **151 项测试通过**，含新增 14 项；50-episode mock、完整 12 批进程契约和失败停派发均通过。测试输出中的 14,500 episode 是 mock 计划矩阵，不是实测。
- 九方法的 **1,100 个已有 split-20 episode / 182,593 步**通过新 raw/safety/预算/witness/oracle 审计原语复核。包含 oracle setup，未增加仿真，未修改旧结果。
- 五 checkpoint 文件/权重、Python 3.7.16、Torch 1.3.1+cpu、SUMO 1.22.0 和完整 frozen pip/runtime 与原 Clean 训练一致。
- 52 个运行源码/场景及所有预登记配置、seed、模型注册表相对 main 冻结点无差异；新增内容只在 scripts/tests/docs。
- 工程证书的 `final_execution_ready=true` 表示工具就绪，登记文件的历史 `false` 不追改。证书记录 `new_simulations=0`，本记录生成时 split 30 尚未启动。
- 元数据原子写入的 Windows 共享占用问题已在并发 mock 中验证处理；不会因结果差而重跑实验。

按规范在工程验证后本地 `--no-ff` 合并，保留功能分支，不创建 tag、不推送公共仓库。正式测试随后只按已登记协议和新的固定 main SHA 执行，实际启动命令/PID、逐批状态、哈希和暴露时间另在 `.local/` 记录。只有完整矩阵通过后才能讨论独立测试效果；防御仍未开始。
