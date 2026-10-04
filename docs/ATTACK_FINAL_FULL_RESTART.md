# 用户授权的完整最终测试重跑

日期：2026-10-04。用户澄清之前的“不重新运行”描述有误，明确要求未完整运行时重新运行，并在后台启动。本次采用**全新完整矩阵**，不再按上一恢复计划复用 21 个配置，也不拼接四个中断配置的 820 条 episode 行。旧目录、日志和轨迹保留，新输出成为本轮结果；原数据不删除或覆盖。

## 固定输入与来源

原未完成实验执行 SHA 为 `85f981d5ac603439a6d5e7195d7fa092f8e0d311`。新增控制工具在独立 `codex/final-attack-restart` 分支开发；实现提交 `3a090f0fd4ee7710d0f54f1d3da927c53fcd66f5`。工具验证后合入 main，实际运行记录当时 main 的完整 SHA。

原 OARL / Environment / Data / `rl_att/` / requirements、全部配置和模型没有变化，仍与原执行及 `d218309...` 方法冻结点一致。原 60 配置 / 12 组 / 14500 原始 episode、五模型、50 个 split-30 交通、三个 attack seed、三档预算和统计规则不变。先完整运行并审计 basic_attack0，再最多两组并发、每组五模型。只有完整矩阵通过才生成原 17 行比较。

split 30 已在原尝试中暴露；本轮不会把重跑称为新独立未见样本，不增加交通样本数量，不使用旧结果调参或选择版本。完整重跑包含原先正确完成的配置，超出原预登记的“仅基础设施中断配置重试”范围；这是用户新指令授权的明确协议偏离，记录于 authorization / preflight / pipeline 和最终结果。不能将它描述为原始一次性测试没有中断。

## 入口与审计

`--restart-authorization` 仅为明确的完整重跑启用：核对原 pipeline、输入预检、interruption receipt / inventory、保存的全部控制证据及 48 份旧搜索轨迹哈希、进程退出记录、原配置字节和新运行的模型 / 环境 / 冻结源码。正常入口仍拒绝已暴露交通。

已完整通过的 pipeline、嵌套 restart、缺少授权/保留证据、哈希改变、已消费的同一授权等均拒绝。新 pipeline 在首次 launcher 前保存 authorization 消费记录；不自动重试失败的批次或不利结果。最终结果保留重跑来源和旧证据哈希，旧尝试不混入本轮结果或样本分母。原完整 episode / step / budget / oracle / Clean / safety / cost 审计保持严格。

原始授权和进程记录：`.local/runs/20261004-final-full-restart-authorization/`。授权 SHA-256 为 `93d1d2cae144370b8e546ead82dc6b1ecca8f3920512a38fadeedff358f3b965`。只读 CIM 检查确认原 dispatcher PID 66484 不存在，本项目旧 Python/SUMO 进程无匹配；其他项目进程没有停止或修改。

## 工程验证与启动

冻结 Python 3.7.16 / Torch 1.3.1+cpu / SUMO 1.22.0，单线程策略计算，Matplotlib 配置在 `.local/`。新增测试覆盖重跑授权、完整矩阵重跑契约、保留旧记录、拒绝完整结果/重复授权/错误配置与旧证据/活动旧进程记录。全部 **164 项单元测试通过**；九方法已有 split-20 fixture **1100 episode / 182593 步**重新通过审计，无新仿真。

工程证书位于 `.local/runs/20261004-final-full-restart-ready/readiness.json`。它固定全部控制工具和测试的 Git blob、当前输入/模型/版本证据及此次授权哈希。实现合入 main 后若工具和测试字节不变，可继续使用；正式 pipeline 再以实际 main SHA 进行输入预检。

后台启动沿用固定环境，隐藏窗口并重定向 stdout/stderr，记录完整命令、PID、实际 SHA 和 readiness / authorization 哈希：

```powershell
$finalRestartSha=(git rev-parse HEAD).Trim()
& .local/envs/oarl-legacy/python.exe scripts/run_attack_final.py `
  --expected-commit $finalRestartSha `
  --readiness .local/runs/20261004-final-full-restart-ready/readiness.json `
  --restart-authorization .local/runs/20261004-final-full-restart-authorization/authorization.json `
  --output .local/runs/20261004-final-attack-full-restart
```

原 `20261004-final-attack-test` 保留；新状态和结果在 `20261004-final-attack-full-restart`，后台启动记录在 `20261004-final-full-restart-launch`。运行期间不编辑 tracked tree 或切换执行 SHA，不新建 milestone tag，不提前发布科学效果结论或开展防御。
