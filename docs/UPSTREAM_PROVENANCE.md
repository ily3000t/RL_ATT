# OARL 上游冻结与环境记录

记录日期：2026-09-12。任何 OARL 源码修改之前已完成以下冻结。

- 上游：<https://github.com/TMIS-Turbo/OARL>
- 上游分支：`master`
- 上游 commit：`29e5c0e2497cd0bd27b6cf83c5bce800a3e2c54a`
- 上游 commit 日期：2025-05-13T15:53:49Z
- 上游 Git tree：`dd659d945ba084c947fc7e316320ed7d2c0d4b19`
- 本地 baseline commit：`e1abb405c147e613a9bc5b247d5ae7c08f423ec8`
- 本地 annotated tag：`upstream-oarl-29e5c0e2497c`
- 开发分支：`baseline/oarl-reproduce`；`main` 从原始 baseline 建立。

本地目录最初没有 `.git`，不能仅凭文件夹名推断 SHA。通过 GitHub REST API 获取上游 commit 和完整 tree，验证原有 12 个文件的 Git blob SHA-1 全部相同。下载并补齐上游原有的 3 个 `Data/._*` macOS 元数据文件，同时恢复 index 中的 executable bits 后，本地 baseline 的 **15 个文件组成的完整 Git tree 与上游完全相同**。这些元数据文件不参与训练。

这是新的本地快照 commit，SHA 与上游 commit 不同；没有假称导入了上游完整提交历史。每个文件的原始存在状态、Git blob、SHA-256 和模式见 [UPSTREAM_PROVENANCE.json](UPSTREAM_PROVENANCE.json)。该 tag 表示源码冻结，不表示实验已复现；目前不创建 `v0.1.0-oarl-reproduced`。

## 许可证与目标仓库

检查完整上游 tree 后没有发现 LICENSE/COPYING 或明确的源码再分发授权；GitHub repository metadata 的 `license` 为空。保留 README 中原作者、论文引用和来源，不擅自添加替代许可证。

目标 <https://github.com/ily3000t/RL_ATT> 首次检查时为 **public**，API 返回 size=0、branches=[]。暂停期间出现了指向 `f99de33` 的远端跟踪分支，随后 API 确认目标仍为 public、默认分支已变为 baseline/oarl-reproduce；这不是本代理执行的 push。因此后续不再按空仓库处理，本代理不继续向公开目标上传完整上游代码。先将目标设为 private，或取得明确的公开再分发授权，再重新检查远端状态后推送。不会 force push、覆盖未知内容或改动 Git credential。

本机 Git HTTPS 到 github.com:443 连接失败；GitHub REST API 可读。这是网络连接失败，未据此断言认证失败或没有写权限。`gh` 未安装。已配置本地 `upstream` 和 `origin` URL，没有修改全局 Git 设置。安全手动上传流程见 [GIT_PUBLISH.md](GIT_PUBLISH.md)。

## 环境审计

| 项目 | 上游 README / requirements | 当前默认 Python | 已有 pytorch Conda 环境 |
|---|---|---|---|
| 操作系统 | Ubuntu 16.04 | Windows 11, build 26200 | 同左 |
| Python | 3.7 | 3.12.7 (Anaconda) | 3.10.16 (conda-forge) |
| PyTorch | 1.3.1+cpu | 未安装 | 2.5.1（distribution metadata） |
| NumPy | 1.21.6 | 1.26.4 | 2.2.6 |
| Gym | 0.15.4 | 未安装 | 0.26.1 |
| bayesian-optimization | 1.2.0 | 未安装 | 未安装 |
| pandas | 1.3.5 | 2.2.2 | 2.2.3 |
| matplotlib | 3.1.1 | 3.9.2 | 3.10.0 |
| SciPy | 未固定 | 1.13.1 | 1.15.0 |
| scikit-learn | 未固定 | 1.5.1 | 1.5.2 |
| SUMO | 1.2.0 | 1.22.0 | 1.22.0 |

- 默认解释器：`E:\Programs\anaconda3\python.exe`
- 已有 PyTorch 解释器：`E:\Programs\EnvAnaconda3\envs\pytorch\python.exe`
- `SUMO_HOME=E:\Program Files\sumo-1.22.0`
- Git：2.50.1.windows.1
- TraCI/sumolib 由 `SUMO_HOME/tools` 提供，不应因为 pip metadata 中未安装便认定它们不存在。
- 原始 requirements 保持原样，完整版本快照见 [ENVIRONMENT_AUDIT.json](ENVIRONMENT_AUDIT.json)。上游没有 lockfile，SciPy/scikit-learn 的原始解析版本未知。

原始 requirements：

```text
gym==0.15.4
numpy==1.21.6
tqdm==4.46.0
pandas==1.3.5
matplotlib==3.1.1
torch==1.3.1+cpu
bayesian-optimization==1.2.0
```

当前默认 Python 不能直接执行上游；旧版依赖也不能直接装入 Python 3.12 来声称恢复了原环境。已有 Gym 0.26.1 与上游七项 step 返回值、旧式 seed/reset API 不兼容。运行检查应使用隔离环境、固定提交的源码副本，并记录实际解析出的全部依赖。现代环境的 smoke 通过不等于历史平台上的论文复现。
