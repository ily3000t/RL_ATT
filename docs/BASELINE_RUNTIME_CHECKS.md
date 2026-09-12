# Stage 0 / Stage 1 前置运行证据

2026-09-12 完成。**原算法的 BO 更新可以运行；原始版本在正常多回合运行中遇到 Auto 被移除即崩溃。一个最小终止 guard 修复后，12 回合检查通过。尚未完成 400 回合训练、论文复现或 Clean Victim。**

## 1. 实际检查结果

| 检查 | 实验 commit | 配置 | 结果 |
|---|---|---|---|
| 默认 Python 启动 | `4e6f006` | 12 episodes × max_step 2，seed 0 | 失败：`ModuleNotFoundError: gym`；未启动 SUMO |
| 原始源码 smoke | `f99de33` | 12×2，seed 0，CPU | 通过：24 env steps，1 次 train_model，5 次 BO objective，4 次 Adam.step，2 次 target 更新 |
| 原始源码 horizon | `f99de33` | 1×200，seed 0，CPU | 通过：200 env steps；按原始 warm-up 条件未调用 train_model |
| 原始源码多回合 | `69b7d12` | 最多 12×200，seed 0，CPU | 失败：第 2 回合第 51 步（总第 251 次 step）`KeyError: 'Auto'`；0 次 train_model |
| 终止 guard 多回合 | `155b1c5` | 同一最多 12×200 配置 | 通过：12 resets、739 次交互、11 次 train_model、55 次 BO objective、44 次 Adam.step、22 次 target 更新 |
| 终止 guard smoke | `6acdb6f` | 同一 12×2 配置 | 通过：与原始 smoke 的两份 CSV **逐字节相同**，关键调用次数相同 |

所有数字来自实际 cProfile 计数，不是依据循环上限推测。739 小于 2400，是因为若干 episode 的 Auto 被移除而提前终止。这里没有报告碰撞率、TTC/DRAC、attack success 或收敛回报；原 main 没有这些可靠输出，短期 CSV 也存在审计文档所述统计问题。

完整 SHA、运行 ID、启动参数、环境、profile 计数和 CSV hash 见 [BASELINE_RUNTIME_SUMMARY.json](BASELINE_RUNTIME_SUMMARY.json)。原始 stdout/stderr、profile、manifest 和配置副本保存在 `.local/runs/<run-id>/`，已被 Git 忽略。

实际运行 ID：

```text
20260912T070158716516Z-default-python-check
20260912T072949556960Z-legacy-smoke
20260912T073038775259Z-legacy-horizon
20260912T073228511824Z-legacy-training
20260912T073538349698Z-terminal-guard-training
20260912T073802909349Z-terminal-guard-smoke
```

## 2. 已确认的阻断原因与修复边界

未修复运行的 SUMO stderr 在 t=132 报告换道碰撞移除 `truck.1` 与 `Auto`。随后调用链为：

```text
main.train -> env.step -> obs_to_state -> _findstate
           -> _findRearVehDistance -> vehicleparameters['Auto'] -> KeyError
```

原 `step()` 在检查 Auto 是否还存在之前便读取下一观测，因此原本写好的终止分支不可达。修复只把这一次 `obs_to_state()` 放进 Auto-exists 条件；缺失时保留 last valid observation，并执行原有 done=True、reward=0、distance=0 分支。没有改动 `oarl.py`、`main.py`、原始 requirements 或 SUMO 数据文件。

这不是全面修复环境，也不把车辆不存在统一解释为碰撞。terminal observation 的 last-valid 约定是对旧代码崩溃路径的补全；由于 BO 同时使用 s/s'，完整训练行为只能从这个明确修复版本开始追踪，不能虚构旧版本崩溃后的等价结果。

两个 unittest 在修复前的结果是 1 fail / 1 pass，修复后是 2 pass。非终止 smoke 的 CSV 和调用计数一致，支持其观测到的存活路径保持原行为，但**没有验证完整 16D 轨迹和模型权重逐位一致**，因为上游没有导出这些内容。没有将缺少证据的内容写成测试通过。

## 3. 环境与重建方式

实际隔离环境：`.local/envs/oarl-legacy`，Python **3.7.16**，PyTorch **1.3.1+cpu**，Gym **0.15.4**，NumPy **1.21.6**，bayesian-optimization **1.2.0**，SciPy **1.7.3**，scikit-learn **0.24.2**，SUMO **1.22.0**。`pip check` 返回 `No broken requirements found`。所有已有全局/Conda 环境未被替换。

原 README 的平台是 Ubuntu 16.04 + Python 3.7 + SUMO 1.2.0。本轮使用 Windows 和 SUMO 1.22.0，因此只是兼容性与有限运行证据，不能当作历史环境的数值复现。旧 Python 的 `platform.platform()` 报告 `Windows-10-10.0.26100-SP0`，外层解释器报告的主机标识与其不同；manifest 保留两者原值，不用一个报告覆盖另一个。

- 原始直接依赖：[requirements.txt](../requirements.txt)，保持上游原样。
- 明确的传递依赖选择：[oarl-py37-constraints.txt](../requirements/oarl-py37-constraints.txt)。这不是作者提供的 lockfile。
- 实际 pip 全量版本：[oarl-py37-win-freeze.txt](../requirements/oarl-py37-win-freeze.txt)。
- Conda 解释器/DLL 的 exact URLs：[oarl-py37-win-conda-explicit.txt](../requirements/oarl-py37-win-conda-explicit.txt)。
- 本轮下载包的 SHA-256：[oarl-py37-win-artifacts.sha256](../requirements/oarl-py37-win-artifacts.sha256)。

正常联网环境的重建命令（PowerShell）：

```powershell
Set-Location -LiteralPath E:\Att\OARL-master
$env:CONDA_PKGS_DIRS = Join-Path (Get-Location).Path '.local/conda-pkgs'
conda create --prefix .local/envs/oarl-legacy --file requirements/oarl-py37-win-conda-explicit.txt --yes
$legacyRoot = (Resolve-Path -LiteralPath .local/envs/oarl-legacy).Path
$env:PATH = "$legacyRoot;$legacyRoot/Library/bin;$legacyRoot/Scripts;$env:PATH"
& ./.local/envs/oarl-legacy/python.exe -m pip install --find-links https://download.pytorch.org/whl/torch_stable.html -r requirements.txt -c requirements/oarl-py37-constraints.txt
& ./.local/envs/oarl-legacy/python.exe -m pip check
```

本机旧 pip 的代理连接不可用，因此本轮使用现代 Python 下载对应 cp37/win_amd64 wheels，Gym/BO/future 使用原始 sdist，之后在旧环境离线安装。OpenCV 从 TUNA 镜像获取，并核对官方 PyPI 的 SHA-256。已下载完整包时，无须联网重复下载，实际安装命令为：

```powershell
$legacyRoot = (Resolve-Path -LiteralPath .local/envs/oarl-legacy).Path
$env:PATH = "$legacyRoot;$legacyRoot/Library/bin;$legacyRoot/Scripts;$env:PATH"
& ./.local/envs/oarl-legacy/python.exe -m pip install --cache-dir .local/pip-cache --disable-pip-version-check --no-index --find-links .local/wheels --no-build-isolation -r requirements.txt -c requirements/oarl-py37-constraints.txt
& ./.local/envs/oarl-legacy/python.exe -m pip check
```

安装下载日志在 `.local/audit/`。这些不是训练实验，也没有提交安装包或虚拟环境。一次 OpenCV 下载曾被自动审批因用量限制拒绝；用户确认继续后，同一请求获批并成功，无剩余审批阻塞。

## 4. 检查命令与记录规则

在干净工作区执行：

```powershell
Set-Location -LiteralPath E:\Att\OARL-master
python scripts/run_baseline.py --python .local/envs/oarl-legacy/python.exe --cpu --profile --label smoke --timeout 180
python scripts/run_baseline.py --python .local/envs/oarl-legacy/python.exe --config configs/oarl_horizon_check.json --cpu --profile --label horizon --timeout 180
python scripts/run_baseline.py --python .local/envs/oarl-legacy/python.exe --config configs/oarl_training_check.json --cpu --profile --label training-check --timeout 300

$legacyRoot = (Resolve-Path -LiteralPath .local/envs/oarl-legacy).Path
$env:PATH = "$legacyRoot;$legacyRoot/Library/bin;$legacyRoot/Scripts;$env:PATH"
& ./.local/envs/oarl-legacy/python.exe -m unittest discover -s tests -v
```

runner 从当前 **已提交且干净的 HEAD** 导出源码，拒绝未提交配置。每次 manifest 在启动主程序前保存完整 commit/tree SHA、配置、Agent defaults、CLI seed、SUMO 0,0,2,2,... seed schedule、命令、路径、环境和文件 hash；完成/失败后保存返回码及运行后 hash。已记录实验对应各自原始 commit，之后的文档提交不会改变这些记录。

只在副本中运行，因为上游 reset 会改写 sumocfg。所有完成的 SUMO 检查中，运行前后唯一变化的原始输入文件均为副本中的 `Data/StraightRoad.sumocfg`。runner 使用 Agg 和本地 matplotlib cache，不改变策略；Agg 无 GUI 警告与 deprecated getCurrentTime 警告已保留。

## 5. 尚未完成

- 没有运行完整 400 回合并验证论文曲线，也没有复现三档交通密度。
- 没有冻结训练完成的 checkpoint、恢复训练或建立 checkpoint 推理评估流程。
- 没有实现 Clean Victim、独立攻击器或评估指标；不能把当前随机初始化/warm-up 的行为称为训练好的 robust policy。
- yaw_rate、距离几何、observation_space、其他车辆消失时的订阅时序等问题仍见 [OARL_CODE_AUDIT.md](OARL_CODE_AUDIT.md)，不得在新算法结果中隐去。
- 未创建 `v0.1.0-oarl-reproduced`；目前唯一 tag 是上游源码冻结 tag。
