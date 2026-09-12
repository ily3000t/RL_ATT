# 安全上传与恢复命令

2026-09-12 首次检查时 `ily3000t/RL_ATT` 为 public、size=0、branches=[]。暂停期间，本地出现指向 `f99de33` 的 `origin/baseline/oarl-reproduce`；15:30 左右再次查询 API，目标仍为 public，default_branch 已变为 `baseline/oarl-reproduce`。不能再按空仓库处理，也不能推断是谁完成了这次同步。

上游没有明确许可证。**先在 GitHub 将目标改为 private，或取得明确的公开再分发授权，再执行上传。** 现有远端代码不构成许可证授权，也不意味着后续可以直接覆盖远端。

本代理未执行 push，没有验证写入权限。重新检查时 Git HTTPS 连接 github.com:443 仍失败，而 REST metadata 可读。未修改 credential、未 force push、未覆盖远端、未改写历史。`origin` 和 `upstream` 已在本地配置。

在 PowerShell 中，从本轮完成后的本地仓库执行以下完整 Git 流程。若当前认证失败，只解决一次认证/网络问题后由用户主动重试，不循环尝试、不替换凭据配置。

```powershell
Set-Location -LiteralPath E:\Att\OARL-master
git status --short --branch
git remote -v
git log --oneline --decorate --all -20
git show --no-patch upstream-oarl-29e5c0e2497c

if (@(git status --porcelain).Count -ne 0) {
    throw '工作区有未保存更改，先审阅并按原子修改提交。'
}

$remoteRefs = @(git ls-remote origin)
if ($LASTEXITCODE -ne 0) {
    throw '远端检查失败：停止，不修改 credential，不反复重试。'
}
git fetch origin
if ($LASTEXITCODE -ne 0) { throw 'fetch 失败，停止。' }
git log --oneline --decorate --graph --all -30
foreach ($branch in @('main', 'baseline/oarl-reproduce')) {
    git show-ref --verify --quiet "refs/remotes/origin/$branch"
    if ($LASTEXITCODE -eq 0) {
        git merge-base --is-ancestor "refs/remotes/origin/$branch" "refs/heads/$branch"
        if ($LASTEXITCODE -ne 0) {
            throw "远端 $branch 含本地未包含的历史：停止，先审阅并在独立分支整合。"
        }
    }
}
$remoteTag = @(git ls-remote --tags origin refs/tags/upstream-oarl-29e5c0e2497c)
if ($LASTEXITCODE -ne 0) { throw 'tag 检查失败，停止。' }
$localTag = git rev-parse refs/tags/upstream-oarl-29e5c0e2497c
if ($remoteTag.Count -gt 0 -and ($remoteTag[0] -split '\s+')[0] -ne $localTag) {
    throw '远端同名 tag 与本地不同，停止，不覆盖 tag。'
}

# 仅在已确认 private 或已获公开再分发授权后执行。
# --atomic 要求分支和 tag 一并成功；不使用任何 force 选项。
git push --atomic origin refs/heads/main refs/heads/baseline/oarl-reproduce refs/tags/upstream-oarl-29e5c0e2497c
if ($LASTEXITCODE -ne 0) { throw 'push 失败：保留本地 commits，停止。' }

git fetch origin
git branch --set-upstream-to=origin/main main
git branch --set-upstream-to=origin/baseline/oarl-reproduce baseline/oarl-reproduce
git ls-remote origin
```

若在最后检查后远端出现并发更新，普通非 force push 会拒绝不符合 fast-forward 的更新；不要通过 force 绕过。以上只允许已有分支的历史完全包含于本地时安全推进，不改动其他远端分支；若出现分叉，单独检查双方历史、在独立分支整合并审阅，不能使用 reset 或重写共享 main。远端目前的默认分支也不会被此脚本更改；确认 main 完整后可由用户在 GitHub 设置默认分支。

需要查验原始代码时，可在干净工作区创建新的检查分支，不回退现有 main：

```powershell
Set-Location -LiteralPath E:\Att\OARL-master
git switch -c audit/upstream-snapshot upstream-oarl-29e5c0e2497c
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
# 预期 tree: dd659d945ba084c947fc7e316320ed7d2c0d4b19
git switch baseline/oarl-reproduce
```

`.local/` 保存隔离解释器、原始运行日志和结果；它们未进 Git。若要在原始 tag 上运行训练，也必须使用独立副本，因为上游 reset 会改写 `Data/StraightRoad.sumocfg`。
