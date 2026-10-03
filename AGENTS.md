# Repository development rules

These rules preserve the user's development and reproducibility requirements.

- Implement each new feature on an independent branch created from the current
  `main`; use `codex/<feature>` unless the user specifies another name. Never
  develop new features directly on `main`.
- Make one meaningful Conventional Commit per atomic change. Before a large
  refactor, create a recoverable commit of the current work.
- Complete the feature's relevant tests and required experiment audits, then
  merge it into `main` with a descriptive `--no-ff` merge commit. If the user
  explicitly requests analysis without merging, follow that instruction for
  that task. A scientifically negative result does not prevent integration of
  a verified engineering feature; document the result and its limits.
- Preserve historical branches and experiment source commits. Do not use
  unnecessary `reset --hard`, force pushes, or rewrite shared main history.
  Merge authorization does not imply public redistribution or publication.
- Create tags only for stable, fully reproducible milestones, never for every
  commit or solely because an engineering smoke passed.
- Commit code, configs, seed protocols, and auditors before experiments. Keep
  the source commit and tracked tree fixed while a batch runs.
- Record the commit SHA, config, role-specific random seeds, launch command,
  environment versions, and checkpoint hashes for every experiment. Use the
  existing launchers and manifests rather than an unrecorded experiment path.
- Keep checkpoints, raw results, logs, large trajectories, and source snapshots
  under ignored `.local/`. Commit configs, scripts, tests, and compact summaries.
- Retain the root OARL experiment files and frozen victim checkpoints. Use
  adapters in `rl_att/`; avoid cosmetic rewrites of upstream algorithms.
- Treat sibling `Zero-OneAttack-main` as read-only reference code.
- Reproduction and established comparisons use no Gate. Do not tune on the
  reserved final traffic split (30) or claim unmeasured safety metrics.
- Separate engineering correctness from scientific efficacy and novelty.
  Report paired outcomes, actual resource costs, negative results, and sample
  limitations. Three attack seeds on one traffic episode are repeated attacks,
  not three independent traffic samples.
- Use the existing frozen Python/Torch/SUMO environment for compatibility tests
  and SUMO experiments; keep Matplotlib's config directory inside `.local/`.
