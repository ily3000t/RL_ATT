# Protocol A: controlled multi-seed reproduction

## Scope and provenance

This experiment uses OARL as a **robust victim**, not a clean RL victim. Its
Bayesian search and dual constraint remain active in training. No attack Gate,
action filter, perturbation clipping, new objective, or early stopping is added.
Zero-One and proposed attacks are outside this experiment.

Upstream: https://github.com/TMIS-Turbo/OARL at
`29e5c0e2497cd0bd27b6cf83c5bce800a3e2c54a`. See `OARL_CODE_AUDIT.md` for
known upstream behavior and the previously committed removed-ego compatibility
guard. No explicit upstream license was found; local development does not grant
permission to redistribute the source publicly.

The user supplied the paper/configuration comparison and selected **only five
Protocol A runs**, with run seeds 0–4. These are controlled-seed reproductions of
the training procedure, not exact reproductions of the unpublished paper seeds
or of upstream's paired SUMO seed schedule. No legacy long-run group is run.

## Fixed training settings

| Setting | Value and implementation |
| --- | --- |
| Traffic | Existing Normal route file: flow probabilities 0.08 + 0.04 + 0.02 = 0.14; unchanged |
| Episodes / maximum steps | 400 / 200; original termination can end an episode earlier |
| Trials | 5; run seeds 0, 1, 2, 3, 4 |
| Network | 16 inputs; one 128-unit ReLU hidden layer; 3 discrete actions |
| Discount / actor LR / critic LR | 0.95 / 0.0001 / 0.001 |
| Dual LR / constraint target | 0.0005 / 0.0001 |
| Polyak | rho = 0.995 (code tau = 0.005) |
| BO | u1 in [0.8, 1.2], u2 in [-0.05, 0.05]; 5 suggestions per update |
| Replay / batch | Capacity 1,000,000; batch 128, sampled with replacement |
| Updates | Every second cumulative interaction when zero-based episode index > 10 |
| Saved actors | Completed episodes 100, 200, 300, 400 |
| Execution | CPU; one numerical-library thread per independent process |
| Gate | Disabled; no Gate implementation |

The traffic value 0.14 is the sum of the three independent flow probabilities,
not an assertion that the probability of at least one insertion is exactly 0.14.
Agent defaults are extracted from the committed `oarl.py` into every manifest.

## Seed contract

For run i, `run_seed`, `python_seed`, `numpy_seed`, `torch_seed`, `policy_seed`,
`sumo_seed`, and `attack_seed` are all i. They name independent roles; the actual
per-episode SUMO seeds are recorded explicitly rather than inferred later.

* Python, NumPy, and Torch are initialized by the unchanged main CLI seed code.
* Network initialization and training keep the Torch training RNG.
* Interaction action sampling uses an isolated CPU Torch stream initialized
  from `policy_seed`. The original `select_action_single` executes unchanged
  inside an RNG-state context; training RNG state is restored afterwards.
* BO uses `attack_seed` as `BayesianOptimization(random_state=...)`. As upstream,
  a new optimizer is constructed each update. This is a fixed initial RNG state
  for each optimizer, not a newly randomized seed on every update. Unused
  categorical samples inside BO still consume the original training Torch RNG.
* SUMO schedule `seedsequence_v1` derives episode e (zero-based) with
  `SeedSequence([run_seed, phase_id, e]).generate_state(1)[0] % (2**31-1)`.
  phase_id is 0 for training and 1 for evaluation. `sumo_seed` is the base seed,
  not the literal value reused on every reset. The deterministic episode-varying
  schedule was the stated default while awaiting the optional seed-schedule
  preference. All generated values are saved; actual SUMO XML values are checked.
* `PYTHONHASHSEED` is set before process startup. CPU thread counts and dependency
  versions are fixed and recorded. No claim of GPU or cross-version bit equality
  is made.

The default source entry points retain legacy behavior unless the controlled
runner is selected. This runner instruments the original `main.py` via runpy;
it does not replace the training loop or its update conditions.

## Artifact and checkpoint contract

Every run exports committed source into its own ignored `.local/runs/.../source`
directory. SUMO rewrites that run's configuration only. Each manifest records
Git commit/tree/branch, original source hashes, complete CLI/configuration,
effective seeds, launch commands, dependency versions, SUMO version, host OS,
runtime overrides, status, and final source hashes. Checkpoints, trajectories,
raw results and logs stay out of Git.

`episodes.jsonl` records actual per-episode returns, lengths, termination,
observed speed, update counts, latest training JS, dual multiplier, and a hash of
the next-observation/action/reward/done sequence. Timing is diagnostic and is not
expected to be deterministic. SUMO's post-step collision ID list is queried
directly. `ego_collision_observed` means Auto was reported in that list during
an interaction step. It does not count events during reset or substitute the
upstream `cn_epi` distance proxy for collisions. TTC/DRAC are not provided here.

At each upstream save point, the actor pickle is reloaded on CPU and checked for
finite weights, 16-to-3 architecture, normalized probabilities, identical weight
hash and exactly equal predictions on the first 64 observed training states.
These are **inference-only actor checkpoints**; replay, critics, optimizers,
dual and RNG state are not saved, so interrupted training cannot resume exactly.

The independent validator rechecks file/probe hashes and predictions in a fresh
process, then freezes the actor and runs greedy argmax inference for 20 held-out
episodes of at most 200 steps at each of the four checkpoints. All four use the
same evaluation seed list for a given run. Its source victim/environment hashes
must equal training hashes. Both weights and file bytes are checked unchanged
after evaluation. Final victim selection is episode 400, preselected rather
than chosen by evaluation performance. Greedy evaluation differs from stochastic
training action selection and is reported separately.

## Commands (PowerShell from repository root)

```powershell
python scripts/run_experiment_batch.py --configs configs/experiments/oarl_protocol_a_seed0.json configs/experiments/oarl_protocol_a_seed1.json configs/experiments/oarl_protocol_a_seed2.json configs/experiments/oarl_protocol_a_seed3.json configs/experiments/oarl_protocol_a_seed4.json --jobs 5 --name protocol-a-oarl-long
python scripts/validate_checkpoints.py --run .local/runs/<batch>/run-0
```

Repeat validation for run-1 through run-4. Launchers require committed changes.
Python defaults to `.local/envs/oarl-legacy/python.exe`; the existing SUMO install
is selected using the current SUMO_HOME/PATH. The locked legacy environment is
documented in `requirements/` and the Stage 0 audit.

## Preflight evidence

Nine unit tests passed: legacy/controlled reset behavior, deterministic seed
schedules, policy RNG isolation/restoration, removed-ego compatibility, and
checkpoint verification/corruption rejection. Two 100-episode × 2-step checks
at commit `4a0e0c7` produced identical episode records (excluding timing) and
identical actor weight hashes and probe probabilities. The fresh-process
checkpoint/evaluation smoke check passed at commit `11e1ccc`.

These checks establish runner operation and short-run repeatability. Long-run
completion, performance, and paper agreement must be reported from actual
results separately; preflight success does not establish those outcomes.
