"""Freeze all nine Return mechanism development cells before running SUMO."""

import json
import subprocess
from pathlib import Path
from prepare_mechanism_controls import ROOT, CONTROL_NAMES, make_config, sha256
from prepare_proposed_configs import save

PROTOCOL = ROOT / "configs/research/mechanism_development.json"
FROZEN_METHOD_COMMIT = "5501e966279eca62344692d436cde17b24b5fe16"
SMOKE_BATCH = ".local/runs/20261003T034414646128Z-attack-benchmark/batch.json"
CLEAN_BATCH = ".local/runs/20260921T085944130990Z-attack-benchmark/batch.json"


def reference(path):
    batch = json.loads((ROOT / path).read_text())
    if batch["status"] != "passed" or len(batch["runs"]) != 5:
        raise ValueError("Reference batch must have passed all five models")
    return dict(path=path, sha256=sha256(ROOT / path), git_commit=batch["git_commit"])


def prepare():
    subprocess.check_call(["git", "diff", "--exit-code", FROZEN_METHOD_COMMIT, "--",
                           "main.py", "oarl.py", "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    base = json.loads((ROOT / "configs/research/mechanism_controls.json").read_text())
    groups = []
    for cap in (100, 200, 400):
        for seed in (0, 1, 2):
            files = []
            for checkpoint in range(5):
                name = "configs/evaluation/mechanism_development_g%d_attack%d_seed%d.json" % (cap, seed, checkpoint)
                save(ROOT / name, make_config(checkpoint, 10, cap, seed))
                files.append(name)
            groups.append(dict(gradient_cap=cap, forward_cap=2 * cap, attack_seed=seed, configs=files, episodes=250))
    record = dict(kind="return_mechanism_development", frozen_method_commit=FROZEN_METHOD_COMMIT,
                  frozen_source_sha256=base["frozen_source_sha256"], references=base["references"],
                  smoke_reference=reference(SMOKE_BATCH), clean_reference=reference(CLEAN_BATCH),
                  seed_splits_sha256=base["seed_splits_sha256"], frozen_victims_sha256=base["frozen_victims_sha256"],
                  methods=list(CONTROL_NAMES), contrasts=base["contrasts"], groups=groups,
                  checkpoint_seeds=list(range(5)), attack_seeds=[0, 1, 2],
                  gradient_forward_caps=[[100, 200], [200, 400], [400, 800]],
                  episodes=10, max_steps=200, split_id=10, total_episodes=2250,
                  attack_episodes=1800, repeated_clean_episodes=450, unique_clean_model_traffic_pairs=50,
                  execution=dict(parallel_groups=2, jobs_per_group=5, max_live_evaluations=10, policy_threads=1),
                  primary_endpoint="paired Return first-minus-second at each fixed budget; lower is stronger attack",
                  secondary_endpoints=["paired collision conversion", "actual gradient/forward/shadow cost", "retry branch yield"],
                  analysis="Per-model and per-traffic averages over correlated attack seeds; wins/losses and leave-one-traffic-out descriptions; no independent N=150 significance claim",
                  limitations="Nested contrasts on reused development traffic, not a full factorial, validation or final test; execution wall time is not a speed benchmark. No method tuning during this grid.")
    save(PROTOCOL, record)
    print("FROZEN_MECHANISM_DEVELOPMENT configs=45 episodes=2250 split=10")


if __name__ == "__main__":
    prepare()
