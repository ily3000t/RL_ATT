"""Freeze a four-condition Return mechanism smoke without touching final traffic."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_att.evaluation.configuration import validate_config
from prepare_proposed_configs import config, save

CONTROL_NAMES = ("none", "zero_one_budgeted_return", "ours_single_return",
                 "ours_return", "ours_progress_return")
REFERENCE_BATCHES = {
    "original": ".local/runs/20260921T071026843375Z-attack-benchmark/batch.json",
    "progress": ".local/runs/20260922T073213636553Z-attack-benchmark/batch.json",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256(path):
    # Git archive applies export line endings; hash exactly what launchers execute.
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", path], cwd=str(ROOT))
    with tarfile.open(fileobj=io.BytesIO(archive)) as exported:
        return hashlib.sha256(exported.extractfile(path).read()).hexdigest()


def make_config(checkpoint_seed, episodes=2, gradient_cap=400, attack_seed=0):
    if checkpoint_seed not in range(5) or episodes not in (2, 10) or gradient_cap not in (100, 200, 400) or attack_seed not in (0, 1, 2):
        raise ValueError("Use the registered development mechanism grid")
    seeds = json.loads((ROOT / "configs/research_seed_splits.json").read_text())
    result = config(checkpoint_seed, episodes, seeds["splits"]["10"])
    base = result["attacks"][3]
    result["attacks"] = [result["attacks"][0]]
    for name in CONTROL_NAMES[1:]:
        spec = copy.deepcopy(base)
        spec["name"] = name
        p = spec["parameters"]
        p["max_attempts"] = 1 if name == "ours_single_return" else 3
        p["resource_limits"].update(gradient_evaluations=gradient_cap, policy_forward_calls=2 * gradient_cap)
        if name == "ours_progress_return":
            p["retry_rule"] = "strict_margin_progress"
        result["attacks"].append(spec)
    result["research_seeds"]["attack_seed"] = attack_seed
    return validate_config(result)


def prepare():
    references, source_paths = {}, set()
    for name, path in REFERENCE_BATCHES.items():
        batch_path = ROOT / path
        batch = json.loads(batch_path.read_text())
        if batch["status"] != "passed":
            raise ValueError("Historical control batch failed")
        manifests = []
        for run in sorted(batch["runs"], key=lambda r: r["config"]):
            manifest_path = Path(run["run_dir"]) / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            source_paths.update(manifest["source_sha256_before"])
            manifests.append(dict(path=manifest_path.relative_to(ROOT).as_posix(), sha256=sha256(manifest_path)))
        references[name] = dict(path=path, sha256=sha256(batch_path), git_commit=batch["git_commit"], manifests=manifests)
    source_paths.add("rl_att/attacks/single_attempt.py")
    paths = []
    for seed in range(5):
        path = "configs/evaluation/mechanism_smoke_seed%d.json" % seed
        save(ROOT / path, make_config(seed))
        paths.append(path)
    protocol = dict(kind="return_mechanism_controls", frozen_base_commit="db50474",
                    methods=list(CONTROL_NAMES), references=references,
                    frozen_source_sha256={p: source_sha256(p) for p in sorted(source_paths)},
                    seed_splits_sha256=sha256(ROOT / "configs/research_seed_splits.json"),
                    frozen_victims_sha256=sha256(ROOT / "configs/frozen_victims.json"),
                    smoke=dict(configs=paths, episodes=2, max_steps=200, split_id=10, attack_seed=0,
                               gradient_cap=400, forward_cap=800, total_episodes=50),
                    contrasts=[dict(first="ours_single_return", second="zero_one_budgeted_return", effect="outer search organization without retries"),
                               dict(first="ours_return", second="ours_single_return", effect="fixed retries within behavior search"),
                               dict(first="ours_progress_return", second="ours_return", effect="progress stopping within behavior search"),
                               dict(first="ours_progress_return", second="zero_one_budgeted_return", effect="current method versus matched baseline")],
                    planned_development=dict(status="not_run", checkpoint_seeds=list(range(5)),
                                             attack_seeds=[0, 1, 2], gradient_forward_caps=[[100, 200], [200, 400], [400, 800]],
                                             episodes=10, split_id=10, total_episodes=2250),
                    limitation="Nested conditional contrasts, not a full factorial or a causal test of history caching alone; smoke is not efficacy evidence. Final split 30 remains unused.")
    save(ROOT / "configs/research/mechanism_controls.json", protocol)
    print("Frozen five mechanism smoke configs (50 episodes); full study remains planned.")


if __name__ == "__main__":
    prepare()
