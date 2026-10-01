"""Freeze existing simple attacks on the same validation traffic as search methods."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from prepare_proposed_configs import save

ROOT = Path(__file__).resolve().parents[1]
SEARCH_PIPELINE = ".local/runs/20260925-shared-compute-validation/validation.json"


def make_config(reference, basic, attack_seed):
    config = copy.deepcopy(reference)
    config["research_seeds"]["attack_seed"] = attack_seed
    config["attacks"] = [copy.deepcopy(a) for a in basic["attacks"] if attack_seed == 0 or a["name"] != "fgsm"]
    return config


def prepare():
    pipeline = json.loads((ROOT / SEARCH_PIPELINE).read_text())
    if pipeline["status"] != "passed" or pipeline["verified_episodes"] != 4500:
        raise ValueError("Complete audited search validation first")
    reference_batch = json.loads(Path(pipeline["groups"][0]["batch"]).read_text())
    references = []
    for run in reference_batch["runs"]:
        path = Path(run["run_dir"]) / "manifest.json"
        manifest = json.loads(path.read_text())
        references.append(dict(checkpoint_seed=manifest["config"]["run_seeds"][0], path=path.relative_to(ROOT).as_posix(),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    groups = []
    for seed in range(3):
        files = []
        for checkpoint in range(5):
            reference = json.loads((ROOT / ("configs/evaluation/budget_validation_g400_attack0_seed%d.json" % checkpoint)).read_text())
            basic = json.loads((ROOT / ("configs/evaluation/benchmark_stage3_seed%d.json" % checkpoint)).read_text())
            path = "configs/evaluation/basic_validation_attack%d_seed%d.json" % (seed, checkpoint)
            save(ROOT / path, make_config(reference, basic, seed))
            files.append(path)
        groups.append(dict(attack_seed=seed, configs=files))
    protocol = dict(kind="basic_validation_controls", research_split_id=20, episodes=20, max_steps=200,
        groups=groups, total_episodes=1300, attack_episodes=1000, clean_control_episodes=300,
        fgsm_unique_episodes=100, stochastic_episodes_per_attack=300,
        frozen_method_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip(),
        search_pipeline=SEARCH_PIPELINE, search_pipeline_sha256=hashlib.sha256((ROOT / SEARCH_PIPELINE).read_bytes()).hexdigest(),
        search_summary="docs/SHARED_COMPUTE_VALIDATION_RESULTS.json",
        search_summary_sha256=hashlib.sha256((ROOT / "docs/SHARED_COMPUTE_VALIDATION_RESULTS.json").read_bytes()).hexdigest(),
        clean_reference_batch=str(Path(pipeline["groups"][0]["batch"]).relative_to(ROOT)),
        reference_manifests=sorted(references, key=lambda r: r["checkpoint_seed"]),
        basic_specs=json.loads((ROOT / "configs/evaluation/benchmark_stage3_seed0.json").read_text())["attacks"],
        interpretation="Fixed simple-attack configurations, actual costs; FGSM/Clean deterministic records are not independent attack replicates")
    save(ROOT / "configs/research/basic_validation_controls.json", protocol)
    print("BASIC_VALIDATION_FROZEN attacks=1000 controls=300 configs=15")


if __name__ == "__main__":
    prepare()
