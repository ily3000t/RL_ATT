"""Freeze the Single candidate on reused validation traffic and audited controls."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_mechanism_controls import ROOT, sha256
from prepare_proposed_configs import save
from rl_att.evaluation.configuration import validate_config

PROTOCOL = ROOT / "configs/research/single_candidate_validation.json"
FROZEN_METHOD_COMMIT = "aac4716fd7d535920f667eabde8af749b2360245"
BASELINE_SUMMARY = ROOT / "docs/SHARED_COMPUTE_VALIDATION_RESULTS.json"
PREFLIGHT = ROOT / ".local/runs/20261003-validation-control-preflight.json"
NEW_NAMES = ("none", "ours_single_return")
COMPARISON_NAMES = ("none", "zero_one_budgeted_return", "ours_single_return", "ours_progress_return")
REGISTRATION_FILES = ("rl_att/attacks/registry.py", "rl_att/evaluation/configuration.py")


def candidate_config(checkpoint, cap, attack_seed):
    if checkpoint not in range(5) or cap not in (100, 200, 400) or attack_seed not in (0, 1, 2):
        raise ValueError("Use the frozen candidate validation grid")
    path = ROOT / ("configs/evaluation/budget_validation_g%d_attack%d_seed%d.json" % (cap, attack_seed, checkpoint))
    baseline = json.loads(path.read_text())
    result = copy.deepcopy(baseline)
    clean = next(a for a in result["attacks"] if a["name"] == "none")
    candidate = next(a for a in result["attacks"] if a["name"] == "ours_progress_return")
    candidate["name"] = "ours_single_return"
    candidate["parameters"].pop("retry_rule")
    candidate["parameters"]["max_attempts"] = 1
    result["attacks"] = [clean, candidate]
    return validate_config(result)


def prepare():
    subprocess.check_call(["git", "diff", "--exit-code", FROZEN_METHOD_COMMIT, "--",
                           "main.py", "oarl.py", "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    baseline = json.loads(BASELINE_SUMMARY.read_text())
    preflight = json.loads(PREFLIGHT.read_text())
    if {k: v for k, v in baseline.items() if k != "summary_commit"} != {k: v for k, v in preflight.items() if k != "summary_commit"}:
        raise ValueError("Rebuilt baseline results differ from their published audit")
    current_sources = json.loads((ROOT / "configs/research/mechanism_development.json").read_text())["frozen_source_sha256"]
    groups, controls, baseline_sources, runtime = [], [], None, None
    for cap in (100, 200, 400):
        for seed in (0, 1, 2):
            proof = next(p for p in baseline["inputs"] if (p["gradient_cap"], p["attack_seed"]) == (cap, seed))
            path = ROOT / proof["path"]
            if sha256(path) != proof["sha256"]:
                raise ValueError("Frozen control audit changed")
            audit = json.loads(path.read_text())
            batch_path = ROOT / audit["batch"]
            batch = json.loads(batch_path.read_text())
            manifests = []
            for run in sorted(batch["runs"], key=lambda r: r["config"]):
                manifest_path = Path(run["run_dir"]) / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                if baseline_sources is None:
                    baseline_sources = manifest["source_sha256_before"]
                if manifest["source_sha256_before"] != baseline_sources:
                    raise ValueError("Baseline controls use different executed sources")
                manifests.append(dict(path=manifest_path.relative_to(ROOT).as_posix(), sha256=sha256(manifest_path)))
            runtime = manifests if runtime is None else runtime
            controls.append(dict(gradient_cap=cap, attack_seed=seed, path=path.relative_to(ROOT).as_posix(),
                                 sha256=proof["sha256"], batch=batch_path.relative_to(ROOT).as_posix(),
                                 batch_sha256=sha256(batch_path), git_commit=batch["git_commit"]))
            paths = []
            for checkpoint in range(5):
                name = "configs/evaluation/single_validation_g%d_attack%d_seed%d.json" % (cap, seed, checkpoint)
                save(ROOT / name, candidate_config(checkpoint, cap, seed))
                paths.append(name)
            groups.append(dict(gradient_cap=cap, forward_cap=2 * cap, attack_seed=seed, configs=paths, episodes=200))
    changed = {p for p, value in baseline_sources.items() if current_sources.get(p) != value}
    if changed != set(REGISTRATION_FILES) or set(current_sources) - set(baseline_sources) != {"rl_att/attacks/single_attempt.py"}:
        raise ValueError("Legacy runtime changes extend beyond the reviewed Single registration")
    registration_diff = subprocess.check_output(["git", "diff", baseline["experiment_commit"], FROZEN_METHOD_COMMIT, "--"] + list(REGISTRATION_FILES), cwd=str(ROOT))
    record = dict(kind="single_candidate_validation", frozen_method_commit=FROZEN_METHOD_COMMIT,
                  frozen_source_sha256=current_sources, baseline_source_sha256=baseline_sources,
                  registration_changes=dict(files=list(REGISTRATION_FILES), diff_sha256=hashlib.sha256(registration_diff).hexdigest(),
                                            interpretation="Only register/validate the Single wrapper; existing policy, environment and search implementation bytes are unchanged"),
                  baseline_summary=dict(path=BASELINE_SUMMARY.relative_to(ROOT).as_posix(), sha256=sha256(BASELINE_SUMMARY)),
                  baseline_preflight=dict(path=PREFLIGHT.relative_to(ROOT).as_posix(), sha256=sha256(PREFLIGHT)),
                  baseline_experiment_commit=baseline["experiment_commit"], controls=controls,
                  references=dict(original=dict(manifests=runtime)), groups=groups,
                  seed_splits_sha256=sha256(ROOT / "configs/research_seed_splits.json"),
                  frozen_victims_sha256=sha256(ROOT / "configs/frozen_victims.json"),
                  new_methods=list(NEW_NAMES), comparison_methods=list(COMPARISON_NAMES),
                  checkpoint_seeds=list(range(5)), attack_seeds=[0, 1, 2],
                  gradient_forward_caps=[[100, 200], [200, 400], [400, 800]], episodes=20, max_steps=200, split_id=20,
                  total_episodes=1800, new_attack_episodes=900, new_clean_episodes=900,
                  reused_attack_episodes=1800, comparison_episodes=3600, unique_model_traffic_pairs=100,
                  execution=dict(parallel_groups=2, jobs_per_group=5, max_live_evaluations=10, policy_threads=1),
                  contrasts=[dict(first="ours_single_return", second="zero_one_budgeted_return", effect="simplified candidate versus matched baseline"),
                             dict(first="ours_single_return", second="ours_progress_return", effect="simplified candidate versus current method"),
                             dict(first="ours_progress_return", second="zero_one_budgeted_return", effect="unchanged current method versus baseline")],
                  primary_endpoint="paired Return first-minus-second per fixed coupled budget; lower is stronger attack",
                  secondary_endpoints=["paired collision conversion", "actual gradient/forward/shadow costs", "model and traffic concentration"],
                  limitations="Validation-driven simplification on already used split 20. Correlated attack seeds, not an unseen or final test. Return only; no Safety, Gate, tuning, or final split 30.")
    save(PROTOCOL, record)
    print("FROZEN_SINGLE_VALIDATION configs=45 new_episodes=1800 reused_attack_episodes=1800 split=20")


if __name__ == "__main__":
    prepare()
