"""Audit held-out split-20 runs against frozen development code and paired Clean."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from summarize_proposed_smoke import ROOT, require, summarize
from prepare_budget_validation import validation_config
from analyze_proposed_development import audit_episode_records, pair_outcomes
from analyze_progress_retry import audit_stops, verify_v0_reference
from analyze_budget_sweep import candidate_availability
from rl_att.evaluation.results import summarize_episodes
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.utils.zero_one_dependency import SHA256 as ZOOPT_SHA256

NAMES = {"none", "zero_one_budgeted_return", "zero_one_budgeted_safety", "ours_progress_return", "ours_progress_safety"}
PROTOCOL = ROOT / "configs/research/shared_compute_validation.json"


def validate_frozen_run(manifest, reference, cap, seed, traffic):
    require(cap in (100, 200, 400) and seed in (0, 1, 2), "Unregistered validation cell")
    for field in ("source_sha256_before", "python_runtime", "pip_freeze", "sumo_version", "victim_references"):
        require(manifest[field] == reference[field], "Frozen development provenance changed: " + field)
    require(manifest["config"] == validation_config(reference["config"], cap, seed, traffic),
            "Validation changes extend beyond frozen traffic/count/attack seed/model caps")


def validate_effective_seeds(entry, config):
    expected = evaluation_seeds(entry["run_seed"], 20, research_seeds=config["research_seeds"])
    expected.update(policy_rng="unused_greedy_argmax", attack_rng=("unused_no_attack" if entry["attack"]["name"] == "none"
                    else "sha256_attack_episode_full_history_target_attempt_local_numpy"))
    require(entry["effective_seeds"] == expected, "Unexpected validation seed values or RNG mechanism")


def analyze(batch_path, cap, seed, clean_reference=None):
    protocol = json.loads(PROTOCOL.read_text())
    splits_path = ROOT / "configs/research_seed_splits.json"
    for path, key in ((splits_path, "seed_protocol_sha256"), (ROOT / "configs/frozen_victims.json", "victim_registry_sha256")):
        require(hashlib.sha256(path.read_bytes()).hexdigest() == protocol[key], "Frozen registry/traffic file changed")
    traffic = json.loads(splits_path.read_text())["splits"]["20"]
    group = next(g for g in protocol["groups"] if (g["gradient_cap"], g["attack_seed"]) == (cap, seed))
    batch = json.loads(batch_path.read_text())
    require(batch["configs"] == group["configs"], "Batch is not the registered validation group")
    report = summarize(batch_path, expected_episodes=20, expected_names=NAMES,
                       expected_attack_seed=seed, expected_split_id=20)
    report.update(kind="shared_compute_validation", gradient_cap=cap, forward_cap=2 * cap,
                  research_split_id=20, raw_episode_verified_steps=0)
    for path in (PROTOCOL, splits_path, ROOT / "configs/frozen_victims.json", batch_path):
        report["source_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    references = {}
    for proof in protocol["development_references"]:
        path = ROOT / proof["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == proof["sha256"], "Frozen development manifest changed")
        report["source_sha256"][str(path.relative_to(ROOT))] = proof["sha256"]
        references[proof["checkpoint_seed"]] = json.loads(path.read_text())
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        checkpoint = manifest["config"]["run_seeds"][0]
        config_path = ROOT / run["config"]
        require(manifest["config"] == json.loads(config_path.read_text()), "Manifest differs from frozen config file")
        report["source_sha256"][str(config_path.relative_to(ROOT))] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        validate_frozen_run(manifest, references[checkpoint], cap, seed, traffic)
        require([d["sha256"] for d in manifest["extra_dependencies"]] == [ZOOPT_SHA256], "Optimizer dependency changed")
        clean = by_key[checkpoint, "none"]["episode_rows"]
        for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
            name = entry["attack"]["name"]
            require(entry["attack"] in manifest["config"]["attacks"], "Attack differs from registered specification")
            require(entry["run_seed"] == checkpoint, "Wrong checkpoint identity")
            for key in ("checkpoint_sha256", "weights_sha256"):
                require(entry[key] == manifest["victim_references"][0][key], "Frozen checkpoint hash changed")
            validate_effective_seeds(entry, manifest["config"])
            row = by_key[checkpoint, name]
            with (directory / entry["results_directory"] / "steps.jsonl").open() as stream:
                steps = [json.loads(line) for line in stream]
            report["raw_episode_verified_steps"] += audit_episode_records(steps, row["episode_rows"], 200)
            expected_summary = summarize_episodes(row["episode_rows"], None if name == "none" else clean)
            require({k: v for k, v in row["summary"].items() if k != "safety"} == expected_summary,
                    "Episode-derived summary or conversion denominator mismatch")
            if name != "none":
                row["candidate_availability"] = candidate_availability(steps)
            if name.startswith("ours_progress_"):
                row["verified_suppressed_retry_targets"] = audit_stops(steps)
    if clean_reference is None:
        require((cap, seed) == (400, 0), "Only the first registered batch may establish Clean reference")
        report["clean_regression"] = dict(passed=True, role="initial_validation_clean_reference", verified_steps=0)
    else:
        reference_batch = json.loads(clean_reference.read_text())
        require(reference_batch["git_commit"] == batch["git_commit"] and
                reference_batch["configs"] == protocol["groups"][0]["configs"], "Wrong validation Clean reference")
        report["clean_regression"] = verify_v0_reference(batch_path, clean_reference, clean_only=True)
        report["source_sha256"][str(clean_reference.relative_to(ROOT))] = hashlib.sha256(clean_reference.read_bytes()).hexdigest()
    report["paired_contrasts"] = []
    for checkpoint in range(5):
        for objective in ("return", "safety"):
            first, second = "ours_progress_" + objective, "zero_one_budgeted_" + objective
            report["paired_contrasts"].append(dict(checkpoint_seed=checkpoint, first=first, second=second,
                **pair_outcomes(by_key[checkpoint, "none"]["episode_rows"], by_key[checkpoint, first]["episode_rows"],
                                by_key[checkpoint, second]["episode_rows"])))
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(report["analysis_commit"] == batch["git_commit"], "Analysis and experiment must use one frozen validation commit")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analyzer before validation")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--gradient-cap", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--clean-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch.resolve(), args.gradient_cap, args.attack_seed,
                     args.clean_reference.resolve() if args.clean_reference else None)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("VALIDATION_AUDIT_PASSED cap=%d seed=%d steps=%d" % (args.gradient_cap, args.attack_seed, result["raw_episode_verified_steps"]))
