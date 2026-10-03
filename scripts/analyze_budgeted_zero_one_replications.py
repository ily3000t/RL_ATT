"""Audit matched budgeted Zero-One controls against frozen progress-retry runs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_proposed_smoke import ROOT, summarize, require
from analyze_proposed_development import audit_episode_records, pair_outcomes
from analyze_progress_retry import verify_v0_reference
from rl_att.utils.zero_one_dependency import SHA256 as ZOOPT_SHA256


def validate_pairing(baseline, reference, baseline_entries, reference_entries, attack_seed):
    for field in ("python_runtime", "pip_freeze", "sumo_version", "source_sha256_before", "victim_references"):
        require(baseline[field] == reference[field], "Cross-batch provenance differs: " + field)
    configs = [{k: v for k, v in m["config"].items() if k != "attacks"} for m in (baseline, reference)]
    require(configs[0] == configs[1], "Cross-batch evaluation config differs")
    require(configs[0]["research_seeds"]["attack_seed"] == attack_seed, "Wrong paired attack seed")
    clean = next(e for e in reference_entries if e["attack"]["name"] == "none")
    for entry in baseline_entries + reference_entries:
        for field in ("run_seed", "effective_seeds", "checkpoint_sha256", "weights_sha256"):
            actual, expected = entry[field], clean[field]
            if field == "effective_seeds":
                require(isinstance(actual, dict), "Cross-batch victim/seed differs: " + field)
                mechanism = ("unused_no_attack" if entry["attack"]["name"] == "none"
                             else "sha256_attack_episode_full_history_target_attempt_local_numpy")
                require(actual.get("attack_rng") == mechanism, "Unexpected attack RNG mechanism")
                actual, expected = [{k: v for k, v in seeds.items() if k != "attack_rng"}
                                    for seeds in (actual, expected)]
            require(actual == expected, "Cross-batch victim/seed differs: " + field)
    for objective in ("return", "safety"):
        zero = next(e["attack"] for e in baseline_entries if e["attack"]["name"] == "zero_one_budgeted_" + objective)
        for prefix in ("ours_", "ours_progress_"):
            ours = next(e["attack"] for e in reference_entries if e["attack"]["name"] == prefix + objective)
            require(zero["budget"] == ours["budget"], "Cross-batch perturbation budget differs")
            parameters = {k: v for k, v in ours["parameters"].items() if k != "retry_rule"}
            require(zero["parameters"] == parameters, "Cross-batch search budget or objective differs")


def analyze(batch_path, reference_path, attack_seed):
    require(attack_seed in (1, 2), "This extension is restricted to attack seeds 1/2")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    require(reference["verified"] and reference["kind"] == "progress_retry_development"
            and reference["attack_seed"] == attack_seed, "Wrong or unverified reference")
    for name, digest in reference["source_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "Frozen reference input changed: " + name)
    report = summarize(batch_path, expected_episodes=10, expected_attack_seed=attack_seed,
                       expected_names={"none", "zero_one_budgeted_return", "zero_one_budgeted_safety"})
    require(report["traffic"] == reference["traffic"], "Cross-batch traffic differs")
    report["kind"] = "budgeted_zero_one_matched_replication"
    report["reference_report"] = dict(path=str(reference_path.relative_to(ROOT)),
                                       sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                                       git_commit=reference["git_commit"], analysis_commit=reference["analysis_commit"])
    report["reference_rows"] = reference["rows"]
    reference_batch_path = ROOT / reference["batch"]
    reference_batch = json.loads(reference_batch_path.read_text())
    require(reference_batch["status"] == "passed", "Reference batch did not pass")
    references = {}
    for run in reference_batch["runs"]:
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        entries = json.loads((directory / "evaluation.json").read_text())["runs"]
        seed = manifest["config"]["run_seeds"][0]
        require(seed not in references, "Repeated reference checkpoint")
        references[seed] = manifest, entries
    require(set(references) == set(range(5)), "Missing reference checkpoint")
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    report["raw_episode_verified_steps"] = 0
    for run in json.loads(batch_path.read_text())["runs"]:
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        entries = json.loads((directory / "evaluation.json").read_text())["runs"]
        seed = manifest["config"]["run_seeds"][0]
        ref_manifest, ref_entries = references[seed]
        validate_pairing(manifest, ref_manifest, entries, ref_entries, attack_seed)
        require([d["sha256"] for d in manifest["extra_dependencies"]] == [ZOOPT_SHA256], "Unpinned Zero-One dependency")
        for entry in entries:
            path = directory / entry["results_directory"] / "steps.jsonl"
            episodes = by_key[(seed, entry["attack"]["name"])]["episode_rows"]
            with path.open(encoding="utf-8") as stream:
                report["raw_episode_verified_steps"] += audit_episode_records((json.loads(line) for line in stream), episodes, 200)
    report["clean_regression"] = verify_v0_reference(batch_path, reference_batch_path, clean_only=True)
    # validate_pairing above additionally requires equal attack seeds for this comparison.
    report["paired_contrasts"] = []
    ref_rows = {(r["checkpoint_seed"], r["attack"]): r for r in reference["rows"]}
    for seed in range(5):
        clean = by_key[(seed, "none")]["episode_rows"]
        for objective in ("return", "safety"):
            second = "zero_one_budgeted_" + objective
            for prefix in ("ours_", "ours_progress_"):
                first = prefix + objective
                pair = pair_outcomes(clean, ref_rows[(seed, first)]["episode_rows"], by_key[(seed, second)]["episode_rows"])
                report["paired_contrasts"].append(dict(checkpoint_seed=seed, first=first, second=second, **pair))
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analysis before publishing")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--attack-seed", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.batch.resolve(), args.reference_report.resolve(), args.attack_seed)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("MATCHED_BASELINE_AUDIT_PASSED=" + report["git_commit"])
    for row in report["rows"]:
        print("checkpoint=%d %-26s return=%.6f collisions=%d/%d" % (
            row["checkpoint_seed"], row["attack"], row["summary"]["episode_return_mean"], row["collisions"], row["episodes"]))
