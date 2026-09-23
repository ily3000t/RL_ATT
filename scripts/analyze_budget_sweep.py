"""Audit a lower-budget paired batch against matched frozen full-budget sources."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_proposed_smoke import ROOT, summarize, require
from analyze_proposed_development import audit_episode_records, pair_outcomes
from analyze_progress_retry import audit_stops, verify_v0_reference
from rl_att.utils.zero_one_dependency import SHA256 as ZOOPT_SHA256


def candidate_availability(steps):
    counts = dict(blocks=0, incomplete_candidates=0, no_complete_search_candidate_blocks=0,
                  selected_complete_fallback_blocks=0, selected_search_candidate_blocks=0, unplanned_fallback_blocks=0)
    for row in steps:
        m = row["attack_metadata"]
        if not m["planned"]:
            continue
        counts["blocks"] += 1
        traces = m["candidate_trace"]
        counts["incomplete_candidates"] += len(m["incomplete_candidates"])
        counts["no_complete_search_candidate_blocks"] += int(not any(t["kind"] != "fallback" for t in traces))
        selected = m["selected_candidate"]
        if selected is None:
            require(not traces and m["oracle_unplanned_fallback"], "Unplanned fallback despite a complete plan")
            counts["unplanned_fallback_blocks"] += 1
        elif traces[selected]["kind"] == "fallback":
            counts["selected_complete_fallback_blocks"] += 1
        else:
            counts["selected_search_candidate_blocks"] += 1
    return counts


def validate_budget_reference(manifest, entry, reference_manifest, reference_entry, gradient_cap):
    require(gradient_cap in (100, 200), "Unregistered lower budget")
    for key in ("source_sha256_before", "python_runtime", "pip_freeze", "sumo_version", "victim_references"):
        require(manifest[key] == reference_manifest[key], "Budget reference provenance differs: " + key)
    require({k: v for k, v in manifest["config"].items() if k != "attacks"} ==
            {k: v for k, v in reference_manifest["config"].items() if k != "attacks"}, "Non-budget evaluation settings differ")
    for key in ("run_seed", "effective_seeds", "checkpoint_sha256", "weights_sha256"):
        require(entry[key] == reference_entry[key], "Budget reference victim/seed differs: " + key)
    require(entry["attack"] in manifest["config"]["attacks"], "Logged attack differs from manifest")
    expected = copy.deepcopy(reference_entry["attack"])
    if expected["name"] != "none":
        limits = expected["parameters"]["resource_limits"]
        require(limits == dict(gradient_evaluations=400, policy_forward_calls=800,
                               new_shadow_transitions=200, shadow_steps=4000), "Reference is not the frozen upper budget")
        limits.update(gradient_evaluations=gradient_cap, policy_forward_calls=2 * gradient_cap)
    require(entry["attack"] == expected, "Changes extend beyond the two model-computation caps")


def load_reference(path, attack_seed, names):
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report["verified"] and report["attack_seed"] == attack_seed, "Wrong/unverified upper reference")
    for name, digest in report["source_sha256"].items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "Reference input changed: " + name)
    entries = {}
    batch = json.loads((ROOT / report["batch"]).read_text())
    require(batch["status"] == "passed", "Upper reference failed")
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
            if entry["attack"]["name"] in names:
                key = entry["run_seed"], entry["attack"]["name"]
                require(key not in entries, "Repeated upper-reference condition")
                entries[key] = manifest, entry
    require(set(entries) == {(s, n) for s in range(5) for n in names}, "Missing upper-reference condition")
    proof = dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), git_commit=report["git_commit"])
    return report, entries, proof


def analyze(batch_path, zero_path, progress_path, attack_seed, gradient_cap):
    zero_names = {"zero_one_budgeted_return", "zero_one_budgeted_safety"}
    progress_names = {"none", "ours_progress_return", "ours_progress_safety"}
    zero, zero_entries, zero_proof = load_reference(zero_path, attack_seed, zero_names)
    progress, progress_entries, progress_proof = load_reference(progress_path, attack_seed, progress_names)
    references = dict(zero_entries)
    references.update(progress_entries)
    report = summarize(batch_path, expected_episodes=10, expected_names=zero_names | progress_names,
                       expected_attack_seed=attack_seed)
    report.update(kind="shared_compute_budget_sweep", gradient_cap=gradient_cap, forward_cap=2 * gradient_cap,
                  upper_references=[zero_proof, progress_proof], raw_episode_verified_steps=0)
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    for run in json.loads(batch_path.read_text())["runs"]:
        directory = Path(run["run_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        require([d["sha256"] for d in manifest["extra_dependencies"]] == [ZOOPT_SHA256], "Unpinned optimizer dependency")
        for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
            key = entry["run_seed"], entry["attack"]["name"]
            ref_manifest, ref_entry = references[key]
            validate_budget_reference(manifest, entry, ref_manifest, ref_entry, gradient_cap)
            path = directory / entry["results_directory"] / "steps.jsonl"
            with path.open() as stream:
                steps = [json.loads(line) for line in stream]
            report["raw_episode_verified_steps"] += audit_episode_records(steps, by_key[key]["episode_rows"], 200)
            if key[1] != "none":
                by_key[key]["candidate_availability"] = candidate_availability(steps)
            if key[1].startswith("ours_progress_"):
                by_key[key]["verified_suppressed_retry_targets"] = audit_stops(steps)
    report["clean_regression"] = verify_v0_reference(batch_path, ROOT / progress["batch"], clean_only=True)
    upper_rows = {(r["checkpoint_seed"], r["attack"]): r for r in zero["rows"] + progress["rows"]}
    report["paired_contrasts"], report["upper_budget_contrasts"] = [], []
    for seed in range(5):
        clean = by_key[(seed, "none")]["episode_rows"]
        for objective in ("return", "safety"):
            first, second = "ours_progress_" + objective, "zero_one_budgeted_" + objective
            report["paired_contrasts"].append(dict(checkpoint_seed=seed, first=first, second=second,
                **pair_outcomes(clean, by_key[(seed, first)]["episode_rows"], by_key[(seed, second)]["episode_rows"])))
        for name in sorted(zero_names | (progress_names - {"none"})):
            report["upper_budget_contrasts"].append(dict(checkpoint_seed=seed, attack=name, first_budget=gradient_cap,
                second_budget=400, **pair_outcomes(clean, by_key[(seed, name)]["episode_rows"], upper_rows[(seed, name)]["episode_rows"])))
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analysis before publishing")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--zero-one-reference", type=Path, required=True)
    parser.add_argument("--progress-reference", type=Path, required=True)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--gradient-cap", type=int, choices=(100, 200), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.batch.resolve(), args.zero_one_reference.resolve(), args.progress_reference.resolve(), args.attack_seed, args.gradient_cap)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("BUDGET_SWEEP_AUDIT_PASSED seed=%d gradient_cap=%d steps=%d" % (args.attack_seed, args.gradient_cap, report["raw_episode_verified_steps"]))
