"""Audit new Single runs and immutable, matched Return controls on reused split 20."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from prepare_single_validation import PROTOCOL, NEW_NAMES, COMPARISON_NAMES, candidate_config, REGISTRATION_FILES
from prepare_mechanism_controls import ROOT, sha256
from summarize_proposed_smoke import summarize, require
from analyze_mechanism_controls import audit_attempts, expected_seeds, verify_shared_witnesses
from analyze_proposed_development import audit_episode_records, pair_outcomes, retry_yield
from analyze_progress_retry import verify_v0_reference, audit_stops
from rl_att.evaluation.results import summarize_episodes
from rl_att.utils.zero_one_dependency import SHA256 as ZOOPT_SHA256


def validate_reuse_sources(current, legacy):
    require({p for p, value in legacy.items() if current.get(p) != value} == set(REGISTRATION_FILES),
            "Legacy execution changes extend beyond Single registration")
    require(set(current) - set(legacy) == {"rl_att/attacks/single_attempt.py"}, "Unexpected added runtime file")


def validate_runtime(manifest, reference):
    for field in ("python_runtime", "pip_freeze", "sumo_version", "victim_references"):
        require(manifest[field] == reference[field], "Frozen runtime/model differs: " + field)
    require(manifest["extra_dependencies"] == [], "Single unexpectedly uses extra optimizer dependencies")
    require([p["sha256"] for p in reference["extra_dependencies"]] == [ZOOPT_SHA256], "Baseline optimizer changed")


def load_controls(protocol, cap, seed):
    proof = next(p for p in protocol["controls"] if (p["gradient_cap"], p["attack_seed"]) == (cap, seed))
    path, batch_path = ROOT / proof["path"], ROOT / proof["batch"]
    require(sha256(path) == proof["sha256"] and sha256(batch_path) == proof["batch_sha256"], "Frozen control proof changed")
    report, batch = json.loads(path.read_text()), json.loads(batch_path.read_text())
    require(report["verified"] and report["kind"] == "shared_compute_validation" and report["research_split_id"] == 20,
            "Wrong control report")
    require(report["git_commit"] == batch["git_commit"] == proof["git_commit"] == protocol["baseline_experiment_commit"],
            "Mixed control execution provenance")
    require((report["gradient_cap"], report["attack_seed"]) == (cap, seed) and len(report["rows"]) == 25,
            "Mislabeled or incomplete control cell")
    for filename, digest in report["source_sha256"].items():
        require(sha256(ROOT / filename) == digest, "Frozen control input changed: " + filename)
    rows = {(r["checkpoint_seed"], r["attack"]): copy.deepcopy(r) for r in report["rows"]}
    manifests, directories, entries = {}, {}, {}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest, evaluation = [json.loads((directory / name).read_text()) for name in ("manifest.json", "evaluation.json")]
        checkpoint = manifest["config"]["run_seeds"][0]
        require(manifest["source_sha256_before"] == protocol["baseline_source_sha256"], "Control source map changed")
        require(manifest["status"] == "passed" and evaluation["git_commit"] == batch["git_commit"], "Unverified control run")
        manifests[checkpoint] = manifest
        for entry in evaluation["runs"]:
            key = checkpoint, entry["attack"]["name"]
            directories[key], entries[key] = directory / entry["results_directory"], entry
    require(set(manifests) == set(range(5)), "Missing control checkpoint")
    return proof, rows, manifests, directories, entries


def analyze(batch_path, cap, seed):
    protocol = json.loads(PROTOCOL.read_text())
    validate_reuse_sources(protocol["frozen_source_sha256"], protocol["baseline_source_sha256"])
    patch = subprocess.check_output(["git", "diff", protocol["baseline_experiment_commit"], protocol["frozen_method_commit"], "--"] + list(REGISTRATION_FILES), cwd=str(ROOT))
    require(hashlib.sha256(patch).hexdigest() == protocol["registration_changes"]["diff_sha256"], "Reviewed registration diff changed")
    sources = {PROTOCOL.relative_to(ROOT).as_posix(): sha256(PROTOCOL)}
    for filename, key in (("configs/research_seed_splits.json", "seed_splits_sha256"), ("configs/frozen_victims.json", "frozen_victims_sha256")):
        require(sha256(ROOT / filename) == protocol[key], "Frozen traffic/model registry changed")
        sources[filename] = protocol[key]
    for key in ("baseline_summary", "baseline_preflight"):
        proof = protocol[key]
        require(sha256(ROOT / proof["path"]) == proof["sha256"], "Baseline summary/preflight changed")
        sources[proof["path"]] = proof["sha256"]
    group = next(g for g in protocol["groups"] if (g["gradient_cap"], g["attack_seed"]) == (cap, seed))
    batch = json.loads(batch_path.read_text())
    require(batch["configs"] == group["configs"] and len(batch["runs"]) == 5, "Unregistered candidate cell")
    params = {a["name"]: a["parameters"] for a in candidate_config(0, cap, seed)["attacks"]}
    report = summarize(batch_path, expected_episodes=20, expected_names=set(NEW_NAMES), expected_attack_seed=seed,
                       expected_split_id=20, expected_parameters=params)
    proof, old_rows, references, old_directories, old_entries = load_controls(protocol, cap, seed)
    report.update(kind="single_candidate_validation", gradient_cap=cap, forward_cap=2 * cap, research_split_id=20,
                  raw_episode_verified_steps=0, reused_return_verified_steps=0, shared_first_attempt_comparisons=[],
                  paired_contrasts=[], control_reference=proof)
    sources[proof["path"]], sources[proof["batch"]] = proof["sha256"], proof["batch_sha256"]
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    for row in report["rows"]:
        row.update(origin="new_run", source_commit=batch["git_commit"])
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest, evaluation = [json.loads((directory / name).read_text()) for name in ("manifest.json", "evaluation.json")]
        checkpoint = manifest["config"]["run_seeds"][0]
        require(manifest["config"] == candidate_config(checkpoint, cap, seed) == evaluation["config"], "Candidate configuration changed")
        sources[run["config"]] = sha256(ROOT / run["config"])
        require(manifest["source_sha256_before"] == protocol["frozen_source_sha256"], "Candidate executed source changed")
        require(all(p == "Data/StraightRoad.sumocfg" for p in manifest["changed_source_files"]), "Unexpected source mutation")
        validate_runtime(manifest, references[checkpoint])
        require([e["attack"]["name"] for e in evaluation["runs"]] == list(NEW_NAMES) and not evaluation["gate_enabled"], "Wrong candidate conditions")
        witnesses, clean = {}, by_key[checkpoint, "none"]["episode_rows"]
        for entry in evaluation["runs"]:
            name = entry["attack"]["name"]
            require(entry["run_seed"] == checkpoint and entry["victim"] == "clean" and entry["frozen_unchanged"], "Frozen identity changed")
            require(entry["attack"] == next(a for a in manifest["config"]["attacks"] if a["name"] == name), "Candidate attack differs from config")
            require(entry["effective_seeds"] == expected_seeds(manifest["config"], checkpoint, name), "Candidate role seeds differ")
            for field in ("checkpoint_sha256", "weights_sha256"):
                require(entry[field] == manifest["victim_references"][0][field], "Candidate weight hash differs")
            path = directory / entry["results_directory"] / "steps.jsonl"
            steps = [json.loads(line) for line in path.read_text().splitlines()]
            row = by_key[checkpoint, name]
            report["raw_episode_verified_steps"] += audit_episode_records(steps, row["episode_rows"], 200)
            for step in steps:
                obs, adv, delta = [np.asarray(step[k], dtype=np.float64) for k in ("observation", "adversarial_observation", "perturbation")]
                require(all(a.shape == (16,) and np.isfinite(a).all() for a in (obs, adv, delta)), "Invalid raw observation")
                require(np.allclose(adv - obs, delta, rtol=0, atol=1e-7) and
                        np.all(np.abs(delta) <= .2 * np.abs(obs) + .05 + 2e-7), "Invalid perturbation envelope")
            if name == "none":
                require(all(not r["attacked"] and not any(r["perturbation"]) for r in steps), "Clean perturbed")
            else:
                witnesses[name], row["attempt_audit"] = audit_attempts(steps, name, seed, params[name])
                row["retry_yield"] = retry_yield(steps)
            require(summarize_episodes(row["episode_rows"], clean if name != "none" else None) ==
                    {k: v for k, v in row["summary"].items() if k != "safety"}, "Candidate episode aggregate differs")
        for name in ("zero_one_budgeted_return", "ours_progress_return"):
            key = checkpoint, name
            row, entry = old_rows[key], old_entries[key]
            require(entry["effective_seeds"] == expected_seeds(manifest["config"], checkpoint, name), "Control role seeds differ")
            for field in ("checkpoint_sha256", "weights_sha256"):
                require(entry[field] == manifest["victim_references"][0][field], "Control weight hash differs")
            p = copy.deepcopy(params["ours_single_return"])
            p["max_attempts"] = 3
            if name == "ours_progress_return":
                p["retry_rule"] = "strict_margin_progress"
            require(entry["attack"]["parameters"] == p, "Matched control parameters changed")
            steps = [json.loads(line) for line in (old_directories[key] / "steps.jsonl").read_text().splitlines()]
            report["reused_return_verified_steps"] += audit_episode_records(steps, row["episode_rows"], 200)
            witnesses[name], row["attempt_audit"] = audit_attempts(steps, name, seed, p)
            if name == "ours_progress_return":
                row["retry_yield"], row["verified_suppressed_retry_targets"] = retry_yield(steps), audit_stops(steps)
            require(summarize_episodes(row["episode_rows"], clean) ==
                    {k: v for k, v in row["summary"].items() if k != "safety"}, "Reused comparison denominator changed")
            row.update(origin="reused_validation", source_commit=proof["git_commit"])
            report["rows"].append(row)
            by_key[key] = row
        names = COMPARISON_NAMES[1:]
        report["shared_first_attempt_comparisons"].append(dict(checkpoint_seed=checkpoint,
                                                               comparisons=verify_shared_witnesses(witnesses, names)))
        for spec in protocol["contrasts"]:
            report["paired_contrasts"].append(dict(checkpoint_seed=checkpoint, **spec,
                **pair_outcomes(clean, by_key[checkpoint, spec["first"]]["episode_rows"], by_key[checkpoint, spec["second"]]["episode_rows"])))
    report["clean_regression"] = verify_v0_reference(batch_path, ROOT / proof["batch"], clean_only=True)
    report["source_sha256"].update(sources)
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    report["new_episodes"] = sum(r["episodes"] for r in report["rows"] if r["origin"] == "new_run")
    require(report["new_episodes"] == group["episodes"] == 200 and len(report["rows"]) == 20, "Incomplete candidate comparison")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit audit code before publishing")
    report["limitation"] = protocol["limitations"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--gradient-cap", type=int, choices=(100, 200, 400), required=True)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.batch.resolve(), args.gradient_cap, args.attack_seed)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("CANDIDATE_VALIDATION_AUDIT_PASSED new_episodes=%d new_steps=%d reused_steps=%d" %
          (report["new_episodes"], report["raw_episode_verified_steps"], report["reused_return_verified_steps"]))
