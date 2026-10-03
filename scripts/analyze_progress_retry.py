"""Audit v0/progress-retry development pairs and logged retry stop decisions."""

import argparse
import json
from pathlib import Path
import subprocess
from collections import defaultdict
import hashlib
from summarize_proposed_smoke import ROOT, summarize, require
from analyze_proposed_development import pair_outcomes, audit_episode_records, retry_yield


def verify_v0_reference(batch_path, reference_path, clean_only=False):
    names = ("none",) if clean_only else ("none", "ours_return", "ours_safety")
    records = []
    for path in (reference_path, batch_path):
        batch = json.loads(path.read_text())
        require(batch["status"] == "passed", "Baseline regression batch did not pass")
        conditions = {}
        for run in batch["runs"]:
            directory = Path(run["run_dir"])
            manifest = json.loads((directory / "manifest.json").read_text())
            evaluation = json.loads((directory / "evaluation.json").read_text())
            require(manifest["status"] == "passed", "Baseline regression run failed")
            for entry in evaluation["runs"]:
                name = entry["attack"]["name"]
                if name in names:
                    require((entry["run_seed"], name) not in conditions, "Repeated regression condition")
                    conditions[(entry["run_seed"], name)] = (directory / entry["results_directory"] / "steps.jsonl", entry, manifest)
        records.append((batch, conditions))
    expected = {(seed, name) for seed in range(5) for name in names}
    require(set(records[0][1]) == set(records[1][1]) == expected, "Missing v0 regression conditions")
    proof = []
    for key in sorted(expected):
        old_path, old_entry, old_manifest = records[0][1][key]
        new_path, new_entry, new_manifest = records[1][1][key]
        for field in ("attack", "effective_seeds", "checkpoint_sha256", "weights_sha256"):
            old_value, new_value = old_entry[field], new_entry[field]
            if clean_only and field == "effective_seeds":
                old_value, new_value = [{k: v for k, v in value.items() if k != "attack_seed"}
                                        for value in (old_value, new_value)]
            require(old_value == new_value, "Baseline regression provenance differs: " + field)
        for field in ("python_runtime", "pip_freeze", "sumo_version"):
            require(old_manifest[field] == new_manifest[field], "Baseline regression runtime differs")
        old_rows, new_rows = [[json.loads(line) for line in p.read_text().splitlines()] for p in (old_path, new_path)]
        require(len(old_rows) == len(new_rows), "Baseline regression trajectory length differs")
        for old, new in zip(old_rows, new_rows):
            old["attack_cost"].pop("wall_seconds")
            new["attack_cost"].pop("wall_seconds")
            require(old == new, "Existing v0 behavior changed: " + str(key))
        proof.append(dict(checkpoint_seed=key[0], attack=key[1], verified_steps=len(new_rows),
                          old_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                          new_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest()))
    return dict(passed=True, reference_batch=str(reference_path.relative_to(ROOT)),
                reference_commit=records[0][0]["git_commit"], comparisons=proof,
                excluded_fields=["attack_cost.wall_seconds"],
                excluded_seed_fields=["attack_seed"] if clean_only else [],
                verified_steps=sum(p["verified_steps"] for p in proof))


def audit_stops(steps):
    stopped = 0
    for row in steps:
        metadata = row["attack_metadata"]
        if not metadata["planned"]:
            continue
        groups = defaultdict(list)
        for attempt in metadata["inner_attempt_trace"]:
            groups[(tuple(attempt["history"]), attempt["target"])].append(attempt)
        trace = metadata["retry_stop_trace"]
        require(metadata["suppressed_retry_targets"] == len(trace), "Suppression count differs")
        seen = set()
        for stop in trace:
            key = tuple(stop["history"]), stop["target"]
            require(key not in seen, "Repeated suppression decision")
            seen.add(key)
            attempts = groups[key]
            require(len(attempts) == stop["attempts_so_far"] >= 2, "Suppressed retry was executed or first retry omitted")
            require([a["attempt"] for a in attempts] == list(range(len(attempts))), "Nonconsecutive attempts")
            previous = max(a["margin"] for a in attempts[:-1])
            require(stop["previous_best"] == previous and stop["margin"] == attempts[-1]["margin"], "Stop margin differs from PGD trace")
            require(stop["margin"] <= previous and stop["improved"] is False, "Improving retry incorrectly suppressed")
            require(all(a["actual"] != stop["target"] for a in attempts), "Found target mislabeled as unresolved")
            require(stop["reason"] == "retry_did_not_improve_best_margin", "Unknown stop reason")
        stopped += len(trace)
    return stopped


def analyze(batch_path, reference_path=None, expected_episodes=2, expected_attack_seed=0, clean_reference_path=None):
    names = {"none", "ours_return", "ours_safety", "ours_progress_return", "ours_progress_safety"}
    report = summarize(batch_path, expected_episodes=expected_episodes, expected_names=names,
                       expected_attack_seed=expected_attack_seed)
    report["kind"] = ("progress_retry_development_smoke" if expected_episodes == 2
                      else "progress_retry_development")
    batch = json.loads(batch_path.read_text())
    directories = {}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        evaluation = json.loads((directory / "evaluation.json").read_text())
        for entry in evaluation["runs"]:
            directories[(entry["run_seed"], entry["attack"]["name"])] = directory / entry["results_directory"]
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    report["raw_episode_verified_steps"] = 0
    for key, directory in directories.items():
        with (directory / "steps.jsonl").open() as stream:
            steps = [json.loads(line) for line in stream]
        row = by_key[key]
        report["raw_episode_verified_steps"] += audit_episode_records(steps, row["episode_rows"], 200)
        if key[1] != "none":
            row["retry_yield"] = retry_yield(steps)
        if key[1].startswith("ours_progress_"):
            row["verified_suppressed_retry_targets"] = audit_stops(steps)
    report["paired_contrasts"] = []
    for seed in range(5):
        for objective in ("return", "safety"):
            old, new = "ours_" + objective, "ours_progress_" + objective
            contrast = pair_outcomes(by_key[(seed, "none")]["episode_rows"],
                                     by_key[(seed, new)]["episode_rows"], by_key[(seed, old)]["episode_rows"])
            report["paired_contrasts"].append(dict(checkpoint_seed=seed, first=new, second=old, **contrast))
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    if reference_path is not None:
        report["v0_regression"] = verify_v0_reference(batch_path, reference_path)
    if clean_reference_path is not None:
        report["clean_regression"] = verify_v0_reference(batch_path, clean_reference_path, clean_only=True)
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analyzer before publishing report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v0-reference", type=Path)
    parser.add_argument("--episodes", type=int, choices=(2, 10), default=2)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--clean-reference", type=Path)
    args = parser.parse_args()
    result = analyze(args.batch.resolve(), args.v0_reference.resolve() if args.v0_reference else None,
                     expected_episodes=args.episodes, expected_attack_seed=args.attack_seed,
                     clean_reference_path=args.clean_reference.resolve() if args.clean_reference else None)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PROGRESS_RETRY_AUDIT_PASSED=" + result["git_commit"])
    for row in result["rows"]:
        print("checkpoint=%d %-24s return=%.6f collisions=%d/%d gradients=%d stopped=%d" % (
            row["checkpoint_seed"], row["attack"], row["summary"]["episode_return_mean"], row["collisions"],
            row["episodes"], row["summary"]["gradient_evaluations"], row.get("verified_suppressed_retry_targets", 0)))
