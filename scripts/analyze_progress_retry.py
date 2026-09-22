"""Audit v0/progress-retry smoke pairs and every logged retry stop decision."""

import argparse
import json
from pathlib import Path
import subprocess
from collections import defaultdict
from summarize_proposed_smoke import ROOT, summarize, require
from analyze_proposed_development import pair_outcomes, audit_episode_records, retry_yield


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


def analyze(batch_path):
    names = {"none", "ours_return", "ours_safety", "ours_progress_return", "ours_progress_safety"}
    report = summarize(batch_path, expected_names=names)
    report["kind"] = "progress_retry_development_smoke"
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
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analyzer before publishing report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("PROGRESS_RETRY_AUDIT_PASSED=" + result["git_commit"])
    for row in result["rows"]:
        print("checkpoint=%d %-24s return=%.6f collisions=%d/%d gradients=%d stopped=%d" % (
            row["checkpoint_seed"], row["attack"], row["summary"]["episode_return_mean"], row["collisions"],
            row["episodes"], row["summary"]["gradient_evaluations"], row.get("verified_suppressed_retry_targets", 0)))
