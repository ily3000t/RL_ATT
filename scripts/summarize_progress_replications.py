"""Combine audited attack-seed replicates and locate changed paired trajectories."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_proposed_smoke import ROOT, require


def first_divergence(old, new):
    fields = ("action", "reward", "terminated", "next_observation")
    for index, (a, b) in enumerate(zip(old, new)):
        require(a["step"] == b["step"] == index, "Nonconsecutive paired episode")
        if any(a[k] != b[k] for k in fields):
            require(a["observation"] == b["observation"], "Different input before first transition divergence")
            result = dict(step_zero_based=index, same_observation=True,
                          v0_action=a["action"], progress_action=b["action"])
            for label, rows in (("v0", old), ("progress", new)):
                block = max(i for i in range(index + 1) if rows[i]["attack_metadata"]["planned"])
                metadata = rows[block]["attack_metadata"]
                selected = metadata["selected_candidate"]
                history = [r["action"] for r in rows[:index]]
                compact = lambda entry: {k: v for k, v in entry.items() if k != "history"}
                result[label] = dict(block_start=block, selected_candidate=selected,
                                     selected_trace=None if selected is None else metadata["candidate_trace"][selected],
                                     complete_candidates=len(metadata["candidate_trace"]),
                                     block_start_cost=dict(rows[block]["attack_cost"]),
                                     attempts_at_divergence=[compact(t) for t in metadata["inner_attempt_trace"] if t["history"] == history],
                                     stops_at_divergence=[compact(t) for t in metadata.get("retry_stop_trace", []) if t["history"] == history])
                result[label]["block_start_cost"].pop("wall_seconds", None)
            return result
    require(len(old) == len(new), "Unequal episode length without logged termination divergence")
    return None


def summarize_reports(paths):
    reports, inputs = {}, []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        seed = data["attack_seed"]
        require(seed not in reports, "Duplicate attack seed")
        require(data["verified"] and data["kind"] == "progress_retry_development", "Unverified development input")
        for name, digest in data["source_sha256"].items():
            require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, "Audited input changed: " + name)
        reports[seed] = data
        inputs.append(dict(attack_seed=seed, path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    require(set(reports) == {0, 1, 2}, "Expected attack seeds 0/1/2")
    baseline = reports[0]
    result = dict(kind="progress_retry_attack_seed_replication", inputs=inputs, replications=[], divergences=[],
                  limitation="Shared development traffic and checkpoints; objectives and attack replicates are correlated, not independent generalization trials")
    baseline_models = {(x["checkpoint_seed"], x["attack"]): x["checkpoint_sha256"] for x in baseline["rows"]}
    reference_configs = None
    for seed, report in sorted(reports.items()):
        require(report["traffic"] == baseline["traffic"], "Traffic differs across attack seeds")
        require({(x["checkpoint_seed"], x["attack"]): x["checkpoint_sha256"] for x in report["rows"]} == baseline_models,
                "Frozen models/conditions differ across attack seeds")
        batch = json.loads((ROOT / report["batch"]).read_text())
        directories, configs = {}, {}
        for run in batch["runs"]:
            directory = Path(run["run_dir"])
            manifest = json.loads((directory / "manifest.json").read_text())
            config = manifest["config"]
            require(config["research_seeds"].pop("attack_seed") == seed, "Manifest attack seed differs")
            configs[tuple(config["run_seeds"])] = config
            evaluation = json.loads((directory / "evaluation.json").read_text())
            for entry in evaluation["runs"]:
                directories[(entry["run_seed"], entry["attack"]["name"])] = directory / entry["results_directory"]
        if reference_configs is None:
            reference_configs = configs
        require(configs == reference_configs, "Cross-replicate configs differ beyond attack seed")
        compact = {k: report[k] for k in ("attack_seed", "git_commit", "analysis_commit", "batch", "traffic", "raw_episode_verified_steps")}
        compact["regression"] = report.get("clean_regression", report.get("v0_regression"))
        compact["rows"] = []
        by_key = {(x["checkpoint_seed"], x["attack"]): x for x in report["rows"]}
        for row in report["rows"]:
            item = {k: row[k] for k in ("checkpoint_seed", "attack", "collisions", "episodes", "audit")}
            item["outcomes"] = {k: row["summary"].get(k) for k in ("episode_return_mean", "attack_successes", "attack_success_eligible_episodes")}
            if "retry_yield" in row:
                item["retry_yield"] = {k: v for k, v in row["retry_yield"].items() if k != "new_branch_examples"}
            if "verified_suppressed_retry_targets" in row:
                item["verified_suppressed_retry_targets"] = row["verified_suppressed_retry_targets"]
            compact["rows"].append(item)
        compact["paired_contrasts"] = [{k: v for k, v in p.items() if k != "episodes"} for p in report["paired_contrasts"]]
        for pair in report["paired_contrasts"]:
            checkpoint, first, second = pair["checkpoint_seed"], pair["first"], pair["second"]
            old, new = [by_key[(checkpoint, name)]["episode_rows"] for name in (second, first)]
            changed = [a["episode"] for a, b in zip(old, new) if a["trajectory_sha256"] != b["trajectory_sha256"]]
            if not changed:
                continue
            paths = [directories[(checkpoint, name)] / "steps.jsonl" for name in (second, first)]
            raw = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
            for ep in changed:
                divergence = first_divergence(*[[r for r in rows if r["episode"] == ep] for rows in raw])
                require(divergence is not None, "Changed digest without transition divergence")
                result["divergences"].append(dict(attack_seed=seed, checkpoint_seed=checkpoint, episode=ep,
                                                 first=first, second=second,
                                                 outcome=next(e for e in pair["episodes"] if e["episode"] == ep),
                                                 raw_paths=[str(p.relative_to(ROOT)) for p in paths], **divergence))
        result["replications"].append(compact)
    result["summary_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit summarizer before publishing")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize_reports([p.resolve() for p in args.reports])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("REPLICATION_SUMMARY_PASSED changed_pairs=%d" % len(report["divergences"]))
