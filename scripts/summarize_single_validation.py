"""Summarize all frozen Single cells against audited Return validation controls."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from prepare_single_validation import PROTOCOL, COMPARISON_NAMES
from prepare_mechanism_controls import ROOT, sha256
from summarize_proposed_smoke import require
from summarize_mechanism_development import CAPS, COSTS, validate_grid, aggregate, aggregate_contrast, transition_divergence
from analyze_single_validation import load_controls
from export_mechanism_development import plot


def validate_reports(reports, protocol, commit):
    validate_grid(reports)
    traffic, models = None, {}
    expected = {(c, name) for c in range(5) for name in COMPARISON_NAMES}
    for report in reports:
        require(report["verified"] and report["kind"] == "single_candidate_validation" and report["git_commit"] == commit,
                "Mixed candidate provenance")
        require(report["research_split_id"] == 20 and report["new_episodes"] == 200 and report["clean_regression"]["passed"],
                "Wrong candidate count, split or regression")
        require(len(report["rows"]) == 20 and {(r["checkpoint_seed"], r["attack"]) for r in report["rows"]} == expected,
                "Missing or repeated candidate comparison")
        traffic = report["traffic"] if traffic is None else traffic
        require(report["traffic"] == traffic and len(traffic["episode_sumo_seeds"]) == 20, "Cross-cell traffic differs")
        for row in report["rows"]:
            new = row["attack"] in protocol["new_methods"]
            require(row["origin"] == ("new_run" if new else "reused_validation") and
                    row["source_commit"] == (commit if new else protocol["baseline_experiment_commit"]), "Wrong row provenance")
            require(row["episodes"] == len(row["episode_rows"]) == 20 and
                    [(e["episode"], e["sumo_seed"]) for e in row["episode_rows"]] == list(enumerate(traffic["episode_sumo_seeds"], 1)),
                    "Missing or unpaired candidate episode")
            models.setdefault(row["checkpoint_seed"], row["checkpoint_sha256"])
            require(models[row["checkpoint_seed"]] == row["checkpoint_sha256"], "Cross-cell model hash differs")
    return traffic, models


def summarize(pipeline_path):
    cache = {}
    def digest(path):
        path = path.resolve()
        if path not in cache:
            cache[path] = sha256(path)
        return cache[path]
    protocol, pipeline = json.loads(PROTOCOL.read_text()), json.loads(pipeline_path.read_text())
    require(pipeline["kind"] == "single_candidate_validation_pipeline" and pipeline["status"] == "passed" and
            pipeline["verified_episodes"] == protocol["total_episodes"] == 1800, "Incomplete candidate pipeline")
    require(digest(PROTOCOL) == pipeline["protocol_sha256"], "Frozen candidate protocol changed")
    require([(g["gradient_cap"], g["attack_seed"]) for g in pipeline["groups"]] ==
            [(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]], "Candidate grid order changed")
    reports, inputs, directories = [], [], {}
    baseline_regression = 0
    for group, registered in zip(pipeline["groups"], protocol["groups"]):
        path = Path(group["audit_path"])
        require(group["status"] == "passed" and group["configs"] == registered["configs"] and
                digest(path) == group["audit_sha256"], "Unaudited candidate cell")
        report = json.loads(path.read_text())
        require((report["gradient_cap"], report["attack_seed"]) == (group["gradient_cap"], group["attack_seed"]), "Mislabeled candidate cell")
        for filename, value in report["source_sha256"].items():
            require(digest(ROOT / filename) == value, "Candidate audit input changed: " + filename)
        proof, _, _, old_dirs, _ = load_controls(protocol, group["gradient_cap"], group["attack_seed"])
        require(report["control_reference"] == proof, "Different reused control cell")
        old_report = json.loads((ROOT / proof["path"]).read_text())
        for pair in report["paired_contrasts"]:
            if pair["first"] == "ours_progress_return" and pair["second"] == "zero_one_budgeted_return":
                original = next(p for p in old_report["paired_contrasts"] if p["checkpoint_seed"] == pair["checkpoint_seed"]
                                and p["first"] == pair["first"] and p["second"] == pair["second"])
                require({k: v for k, v in pair.items() if k != "effect"} == original, "Reused Progress/Zero-One outcome changed")
                baseline_regression += 1
        for (checkpoint, name), directory in old_dirs.items():
            if name in ("zero_one_budgeted_return", "ours_progress_return"):
                directories[group["gradient_cap"], group["attack_seed"], checkpoint, name] = directory
        for run in json.loads((ROOT / report["batch"]).read_text())["runs"]:
            directory = Path(run["run_dir"])
            for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
                directories[group["gradient_cap"], group["attack_seed"], entry["run_seed"], entry["attack"]["name"]] = directory / entry["results_directory"]
        reports.append(report)
        inputs.append(dict(gradient_cap=group["gradient_cap"], attack_seed=group["attack_seed"], path=path.relative_to(ROOT).as_posix(),
                           sha256=digest(path), control_reference=proof, raw_verified_steps=report["raw_episode_verified_steps"],
                           reused_return_verified_steps=report["reused_return_verified_steps"]))
    traffic, models = validate_reports(reports, protocol, pipeline["git_commit"])
    require(baseline_regression == 45, "Incomplete baseline outcome regression")
    first_batch = Path(pipeline["groups"][0]["batch"]).parent
    manifest_path = first_batch / "run-0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result = dict(kind="single_candidate_validation_results", verified=True, experiment_commit=pipeline["git_commit"],
                  baseline_experiment_commit=protocol["baseline_experiment_commit"],
                  summary_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip(),
                  pipeline=pipeline_path.relative_to(ROOT).as_posix(), pipeline_sha256=digest(pipeline_path),
                  inputs=inputs, traffic=traffic, checkpoint_sha256=models, new_episodes=1800, new_attack_episodes=900,
                  reused_attack_episodes=1800, comparison_episodes=3600, unique_traffic_episodes=20, unique_model_traffic_pairs=100,
                  raw_verified_steps=sum(r["raw_episode_verified_steps"] for r in reports),
                  reused_return_verified_steps=sum(r["reused_return_verified_steps"] for r in reports),
                  clean_regression_steps=sum(r["clean_regression"]["verified_steps"] for r in reports),
                  baseline_contrast_regression=dict(passed=True, model_cells=baseline_regression),
                  runtime=json.loads(manifest["python_runtime"]["stdout"]), host_os=manifest["host_os"],
                  sumo_version=manifest["sumo_version"]["stdout"].splitlines()[0],
                  launch_command=pipeline["command"], started_at_utc=pipeline["started_at_utc"], finished_at_utc=pipeline["finished_at_utc"],
                  gate_enabled=False, final_split_used=False, budgets=[], key_cases=[], limitation=protocol["limitations"])
    for cap in CAPS:
        selected = [r for r in reports if r["gradient_cap"] == cap]
        rows = [row for report in selected for row in report["rows"]]
        conditions = [dict(attack=name, **aggregate([r for r in rows if r["attack"] == name])) for name in COMPARISON_NAMES]
        by_name = {r["attack"]: r for r in conditions}
        budget = dict(gradient_cap=cap, forward_cap=2 * cap, conditions=conditions, contrasts=[], attack_seed_rows=[], checkpoint_rows=[])
        for seed in (0, 1, 2):
            report = next(r for r in selected if r["attack_seed"] == seed)
            budget["attack_seed_rows"].extend(dict(attack_seed=seed, attack=name, **aggregate([r for r in report["rows"] if r["attack"] == name])) for name in COMPARISON_NAMES)
        for checkpoint in range(5):
            budget["checkpoint_rows"].extend(dict(checkpoint_seed=checkpoint, attack=name,
                **aggregate([r for r in rows if r["attack"] == name and r["checkpoint_seed"] == checkpoint])) for name in COMPARISON_NAMES)
        for spec in protocol["contrasts"]:
            pairs = [dict(attack_seed=r["attack_seed"], **p) for r in selected for p in r["paired_contrasts"]
                     if (p["first"], p["second"]) == (spec["first"], spec["second"])]
            contrast, records = aggregate_contrast(pairs, expected_episodes=20)
            contrast.update(spec)
            contrast["cost_delta_totals"] = {k: by_name[spec["first"]]["costs"][k] - by_name[spec["second"]]["costs"][k] for k in COSTS}
            budget["contrasts"].append(contrast)
            units = defaultdict(list)
            for row in records:
                units[row["checkpoint_seed"], row["episode"]].append(row)
            ordered = sorted(units, key=lambda k: (sum(r["return_first_minus_second"] for r in units[k]) / 3, k))
            for role, key in (("smallest_return_delta", ordered[0]), ("largest_return_delta", ordered[-1])):
                representative = max(units[key], key=lambda r: (abs(r["return_first_minus_second"]), -r["attack_seed"]))
                paths = [directories[cap, representative["attack_seed"], key[0], name] / "steps.jsonl" for name in (spec["first"], spec["second"])]
                trajectories = []
                for path in paths:
                    with path.open() as stream:
                        trajectories.append([r for r in (json.loads(line) for line in stream) if r["episode"] == key[1]])
                result["key_cases"].append(dict(gradient_cap=cap, **spec, role=role, checkpoint_seed=key[0], episode=key[1],
                    sumo_seed=representative["sumo_seed"], attack_seed=representative["attack_seed"], representative_outcome=representative,
                    mean_return_delta_across_attack_seeds=sum(r["return_first_minus_second"] for r in units[key]) / 3,
                    raw_paths=[p.relative_to(ROOT).as_posix() for p in paths], first_divergence=transition_divergence(*trajectories)))
        result["budgets"].append(budget)
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit summary tools before publishing")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    result = summarize(args.pipeline.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figure:
        chart = dict(result, figure_title="Single candidate: reused validation traffic")
        plot(chart, args.figure, methods=(("zero_one_budgeted_return", "Budgeted Zero-One", "#2864ad", "o"),
                                         ("ours_single_return", "Single attempt", "#24865d", "s"),
                                         ("ours_progress_return", "Progress retries", "#d56a24", "D")),
             contrast_labels=("Single - Zero-One", "Single - Progress", "Progress - Zero-One"))
    print("SINGLE_VALIDATION_SUMMARY_PASSED new_episodes=%d new_steps=%d reused_steps=%d" %
          (result["new_episodes"], result["raw_verified_steps"], result["reused_return_verified_steps"]))
