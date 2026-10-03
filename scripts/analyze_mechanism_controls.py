"""Audit the registered Return mechanism smoke and unchanged historical controls."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
from summarize_proposed_smoke import ROOT, summarize, require
from analyze_proposed_development import audit_episode_records, pair_outcomes, retry_yield
from analyze_progress_retry import audit_stops, verify_v0_reference
from prepare_mechanism_controls import CONTROL_NAMES, make_config, sha256
from rl_att.attacks.behavior_search import witness_seed
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.evaluation.results import summarize_episodes

PROTOCOL = ROOT / "configs/research/mechanism_controls.json"


def expected_seeds(config, checkpoint, name):
    seeds = evaluation_seeds(checkpoint, config["episodes"], research_seeds=config["research_seeds"])
    seeds.update(policy_rng="unused_greedy_argmax", attack_rng="unused_no_attack" if name == "none"
                 else "sha256_attack_episode_full_history_target_attempt_local_numpy")
    return seeds


def audit_attempts(steps, name, attack_seed, parameters):
    witnesses, retries, attempts_count = {}, 0, 0
    single = name in ("ours_single_return", "zero_one_budgeted_return")
    for row in steps:
        m = row["attack_metadata"]
        require(m["name"] == name and m["objective"] == "return" and not m["gate_enabled"], "Raw mechanism label changed")
        if not m["planned"]:
            continue
        require(m["search_kind"] == ("target_sequence" if name == "zero_one_budgeted_return" else "behavior"), "Wrong outer search")
        groups = defaultdict(list)
        trace = m["inner_attempt_trace"]
        for a in trace:
            index = a["attempt"]
            require(type(index) is int and 0 <= index < parameters["max_attempts"], "Attempt cap violated")
            require(not single or index == 0, "Single-attempt condition retried")
            require(a["seed"] == witness_seed(attack_seed, row["episode"] - 1, a["history"], a["target"], index), "Witness seed differs")
            require(a["succeeded"] == (a["target"] == a["actual"]), "Target hit mislabeled")
            key = row["episode"], tuple(a["history"]), a["target"], index
            value = a["seed"], a["actual"], a["margin"]
            require(key not in witnesses or witnesses[key] == value, "Same seeded primitive changed within condition")
            witnesses[key] = value
            groups[tuple(a["history"]), a["target"]].append(index)
        for indices in groups.values():
            require(indices == list(range(len(indices))), "Nonconsecutive or duplicate PGD attempts")
        count = sum(a["attempt"] > 0 for a in trace)
        require(m["retry_attempts"] == count and m["failed_target_attempts"] == sum(not a["succeeded"] for a in trace), "Attempt counters differ")
        require(row["attack_cost"]["gradient_evaluations"] == len(trace) * parameters["inner_steps"], "Gradient count differs from attempted primitives")
        retries += count
        attempts_count += len(trace)
    return witnesses, dict(attempts=attempts_count, retries=retries)


def verify_shared_witnesses(witnesses):
    comparisons = []
    for first, second in itertools.combinations(CONTROL_NAMES[1:], 2):
        a, b = witnesses[first], witnesses[second]
        common = {k for k in set(a) & set(b) if k[-1] == 0}
        require(all(a[k] == b[k] for k in common), "Shared first-attempt primitive differs between controls")
        comparisons.append(dict(first=first, second=second, shared_first_attempts=len(common)))
    return comparisons


def analyze(batch_path):
    sources = {}
    def read(path):
        sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))
    protocol = read(PROTOCOL)
    require(sha256(ROOT / "configs/research_seed_splits.json") == protocol["seed_splits_sha256"], "Seed split changed")
    require(sha256(ROOT / "configs/frozen_victims.json") == protocol["frozen_victims_sha256"], "Frozen registry changed")
    references = {}
    for label, proof in protocol["references"].items():
        path = ROOT / proof["path"]
        require(sha256(path) == proof["sha256"], "Historical batch changed")
        reference_batch = read(path)
        require(reference_batch["git_commit"] == proof["git_commit"], "Historical commit changed")
        references[label] = {}
        for item in proof["manifests"]:
            path = ROOT / item["path"]
            require(sha256(path) == item["sha256"], "Historical manifest changed")
            manifest = read(path)
            references[label][manifest["config"]["run_seeds"][0]] = manifest
    batch = read(batch_path)
    require(batch["status"] == "passed" and batch["configs"] == protocol["smoke"]["configs"] and len(batch["runs"]) == 5,
            "Unregistered or incomplete mechanism smoke")
    params = {a["name"]: a["parameters"] for a in make_config(0)["attacks"]}
    report = summarize(batch_path, expected_names=set(CONTROL_NAMES), expected_parameters=params)
    report.update(kind="return_mechanism_engineering_smoke", raw_episode_verified_steps=0,
                  shared_first_attempt_comparisons=[], paired_contrasts=[])
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest, evaluation = read(directory / "manifest.json"), read(directory / "evaluation.json")
        seed = manifest["config"]["run_seeds"][0]
        require(run["returncode"] == 0 and manifest["config"] == make_config(seed), "Wrong checkpoint or smoke configuration")
        require(evaluation["config"] == manifest["config"] and not evaluation["gate_enabled"], "Evaluation protocol changed")
        require(manifest["source_sha256_before"] == protocol["frozen_source_sha256"], "Frozen mechanism source changed")
        require(all(p == "Data/StraightRoad.sumocfg" for p in manifest["changed_source_files"]), "Unexpected runtime source mutation")
        reference = references["original"][seed]
        for field in ("python_runtime", "pip_freeze", "sumo_version", "victim_references", "extra_dependencies"):
            require(manifest[field] == reference[field], "Historical runtime or checkpoint differs: " + field)
        require([e["attack"]["name"] for e in evaluation["runs"]] == list(CONTROL_NAMES), "Wrong mechanism order")
        witnesses, clean = {}, None
        for entry in evaluation["runs"]:
            name = entry["attack"]["name"]
            require(entry["attack"] == next(a for a in manifest["config"]["attacks"] if a["name"] == name), "Declared attack changed")
            require(entry["effective_seeds"] == expected_seeds(manifest["config"], seed, name), "Effective mechanism seeds differ")
            path = directory / entry["results_directory"] / "steps.jsonl"
            sources[path.relative_to(ROOT).as_posix()] = sha256(path)
            steps = [json.loads(line) for line in path.read_text().splitlines()]
            row = by_key[seed, name]
            report["raw_episode_verified_steps"] += audit_episode_records(steps, row["episode_rows"], 200)
            for step in steps:
                obs, adv, delta = [np.asarray(step[k], dtype=np.float64) for k in ("observation", "adversarial_observation", "perturbation")]
                require(all(x.shape == (16,) and np.isfinite(x).all() for x in (obs, adv, delta)), "Invalid raw observation")
                require(np.allclose(adv - obs, delta, rtol=0, atol=1e-7), "Raw perturbation inconsistent")
                require(np.all(np.abs(delta) <= .2 * np.abs(obs) + .05 + 2e-7), "Raw perturbation escaped envelope")
            if name == "none":
                clean = row["episode_rows"]
                require(all(not s["attacked"] and not any(s["perturbation"]) for s in steps), "Clean condition perturbed")
            else:
                witnesses[name], row["attempt_audit"] = audit_attempts(steps, name, 0, params[name])
                if name.startswith("ours_"):
                    row["retry_yield"] = retry_yield(steps)
                if name == "ours_progress_return":
                    row["verified_suppressed_retry_targets"] = audit_stops(steps)
            require(summarize_episodes(row["episode_rows"], clean if name != "none" else None) ==
                    {k: v for k, v in row["summary"].items() if k != "safety"}, "Episode aggregate differs")
        report["shared_first_attempt_comparisons"].append(dict(checkpoint_seed=seed, comparisons=verify_shared_witnesses(witnesses)))
        for contrast in protocol["contrasts"]:
            a, b = contrast["first"], contrast["second"]
            report["paired_contrasts"].append(dict(checkpoint_seed=seed, **contrast,
                                                 **pair_outcomes(clean, by_key[seed, a]["episode_rows"], by_key[seed, b]["episode_rows"])))
    report["historical_regression"] = dict(
        original=verify_v0_reference(batch_path, ROOT / protocol["references"]["original"]["path"],
                                     names=("none", "zero_one_budgeted_return", "ours_return")),
        progress=verify_v0_reference(batch_path, ROOT / protocol["references"]["progress"]["path"],
                                     names=("ours_progress_return",)))
    report["source_sha256"].update(sources)
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analyzer before publishing report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.batch.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("MECHANISM_AUDIT_PASSED=" + report["git_commit"])
    print("RAW_VERIFIED_STEPS=" + str(report["raw_episode_verified_steps"]))
