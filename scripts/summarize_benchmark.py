"""Validate all five paired trials and export run-level benchmark statistics."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from run_baseline import ROOT, git


METHODS = ("none", "random", "fgsm", "pgd", "oarl_bo")
COSTS = {"none": (0, 0, 0), "random": (0, 0, 0), "fgsm": (2, 2, 1),
         "pgd": (11, 12, 10), "oarl_bo": (5, 11, 0)}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return {"mean": statistics.mean(finite) if finite else None,
            "sample_sd": statistics.stdev(finite) if len(finite) > 1 else None,
            "valid_runs": len(finite)}


def verify_steps(directory, name, episodes):
    counts = [0] * 20
    with (directory / "steps.jsonl").open(encoding="utf-8") as raw:
        for line in raw:
            row = json.loads(line)
            index = row["episode"] - 1
            if not 0 <= index < 20 or row["step"] != counts[index]:
                raise ValueError("Missing, duplicated, or misordered step records")
            counts[index] += 1
            obs, adv, delta = row["observation"], row["adversarial_observation"], row["perturbation"]
            if any(len(x) != 16 or not all(math.isfinite(v) for v in x) for x in (obs, adv, delta)):
                raise ValueError("Invalid raw observation or perturbation")
            if any(abs(a - o - d) > 1e-7 for o, a, d in zip(obs, adv, delta)):
                raise ValueError("Perturbation and returned observation disagree")
            if any(abs(d) > 0.2 * abs(o) + 0.05 + 2e-7 for o, d in zip(obs, delta)):
                raise ValueError("Raw perturbation exceeds the declared common envelope")
            if row["attacked"] != (name != "none") or (name == "none" and any(delta)):
                raise ValueError("Attack frequency or clean control changed")
            cost = row["attack_cost"]
            if (cost["objective_evaluations"], cost["policy_forward_calls"], cost.get("gradient_evaluations", 0)) != COSTS[name]:
                raise ValueError("Attack cost differs from the predeclared steps")
            if name == "oarl_bo":
                trace = row["attack_metadata"]["trace"]
                if len(trace) != 5 or row["attack_metadata"]["objective_value"] != max(p["objective"] for p in trace):
                    raise ValueError("Invalid BO trace/winner")
                for point in trace:
                    u1, u2 = point["parameters"]["u1"], point["parameters"]["u2"]
                    if not 0.8 <= u1 <= 1.2 or not -0.05 <= u2 <= 0.05:
                        raise ValueError("BO search bounds changed")
    if counts != [r["steps"] for r in episodes]:
        raise ValueError("Raw step count differs from episode records")
    return sum(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = read(args.batch / "batch.json")
    if batch["status"] != "passed" or len(batch["runs"]) != 5:
        raise ValueError("Require a completed five-run batch")
    references = {(r["victim"], r["run_seed"]): r for r in read(ROOT / "configs/frozen_victims.json")["victims"]}
    rows, seeds, shared_config, raw_steps = [], set(), None, 0
    for child in batch["runs"]:
        run_dir = Path(child["run_dir"])
        manifest, evaluation = read(run_dir / "manifest.json"), read(run_dir / "evaluation.json")
        if child["returncode"] != 0 or manifest["status"] != "passed" or manifest["git_commit"] != batch["git_commit"] or evaluation["git_commit"] != batch["git_commit"]:
            raise ValueError("Failed or mixed-commit benchmark trial")
        if any(p != "Data/StraightRoad.sumocfg" for p in manifest["changed_source_files"]):
            raise ValueError("Unexpected runtime source modification")
        config = copy.deepcopy(manifest["config"])
        if config["victims"] != ["clean"] or len(config["run_seeds"]) != 1 or (config["episodes"], config["max_steps"]) != (20, 200):
            raise ValueError("Benchmark victim or horizon mismatch")
        seed = config.pop("run_seeds")[0]
        if seed in seeds or seed not in range(5):
            raise ValueError("Duplicate or unexpected run seed")
        seeds.add(seed)
        if shared_config is None:
            shared_config = config
        if config != shared_config:
            raise ValueError("Configs differ beyond run_seed")
        expected_budget = {"norm": "observation_scaled_linf", "epsilon": 1.0, "relative_scale": 0.2, "absolute_scale": 0.05}
        if any(a["budget"] != expected_budget or a["parameters"]["every_n_steps"] != 1 for a in config["attacks"] if a["name"] != "none"):
            raise ValueError("Methods do not share the predeclared envelope/frequency")
        if [r["attack"]["name"] for r in evaluation["runs"]] != list(METHODS):
            raise ValueError("Missing or reordered benchmark conditions")
        reference = references[("clean", seed)]
        if hashlib.sha256((ROOT / reference["checkpoint"]).read_bytes()).hexdigest() != reference["checkpoint_sha256"]:
            raise ValueError("Frozen checkpoint file changed")
        baseline_entry = evaluation["runs"][0]
        if not baseline_entry["legacy_equivalence"]["passed"]:
            raise ValueError("NoAttack legacy equivalence missing")
        baseline = read(run_dir / baseline_entry["results_directory"] / "episodes.json")
        for entry in evaluation["runs"]:
            name = entry["attack"]["name"]
            if entry["run_seed"] != seed or entry["victim"] != "clean" or not entry["frozen_unchanged"]:
                raise ValueError("Victim changed across methods")
            if entry["checkpoint_sha256"] != reference["checkpoint_sha256"] or entry["weights_sha256"] != reference["weights_sha256"]:
                raise ValueError("Frozen model hashes differ between attacks")
            if entry["effective_seeds"]["episode_sumo_seeds"] != baseline_entry["effective_seeds"]["episode_sumo_seeds"]:
                raise ValueError("Evaluation traffic seeds differ between attacks")
            directory = run_dir / entry["results_directory"]
            episodes = read(directory / "episodes.json")
            if len(episodes) != 20 or any((r["episode"], r["sumo_seed"]) != (b["episode"], b["sumo_seed"])
                                         for r, b in zip(episodes, baseline)):
                raise ValueError("Episode pairing mismatch")
            raw_steps += verify_steps(directory, name, episodes)
            summary = copy.deepcopy(entry["summary"])
            drops = [b["episode_return"] - r["episode_return"] for r, b in zip(episodes, baseline)]
            eligible = [(r, b) for r, b in zip(episodes, baseline) if not b["ego_collision_observed"]]
            successes = sum(r["ego_collision_observed"] for r, _ in eligible)
            if name == "none":
                # The clean control paired with itself has known zero perturbation/drop.
                summary.update(return_drop_mean=0.0, attack_success_rate=0.0 if eligible else None,
                               attack_successes=0, attack_success_eligible_episodes=len(eligible), scaled_linf_max=0.0)
            if abs(summary["return_drop_mean"] - statistics.mean(drops)) > 1e-9 or summary["attack_successes"] != successes:
                raise ValueError("Paired outcome summary disagrees with episode records")
            steps = sum(r["steps"] for r in episodes)
            if (summary["objective_evaluations"], summary["attack_policy_forward_calls"], summary["gradient_evaluations"]) != tuple(c * steps for c in COSTS[name]):
                raise ValueError("Summary query/gradient accounting mismatch")
            rows.append({"run_seed": seed, "attack": name, "checkpoint_sha256": reference["checkpoint_sha256"],
                         "result_directory": directory.resolve().relative_to(ROOT).as_posix(),
                         "summary": summary, "return_drops": drops})
    metrics = ["episode_return_mean", "return_drop_mean", "collision_rate", "attack_success_rate", "attack_rate",
               "action_change_rate", "linf_max", "l2_max", "scaled_linf_max"]
    safety_metrics = ["minimum_ttc_s", "ttc_low_percentile_s", "drac_high_percentile_mps2"]
    aggregate = {}
    for name in METHODS:
        method_rows = [r for r in rows if r["attack"] == name]
        aggregate[name] = {k: stats([r["summary"][k] for r in method_rows]) for k in metrics}
        aggregate[name].update({k: stats([r["summary"]["safety"][k] for r in method_rows]) for k in safety_metrics})
        aggregate[name]["totals"] = {k: sum(r["summary"][k] for r in method_rows) for k in
                                     ("episodes", "steps", "objective_evaluations", "attack_policy_forward_calls",
                                      "gradient_evaluations", "attack_wall_seconds", "attack_successes", "attack_success_eligible_episodes")}
    result = {"stage": 3, "batch": args.batch.resolve().relative_to(ROOT).as_posix(),
              "benchmark_commit": batch["git_commit"], "summary_commit": git("rev-parse", "HEAD"),
              "summary_command": [sys.executable, *sys.argv], "config": shared_config,
              "all_5_seeds_complete": True, "all_5_no_attack_legacy_matches": True,
              "all_frozen_hashes_verified": True, "all_raw_step_budgets_verified": True,
              "raw_steps_checked": raw_steps, "episodes": 500,
              "statistics": "Mean and sample SD over five run-level values; conditional safety percentiles retain valid sample counts",
              "collision_scope": "Original SUMO reported events, including possible minGap violations; not verified physical contact",
              "aggregate": aggregate, "runs": sorted(rows, key=lambda r: (r["run_seed"], METHODS.index(r["attack"])))}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("BENCHMARK_SUMMARY=" + str(args.output))


if __name__ == "__main__":
    main()
