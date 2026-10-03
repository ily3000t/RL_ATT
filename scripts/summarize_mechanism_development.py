"""Summarize the entire audited nested mechanism grid without treating seeds as traffic."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_proposed_smoke import ROOT, require
from prepare_mechanism_controls import CONTROL_NAMES

CAPS = (100, 200, 400)
COSTS = ("gradient_evaluations", "policy_forward_calls", "new_shadow_transitions", "shadow_steps",
         "replay_steps", "warmup_steps", "shadow_resets", "cache_hits", "ipc_requests")


def validate_grid(reports):
    keys = [(r["gradient_cap"], r["attack_seed"]) for r in reports]
    require(len(keys) == 9 and len(set(keys)) == 9 and set(keys) == {(c, s) for c in CAPS for s in (0, 1, 2)},
            "Incomplete or duplicate mechanism grid")


def win_tie_loss(values):
    return dict(lower=sum(v < -1e-9 for v in values), tied=sum(abs(v) <= 1e-9 for v in values),
                higher=sum(v > 1e-9 for v in values))


def aggregate(rows):
    episodes = [e for r in rows for e in r["episode_rows"]]
    require(episodes, "Empty mechanism outcome")
    eligible = sum(r["summary"].get("attack_success_eligible_episodes", 0) for r in rows)
    conversions = sum(r["summary"].get("attack_successes", 0) for r in rows)
    costs = {k: sum(r["audit"].get("costs", {}).get(k, 0) for r in rows) for k in COSTS}
    setup = sum(e.get("research_audit", {}).get("oracle_episode_setup_cost", {}).get("shadow_steps", 0) for e in episodes)
    result = dict(episodes=len(episodes), live_steps=sum(e["steps"] for e in episodes),
                  mean_return=sum(e["episode_return"] for e in episodes) / len(episodes),
                  collisions=sum(r["collisions"] for r in rows),
                  eligible_clean_noncollision=eligible if eligible else None,
                  collision_conversions=conversions if eligible else None, conversion_rate=conversions / eligible if eligible else None,
                  costs=costs, costs_per_episode={k: v / len(episodes) for k, v in costs.items()},
                  oracle_setup_shadow_steps=setup, physical_shadow_steps_with_setup=costs["shadow_steps"] + setup,
                  attempts=sum(r.get("attempt_audit", {}).get("attempts", 0) for r in rows),
                  retries=sum(r.get("attempt_audit", {}).get("retries", 0) for r in rows),
                  suppressed_retry_targets=sum(r.get("verified_suppressed_retry_targets", 0) for r in rows))
    for key in ("complete_search_candidates", "unique_actual_sequences", "duplicate_actual_sequences", "unique_actual_prefixes",
                "failed_target_attempts", "budget_exhausted_blocks", "unplanned_fallback_steps", "live_unverified_fallback_steps"):
        result[key] = sum(r["audit"].get(key, 0) for r in rows)
    result["retry_yield"] = {k: sum(r.get("retry_yield", {}).get(k, 0) for r in rows)
                             for k in ("retry_target_hits", "retry_new_branches", "retry_new_branches_on_selected_plan", "retries_without_new_branch")}
    return result


def aggregate_contrast(pairs):
    records, seen = [], set()
    for pair in pairs:
        for episode in pair["episodes"]:
            key = pair["attack_seed"], pair["checkpoint_seed"], episode["episode"]
            require(key not in seen, "Duplicate correlated pair")
            seen.add(key)
            records.append(dict(attack_seed=pair["attack_seed"], checkpoint_seed=pair["checkpoint_seed"], **episode))
    require(seen == {(a, c, e) for a in (0, 1, 2) for c in range(5) for e in range(1, 11)}, "Missing mechanism paired unit")
    by_model, by_traffic = defaultdict(list), defaultdict(list)
    for r in records:
        by_model[r["checkpoint_seed"]].append(r)
        by_traffic[r["episode"]].append(r)
    total_return = sum(r["return_first_minus_second"] for r in records)
    def describe(group):
        eligible = [r for r in group if not r["clean_collision"]]
        gain = sum(r["first_collision"] and not r["second_collision"] for r in eligible)
        loss = sum(r["second_collision"] and not r["first_collision"] for r in eligible)
        return dict(paired_records=len(group), mean_return_delta=sum(r["return_first_minus_second"] for r in group) / len(group),
                    eligible=len(eligible), first_only_conversion=gain, second_only_conversion=loss,
                    net_conversions=gain - loss,
                    gradient_delta=sum(r["gradient_first_minus_second"] for r in group),
                    forward_delta=sum(r["forward_first_minus_second"] for r in group))
    models = [dict(checkpoint_seed=c, **describe(by_model[c])) for c in range(5)]
    traffic = []
    total_net = sum(r["net_conversions"] for r in models)
    for e in range(1, 11):
        group = by_traffic[e]
        require(len({r["sumo_seed"] for r in group}) == 1, "Unpaired traffic cluster")
        item = dict(episode=e, sumo_seed=group[0]["sumo_seed"], **describe(group))
        item.update(mean_return_delta_without_traffic=(total_return - sum(r["return_first_minus_second"] for r in group)) / (len(records) - len(group)),
                    net_conversions_without_traffic=total_net - item["net_conversions"])
        traffic.append(item)
    result = describe(records)
    result.update(per_model=models, per_traffic=traffic,
                  per_model_return_win_tie_loss=win_tie_loss([r["mean_return_delta"] for r in models]),
                  per_traffic_return_win_tie_loss=win_tie_loss([r["mean_return_delta"] for r in traffic]),
                  correlated_pair_return_win_tie_loss=win_tie_loss([r["return_first_minus_second"] for r in records]),
                  identical_trajectories=sum(p["identical_trajectories"] for p in pairs),
                  leave_one_traffic_out_return_delta_range=[min(r["mean_return_delta_without_traffic"] for r in traffic),
                                                          max(r["mean_return_delta_without_traffic"] for r in traffic)],
                  leave_one_traffic_out_net_conversion_range=[min(r["net_conversions_without_traffic"] for r in traffic),
                                                             max(r["net_conversions_without_traffic"] for r in traffic)])
    return result, records


def transition_divergence(first, second):
    for index, (a, b) in enumerate(zip(first, second)):
        if any(a[k] != b[k] for k in ("action", "reward", "terminated", "next_observation")):
            require(a["observation"] == b["observation"], "Different state before first divergence")
            details = dict(step_zero_based=index, first_action=a["action"], second_action=b["action"], same_observation=True)
            for label, rows in (("first", first), ("second", second)):
                block = max(i for i in range(index + 1) if rows[i]["attack_metadata"]["planned"])
                m = rows[block]["attack_metadata"]
                history = [r["action"] for r in rows[:index]]
                details[label] = dict(block_start=block, selected_candidate=m["selected_candidate"],
                                      retries_in_block=m["retry_attempts"],
                                      used_witness_seed=rows[index]["attack_metadata"]["witness_seed"],
                                      attempts_at_divergence=[{k: v for k, v in t.items() if k != "history"}
                                                              for t in m["inner_attempt_trace"] if t["history"] == history])
            return details
    require(len(first) == len(second), "Unequal trajectory lengths without termination divergence")
    return None


def summarize(pipeline_path):
    cache = {}
    def digest(path):
        path = path.resolve()
        if path not in cache:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            cache[path] = h.hexdigest()
        return cache[path]
    protocol_path = ROOT / "configs/research/mechanism_development.json"
    protocol = json.loads(protocol_path.read_text())
    pipeline = json.loads(pipeline_path.read_text())
    require(pipeline["status"] == "passed" and pipeline["verified_episodes"] == protocol["total_episodes"] == 2250, "Incomplete mechanism pipeline")
    require(digest(protocol_path) == pipeline["protocol_sha256"], "Registered mechanism protocol changed")
    require([(g["gradient_cap"], g["attack_seed"]) for g in pipeline["groups"]] ==
            [(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]], "Mechanism group order or coverage changed")
    reports, inputs, directories = [], [], {}
    traffic, models = None, {}
    for group, registered in zip(pipeline["groups"], protocol["groups"]):
        path = Path(group["audit_path"])
        require(group["status"] == "passed" and group["configs"] == registered["configs"] and digest(path) == group["audit_sha256"], "Unverified group")
        report = json.loads(path.read_text())
        require(report["verified"] and report["kind"] == "return_mechanism_development" and report["git_commit"] == pipeline["git_commit"], "Mixed experiment provenance")
        require((report["gradient_cap"], report["attack_seed"]) == (group["gradient_cap"], group["attack_seed"]), "Mislabeled cell")
        for name, sha in report["source_sha256"].items():
            require(digest(ROOT / name) == sha, "Audited source changed: " + name)
        require(report["clean_regression"]["passed"], "Clean regression not verified")
        if (group["gradient_cap"], group["attack_seed"]) == (400, 0):
            require(report["smoke_regression"]["passed"], "Five-condition prefix regression missing")
        traffic = report["traffic"] if traffic is None else traffic
        require(report["traffic"] == traffic, "Cross-cell traffic changed")
        require(len(report["rows"]) == 25 and {(r["checkpoint_seed"], r["attack"]) for r in report["rows"]} ==
                {(c, n) for c in range(5) for n in CONTROL_NAMES}, "Missing or duplicate model/condition")
        for row in report["rows"]:
            models.setdefault(row["checkpoint_seed"], row["checkpoint_sha256"])
            require(row["checkpoint_sha256"] == models[row["checkpoint_seed"]] and row["episodes"] == len(row["episode_rows"]) == 10,
                    "Frozen weights or episode series changed")
        batch = json.loads((ROOT / report["batch"]).read_text())
        for run in batch["runs"]:
            directory = Path(run["run_dir"])
            for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
                directories[group["gradient_cap"], group["attack_seed"], entry["run_seed"], entry["attack"]["name"]] = directory / entry["results_directory"]
        reports.append(report)
        inputs.append(dict(gradient_cap=group["gradient_cap"], attack_seed=group["attack_seed"],
                           path=path.relative_to(ROOT).as_posix(), sha256=digest(path), raw_verified_steps=report["raw_episode_verified_steps"]))
    validate_grid(reports)
    result = dict(kind="return_mechanism_development_summary", verified=True,
                  experiment_commit=pipeline["git_commit"], summary_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip(),
                  pipeline=pipeline_path.relative_to(ROOT).as_posix(), pipeline_sha256=digest(pipeline_path),
                  inputs=inputs, traffic=traffic, checkpoint_sha256=models, episodes=2250,
                  raw_verified_steps=sum(r["raw_episode_verified_steps"] for r in reports),
                  unique_traffic_episodes=10, unique_model_traffic_pairs=50, budgets=[], key_cases=[],
                  limitation="Nested conditional contrasts on reused development traffic; three attacks per traffic are correlated. No significance, full factorial, physical plausibility or final-test claim. Wall times are not a speed benchmark.")
    for cap in CAPS:
        selected = [r for r in reports if r["gradient_cap"] == cap]
        rows = [r for report in selected for r in report["rows"]]
        conditions = [dict(attack=n, **aggregate([r for r in rows if r["attack"] == n])) for n in CONTROL_NAMES]
        by_name = {r["attack"]: r for r in conditions}
        budget = dict(gradient_cap=cap, forward_cap=2 * cap, conditions=conditions, contrasts=[],
                      attack_seed_rows=[], checkpoint_rows=[])
        for seed in (0, 1, 2):
            report = next(r for r in selected if r["attack_seed"] == seed)
            budget["attack_seed_rows"].extend(dict(attack_seed=seed, attack=n, **aggregate([r for r in report["rows"] if r["attack"] == n])) for n in CONTROL_NAMES)
        for checkpoint in range(5):
            budget["checkpoint_rows"].extend(dict(checkpoint_seed=checkpoint, attack=n,
                                                 **aggregate([r for r in rows if r["attack"] == n and r["checkpoint_seed"] == checkpoint])) for n in CONTROL_NAMES)
        for specification in protocol["contrasts"]:
            first, second = specification["first"], specification["second"]
            pairs = [dict(attack_seed=r["attack_seed"], **p) for r in selected for p in r["paired_contrasts"] if (p["first"], p["second"]) == (first, second)]
            contrast, records = aggregate_contrast(pairs)
            contrast.update(specification)
            contrast["cost_delta_totals"] = {k: by_name[first]["costs"][k] - by_name[second]["costs"][k] for k in COSTS}
            budget["contrasts"].append(contrast)
            units = defaultdict(list)
            for row in records:
                units[row["checkpoint_seed"], row["episode"]].append(row)
            ordered = sorted(units, key=lambda key: (sum(r["return_first_minus_second"] for r in units[key]) / 3, key))
            for role, key in (("smallest_return_delta", ordered[0]), ("largest_return_delta", ordered[-1])):
                representative = max(units[key], key=lambda r: (abs(r["return_first_minus_second"]), -r["attack_seed"]))
                paths = [directories[cap, representative["attack_seed"], key[0], name] / "steps.jsonl" for name in (first, second)]
                episodes = []
                for path in paths:
                    with path.open() as stream:
                        episode = [row for row in (json.loads(line) for line in stream) if row["episode"] == key[1]]
                    episodes.append(episode)
                result["key_cases"].append(dict(gradient_cap=cap, first=first, second=second, role=role,
                                                checkpoint_seed=key[0], episode=key[1], sumo_seed=representative["sumo_seed"],
                                                mean_return_delta_across_attack_seeds=sum(r["return_first_minus_second"] for r in units[key]) / 3,
                                                attack_seed=representative["attack_seed"], representative_outcome=representative,
                                                raw_paths=[p.relative_to(ROOT).as_posix() for p in paths],
                                                first_divergence=transition_divergence(*episodes)))
        result["budgets"].append(budget)
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit summary code before publishing")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.pipeline.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("MECHANISM_SUMMARY_PASSED episodes=%d raw_steps=%d" % (report["episodes"], report["raw_verified_steps"]))
