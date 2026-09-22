"""Audit committed Ours-v0 smoke runs without interpreting them as efficacy tests."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_att.attacks.rollout_objectives import rollout_key

RESOURCES = ("gradient_evaluations", "policy_forward_calls", "new_shadow_transitions", "shadow_steps")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_steps(steps, parameters):
    selected, position, used, block = None, 0, None, None
    totals = dict.fromkeys(RESOURCES + ("replay_steps", "warmup_steps", "shadow_resets", "cache_hits", "ipc_requests"), 0)
    stats = dict(blocks=0, complete_search_candidates=0, unique_actual_sequences=0,
                 duplicate_actual_sequences=0, unique_actual_prefixes=0, failed_target_attempts=0,
                 retry_attempts=0, budget_exhausted_blocks=0, unplanned_fallback_steps=0)
    for row in steps:
        m, cost = row["attack_metadata"], row["attack_cost"]
        if m["planned"]:
            stats["blocks"] += 1
            block = row["episode"], row["step"]
            used = dict.fromkeys(RESOURCES, 0)
            position = 0
            traces = m["candidate_trace"]
            require(len(traces) <= parameters["evaluations"] + 1, "Too many complete rollouts")
            for t in traces:
                require(len(t["actions"]) == len(t["rewards"]) == len(t["collisions"]), "Trace lengths differ")
                require(1 <= len(t["actions"]) <= m["horizon"], "Invalid complete rollout length")
                require(t["terminated"] or len(t["actions"]) == m["horizon"], "Partial rollout scored as complete")
                require(t["horizon_truncated"] == (not t["terminated"]), "Termination/truncation conflated")
                require(list(rollout_key(t["rewards"], t["collisions"], parameters["objective"])) == t["score"], "Objective mismatch")
            if m["selected_candidate"] is None:
                require(not traces and m["oracle_unplanned_fallback"], "Incomplete fallback selected as complete")
                selected = None
            else:
                require(m["selected_candidate"] == min(range(len(traces)), key=lambda i: traces[i]["score"]), "Did not select best complete rollout")
                selected = traces[m["selected_candidate"]]
                require(selected["value"] == m["selected_return"], "Selected return mismatch")
            for key in ("complete_search_candidates", "unique_actual_sequences", "duplicate_actual_sequences",
                        "unique_actual_prefixes", "failed_target_attempts", "retry_attempts"):
                stats[key] += m[key]
            stats["budget_exhausted_blocks"] += int(m["budget_exhausted"])
            expected_limits = dict(parameters["resource_limits"])
            for key in ("gradient_evaluations", "new_shadow_transitions"):
                expected_limits[key] = expected_limits[key] * m["horizon"] // parameters["horizon"]
            require(m["budget"]["limits"] == expected_limits, "Block resource limits differ from config")
        require(block is not None and block[0] == row["episode"], "Plan crossed episode boundary")
        for key in RESOURCES:
            charge = cost.get(key, 0)
            require(type(charge) is int and charge >= 0, "Invalid resource charge")
            used[key] += charge
            require(used[key] == m["budget"]["used"][key], "Ledger differs from per-step costs")
            require(used[key] + m["budget"]["reserved"][key] <= m["budget"]["limits"][key], "Resource cap exceeded")
        require(cost.get("shadow_steps", 0) == cost.get("new_shadow_transitions", 0) + cost.get("replay_steps", 0) + cost.get("warmup_steps", 0), "Physical simulator cost not reconciled")
        require(row["action"] == m["planned_action"], "Executed action differs from witness")
        require(row["attacked"] == any(d != 0 for d in row["perturbation"]), "Clean witness mislabeled as applied attack")
        require(row["scaled_linf"] <= 1.0000001 * parameters["epsilon"], "Perturbation escaped box")
        if selected is not None:
            require(position < len(selected["actions"]), "Execution exceeded selected plan")
            require(row["action"] == selected["actions"][position] and row["reward"] == selected["rewards"][position], "Plan execution reward/action mismatch")
            require(row["safety"]["ego_collision_observed"] == selected["collisions"][position], "Collision trace mismatch")
            require(row["terminated"] == (position == len(selected["actions"]) - 1 and selected["terminated"]), "Plan termination mismatch")
        else:
            stats["unplanned_fallback_steps"] += 1
            require(not row["attacked"] and row["action"] == row["clean_action_at_visited_state"], "Fallback is not clean")
        position += 1
        for key in totals:
            totals[key] += cost.get(key, 0)
    return dict(costs=totals, **stats)


def summarize(batch_path, expected_episodes=2, expected_names=None):
    require(expected_episodes in (2, 10), "Only frozen smoke/development protocols are supported")
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    require(batch["status"] == "passed", "Batch did not pass")
    rows, sources, checkpoint_seeds, common_traffic = [], {}, set(), None
    custom_methods = expected_names is not None
    if expected_names is None:
        expected_names = {"none", "zero_one_budgeted_return", "zero_one_budgeted_safety", "ours_return", "ours_safety"}
    for run in sorted(batch["runs"], key=lambda r: r["config"]):
        directory = Path(run["run_dir"])
        def read(path):
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            return json.loads(path.read_text(encoding="utf-8"))
        manifest, evaluation = read(directory / "manifest.json"), read(directory / "evaluation.json")
        require(manifest["status"] == "passed" and manifest["git_commit"] == batch["git_commit"], "Run provenance mismatch")
        require(evaluation["git_commit"] == batch["git_commit"], "Evaluation commit mismatch")
        config = manifest["config"]
        require(config["episodes"] == expected_episodes and config["max_steps"] == 200 and config["research_seeds"]["split_id"] == 10,
                "This report is restricted to the requested development protocol")
        require(set(e["attack"]["name"] for e in evaluation["runs"]) == expected_names, "Incomplete 2x2 controls")
        comparison = None
        clean_steps = None
        for entry in evaluation["runs"]:
            require(entry["frozen_unchanged"], "Frozen victim changed")
            seeds = entry["effective_seeds"]
            traffic = {k: seeds[k] for k in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed", "episode_sumo_seeds")}
            if common_traffic is None:
                common_traffic = traffic
            require(common_traffic == traffic and seeds["attack_seed"] == 0, "Unpaired traffic or attack replicate")
            checkpoint_seeds.add(entry["run_seed"])
            name = entry["attack"]["name"]
            result_dir = directory / entry["results_directory"]
            episodes = read(result_dir / "episodes.json")
            require(len(episodes) == expected_episodes and [e["episode"] for e in episodes] == list(range(1, expected_episodes + 1)),
                    "Missing, repeated or misordered episode records")
            require([e["sumo_seed"] for e in episodes] == traffic["episode_sumo_seeds"], "Episode traffic mismatch")
            raw = result_dir / "steps.jsonl"
            sources[str(raw.relative_to(ROOT))] = hashlib.sha256(raw.read_bytes()).hexdigest()
            steps = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
            require(len(steps) == sum(e["steps"] for e in episodes), "Missing episode steps")
            audit = {}
            if name == "none":
                clean_steps = {(s["episode"], s["step"]): s for s in steps}
            else:
                p = entry["attack"]["parameters"]
                comparable = {k: v for k, v in p.items() if k not in ("objective", "retry_rule")}
                if comparison is None:
                    comparison = comparable
                require(comparable == comparison, "Controls have different search/perturbation budgets")
                audit = audit_steps(steps, p)
                oracle = read(directory / (entry["results_directory"] + "-oracle") / "manifest.json")
                require(oracle["status"] == "passed" and oracle["account_reset_warmup"], "Oracle not verified with warmup accounting")
                require(oracle["counts"]["live_verified_steps"] + oracle["counts"].get("live_unverified_fallback_steps", 0) == len(steps), "Live verification count mismatch")
                require(oracle["counts"].get("live_unverified_fallback_steps", 0) <= audit["unplanned_fallback_steps"], "Fallback verification mismatch")
                audit["live_unverified_fallback_steps"] = oracle["counts"].get("live_unverified_fallback_steps", 0)
                for key in ("shadow_steps", "replay_steps", "warmup_steps", "shadow_resets", "cache_hits"):
                    setup = sum(e["research_audit"]["oracle_episode_setup_cost"].get(key, 0) for e in episodes)
                    require(oracle["counts"].get(key, 0) == audit["costs"][key] + setup, "Oracle/attack ledger mismatch: " + key)
                diverged = set()
                for step in steps:
                    ep = step["episode"]
                    if ep in diverged:
                        continue
                    clean = clean_steps.get((ep, step["step"]))
                    require(clean is not None and clean["observation"] == step["observation"], "Traffic differs before first action divergence")
                    if clean["action"] != step["action"]:
                        diverged.add(ep)
                    else:
                        require(all(clean[k] == step[k] for k in ("reward", "terminated", "next_observation", "safety")), "Same prefix produced different transition")
            rows.append(dict(checkpoint_seed=entry["run_seed"], attack=name,
                             checkpoint_sha256=entry["checkpoint_sha256"], summary=entry["summary"],
                             collisions=sum(e["ego_collision_observed"] for e in episodes),
                             episodes=len(episodes), audit=audit))
            if expected_episodes == 10 or custom_methods:
                rows[-1]["episode_rows"] = episodes
    require(checkpoint_seeds == set(range(5)), "Missing frozen checkpoint")
    return dict(kind="proposed_v0_engineering_smoke" if expected_episodes == 2 else "proposed_v0_development", git_commit=batch["git_commit"],
                batch=str(batch_path.relative_to(ROOT)), traffic=common_traffic, rows=rows,
                verified=True, source_sha256=sources,
                limitation=("Two development episodes per checkpoint, one attack seed: not an efficacy/generalization test" if expected_episodes == 2
                            else "Ten shared development traffic episodes and one attack seed; overlapping smoke, not held-out validation or final test"))


def verify_repeat(original_path, repeat_path):
    """Compare every step field except the explicitly nondeterministic timer."""
    original, repeated = [json.loads(p.read_text(encoding="utf-8")) for p in (original_path, repeat_path)]
    require(original["status"] == repeated["status"] == "passed", "Repeat batch did not pass")
    old_runs = {r["config"]: Path(r["run_dir"]) for r in original["runs"]}
    comparisons = []
    for run in repeated["runs"]:
        new_dir, old_dir = Path(run["run_dir"]), old_runs[run["config"]]
        old_manifest, new_manifest = [json.loads((d / "manifest.json").read_text()) for d in (old_dir, new_dir)]
        require(old_manifest["status"] == new_manifest["status"] == "passed", "Repeat run failed")
        for key in ("config", "victim_references", "python_runtime", "pip_freeze", "sumo_version"):
            require(old_manifest[key] == new_manifest[key], "Repeat provenance differs: " + key)
        for new_file in sorted(new_dir.glob("clean-*/steps.jsonl")):
            old_file = old_dir / new_file.relative_to(new_dir)
            old_rows, new_rows = [[json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
                                  for p in (old_file, new_file)]
            require(len(old_rows) == len(new_rows), "Repeat trajectory length differs")
            for old, new in zip(old_rows, new_rows):
                for row in (old, new):
                    row["attack_cost"].pop("wall_seconds")
                require(old == new, "Repeat step differs: " + str(new_file))
            comparisons.append(dict(condition=new_file.parent.name, steps=len(new_rows),
                                     original_sha256=hashlib.sha256(old_file.read_bytes()).hexdigest(),
                                     repeat_sha256=hashlib.sha256(new_file.read_bytes()).hexdigest()))
    require(comparisons, "No repeat trajectories found")
    return dict(passed=True, original_commit=original["git_commit"], repeat_commit=repeated["git_commit"],
                repeat_batch=str(repeat_path.relative_to(ROOT)), ignored_fields=["attack_cost.wall_seconds"],
                comparisons=comparisons, verified_steps=sum(c["steps"] for c in comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-batch", type=Path)
    args = parser.parse_args()
    report = summarize(args.batch.resolve())
    if args.repeat_batch:
        report["repeat_verification"] = verify_repeat(args.batch.resolve(), args.repeat_batch.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified %d conditions at %s" % (len(report["rows"]), report["git_commit"]))
    for row in report["rows"]:
        print("checkpoint=%d %-26s return=%.6f collisions=%d/%d" % (
            row["checkpoint_seed"], row["attack"], row["summary"]["episode_return_mean"], row["collisions"], row["episodes"]))


if __name__ == "__main__":
    main()
