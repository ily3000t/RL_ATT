"""Strict final-50 group audit; reuse primitives without loosening old protocols."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import numpy as np

from prepare_attack_final_protocol import ROOT, PROTOCOL, read as config_read
from prepare_mechanism_controls import sha256
from summarize_proposed_smoke import require, audit_steps
from summarize_benchmark import COSTS
from analyze_proposed_development import audit_episode_records
from analyze_progress_retry import audit_stops, verify_v0_reference
from analyze_basic_validation import verify_random_seed
from analyze_mechanism_controls import verify_shared_witnesses
from rl_att.attacks.behavior_search import witness_seed
from rl_att.attacks.box import box_scales
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.evaluation.results import summarize_episodes
from rl_att.evaluation.sumo_metrics import following_measures, summarize_safety
from run_baseline import git

SEARCH = ("zero_one_budgeted_return", "zero_one_budgeted_safety", "ours_single_return", "ours_progress_return")
SEARCH_COSTS = ("new_shadow_transitions", "shadow_steps", "replay_steps", "warmup_steps", "shadow_resets", "cache_hits", "ipc_requests")


class Reader:
    def __init__(self):
        self.sources = {}

    def path(self, path):
        path = Path(path).resolve()
        require((ROOT / ".local/runs").resolve() in path.parents or ROOT.resolve() in path.parents,
                "Audit input outside repository")
        self.sources[path.relative_to(ROOT).as_posix()] = sha256(path)
        return path

    def json(self, path):
        return json.loads(self.path(path).read_text(encoding="utf-8"))

    def steps(self, path):
        with self.path(path).open(encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]


def expected_seeds(config, checkpoint, name):
    seeds = evaluation_seeds(checkpoint, config["episodes"], research_seeds=config["research_seeds"])
    mechanism = ("unused_no_attack" if name == "none" else "unused_deterministic_gradient" if name == "fgsm"
                 else "seedsequence_attack_phase3_episode_step_local_numpy" if name in ("random", "pgd")
                 else "sha256_attack_episode_full_history_target_attempt_local_numpy" if name in SEARCH
                 else "optimizer_and_discarded_torch_draws_reinitialized_each_attack")
    seeds.update(policy_rng="unused_greedy_argmax", attack_rng=mechanism)
    return seeds


def audit_witnesses(steps, name, seed, parameters):
    signatures = {}
    single = name != "ours_progress_return"
    for row in steps:
        m = row["attack_metadata"]
        require(m["name"] == name and m["objective"] == parameters["objective"] and not m["gate_enabled"],
                "Search label/objective/Gate mismatch")
        if not m["planned"]:
            continue
        require(m["search_kind"] == ("target_sequence" if name.startswith("zero_one_") else "behavior"),
                "Wrong outer search")
        grouped = defaultdict(list)
        trace = m["inner_attempt_trace"]
        for attempt in trace:
            index = attempt["attempt"]
            require(type(index) is int and 0 <= index < parameters["max_attempts"] and (not single or index == 0),
                    "Attempt/retry cap violated")
            require(attempt["seed"] == witness_seed(seed, row["episode"] - 1, attempt["history"], attempt["target"], index),
                    "Wrong independent witness seed")
            require(attempt["succeeded"] == (attempt["actual"] == attempt["target"]), "Target hit mislabeled")
            key = row["episode"], tuple(attempt["history"]), attempt["target"], index
            signature = attempt["seed"], attempt["actual"], attempt["margin"]
            require(key not in signatures or signatures[key] == signature, "Same witness changed")
            signatures[key] = signature
            grouped[(tuple(attempt["history"]), attempt["target"])].append(index)
        require(all(indices == list(range(len(indices))) for indices in grouped.values()), "Nonconsecutive attempts")
        require(m["retry_attempts"] == sum(a["attempt"] > 0 for a in trace) and
                m["failed_target_attempts"] == sum(not a["succeeded"] for a in trace), "Attempt counters differ")
        require(row["attack_cost"]["gradient_evaluations"] == len(trace) * parameters["inner_steps"],
                "Gradient count differs from PGD trace")
    if name == "ours_progress_return":
        audit_stops(steps)
    return signatures


def audit_raw(steps, episodes, config, spec, reference=None):
    """Generic raw primitive also usable on *existing* development fixtures.

    Production entry below requires the exact registered final-50 config. This
    primitive has no authority to run a simulation or accept a different grid.
    """
    name = spec["name"]
    require(len(episodes) == config["episodes"] and [e["episode"] for e in episodes] == list(range(1, config["episodes"] + 1)),
            "Missing/reordered episodes")
    require([e["sumo_seed"] for e in episodes] == config["research_seeds"]["episode_sumo_seeds"], "Unpaired traffic")
    by_episode = defaultdict(list)
    for row in steps:
        for key in ("observation", "adversarial_observation", "perturbation", "next_observation"):
            require(len(row[key]) == 16 and np.isfinite(row[key]).all(), "Invalid raw observation")
        obs, adv, delta = [np.asarray(row[k], dtype=np.float64) for k in ("observation", "adversarial_observation", "perturbation")]
        require(np.allclose(adv - obs, delta, atol=2e-7, rtol=0), "Observation/delta disagree")
        require(np.all(np.abs(delta) <= box_scales(obs, .2, .05) + 2e-7), "Perturbation escaped common box")
        require(row["scaled_linf"] == (float(np.max(np.abs(delta) / box_scales(obs, .2, .05))) if name != "none" else None),
                "Raw scaled perturbation norm differs")
        require(type(row["action"]) is int and row["action"] in (0, 1, 2) and np.isfinite(row["reward"]), "Invalid action/reward")
        if name in COSTS:
            cost = row["attack_cost"]
            require(tuple(cost.get(k, 0) for k in ("objective_evaluations", "policy_forward_calls", "gradient_evaluations")) == COSTS[name],
                    "Simple attack costs changed")
            require(row["attacked"] == (name != "none") and (name != "none" or not np.any(delta)), "Simple frequency/clean changed")
            verify_random_seed(row, config["research_seeds"]["attack_seed"], name)
            if name == "oarl_bo":
                trace = row["attack_metadata"]["trace"]
                require(len(trace) == 5 and row["attack_metadata"]["objective_value"] == max(p["objective"] for p in trace), "BO winner differs")
                require(all(.8 <= p["parameters"]["u1"] <= 1.2 and -.05 <= p["parameters"]["u2"] <= .05 for p in trace), "BO bounds differ")
        for pair in row["safety"]["pairs"]:
            expected = following_measures(pair["gap_m"], pair["closing_speed_mps"], 0.)
            require(all(pair[k] == value for k, value in expected.items()), "TTC/DRAC formula changed")
        by_episode[row["episode"]].append(row)
    raw_count = audit_episode_records(steps, episodes, config["max_steps"])
    for ep in episodes:
        raw = by_episode[ep["episode"]]
        delta = [np.asarray(r["perturbation"], dtype=np.float64) for r in raw]
        expected = dict(attacked_steps=sum(r["attacked"] for r in raw), changed_steps=sum(bool(np.any(d)) for d in delta),
                        action_changed_steps=sum(r["action"] != r["clean_action_at_visited_state"] for r in raw),
                        linf_max=max(float(np.linalg.norm(d, ord=np.inf)) for d in delta),
                        l2_max=max(float(np.linalg.norm(d)) for d in delta),
                        attack_wall_seconds=sum(r["attack_cost"]["wall_seconds"] for r in raw),
                        scaled_linf_max=max(r["scaled_linf"] for r in raw) if name != "none" else None)
        require(all(ep[k] == v for k, v in expected.items()), "Episode perturbation/attack accounting mismatch")
        require(ep["safety"] == summarize_safety([r["safety"] for r in raw], **config["metric_percentiles"]), "Episode safety summary differs")
        audit = ep["research_audit"]
        require(audit["evaluator_policy_forward_calls"] == 2 * ep["steps"] and
                audit["eligible_steps"] == ep["steps"] and audit["applied_steps"] == ep["attacked_steps"] and
                audit["changed_steps"] == ep["changed_steps"] and
                audit["attempted_steps"] == sum(r["attack_metadata"].get("attempted", r["attacked"]) for r in raw),
                "Evaluator/research cost mismatch")
        for key in SEARCH_COSTS + ("inner_cache_hits",):
            require(audit[key] == sum(r["attack_cost"].get(key, 0) for r in raw), "Episode shadow ledger differs: " + key)
    return raw_count, summarize_episodes(episodes, reference)


def verify_prefix(steps, clean_steps):
    baseline = {(s["episode"], s["step"]): s for s in clean_steps}
    diverged = set()
    for row in steps:
        ep = row["episode"]
        if ep in diverged:
            continue
        clean = baseline.get((ep, row["step"]))
        require(clean is not None and clean["observation"] == row["observation"], "Traffic differs before action divergence")
        if row["action"] != clean["action"]:
            diverged.add(ep)
        else:
            require(all(row[k] == clean[k] for k in ("reward", "terminated", "next_observation", "safety")), "Same prefix different transition")


def verify_sources(manifest, frozen, directory):
    require(manifest["source_sha256_before"] == frozen, "Runtime source differs from frozen algorithm")
    after = manifest["source_sha256_after"]
    require(set(after) == set(frozen), "Runtime source file set changed")
    changed = [p for p in frozen if frozen[p] != after[p]]
    require(set(changed) == set(manifest["changed_source_files"]) and all(p == "Data/StraightRoad.sumocfg" for p in changed),
            "Unexpected runtime source mutation")
    require(all(sha256(directory / "source" / p) == value for p, value in after.items()), "Executed snapshot bytes changed after run")


def audit_oracle(reader, directory, entry, steps, episodes, frozen):
    audit = audit_steps(steps, entry["attack"]["parameters"])
    oracle_dir = directory / (entry["results_directory"] + "-oracle")
    oracle = reader.json(oracle_dir / "manifest.json")
    # The private copy inherits the already-running SUMO config, not necessarily
    # its repository initial seed. All other algorithm/road bytes stay frozen.
    initial = dict(frozen)
    initial["Data/StraightRoad.sumocfg"] = oracle["source_sha256_before"]["Data/StraightRoad.sumocfg"]
    verify_sources(oracle, initial, oracle_dir)
    require(oracle["status"] == "passed" and oracle["returncode"] == 0 and oracle["account_reset_warmup"], "Oracle failed")
    counts = oracle["counts"]
    require(counts["live_verified_steps"] + counts.get("live_unverified_fallback_steps", 0) == len(steps) and
            counts.get("live_unverified_fallback_steps", 0) <= audit["unplanned_fallback_steps"], "Live/oracle verification differs")
    costs = {}
    for key in SEARCH_COSTS:
        costs[key] = sum(r["attack_cost"].get(key, 0) for r in steps) + sum(
            ep["research_audit"]["oracle_episode_setup_cost"].get(key, 0) for ep in episodes)
        # IPC includes additional evaluator reset/observe/count/close requests;
        # novel-transition count is derived from the physical decomposition.
        if key not in ("ipc_requests", "new_shadow_transitions"):
            require(counts.get(key, 0) == costs[key], "Oracle ledger differs: " + key)
    require(costs["shadow_steps"] == costs["new_shadow_transitions"] + costs["replay_steps"] + costs["warmup_steps"],
            "Whole-episode physical shadow ledger differs")
    return costs


def analyze(batch_path, group_id, clean_reference=None):
    commit = git("rev-parse", "HEAD")
    require(not git("status", "--porcelain"), "Audit requires clean tracked tree")
    reader = Reader()
    protocol, batch = reader.json(PROTOCOL), reader.json(batch_path)
    group = next(g for g in protocol["groups"] if g["id"] == group_id)
    require(batch["git_commit"] == commit and batch["status"] == "passed" and batch["configs"] == group["configs"] and
            len(batch["runs"]) == 5 and {r["config"] for r in batch["runs"]} == set(group["configs"]), "Incomplete/mixed final batch")
    anchor = group_id == protocol["execution"]["clean_reference_group"]
    require(anchor == (clean_reference is None), "Final groups require the canonical Clean anchor")
    rows, raw_steps, comparisons = [], 0, []
    for run in sorted(batch["runs"], key=lambda r: r["config"]):
        directory = Path(run["run_dir"])
        m, evaluation = reader.json(directory / "manifest.json"), reader.json(directory / "evaluation.json")
        config = reader.json(ROOT / run["config"])
        require(config == config_read(run["config"]) and config["episodes"] == 50 and config["research_seeds"]["split_id"] == 30,
                "Final audit requires registered final-50 config")
        from prepare_attack_final_protocol import canonical_sha256
        require(canonical_sha256(config) == protocol["configs"][run["config"]], "Registered config changed")
        checkpoint = config["run_seeds"][0]
        victim = next(v for v in protocol["victims"] if v["run_seed"] == checkpoint)
        require(run["returncode"] == 0 and m["status"] == "passed" and m["git_commit"] == evaluation["git_commit"] == commit and
                m["config"] == evaluation["config"] == config and not evaluation["gate_enabled"] and m["victim_references"] == [victim], "Run/model/config provenance differs")
        require(sha256(ROOT / victim["checkpoint"]) == victim["checkpoint_sha256"], "Frozen checkpoint changed")
        verify_sources(m, protocol["frozen_source_sha256"], directory)
        training = reader.json(ROOT / protocol["runtime_references"][checkpoint]["path"])
        for key in ("python_runtime", "pip_freeze", "sumo_version"):
            require(m[key]["returncode"] == 0 and m[key]["stdout"] == training[key]["stdout"], "Frozen runtime mismatch")
        require([d["sha256"] for d in m["extra_dependencies"]] ==
                ([protocol["zero_one_wheel"]["sha256"]] if group["gradient_cap"] is not None else []), "Dependency protocol changed")
        require([e["attack"] for e in evaluation["runs"]] == config["attacks"], "Incomplete/reordered final methods")
        clean, clean_steps, witnesses = None, None, {}
        for entry, spec in zip(evaluation["runs"], config["attacks"]):
            name = spec["name"]
            require(entry["victim"] == "clean" and entry["run_seed"] == checkpoint and entry["frozen_unchanged"] and
                    all(entry[k] == victim[k] for k in ("checkpoint_sha256", "weights_sha256")), "Frozen victim differs")
            require(entry["effective_seeds"] == expected_seeds(config, checkpoint, name), "Effective role seeds differ")
            result = directory / entry["results_directory"]
            episodes, steps = reader.json(result / "episodes.json"), reader.steps(result / "steps.jsonl")
            count, summary = audit_raw(steps, episodes, config, spec, clean)
            summary["safety"] = summarize_safety([s["safety"] for s in steps], **config["metric_percentiles"])
            require(entry["summary"] == reader.json(result / "summary.json") == summary, "Final summary differs from raw records")
            raw_steps += count
            costs = {key: sum(s["attack_cost"].get(key, 0) for s in steps)
                     for key in ("objective_evaluations", "gradient_evaluations", "policy_forward_calls") + SEARCH_COSTS}
            if name == "none":
                clean, clean_steps = episodes, steps
            else:
                verify_prefix(steps, clean_steps)
            if name in SEARCH:
                witnesses[name] = audit_witnesses(steps, name, group["attack_seed"], spec["parameters"])
                costs.update(audit_oracle(reader, directory, entry, steps, episodes, protocol["frozen_source_sha256"]))
            costs["evaluator_policy_forward_calls"] = 2 * len(steps)
            costs["total_policy_forward_calls"] = costs["policy_forward_calls"] + costs["evaluator_policy_forward_calls"]
            rows.append(dict(checkpoint_seed=checkpoint, attack=name, episodes=50, checkpoint_sha256=victim["checkpoint_sha256"],
                             summary=summary, costs=costs, episode_rows=episodes))
        if witnesses:
            comparisons.extend(dict(checkpoint_seed=checkpoint, **c) for c in verify_shared_witnesses(witnesses, SEARCH))
    regression = (dict(passed=True, canonical=True, verified_steps=sum(r["summary"]["steps"] for r in rows if r["attack"] == "none"))
                  if anchor else verify_v0_reference(batch_path, clean_reference, clean_only=True))
    if clean_reference is not None:
        reference = reader.json(clean_reference)
        for run in reference["runs"]:
            directory = Path(run["run_dir"])
            reader.json(directory / "manifest.json")
            evaluation = reader.json(directory / "evaluation.json")
            entry = next(e for e in evaluation["runs"] if e["attack"]["name"] == "none")
            reader.path(directory / entry["results_directory"] / "steps.jsonl")
    require(sum(r["episodes"] for r in rows) == group["episodes"] and git("rev-parse", "HEAD") == commit and
            not git("status", "--porcelain"), "Incomplete final audit or source changed")
    return dict(kind="attack_final_group", verified=True, git_commit=commit, group_id=group_id,
                gradient_cap=group["gradient_cap"], attack_seed=group["attack_seed"], split_id=30,
                protocol_sha256=sha256(PROTOCOL), batch=batch_path.relative_to(ROOT).as_posix(),
                episodes=group["episodes"], raw_episode_verified_steps=raw_steps, rows=rows,
                clean_regression=regression, shared_first_attempt_comparisons=comparisons, source_sha256=reader.sources)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--clean-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch.resolve(), args.group, args.clean_reference.resolve() if args.clean_reference else None)
    require((ROOT / ".local/runs").resolve() in args.output.resolve().parents and not args.output.exists(), "Use a new ignored audit output")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("FINAL_GROUP_VERIFIED group=%s episodes=%d steps=%d" % (args.group, result["episodes"], result["raw_episode_verified_steps"]))
