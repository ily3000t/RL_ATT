"""Audit adaptive attacks on both frozen cohorts and report defense tradeoffs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_defense_baseline import canonical_hash
from analyze_attack_final import (Reader, SEARCH, SEARCH_COSTS, audit_raw, audit_oracle,
                                 audit_witnesses, expected_seeds, verify_prefix, verify_sources)
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.evaluation.sumo_metrics import summarize_safety
from run_baseline import git


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_tradeoffs(rows, config):
    """Raw collision and within-policy ASR have distinct reference populations."""
    names = [a["name"] for a in config["attacks"]]
    expected = {(v, s, a) for v in config["victims"] for s in config["run_seeds"] for a in names}
    indexed = {(r["victim"], r["checkpoint_seed"], r["attack"]): r for r in rows}
    require(len(indexed) == len(rows) and set(indexed) == expected, "Incomplete/duplicate defense grid")
    traffic = config["research_seeds"]["episode_sumo_seeds"]
    for r in rows:
        require([e["sumo_seed"] for e in r["episode_rows"]] == traffic, "Defense traffic not paired")
        require(len(r["episode_rows"]) == config["episodes"], "Defense episode count differs")
    comparisons = []
    for name in names:
        per_pair = []
        for s in config["run_seeds"]:
            clean, robust = indexed[("clean", s, name)], indexed[("oarl", s, name)]
            clean_base, robust_base = indexed[("clean", s, "none")], indexed[("oarl", s, "none")]
            for i, seed in enumerate(traffic):
                c, r = clean["episode_rows"][i], robust["episode_rows"][i]
                cb, rb = clean_base["episode_rows"][i], robust_base["episode_rows"][i]
                per_pair.append(dict(checkpoint_seed=s, sumo_seed=seed,
                    return_robust_minus_clean=r["episode_return"]-c["episode_return"],
                    drop_robust_minus_clean=(rb["episode_return"]-r["episode_return"])-(cb["episode_return"]-c["episode_return"]),
                    collision_robust_minus_clean=int(r["ego_collision_observed"])-int(c["ego_collision_observed"]),
                    both_unattacked_noncollision=not cb["ego_collision_observed"] and not rb["ego_collision_observed"]))
        per_traffic = [dict(sumo_seed=s, return_robust_minus_clean=float(np.mean([
            p["return_robust_minus_clean"] for p in per_pair if p["sumo_seed"] == s]))) for s in traffic]
        populations = {}
        for v in ("clean", "oarl"):
            attacked = [e for s in config["run_seeds"] for e in indexed[(v,s,name)]["episode_rows"]]
            baseline = [e for s in config["run_seeds"] for e in indexed[(v,s,"none")]["episode_rows"]]
            eligible = [e for e,b in zip(attacked,baseline) if not b["ego_collision_observed"]]
            costs = {key: sum(indexed[(v,s,name)]["costs"][key] for s in config["run_seeds"])
                     for key in indexed[(v,config["run_seeds"][0],name)]["costs"]}
            real_steps = sum(e["steps"] for e in attacked)
            populations[v] = dict(actual_episodes=len(attacked), real_steps=real_steps,
                return_mean=float(np.mean([e["episode_return"] for e in attacked])),
                return_drop_mean=float(np.mean([b["episode_return"]-e["episode_return"] for e,b in zip(attacked,baseline)])),
                collisions=sum(e["ego_collision_observed"] for e in attacked),
                collision_rate=sum(e["ego_collision_observed"] for e in attacked)/len(attacked),
                asr_eligible=len(eligible) if name != "none" else None,
                asr_successes=sum(e["ego_collision_observed"] for e in eligible) if name != "none" else None,
                asr=sum(e["ego_collision_observed"] for e in eligible)/len(eligible) if eligible and name != "none" else None,
                costs_total=costs, costs_per_actual_episode={k:v/len(attacked) for k,v in costs.items()},
                costs_per_real_step={k:v/real_steps for k,v in costs.items()},
                added_online_defense_network_calls=0,
                safety={k: dict(valid_episodes=sum(e["safety"][k] is not None for e in attacked),
                                mean_of_valid_episode_values=float(np.mean([e["safety"][k] for e in attacked if e["safety"][k] is not None]))
                                if any(e["safety"][k] is not None for e in attacked) else None)
                        for k in ("minimum_ttc_s","ttc_low_percentile_s","drac_high_percentile_mps2")})
        comparisons.append(dict(attack=name, populations=populations,
            return_robust_minus_clean_mean=float(np.mean([p["return_robust_minus_clean"] for p in per_pair])),
            drop_robust_minus_clean_mean=float(np.mean([p["drop_robust_minus_clean"] for p in per_pair])),
            collision_rate_robust_minus_clean=float(np.mean([p["collision_robust_minus_clean"] for p in per_pair])),
            common_clean_noncollision_pairs=sum(p["both_unattacked_noncollision"] for p in per_pair),
            traffic_clusters=len(traffic), fixed_checkpoint_pairs=len(config["run_seeds"]),
            per_traffic=per_traffic, pairs=per_pair))
    return comparisons


def analyze(directory, group_id):
    directory = directory.resolve()
    require((ROOT / ".local/runs").resolve() in directory.parents, "Use an ignored run directory")
    require(not git("status", "--porcelain"), "Commit source before defense audit")
    reader = Reader()
    protocol = reader.json(ROOT / "configs/research/defense_baseline.json")
    group = next(g for g in protocol["groups"] if g["id"] == group_id)
    config = reader.json(ROOT / group["config"])
    require(canonical_hash(config) == group["config_sha256"], "Registered defense configuration changed")
    manifest, evaluation = reader.json(directory/"manifest.json"), reader.json(directory/"evaluation.json")
    require(manifest["status"] == "passed" and manifest["returncode"] == 0 and
            manifest["config"] == evaluation["config"] == config and
            manifest["git_commit"] == evaluation["git_commit"] == git("rev-parse", "HEAD"), "Defense provenance differs")
    require(manifest["victim_references"] == protocol["victim_references"] and not evaluation["gate_enabled"], "Frozen defense cohort or Gate changed")
    frozen = {name: hashlib.sha256(subprocess.check_output(["git","show",manifest["git_commit"]+":"+name], cwd=str(ROOT))).hexdigest()
              for name in manifest["source_sha256_before"]}
    verify_sources(manifest, frozen, directory)
    rows, total_steps, entries = [], 0, iter(evaluation["runs"])
    for ref in protocol["victim_references"]:
        checkpoint = ROOT/ref["checkpoint"]
        reader.path(checkpoint)
        training = reader.json(checkpoint.parents[2]/"manifest.json")
        require(training["status"] == "passed" and training["git_commit"] == ref["training_commit"], "Frozen training provenance changed")
        for key in ("python_runtime","pip_freeze","sumo_version"):
            require(manifest[key]["returncode"] == 0 and manifest[key]["stdout"] == training[key]["stdout"], "Frozen runtime changed")
        victim = VictimAdapter.from_reference(ref, ROOT)
        clean, clean_steps = None, None
        for spec in config["attacks"]:
            entry = next(entries, None)
            require(entry is not None and entry["attack"] == spec and entry["victim"] == ref["victim"] and
                    entry["run_seed"] == ref["run_seed"] and entry["frozen_unchanged"] and
                    all(entry[k] == ref[k] for k in ("checkpoint_sha256","weights_sha256")), "Defense grid or checkpoint changed")
            name = spec["name"]
            require(entry["effective_seeds"] == expected_seeds(config,ref["run_seed"],name), "Effective seed roles changed")
            result = directory/entry["results_directory"]
            episodes, steps = reader.json(result/"episodes.json"), reader.steps(result/"steps.jsonl")
            count, summary = audit_raw(steps,episodes,config,spec,clean)
            summary["safety"] = summarize_safety([s["safety"] for s in steps], **config["metric_percentiles"])
            require(entry["summary"] == reader.json(result/"summary.json") == summary, "Defense summary differs from raw")
            for s in steps:
                require(victim.action(s["observation"]) == s["clean_action_at_visited_state"] and
                        victim.action(s["adversarial_observation"]) == s["action"], "Actual policy differs from registered defended checkpoint")
            total_steps += count
            costs = {k:sum(s["attack_cost"].get(k,0) for s in steps)
                     for k in ("objective_evaluations","gradient_evaluations","policy_forward_calls","wall_seconds")+SEARCH_COSTS}
            if name == "none":
                clean, clean_steps = episodes, steps
            else:
                verify_prefix(steps,clean_steps)
            if name in SEARCH:
                audit_witnesses(steps,name,config["research_seeds"]["attack_seed"],spec["parameters"])
                costs.update(audit_oracle(reader,directory,entry,steps,episodes,frozen))
            costs["evaluator_policy_forward_calls"] = 2*len(steps)
            costs["total_policy_forward_calls"] = costs["policy_forward_calls"]+costs["evaluator_policy_forward_calls"]
            rows.append(dict(victim=ref["victim"],checkpoint_seed=ref["run_seed"],attack=name,
                             costs=costs,episode_rows=episodes))
        victim.assert_frozen()
    require(next(entries,None) is None and sum(len(r["episode_rows"]) for r in rows) == group["actual_episodes"], "Unexpected/incomplete defense runs")
    require(not git("status","--porcelain") and git("rev-parse","HEAD") == manifest["git_commit"], "Defense source changed while auditing")
    return dict(kind="oarl_defense_baseline_audit", verified=True, group_id=group_id,
        git_commit=manifest["git_commit"], protocol_sha256=reader.sources["configs/research/defense_baseline.json"],
        attack_mode=protocol["attack_mode"], actual_episodes=group["actual_episodes"], real_steps=total_steps,
        comparisons=paired_tradeoffs(rows,config), source_sha256=reader.sources,
        scientific_evidence=group_id != "engineering_smoke",
        limitation=protocol["statistics"]["efficacy_limit"], defense_training_cost="not measured by evaluation; report training independently")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--group",choices=("engineering_smoke","development_pilot"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    require((ROOT/".local/runs").resolve() in args.output.resolve().parents and not args.output.exists(), "Use new ignored audit output")
    report = analyze(args.run,args.group)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("DEFENSE_AUDIT_PASSED episodes=%d steps=%d"%(report["actual_episodes"],report["real_steps"]))
