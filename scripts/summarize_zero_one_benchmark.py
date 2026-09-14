"""Revalidate Stage 3 and join Zero-One only after exact paired-control agreement."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess
import sys
from run_baseline import ROOT, git
from summarize_benchmark import read, stats, verify_steps


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_zero_one(directory, episodes, oracle):
    counters = {k: 0 for k in ("shadow_steps", "replay_steps", "cache_hits", "shadow_resets",
                               "candidate_rollouts", "inner_cache_hits", "gradient_evaluations",
                               "policy_forward_calls", "objective_evaluations")}
    counts, returns, collisions = [0] * 20, [0.] * 20, [False] * 20
    digests = [hashlib.sha256() for _ in range(20)]
    episode_costs = [[0, 0, 0] for _ in range(20)]
    selected = None
    with (directory / "steps.jsonl").open(encoding="utf-8") as raw:
        for line in raw:
            row = json.loads(line)
            i, step = row["episode"] - 1, row["step"]
            if not 0 <= i < 20 or step != counts[i]:
                raise ValueError("Nonconsecutive Zero-One records")
            counts[i] += 1
            obs, adv, delta = (row[k] for k in ("observation", "adversarial_observation", "perturbation"))
            if any(len(x) != 16 or not all(math.isfinite(v) for v in x) for x in (obs, adv, delta)):
                raise ValueError("Invalid Zero-One observation")
            if any(abs(a-o-d) > 1e-7 or abs(d) > .2*abs(o)+.05+2e-7 for o,a,d in zip(obs,adv,delta)):
                raise ValueError("Zero-One common perturbation budget violated")
            meta, cost = row["attack_metadata"], row["attack_cost"]
            if not row["attacked"] or meta["gate_enabled"] or meta["planned"] != (step % 20 == 0):
                raise ValueError("Block schedule or no-Gate protocol changed")
            if meta["planned"]:
                trace = meta["candidate_trace"]
                expected_calls = min(10, 3 ** min(20, 200-step))
                if len(trace) != expected_calls or cost["objective_evaluations"] != expected_calls:
                    raise ValueError("Candidate budget mismatch")
                best = min(range(len(trace)), key=lambda j: trace[j]["value"])
                if best != meta["selected_candidate"] or trace[best]["value"] != meta["selected_return"]:
                    raise ValueError("Incorrect candidate winner")
                selected = trace[best]
                for candidate in trace:
                    if candidate["value"] != sum(candidate["rewards"]) or len(candidate["actions"]) != len(candidate["rewards"]):
                        raise ValueError("Candidate return differs from actual rollout rewards")
                    if len(candidate["actions"]) > min(20, 200-step) or any(a not in (0,1,2) for a in candidate["actions"]):
                        raise ValueError("Invalid candidate horizon or actions")
                transitions = sum(len(t["actions"]) for t in trace)
                if cost["candidate_rollouts"] != len(trace) or cost["gradient_evaluations"] % 2:
                    raise ValueError("Invalid outer or inner cost")
                if transitions != cost["gradient_evaluations"] // 2 + cost["inner_cache_hits"]:
                    raise ValueError("Inner cache and gradient cost do not reconcile")
                if transitions != cost["shadow_steps"] - cost["replay_steps"] + cost["cache_hits"]:
                    raise ValueError("Shadow replay cost does not reconcile")
            elif any(cost[k] for k in counters if k != "policy_forward_calls"):
                raise ValueError("Cached block unexpectedly performed optimization")
            if cost["policy_forward_calls"] != cost["gradient_evaluations"] // 2 * 3 + 1:
                raise ValueError("Policy forward cost mismatch")
            offset = step % 20
            if row["action"] != selected["actions"][offset] or row["reward"] != selected["rewards"][offset]:
                raise ValueError("Real execution differs from winning candidate")
            if row["terminated"] != (selected["terminated"] and offset == len(selected["actions"])-1):
                raise ValueError("Real termination differs from winning candidate")
            for key in counters:
                counters[key] += cost[key]
            for j,key in enumerate(("objective_evaluations", "policy_forward_calls", "gradient_evaluations")):
                episode_costs[i][j] += cost[key]
            returns[i] += row["reward"]
            collisions[i] |= row["safety"]["ego_collision_observed"]
            digests[i].update(struct.pack("<16d", *row["next_observation"]))
            digests[i].update(struct.pack("<3d", row["action"], row["reward"], row["terminated"]))
    for i,episode in enumerate(episodes):
        if (counts[i], returns[i], collisions[i], digests[i].hexdigest()) != (
                episode["steps"], episode["episode_return"], episode["ego_collision_observed"], episode["trajectory_sha256"]):
            raise ValueError("Raw trajectory and episode outcome differ")
        if episode_costs[i] != [episode[k] for k in ("objective_evaluations", "attack_policy_forward_calls", "gradient_evaluations")]:
            raise ValueError("Episode costs differ from raw trace")
    if oracle["status"] != "passed" or oracle["returncode"] != 0 or oracle["counts"]["live_verified_steps"] != sum(counts):
        raise ValueError("Incomplete live oracle verification")
    for key in ("shadow_steps", "replay_steps", "cache_hits", "candidate_rollouts", "shadow_resets"):
        if counters[key] + (20 if key == "shadow_resets" else 0) != oracle["counts"][key]:
            raise ValueError("Oracle manifest does not reconcile with attack costs: " + key)
    if any(name != "Data/StraightRoad.sumocfg" for name in oracle["changed_source_files"]):
        raise ValueError("Oracle modified its source")
    counters.update(live_steps=sum(counts), live_verified_steps=sum(counts),
                    initial_episode_resets=20, raw_steps_sha256=sha(directory / "steps.jsonl"))
    return counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage3-batch", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit summary implementation before recording analysis provenance")
    previous_path = ROOT / ".local/reports/stage3_reverified_for_zero_one.json"
    previous_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_benchmark.py"),
                    "--batch", str(args.stage3_batch), "--output", str(previous_path)], check=True)
    result = read(previous_path)
    published = read(ROOT / "docs/STAGE3_BENCHMARK_RESULTS.json")
    if any(result[k] != published[k] for k in ("aggregate", "runs", "raw_steps_checked", "benchmark_commit")):
        raise ValueError("Revalidated Stage 3 differs from recorded results")
    old_batch, batch = read(args.stage3_batch / "batch.json"), read(args.batch / "batch.json")
    if batch["status"] != "passed" or len(batch["runs"]) != 5:
        raise ValueError("Require all five completed Zero-One trials")
    old_dirs = {read(Path(r["run_dir"]) / "manifest.json")["config"]["run_seeds"][0]: Path(r["run_dir"])
                for r in old_batch["runs"]}
    seeds, costs, joins = set(), [], []
    shared = None
    for child in batch["runs"]:
        directory = Path(child["run_dir"])
        manifest, evaluation = read(directory / "manifest.json"), read(directory / "evaluation.json")
        config = manifest["config"]
        seed = config["run_seeds"][0]
        if seed in seeds or seed not in range(5) or manifest["status"] != "passed" or child["returncode"] != 0:
            raise ValueError("Invalid Zero-One trial")
        seeds.add(seed)
        if manifest["git_commit"] != batch["git_commit"] or evaluation["git_commit"] != batch["git_commit"]:
            raise ValueError("Mixed Zero-One experiment commits")
        if config["victims"] != ["clean"] or config["run_seeds"] != [seed]:
            raise ValueError("Expected one frozen Clean victim per trial")
        protocol = copy.deepcopy(config)
        protocol.pop("run_seeds")
        if shared is None:
            shared = protocol
        if shared != protocol:
            raise ValueError("Zero-One configs differ beyond run seed")
        old_manifest, old_evaluation = read(old_dirs[seed] / "manifest.json"), read(old_dirs[seed] / "evaluation.json")
        if any(config[k] != old_manifest["config"][k] for k in config if k != "attacks"):
            raise ValueError("Evaluation protocol differs from Stage 3")
        for k in ("python_runtime", "pip_freeze", "sumo_version", "victim_references"):
            if manifest[k] != old_manifest[k]:
                raise ValueError("Frozen reference/runtime changed: " + k)
        if any(manifest["source_sha256_before"][k] != v for k,v in old_manifest["source_sha256_before"].items()
               if k.startswith(("Environment/", "Data/")) or k in ("main.py", "oarl.py")):
            raise ValueError("Original environment or victim code changed")
        if any(k != "Data/StraightRoad.sumocfg" for k in manifest["changed_source_files"]):
            raise ValueError("Unexpected evaluation source mutation")
        if [r["attack"]["name"] for r in evaluation["runs"]] != ["none", "zero_one"]:
            raise ValueError("Unexpected Stage 4 methods")
        clean, attacked = evaluation["runs"]
        old_clean = old_evaluation["runs"][0]
        baseline = read(directory / clean["results_directory"] / "episodes.json")
        if baseline != read(old_dirs[seed] / old_clean["results_directory"] / "episodes.json"):
            raise ValueError("New Clean episodes do not exactly match Stage 3")
        if not clean["legacy_equivalence"]["passed"]:
            raise ValueError("Legacy control verification failed")
        for entry in (clean, attacked):
            if any(entry[k] != old_clean[k] for k in ("checkpoint_sha256", "weights_sha256", "victim", "run_seed")) or not entry["frozen_unchanged"]:
                raise ValueError("Attack did not use the same frozen model")
            if any(entry["effective_seeds"][k] != old_clean["effective_seeds"][k]
                   for k in old_clean["effective_seeds"] if k != "attack_rng"):
                raise ValueError("Seed protocol changed")
        verify_steps(directory / clean["results_directory"], "none", baseline)
        attacked_dir = directory / attacked["results_directory"]
        episodes = read(attacked_dir / "episodes.json")
        if len(episodes) != 20 or any((e["episode"],e["sumo_seed"]) != (b["episode"],b["sumo_seed"])
                                      for e,b in zip(episodes,baseline)):
            raise ValueError("Unpaired Zero-One episodes")
        oracle_dir = directory / (attacked["results_directory"] + "-oracle")
        cost = verify_zero_one(attacked_dir, episodes, read(oracle_dir / "manifest.json"))
        cost["run_seed"] = seed
        costs.append(cost)
        summary = attacked["summary"]
        drops = [b["episode_return"] - e["episode_return"] for e,b in zip(episodes,baseline)]
        eligible = [(e,b) for e,b in zip(episodes,baseline) if not b["ego_collision_observed"]]
        if abs(summary["return_drop_mean"] - statistics.mean(drops)) > 1e-9 or summary["attack_successes"] != sum(e["ego_collision_observed"] for e,b in eligible):
            raise ValueError("Paired Zero-One summary mismatch")
        for key, raw_key in (("objective_evaluations", "objective_evaluations"),
                             ("attack_policy_forward_calls", "policy_forward_calls"), ("gradient_evaluations", "gradient_evaluations")):
            if summary[key] != cost[raw_key]:
                raise ValueError("Summary attack cost mismatch")
        result["runs"].append(dict(run_seed=seed, attack="zero_one", checkpoint_sha256=attacked["checkpoint_sha256"],
            result_directory=attacked_dir.resolve().relative_to(ROOT).as_posix(), summary=summary, return_drops=drops))
        joins.append(dict(run_seed=seed, exact_clean_episode_match=True, manifest_sha256=sha(directory / "manifest.json"),
                          oracle_manifest_sha256=sha(oracle_dir / "manifest.json"), extra_dependencies=manifest["extra_dependencies"]))
    method_rows = [r for r in result["runs"] if r["attack"] == "zero_one"]
    aggregate = {}
    for key in result["aggregate"]["none"]:
        if key == "totals":
            aggregate[key] = {k: sum(r["summary"][k] for r in method_rows) for k in result["aggregate"]["none"][key]}
        else:
            aggregate[key] = stats([(r["summary"]["safety"] if key in r["summary"]["safety"] else r["summary"])[key] for r in method_rows])
    result["aggregate"]["zero_one"] = aggregate
    result["config"]["attacks"].append(shared["attacks"][1])
    result.update(stage=4, episodes=600, additional_clean_control_episodes=100,
                  stage3_benchmark_commit=result.pop("benchmark_commit"), zero_one_benchmark_commit=batch["git_commit"],
                  stage3_batch=result.pop("batch"), zero_one_batch=args.batch.resolve().relative_to(ROOT).as_posix(),
                  summary_commit=git("rev-parse", "HEAD"), summary_command=[sys.executable,*sys.argv],
                  stage3_raw_steps_checked=result["raw_steps_checked"],
                  raw_steps_checked=result["raw_steps_checked"]+sum(c["live_steps"] for c in costs),
                  all_5_new_clean_controls_exactly_match=True, all_zero_one_live_steps_oracle_verified=True,
                  zero_one_oracle_costs=costs, pairing_proofs=joins,
                  threat_model_note="Zero-One categorical state-only adaptation has privileged simulator queries; no equal-information or equal-compute claim")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("ZERO_ONE_SUMMARY=" + str(args.output))


if __name__ == "__main__":
    main()
