"""Export a compact, descriptive mechanism smoke report from its verified audit."""

import argparse
import hashlib
import json
from pathlib import Path
from summarize_proposed_smoke import ROOT, require
from prepare_mechanism_controls import CONTROL_NAMES


def compact(path):
    report = json.loads(path.read_text())
    require(report["verified"] and report["kind"] == "return_mechanism_engineering_smoke", "Require verified mechanism smoke")
    for source, digest in report["source_sha256"].items():
        if source.replace("\\", "/").startswith(".local/"):
            require(hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest, "Raw audit input changed")
    rows = report["rows"]
    require(len(rows) == 25 and sum(r["episodes"] for r in rows) == 50, "Incomplete mechanism smoke")
    conditions = []
    for name in CONTROL_NAMES:
        selected = [r for r in rows if r["attack"] == name]
        require({r["checkpoint_seed"] for r in selected} == set(range(5)), "Incomplete condition")
        costs = {k: sum(r["audit"].get("costs", {}).get(k, 0) for r in selected)
                 for k in ("gradient_evaluations", "policy_forward_calls", "new_shadow_transitions",
                           "shadow_steps", "replay_steps", "warmup_steps")}
        episode_rows = [e for r in selected for e in r["episode_rows"]]
        eligible = sum(r["summary"].get("attack_success_eligible_episodes", 0) for r in selected)
        successes = sum(r["summary"].get("attack_successes", 0) for r in selected)
        conditions.append(dict(attack=name, episodes=len(episode_rows), steps=sum(e["steps"] for e in episode_rows),
                               episode_return_mean=sum(e["episode_return"] for e in episode_rows) / len(episode_rows),
                               collisions=sum(r["collisions"] for r in selected),
                               attack_successes=successes if name != "none" else None,
                               attack_success_eligible_episodes=eligible if name != "none" else None,
                               attack_success_rate=successes / eligible if eligible else None,
                               attempts=sum(r.get("attempt_audit", {}).get("attempts", 0) for r in selected),
                               retries=sum(r.get("attempt_audit", {}).get("retries", 0) for r in selected),
                               suppressed_retry_targets=sum(r.get("verified_suppressed_retry_targets", 0) for r in selected),
                               unique_actual_sequences=sum(r["audit"].get("unique_actual_sequences", 0) for r in selected),
                               duplicate_actual_sequences=sum(r["audit"].get("duplicate_actual_sequences", 0) for r in selected),
                               costs=costs))
    contrasts = []
    for first, second in (("ours_single_return", "zero_one_budgeted_return"),
                          ("ours_return", "ours_single_return"), ("ours_progress_return", "ours_return"),
                          ("ours_progress_return", "zero_one_budgeted_return")):
        selected = [r for r in report["paired_contrasts"] if r["first"] == first and r["second"] == second]
        require({r["checkpoint_seed"] for r in selected} == set(range(5)), "Incomplete contrast")
        contrasts.append(dict(first=first, second=second,
                              mean_return_first_minus_second=sum(r["return_first_minus_second_mean"] for r in selected) / 5,
                              gradient_first_minus_second=sum(e["gradient_first_minus_second"] for r in selected for e in r["episodes"]),
                              forward_first_minus_second=sum(e["forward_first_minus_second"] for r in selected for e in r["episodes"]),
                              identical_trajectories=sum(r["identical_trajectories"] for r in selected),
                              first_only_collision=sum(r["first_only_collision"] for r in selected),
                              second_only_collision=sum(r["second_only_collision"] for r in selected)))
    return dict(kind="return_mechanism_smoke_results", experiment_commit=report["git_commit"],
                analysis_commit=report["analysis_commit"], batch=Path(report["batch"]).as_posix(),
                full_audit=path.relative_to(ROOT).as_posix(), full_audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                verified=True, episodes=50, raw_verified_steps=report["raw_episode_verified_steps"],
                traffic_seeds=report["traffic"]["episode_sumo_seeds"], attack_seed=report["attack_seed"],
                conditions=conditions, contrasts=contrasts,
                checkpoint_rows=[dict(checkpoint_seed=r["checkpoint_seed"], attack=r["attack"],
                                      checkpoint_sha256=r["checkpoint_sha256"],
                                      episode_return_mean=r["summary"]["episode_return_mean"],
                                      gradient_evaluations=r["summary"]["gradient_evaluations"], collisions=r["collisions"])
                                 for r in rows],
                historical_regression=report["historical_regression"],
                shared_first_attempt_comparisons=report["shared_first_attempt_comparisons"],
                limitation="Five models on two previously used development traffic episodes, one attack seed, upper compute cap only; descriptive engineering check, not efficacy or generalization evidence. Full nested study remains unrun; final split 30 unused.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compact(args.audit.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Exported audited mechanism smoke; no efficacy conclusion inferred.")
