"""Paired development outcomes and observed retry yield, after full run auditing."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_proposed_smoke import ROOT, summarize, require


def pair_outcomes(clean, first, second):
    require(len(clean) == len(first) == len(second), "Paired episode count differs")
    counts = dict(first_only_collision=0, second_only_collision=0, both_collision=0, neither_collision=0,
                  eligible_clean_noncollision=0, first_conversion=0, second_conversion=0,
                  first_only_conversion=0, second_only_conversion=0, identical_trajectories=0)
    contrasts = []
    for baseline, a, b in zip(clean, first, second):
        require((baseline["episode"], baseline["sumo_seed"]) == (a["episode"], a["sumo_seed"]) == (b["episode"], b["sumo_seed"]),
                "Pair keys differ")
        ac, bc = a["ego_collision_observed"], b["ego_collision_observed"]
        counts["both_collision" if ac and bc else "first_only_collision" if ac else "second_only_collision" if bc else "neither_collision"] += 1
        if not baseline["ego_collision_observed"]:
            counts["eligible_clean_noncollision"] += 1
            counts["first_conversion"] += int(ac)
            counts["second_conversion"] += int(bc)
            counts["first_only_conversion"] += int(ac and not bc)
            counts["second_only_conversion"] += int(bc and not ac)
        counts["identical_trajectories"] += int(a["trajectory_sha256"] == b["trajectory_sha256"])
        contrasts.append(dict(episode=a["episode"], sumo_seed=a["sumo_seed"], clean_collision=baseline["ego_collision_observed"],
                              first_collision=ac, second_collision=bc,
                              return_first_minus_second=a["episode_return"] - b["episode_return"],
                              first_steps=a["steps"], second_steps=b["steps"],
                              gradient_first_minus_second=a["gradient_evaluations"] - b["gradient_evaluations"],
                              forward_first_minus_second=a["attack_policy_forward_calls"] - b["attack_policy_forward_calls"]))
    return dict(counts, episodes=contrasts,
                return_first_minus_second_mean=sum(e["return_first_minus_second"] for e in contrasts) / len(contrasts))


def retry_yield(steps):
    """Observed branch discovery, not an ablation or proof that retries are needed.

    In Ours-v0 only the two non-clean targets are attempted. A node with retries
    must have tried both, allowing its initial clean branch to be reconstructed.
    The first discovery of an actual branch is then attributable to one attempt.
    """
    counts = dict(attempts=0, retries=0, retry_target_hits=0, retry_new_branches=0,
                  retry_new_branches_on_selected_plan=0, retry_repeated_margin_and_action=0,
                  nodes_with_retries=0, retries_without_new_branch=0)
    examples = []
    live_history, episode = [], None
    for row in steps:
        if row["episode"] != episode:
            live_history, episode = [], row["episode"]
        require(len(live_history) == row["step"], "Missing live action prefix")
        m = row["attack_metadata"]
        if m["planned"]:
            require(m["search_kind"] == "behavior", "Retry reconstruction applies only to Ours-v0")
            attempts = m["inner_attempt_trace"]
            counts["attempts"] += len(attempts)
            selected_branches = set()
            selected = m["selected_candidate"]
            prefix = tuple(live_history)
            if selected is not None:
                for action in m["candidate_trace"][selected]["actions"]:
                    selected_branches.add((prefix, action))
                    prefix += (action,)
            grouped = defaultdict(list)
            for attempt in attempts:
                grouped[tuple(attempt["history"])].append(attempt)
            for history, group in grouped.items():
                if not any(a["attempt"] > 0 for a in group):
                    continue
                counts["nodes_with_retries"] += 1
                targets = {a["target"] for a in group}
                require(len(targets) == 2 and targets <= {0, 1, 2}, "Retry node did not try both non-clean targets")
                known = {0, 1, 2} - targets
                previous = {}
                for attempt in group:
                    target, actual, index = attempt["target"], attempt["actual"], attempt["attempt"]
                    last = previous.get(target)
                    require(index == (last["attempt"] + 1 if last is not None else 0), "Attempt index is not consecutive")
                    if index:
                        counts["retries"] += 1
                        counts["retry_target_hits"] += int(target == actual)
                        new = actual not in known
                        selected_new = new and (history, actual) in selected_branches
                        counts["retry_new_branches"] += int(new)
                        counts["retry_new_branches_on_selected_plan"] += int(selected_new)
                        counts["retries_without_new_branch"] += int(not new)
                        counts["retry_repeated_margin_and_action"] += int(last["margin"] == attempt["margin"] and last["actual"] == actual)
                        if new and len(examples) < 12:
                            examples.append(dict(episode=episode, block_start=row["step"], history=list(history),
                                                 target=target, actual=actual, attempt=index, seed=attempt["seed"],
                                                 used_in_selected_plan=selected_new))
                    known.add(actual)
                    previous[target] = attempt
        live_history.append(row["action"])
    return dict(counts, new_branch_examples=examples,
                interpretation="Post-hoc discovery attribution; selected does not establish a causal collision contribution")


def analyze(batch_path):
    report = summarize(batch_path, expected_episodes=10)
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    directories = {}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        for entry in evaluation["runs"]:
            directories[(entry["run_seed"], entry["attack"]["name"])] = directory / entry["results_directory"]
    by_key = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
    report["paired_contrasts"] = []
    for seed in range(5):
        clean = by_key[(seed, "none")]["episode_rows"]
        for first, second in (("ours_return", "zero_one_budgeted_return"), ("ours_safety", "zero_one_budgeted_safety"),
                              ("ours_safety", "ours_return"), ("zero_one_budgeted_safety", "zero_one_budgeted_return")):
            pair = pair_outcomes(clean, by_key[(seed, first)]["episode_rows"], by_key[(seed, second)]["episode_rows"])
            report["paired_contrasts"].append(dict(checkpoint_seed=seed, first=first, second=second, **pair))
        for name in ("ours_return", "ours_safety"):
            path = directories[(seed, name)] / "steps.jsonl"
            with path.open(encoding="utf-8") as source:
                audit = retry_yield(json.loads(line) for line in source)
            row = by_key[(seed, name)]
            require(audit["retries"] == row["audit"]["retry_attempts"], "Retry audit total differs")
            row["retry_yield"] = audit
    report["analysis_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit analysis code before publishing its report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("DEVELOPMENT_AUDIT_PASSED=" + result["git_commit"])
    for row in result["rows"]:
        print("checkpoint=%d %-26s return=%.6f collisions=%d/%d retries=%s" % (
            row["checkpoint_seed"], row["attack"], row["summary"]["episode_return_mean"], row["collisions"],
            row["episodes"], row.get("retry_yield", {}).get("retries", "n/a")))
