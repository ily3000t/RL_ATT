"""Combine the complete preregistered budget grid, retaining paired losses and costs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from analyze_budget_sweep import candidate_availability
from analyze_proposed_development import pair_outcomes
from summarize_proposed_smoke import ROOT, require


ATTACKS = tuple(m + o for m in ("zero_one_budgeted_", "ours_progress_") for o in ("return", "safety"))
CAPS = (100, 200, 400)


def check_grid(reports):
    keys = [(d["gradient_cap"], d["attack_seed"]) for d in reports]
    require(len(keys) == len(set(keys)), "Duplicate budget/seed report")
    require(set(keys) == {(c, s) for c in (100, 200) for s in range(3)}, "Incomplete preregistered grid")


def aggregate(rows):
    """Episode-weighted outcomes; costs are totals, never averages of rates."""
    episodes = sum(r["episodes"] for r in rows)
    eligible = sum(r["summary"]["attack_success_eligible_episodes"] for r in rows)
    conversions = sum(r["summary"]["attack_successes"] for r in rows)
    result = dict(episodes=episodes, collisions=sum(r["collisions"] for r in rows),
                  clean_noncollision_episodes=eligible, collision_conversions=conversions,
                  conversion_rate=conversions / eligible if eligible else None,
                  mean_return=sum(r["summary"]["episode_return_mean"] * r["episodes"] for r in rows) / episodes,
                  live_steps=sum(r["summary"]["steps"] for r in rows))
    result["costs"] = {k: sum(r["audit"]["costs"][k] for r in rows) for k in rows[0]["audit"]["costs"]}
    result["availability"] = {k: sum(r["candidate_availability"][k] for r in rows)
                              for k in rows[0]["candidate_availability"]}
    result["complete_search_candidates"] = sum(r["audit"]["complete_search_candidates"] for r in rows)
    result["live_unverified_fallback_steps"] = sum(r["audit"]["live_unverified_fallback_steps"] for r in rows)
    result["oracle_setup_shadow_steps"] = sum(e["research_audit"]["oracle_episode_setup_cost"]["shadow_steps"]
                                               for r in rows for e in r["episode_rows"])
    return result


def summarize(paths):
    cache, loaded, proofs = {}, {}, []

    def digest(path):
        path = path.resolve()
        if path not in cache:
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            cache[path] = h.hexdigest()
        return cache[path]

    def load(path):
        path = path.resolve()
        if path not in loaded:
            data = json.loads(path.read_text(encoding="utf-8"))
            require(data["verified"], "Unverified input")
            require(data["source_sha256"], "Missing audit input hashes")
            for name, expected in data["source_sha256"].items():
                require(digest(ROOT / name) == expected, "Audited input changed: " + name)
            loaded[path] = data
            proofs.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                               git_commit=data["git_commit"], analysis_commit=data.get("analysis_commit")))
        return loaded[path]

    lows = [load(p) for p in paths]
    check_grid(lows)
    require(all(d["kind"] == "shared_compute_budget_sweep" for d in lows), "Wrong audit kind")
    cells, high_reports, reference_sets = {}, {}, {}
    for low in lows:
        cap, seed = low["gradient_cap"], low["attack_seed"]
        references = sorted((p["path"], p["sha256"]) for p in low["upper_references"])
        require(len(references) == 2, "Expected two upper method references")
        reference_sets.setdefault(seed, references)
        require(references == reference_sets[seed], "Lower budgets use different upper references")
        require(low["forward_cap"] == 2 * cap and low["clean_regression"]["passed"], "Budget/Clean audit mismatch")
        cells[cap, seed] = low["rows"]
        for proof in low["upper_references"]:
            path = ROOT / proof["path"]
            require(digest(path) == proof["sha256"], "Upper reference report changed")
            high = load(path)
            require(high["attack_seed"] == seed, "Upper reference seed differs")
            high_reports[path.resolve()] = high
        if (400, seed) not in cells:
            high_rows = []
            for proof in low["upper_references"]:
                high = loaded[(ROOT / proof["path"]).resolve()]
                is_zero = any(r["attack"].startswith("zero_one_budgeted_") for r in high["rows"])
                names = set(ATTACKS[:2] if is_zero else ("none",) + ATTACKS[2:])
                high_rows.extend(r for r in high["rows"] if r["attack"] in names)
                # Upper audits predate availability counts; derive them from hashed raw records.
                batch = json.loads((ROOT / high["batch"]).read_text())
                for run in batch["runs"]:
                    directory = Path(run["run_dir"])
                    for entry in json.loads((directory / "evaluation.json").read_text())["runs"]:
                        name = entry["attack"]["name"]
                        if name not in names or name == "none":
                            continue
                        raw_path = directory / entry["results_directory"] / "steps.jsonl"
                        require(str(raw_path.relative_to(ROOT)) in high["source_sha256"], "Unhashed upper trajectory")
                        row = next(r for r in high_rows if r["checkpoint_seed"] == entry["run_seed"] and r["attack"] == name)
                        with raw_path.open() as stream:
                            row["candidate_availability"] = candidate_availability(json.loads(line) for line in stream)
            cells[400, seed] = high_rows

    expected = {(s, n) for s in range(5) for n in ("none",) + ATTACKS}
    models, traffic = {}, lows[0]["traffic"]
    for (cap, seed), rows in cells.items():
        require(len(rows) == 25 and {(r["checkpoint_seed"], r["attack"]) for r in rows} == expected,
                "Missing/duplicate checkpoint condition")
        for row in rows:
            checkpoint = row["checkpoint_seed"]
            models.setdefault(checkpoint, row["checkpoint_sha256"])
            require(row["checkpoint_sha256"] == models[checkpoint], "Frozen checkpoint changed")
            require(row["episodes"] == len(row["episode_rows"]) == 10, "Incomplete episode series")
        require(all(d["traffic"] == traffic for d in loaded.values()), "Traffic differs")

    result = dict(kind="shared_compute_budget_sensitivity", inputs=proofs, traffic=traffic,
                  limitation="Development traffic/checkpoints and attack seeds are correlated; no independent significance or generalization claim.",
                  cells=[], pooled=[], paired=[], upper_budget_pairs=[], key_cases=[])
    for cap in CAPS:
        for seed in range(3):
            rows = cells[cap, seed]
            by_key = {(r["checkpoint_seed"], r["attack"]): r for r in rows}
            for attack in ATTACKS:
                selected = [r for r in rows if r["attack"] == attack]
                result["cells"].append(dict(gradient_cap=cap, forward_cap=cap * 2, attack_seed=seed, attack=attack,
                    **aggregate(selected), checkpoints=[dict(checkpoint_seed=r["checkpoint_seed"], collisions=r["collisions"],
                        mean_return=r["summary"]["episode_return_mean"], costs=r["audit"]["costs"],
                        availability=r["candidate_availability"]) for r in selected]))
            for checkpoint in range(5):
                clean = by_key[checkpoint, "none"]["episode_rows"]
                reference_clean = next(r for r in cells[400, 0] if r["checkpoint_seed"] == checkpoint and r["attack"] == "none")
                require([(e["sumo_seed"], e["trajectory_sha256"]) for e in clean] ==
                        [(e["sumo_seed"], e["trajectory_sha256"]) for e in reference_clean["episode_rows"]], "Clean trajectories differ")
                for objective in ("return", "safety"):
                    first, second = "ours_progress_" + objective, "zero_one_budgeted_" + objective
                    p = pair_outcomes(clean, by_key[checkpoint, first]["episode_rows"], by_key[checkpoint, second]["episode_rows"])
                    p["collision_discordances"] = [e for e in p.pop("episodes") if e["first_collision"] != e["second_collision"]]
                    result["paired"].append(dict(gradient_cap=cap, attack_seed=seed, checkpoint_seed=checkpoint,
                                                 objective=objective, first=first, second=second, **p))
                if cap != 400:
                    for name in ATTACKS:
                        upper = next(r for r in cells[400, seed] if r["checkpoint_seed"] == checkpoint and r["attack"] == name)
                        p = pair_outcomes(clean, by_key[checkpoint, name]["episode_rows"], upper["episode_rows"])
                        p["collision_discordances"] = [e for e in p.pop("episodes") if e["first_collision"] != e["second_collision"]]
                        result["upper_budget_pairs"].append(dict(gradient_cap=cap, reference_cap=400, attack_seed=seed,
                                                               checkpoint_seed=checkpoint, attack=name, **p))
            for case_seed, episode in ((1, 5), (2, 9)):
                if seed == case_seed:
                    for name in ATTACKS:
                        e = by_key[2, name]["episode_rows"][episode - 1]
                        result["key_cases"].append(dict(gradient_cap=cap, attack_seed=seed, checkpoint_seed=2, attack=name,
                            **{k: e[k] for k in ("episode", "sumo_seed", "ego_collision_observed", "steps", "episode_return",
                                                "gradient_evaluations", "attack_policy_forward_calls")}))
        for name in ATTACKS:
            result["pooled"].append(dict(gradient_cap=cap, attack=name,
                **aggregate([r for seed in range(3) for r in cells[cap, seed] if r["attack"] == name])))
    fresh_high = next(d for d in high_reports.values() if d["attack_seed"] == 0 and
                      any(r["attack"].startswith("zero_one_budgeted_") for r in d["rows"]))
    new = lows + [fresh_high]
    protocol = json.loads((ROOT / "configs/research/shared_compute_budget_sweep.json").read_text())
    count = sum(sum(r["episodes"] for r in d["rows"]) for d in new)
    require(count == protocol["additional_episodes"], "New episode count differs from preregistration")
    require(len({d["git_commit"] for d in new}) == 1, "New experiments span multiple source commits")
    result["new_experiments"] = dict(episodes=count, batches=len(new), git_commit=new[0]["git_commit"],
        raw_verified_steps=sum(d["raw_episode_verified_steps"] for d in new),
        clean_regression_steps=sum(d["clean_regression"]["verified_steps"] for d in new))
    result["summary_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit summarizer before publishing")
    return result


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, objective in enumerate(("return", "safety")):
        for prefix, label, color in (("zero_one_budgeted_", "Budgeted Zero-One", "#2864ad"),
                                      ("ours_progress_", "Progress retry", "#cc5928")):
            rows = [r for r in report["pooled"] if r["attack"] == prefix + objective]
            xs = [r["costs"]["gradient_evaluations"] / r["episodes"] for r in rows]
            for ax, values in ((axes[0, col], [100 * r["conversion_rate"] for r in rows]),
                               (axes[1, col], [r["mean_return"] for r in rows])):
                ax.plot(xs, values, "o-", color=color, label=label)
                for x, y, r in zip(xs, values, rows):
                    ax.annotate(str(r["gradient_cap"]), (x, y), xytext=(4, 6), textcoords="offset points", fontsize=8, color=color)
                ax.grid(alpha=.2)
            axes[0, col].set_title(objective.capitalize() + " objective")
        axes[1, col].set_xlabel("Actual gradient evaluations / episode")
        axes[0, col].legend(fontsize=8, loc="lower right")
    axes[0, 0].set_ylabel("Collision conversion (%)")
    axes[1, 0].set_ylabel("Mean episode return")
    fig.suptitle("Shared computation budgets: development set only", fontsize=14)
    fig.text(.5, .015, "Stronger attack: higher conversion, lower return. Labels: gradient cap (forward cap = 2x).\n"
             "3 correlated attack seeds; no confidence intervals. Cost includes early termination effects.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs=6, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    report = summarize([p.resolve() for p in args.reports])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figure:
        plot(report, args.figure)
    print("BUDGET_GRID_SUMMARY_PASSED cells=%d new_episodes=%d" % (len(report["cells"]), report["new_experiments"]["episodes"]))
