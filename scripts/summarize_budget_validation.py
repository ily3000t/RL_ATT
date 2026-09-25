"""Summarize the complete validation grid and preregistered traffic concentration."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from summarize_proposed_smoke import ROOT, require
from summarize_budget_sweep import ATTACKS, CAPS, aggregate
from analyze_proposed_development import pair_outcomes


def validate_grid(reports):
    keys = [(r["gradient_cap"], r["attack_seed"]) for r in reports]
    require(len(keys) == len(set(keys)), "Duplicate validation budget/seed")
    require(set(keys) == {(cap, seed) for cap in CAPS for seed in range(3)}, "Incomplete validation grid")


def concentration(pairs, traffic):
    """Cluster by traffic, retaining correlated replicates without claiming independence."""
    counts = {s: dict(episode=i + 1, sumo_seed=s, paired_records=0, eligible=0,
                     first_only_conversion=0, second_only_conversion=0,
                     first_one_step_collision=0, second_one_step_collision=0,
                     first_only_one_step_conversion=0, second_only_one_step_conversion=0)
              for i, s in enumerate(traffic)}
    require(len(counts) == len(traffic), "Duplicate traffic seed")
    seen = set()
    for pair in pairs:
        for e in pair["episodes"]:
            key = pair["attack_seed"], pair["checkpoint_seed"], e["episode"]
            require(key not in seen, "Duplicate paired record")
            seen.add(key)
            require(1 <= e["episode"] <= len(traffic) and traffic[e["episode"] - 1] == e["sumo_seed"], "Traffic order mismatch")
            row = counts[e["sumo_seed"]]
            row["paired_records"] += 1
            a, b = e["first_collision"], e["second_collision"]
            row["first_one_step_collision"] += int(a and e["first_steps"] == 1)
            row["second_one_step_collision"] += int(b and e["second_steps"] == 1)
            if not e["clean_collision"]:
                row["eligible"] += 1
                row["first_only_conversion"] += int(a and not b)
                row["second_only_conversion"] += int(b and not a)
                row["first_only_one_step_conversion"] += int(a and not b and e["first_steps"] == 1)
                row["second_only_one_step_conversion"] += int(b and not a and e["second_steps"] == 1)
    require(all(r["paired_records"] for r in counts.values()), "Missing traffic group")
    total = sum(r["first_only_conversion"] - r["second_only_conversion"] for r in counts.values())
    eligible = sum(r["eligible"] for r in counts.values())
    for row in counts.values():
        row["net_conversions"] = row["first_only_conversion"] - row["second_only_conversion"]
        row["net_without_this_traffic"] = total - row["net_conversions"]
        row["eligible_without_this_traffic"] = eligible - row["eligible"]
        row["rate_difference_without_this_traffic"] = (row["net_without_this_traffic"] / row["eligible_without_this_traffic"]
                                                      if row["eligible_without_this_traffic"] else None)
    return dict(net_conversions=total, eligible=eligible, traffic=list(counts.values()),
                leave_one_traffic_out_net_range=[min(r["net_without_this_traffic"] for r in counts.values()),
                                                max(r["net_without_this_traffic"] for r in counts.values())],
                interpretation="Descriptive concentration check; correlated attack seeds/models, not a module ablation or confidence interval")


def compact_pair(pair):
    result = {k: v for k, v in pair.items() if k != "episodes"}
    result["collision_discordances"] = [e for e in pair["episodes"] if e["first_collision"] != e["second_collision"]]
    return result


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

    pipeline = json.loads(pipeline_path.read_text())
    protocol_path = ROOT / "configs/research/shared_compute_validation.json"
    protocol = json.loads(protocol_path.read_text())
    require(pipeline["status"] == "passed" and pipeline["verified_episodes"] == protocol["total_episodes"] == 4500,
            "Validation pipeline incomplete")
    require(digest(protocol_path) == pipeline["protocol_sha256"], "Preregistered protocol changed")
    require([(g["gradient_cap"], g["attack_seed"]) for g in pipeline["groups"]] ==
            [(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]], "Pipeline group order/coverage differs")
    reports, inputs = [], []
    for group, registered in zip(pipeline["groups"], protocol["groups"]):
        require(group["status"] == "passed", "Group failed")
        path = Path(group["audit_path"])
        require(digest(path) == group["audit_sha256"], "Validation audit changed")
        report = json.loads(path.read_text())
        require(report["verified"] and report["kind"] == "shared_compute_validation" and report["research_split_id"] == 20,
                "Wrong validation report")
        require(report["git_commit"] == report["analysis_commit"] == pipeline["git_commit"], "Validation provenance differs")
        require((report["gradient_cap"], report["attack_seed"]) == (group["gradient_cap"], group["attack_seed"]) and
                report["forward_cap"] == 2 * group["gradient_cap"], "Wrong group budget/seed")
        require(report["raw_episode_verified_steps"] == group["raw_verified_steps"] and report["clean_regression"]["passed"],
                "Raw or Clean audit mismatch")
        batch_path = ROOT / report["batch"]
        require(batch_path.resolve() == Path(group["batch"]).resolve(), "Wrong batch reference")
        batch = json.loads(batch_path.read_text())
        require(batch["status"] == "passed" and batch["configs"] == registered["configs"], "Unregistered batch")
        require(report["source_sha256"], "Missing raw hashes")
        for name, expected in report["source_sha256"].items():
            require(digest(ROOT / name) == expected, "Audited input changed: " + name)
        reports.append(report)
        inputs.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path), batch=report["batch"],
                           gradient_cap=report["gradient_cap"], attack_seed=report["attack_seed"]))
    validate_grid(reports)
    by_cell = {(r["gradient_cap"], r["attack_seed"]): r for r in reports}
    traffic = reports[0]["traffic"]["episode_sumo_seeds"]
    require(traffic == json.loads((ROOT / "configs/research_seed_splits.json").read_text())["splits"]["20"], "Wrong traffic split")
    reference_clean = {r["checkpoint_seed"]: r for r in reports[0]["rows"] if r["attack"] == "none"}
    expected_keys = {(s, n) for s in range(5) for n in ("none",) + ATTACKS}
    for report in reports:
        require(report["traffic"] == reports[0]["traffic"], "Unpaired traffic")
        rows = report["rows"]
        require(len(rows) == 25 and {(r["checkpoint_seed"], r["attack"]) for r in rows} == expected_keys, "Missing model/attack")
        for row in rows:
            require(row["episodes"] == len(row["episode_rows"]) == 20 and
                    [e["episode"] for e in row["episode_rows"]] == list(range(1, 21)) and
                    [e["sumo_seed"] for e in row["episode_rows"]] == traffic, "Incomplete or unpaired episode series")
            reference = reference_clean[row["checkpoint_seed"]]
            require(row["checkpoint_sha256"] == reference["checkpoint_sha256"], "Frozen model differs")
            if row["attack"] == "none":
                require([e["trajectory_sha256"] for e in row["episode_rows"]] ==
                        [e["trajectory_sha256"] for e in reference["episode_rows"]], "Clean trajectory changed")
    result = dict(kind="shared_compute_validation_results", research_split_id=20, inputs=inputs,
                  pipeline=dict(path=str(pipeline_path.relative_to(ROOT)), sha256=digest(pipeline_path)),
                  experiment_commit=pipeline["git_commit"], protocol_sha256=digest(protocol_path),
                  episodes=4500, unique_clean_pairs=100, raw_verified_steps=sum(r["raw_episode_verified_steps"] for r in reports),
                  clean_regression_steps=sum(r["clean_regression"]["verified_steps"] for r in reports),
                  clean=[dict(checkpoint_seed=s, collisions=r["collisions"], episodes=r["episodes"], summary=r["summary"])
                         for s, r in sorted(reference_clean.items())],
                  cells=[], pooled=[], paired=[], upper_budget_pairs=[], concentration=[],
                  limitation="New traffic in the same scenario with the same frozen models; attack seeds/objectives are correlated. Final test unused.")
    all_pairs = []
    for cap in CAPS:
        for seed in range(3):
            report = by_cell[cap, seed]
            rows = {(r["checkpoint_seed"], r["attack"]): r for r in report["rows"]}
            for name in ATTACKS:
                selected = [rows[s, name] for s in range(5)]
                result["cells"].append(dict(gradient_cap=cap, attack_seed=seed, attack=name, **aggregate(selected),
                    checkpoints=[dict(checkpoint_seed=r["checkpoint_seed"], collisions=r["collisions"], summary=r["summary"],
                                      costs=r["audit"]["costs"], availability=r["candidate_availability"]) for r in selected]))
            for checkpoint in range(5):
                clean = rows[checkpoint, "none"]["episode_rows"]
                for objective in ("return", "safety"):
                    first, second = "ours_progress_" + objective, "zero_one_budgeted_" + objective
                    outcomes = pair_outcomes(clean, rows[checkpoint, first]["episode_rows"], rows[checkpoint, second]["episode_rows"])
                    existing = next(p for p in report["paired_contrasts"] if p["checkpoint_seed"] == checkpoint and p["first"] == first)
                    require(outcomes == {k: v for k, v in existing.items() if k not in ("checkpoint_seed", "first", "second")},
                            "Paired audit differs from recomputed outcomes")
                    pair = dict(gradient_cap=cap, attack_seed=seed, checkpoint_seed=checkpoint, objective=objective,
                                first=first, second=second, **outcomes)
                    all_pairs.append(pair)
                    result["paired"].append(compact_pair(pair))
                if cap != 400:
                    for name in ATTACKS:
                        upper = next(r for r in by_cell[400, seed]["rows"] if (r["checkpoint_seed"], r["attack"]) == (checkpoint, name))
                        result["upper_budget_pairs"].append(compact_pair(dict(gradient_cap=cap, reference_cap=400, attack_seed=seed,
                            checkpoint_seed=checkpoint, attack=name, **pair_outcomes(clean, rows[checkpoint, name]["episode_rows"], upper["episode_rows"]))))
        for name in ATTACKS:
            selected = [r for s in range(3) for r in by_cell[cap, s]["rows"] if r["attack"] == name]
            result["pooled"].append(dict(gradient_cap=cap, attack=name, **aggregate(selected)))
        for objective in ("return", "safety"):
            pairs = [p for p in all_pairs if p["gradient_cap"] == cap and p["objective"] == objective]
            result["concentration"].append(dict(gradient_cap=cap, objective=objective, **concentration(pairs, traffic)))
    result["summary_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)), "Commit summarizer before publishing")
    return result


def plot(report, directory):
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
                    ax.annotate(str(r["gradient_cap"]), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8, color=color)
                ax.grid(alpha=.2)
                ax.margins(.08)
            axes[0, col].set_title(objective.capitalize() + " objective")
        axes[1, col].set_xlabel("Actual gradient evaluations / episode")
        axes[0, col].legend(fontsize=8, loc="lower right")
    axes[0, 0].set_ylabel("Collision conversion (%)")
    axes[1, 0].set_ylabel("Mean episode return")
    fig.suptitle("Frozen methods on 20 new validation traffic seeds", fontsize=14)
    fig.text(.5, .015, "Labels: gradient cap (forward cap = 2x). Same five models; three correlated attack seeds.\n"
             "Stronger attack: higher conversion, lower return. Cost includes early termination effects.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .95))
    fig.savefig(str(directory / "validation-budget-curve.png"), dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True)
    for ax, cap in zip(axes, CAPS):
        for objective, offset, color in (("return", -.18, "#2864ad"), ("safety", .18, "#cc5928")):
            item = next(c for c in report["concentration"] if c["gradient_cap"] == cap and c["objective"] == objective)
            ax.bar([r["episode"] + offset for r in item["traffic"]], [r["net_conversions"] for r in item["traffic"]],
                   width=.36, label=objective.capitalize(), color=color)
        ax.axhline(0, color="black", linewidth=.7)
        ax.set_ylabel("Net conversions")
        ax.set_title("Gradient / forward cap: %d / %d" % (cap, cap * 2), fontsize=11)
        ax.set_yticks([-3, -2, -1, 0, 1, 2, 3, 4])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(list(range(1, 21)))
    axes[-1].set_xlabel("Validation traffic episode (same traffic shared by all models and attack seeds)")
    fig.suptitle("Paired collision gains and losses by traffic", fontsize=14)
    fig.text(.5, .01, "Positive = Progress-only conversion; negative = Zero-One-only conversion. Counts include correlated replicates.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    fig.savefig(str(directory / "validation-traffic-concentration.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    args = parser.parse_args()
    result = summarize(args.pipeline.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        plot(result, args.figures)
    print("VALIDATION_SUMMARY_PASSED episodes=%d steps=%d" % (result["episodes"], result["raw_verified_steps"]))
