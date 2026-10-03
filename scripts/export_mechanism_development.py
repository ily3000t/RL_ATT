"""Export a compact mechanism report and descriptive scientific figures."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from summarize_mechanism_development import summarize, ROOT, require


def compact(path):
    full = json.loads(path.read_text(encoding="utf-8"))
    require(full["verified"] and full["kind"] == "return_mechanism_development_summary", "Unverified mechanism summary")
    rebuilt = json.loads(json.dumps(summarize(ROOT / full["pipeline"]), allow_nan=False))
    require({k: v for k, v in full.items() if k != "summary_commit"} ==
            {k: v for k, v in rebuilt.items() if k != "summary_commit"}, "Published mechanism summary differs from audited inputs")
    result = copy.deepcopy(full)
    for budget in result["budgets"]:
        for key in ("attack_seed_rows", "checkpoint_rows"):
            budget[key] = [{k: v for k, v in row.items() if k in
                           ("attack_seed", "checkpoint_seed", "attack", "episodes", "live_steps", "mean_return", "collisions",
                            "eligible_clean_noncollision", "collision_conversions", "conversion_rate", "costs",
                            "oracle_setup_shadow_steps", "physical_shadow_steps_with_setup")} for row in budget[key]]
    pipeline = json.loads((ROOT / full["pipeline"]).read_text())
    manifest_path = Path(pipeline["groups"][0]["batch"]).parent / "run-0/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    result.update(kind="return_mechanism_development_results", full_summary=path.relative_to(ROOT).as_posix(),
                  full_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  export_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip(),
                  runtime=json.loads(manifest["python_runtime"]["stdout"]),
                  sumo_version=manifest["sumo_version"]["stdout"].splitlines()[0], host_os=manifest["host_os"],
                  runtime_manifest=manifest_path.relative_to(ROOT).as_posix(),
                  runtime_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  launch_command=pipeline["command"], started_at_utc=pipeline["started_at_utc"],
                  finished_at_utc=pipeline["finished_at_utc"], gate_enabled=False, final_split_used=False)
    return result


def plot(report, path, methods=None, contrast_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if methods is None:
        methods = (("zero_one_budgeted_return", "Budgeted Zero-One", "#2864ad", "o"),
               ("ours_single_return", "Single attempt", "#24865d", "s"),
               ("ours_return", "Fixed retries", "#a35caa", "^"),
               ("ours_progress_return", "Progress retries", "#d56a24", "D"))
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    xs = list(range(3))
    panels = ((axes[0, 0], "Mean episode return", lambda r: r["mean_return"]),
              (axes[0, 1], "Collision conversion (%)", lambda r: 100 * r["conversion_rate"]),
              (axes[0, 2], "Gradient evaluations / episode", lambda r: r["costs_per_episode"]["gradient_evaluations"]),
              (axes[1, 0], "Policy forward calls / episode", lambda r: r["costs_per_episode"]["policy_forward_calls"]),
              (axes[1, 1], "Physical shadow steps / episode", lambda r: r["physical_shadow_steps_with_setup"] / r["episodes"]))
    for name, label, color, marker in methods:
        rows = [next(r for r in b["conditions"] if r["attack"] == name) for b in report["budgets"]]
        for ax, title, value in panels:
            ax.plot(xs, [value(r) for r in rows], marker=marker, label=label, color=color, linewidth=1.5, markersize=5)
            ax.set_title(title, fontsize=11)
    ax = axes[1, 2]
    if contrast_labels is None:
        contrast_labels = ("Single - Zero-One", "Fixed - Single", "Progress - Fixed", "Progress - Zero-One")
    for index, (label, style) in enumerate(zip(contrast_labels, ("o-", "s--", "^:", "D-."))):
        ax.plot(xs, [b["contrasts"][index]["mean_return_delta"] for b in report["budgets"]], style, label=label, markersize=5)
    ax.axhline(0, color="#666666", linewidth=.8)
    ax.set_title("Paired return delta (first - second)", fontsize=11)
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xticks(xs)
        ax.set_xticklabels(["100/200", "200/400", "400/800"])
        ax.set_xlabel("Gradient / forward cap per block", fontsize=9)
        ax.grid(alpha=.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 1].set_ylim(bottom=0)
    fig.suptitle(report.get("figure_title", "Return mechanism controls: reused development traffic"), fontsize=15, y=.98)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(methods), loc="upper center", bbox_to_anchor=(.5, .947), fontsize=10, frameon=False)
    clean_return = report["budgets"][0]["conditions"][0]["mean_return"]
    fig.text(.5, .025, "5 frozen models, %d traffic seeds, 3 correlated attack seeds; no confidence intervals or final-test claim.\n"
             "Lower return / higher conversion: stronger attack. Clean mean return: %.4f.\n"
             "Shadow includes replay, resets and episode setup; total costs include early termination effects." % (report["unique_traffic_episodes"], clean_return),
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .11, 1, .90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    report = compact(args.summary.resolve())
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.figure:
        plot(report, args.figure)
    print("MECHANISM_EXPORT_PASSED episodes=%d raw_steps=%d" % (report["episodes"], report["raw_verified_steps"]))
