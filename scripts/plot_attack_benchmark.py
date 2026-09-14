"""Scientific run-level attack comparison; raw trials shown with mean +/- SD."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    methods = ["none", "random", "fgsm", "pgd", "oarl_bo"]
    labels = ["Clean", "Random", "FGSM\nmargin", "PGD\nmargin", "OARL-BO"]
    panels = [("episode_return_mean", "Episode return", 1), ("return_drop_mean", "Paired return drop", 1),
              ("collision_rate", "SUMO collision episodes (%)", 100), ("attack_success_rate", "Collision-conversion ASR (%)", 100),
              ("ttc_low_percentile_s", "Conditional TTC p05 (s)", 1), ("drac_high_percentile_mps2", "DRAC p95 (m/s²)", 1)]
    colors = ["#486581", "#809c73", "#d79745", "#a65353", "#6d63a8"]
    if "zero_one" in data["aggregate"]:
        methods.append("zero_one")
        labels.append("Zero-One\nadapter")
        colors.append("#29898a")
    fig, axes = plt.subplots(2, 3, figsize=(15 if len(methods) == 6 else 13, 8))
    for ax, (metric, title, factor) in zip(axes.flat, panels):
        for i, method in enumerate(methods):
            values = []
            for r in data["runs"]:
                if r["attack"] == method:
                    group = r["summary"]["safety"] if metric in r["summary"]["safety"] else r["summary"]
                    if group[metric] is not None:
                        values.append(group[metric] * factor)
            aggregate = data["aggregate"][method][metric]
            if aggregate["mean"] is None:
                continue
            ax.scatter(i + np.linspace(-0.13, 0.13, len(values)), values, color=colors[i], alpha=0.45, s=24)
            ax.errorbar(i, aggregate["mean"] * factor, yerr=aggregate["sample_sd"] * factor,
                        fmt="o", color=colors[i], capsize=5, markersize=7, linewidth=1.6)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Frozen Clean Victim: five-seed attack benchmark", fontsize=16, y=0.98)
    if "zero_one" in data["aggregate"]:
        fig.text(0.5, 0.012, "Zero-One uses privileged simulator lookahead; information and query budgets are not equalized.",
                 ha="center", fontsize=8, color="#28696a")
    fig.text(0.5, 0.04, "Points: run-level values; error bars: sample SD. Common bound: |δᵢ| ≤ 0.2|sᵢ| + 0.05.\n"
             "BO retains a shared affine subspace. SUMO events may include minGap violations; TTC/DRAC cover longitudinal pairs only.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=180)
    fig.savefig(str(args.output.with_suffix(".svg")))
    plt.close(fig)


if __name__ == "__main__":
    main()
