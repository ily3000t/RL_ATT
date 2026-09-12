"""Export descriptive learning/evaluation curves from completed run summaries."""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_baseline import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    window = 20
    for index, path in enumerate(args.summaries):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary["all_2000_episodes_completed"] or not summary["all_20_checkpoints_verified"]:
            raise ValueError("Curves require complete validated results")
        rows = []
        for run in summary["runs"]:
            directory = (ROOT / run["training_manifest"]).parent
            rows.append([json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()])
        label = "OARL robust victim" if summary["runs"][0]["victim"] == "oarl" else "Clean victim"
        color = colors(index)
        for axis, field in zip(axes[0], ("episode_return", "ego_collision_observed")):
            values = np.array([[row[field] for row in seed] for seed in rows], dtype=float)
            smooth = np.array([np.convolve(seed, np.ones(window) / window, mode="valid") for seed in values])
            x = np.arange(window, 401)
            mean, sd = smooth.mean(0), smooth.std(0, ddof=1)
            axis.plot(x, mean, color=color, label=label)
            lower, upper = mean - sd, mean + sd
            if field == "ego_collision_observed":
                lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
            axis.fill_between(x, lower, upper, color=color, alpha=0.16)
        for axis, field in zip(axes[1], ("evaluation_mean_return", "evaluation_collision_episode_rate")):
            values = np.array([[checkpoint[field] for checkpoint in run["checkpoints"]] for run in summary["runs"]])
            axis.errorbar([100, 200, 300, 400], values.mean(0), yerr=values.std(0, ddof=1),
                          marker="o", capsize=3, color=color, label=label)
    axes[0, 0].set(title="Training return (20-episode moving mean)", ylabel="Episode return")
    axes[0, 1].set(title="Training: observed ego collision episodes", ylabel="Episode fraction", ylim=(0, 1))
    axes[1, 0].set(title="Frozen checkpoint: held-out greedy evaluation", ylabel="Mean episode return")
    axes[1, 1].set(title="Frozen checkpoint: observed ego collisions", ylabel="Episode fraction", ylim=(-0.05, 1.05))
    for axis in axes[0]:
        axis.set_xlabel("Completed training episode")
    for axis in axes[1]:
        axis.set_xlabel("Checkpoint training episode")
        axis.set_xticks([100, 200, 300, 400])
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Protocol A: five run seeds; shading/error bars = sample SD across runs\nNo Gate; no observation attack during evaluation", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.output), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
