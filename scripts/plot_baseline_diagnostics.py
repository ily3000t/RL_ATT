"""Descriptive crossed checkpoint/traffic matrices; no significance claims."""

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
    panels = [("clean_return", "Clean return", 1), ("fgsm_return", "FGSM return", 1),
              ("return_drop", "Paired return drop", 1), ("clean_collision_rate", "Clean SUMO events (%)", 100),
              ("fgsm_collision_rate", "FGSM SUMO events (%)", 100), ("collision_increase", "Event-rate increase (percentage points)", 100)]
    fig, axes = plt.subplots(2,3,figsize=(14,8))
    for ax,(key,title,scale) in zip(axes.flat,panels):
        matrix = np.zeros((5,5))
        for cell in data["crossed_cells"]:
            value = cell.get(key,cell["fgsm_collision_rate"]-cell["clean_collision_rate"])
            matrix[cell["checkpoint_seed"],cell["traffic_group"]] = value*scale
        cmap = "YlOrRd" if key not in ("clean_return","fgsm_return") else "YlGnBu"
        limits = dict(vmin=0,vmax=100) if key.endswith("collision_rate") else {}
        im = ax.imshow(matrix,cmap=cmap,**limits)
        for y in range(5):
            for x in range(5):
                ax.text(x,y,"%.1f"%matrix[y,x],ha="center",va="center",
                        color="white" if im.norm(matrix[y,x])>.65 else "black",fontsize=10)
        ax.set(xticks=range(5),yticks=range(5),xlabel="Traffic seed group",ylabel="Checkpoint training seed",title=title)
        fig.colorbar(im,ax=ax,fraction=.045,pad=.03)
    fig.suptitle("Baseline diagnosis: checkpoint identity crossed with traffic seeds",fontsize=16)
    fig.text(.5,.02,"Each cell: 20 paired episodes, frozen models, unchanged FGSM-margin and perturbation budget.\n"
             "Diagonal matches the original benchmark. SUMO events may include minGap violations. Descriptive results, not ablations.",
             ha="center",fontsize=10)
    fig.tight_layout(rect=(0,.08,1,.96))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(str(args.output),dpi=170)
    fig.savefig(str(args.output.with_suffix(".svg")))
    plt.close(fig)


if __name__ == "__main__":
    main()
