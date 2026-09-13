"""Export hash-pinned episode-400 victim references from verified summaries."""

import argparse
import hashlib
import json
from pathlib import Path
from run_baseline import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    victims = []
    identities = set()
    for path in args.summaries:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report["all_2000_episodes_completed"] or not report["all_20_checkpoints_verified"]:
            raise ValueError("Victim export requires complete validated results")
        for run in report["runs"]:
            checkpoint = next(item for item in run["checkpoints"] if item["episode"] == 400)
            actual = hashlib.sha256((ROOT / checkpoint["checkpoint"]).read_bytes()).hexdigest()
            if actual != checkpoint["checkpoint_sha256"]:
                raise ValueError("Checkpoint file changed since verification")
            identity = (run["victim"], run["run_seed"])
            if identity in identities:
                raise ValueError("Duplicate victim identity")
            identities.add(identity)
            victims.append({"victim": run["victim"], "run_seed": run["run_seed"],
                            "training_episode": 400, "checkpoint": checkpoint["checkpoint"],
                            "checkpoint_sha256": actual, "weights_sha256": checkpoint["weights_sha256"],
                            "training_commit": run["training_commit"], "training_config": run["config_file"],
                            "verification_summary": path.resolve().relative_to(ROOT).as_posix()})
    result = {"schema_version": 1, "protocol": "controlled", "checkpoint_selection": "episode_400_predeclared",
              "checkpoint_type": "inference_actor_only", "path_base": "repository_root",
              "load_contract": "Verify the file hash, load using the pinned runtime, set eval mode and requires_grad_(False); do not train these checkpoints",
              "victims": sorted(victims, key=lambda item: (item["victim"], item["run_seed"]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
