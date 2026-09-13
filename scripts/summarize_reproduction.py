"""Build a small audited summary from a complete five-seed batch and evaluations."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
from run_baseline import ROOT, git, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--evaluations", type=Path, nargs=5, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch = args.batch.resolve()
    batch_info = read(batch / "batch.json")
    if batch_info["status"] != "passed" or len(batch_info["runs"]) != 5:
        raise ValueError("Expected a complete successful five-run batch")
    evaluations = {}
    for directory in args.evaluations:
        directory = directory.resolve()
        manifest = read(directory / "manifest.json")
        result = read(directory / "evaluation.json")
        if manifest["status"] != "passed" or result["run_seed"] in evaluations:
            raise ValueError("Failed or duplicate evaluation")
        evaluations[result["run_seed"]] = (directory, manifest, result)
    summaries = []
    for run_record in batch_info["runs"]:
        run = Path(run_record["run_dir"])
        manifest = read(run / "manifest.json")
        config = manifest["config"]
        rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
        if manifest["status"] != "passed" or config["training"]["episodes"] != 400 or config["training"]["max_step"] != 200:
            raise ValueError("Incomplete or unexpected long-run configuration")
        if [row["episode"] for row in rows] != list(range(1, 401)):
            raise ValueError("Episode sequence is incomplete")
        if [row["sumo_seed"] for row in rows] != manifest["effective_seeds"]["episode_sumo_seeds"]:
            raise ValueError("Recorded episode seeds differ from manifest")
        if any(not 1 <= row["steps"] <= 200 for row in rows):
            raise ValueError("Invalid episode length")
        for name, digest in manifest["source_sha256_before"].items():
            if name != "Data/StraightRoad.sumocfg" and manifest["source_sha256_after"].get(name) != digest:
                raise ValueError("Unexpected training source mutation: " + name)
        directory, eval_manifest, evaluation = evaluations[config["run_seed"]]
        if Path(eval_manifest["training_run"]).resolve() != run.resolve() or evaluation["training_commit"] != manifest["git_commit"]:
            raise ValueError("Evaluation does not correspond to this training run")
        if [item["episode"] for item in evaluation["checkpoints"]] != [100, 200, 300, 400]:
            raise ValueError("Missing checkpoint evaluation")
        checks = []
        for item in evaluation["checkpoints"]:
            proof = read(run / "checkpoints" / ("policy%d.json" % item["episode"]))
            if not item["frozen_unchanged"] or len(item["episodes"]) != 20:
                raise ValueError("Frozen evaluation is incomplete")
            if [row["sumo_seed"] for row in item["episodes"]] != evaluation["effective_seeds"]["episode_sumo_seeds"]:
                raise ValueError("Evaluation seed mismatch")
            actual_digest = hashlib.sha256(Path(proof["checkpoint_path"]).read_bytes()).hexdigest()
            if actual_digest != proof["checkpoint_sha256"] or actual_digest != item["verification"]["checkpoint_sha256"]:
                raise ValueError("Checkpoint hash changed")
            checks.append({"episode": item["episode"], "checkpoint": relative(Path(proof["checkpoint_path"])),
                           "checkpoint_sha256": actual_digest, "weights_sha256": proof["weights_sha256"],
                           "fresh_process_verified": True, "frozen_unchanged": True,
                           "evaluation_mean_return": item["mean_return"],
                           "evaluation_collision_episode_rate": item["collision_episode_rate"]})
        summaries.append({"run_seed": config["run_seed"], "victim": config["victim"],
                          "training_manifest": relative(run / "manifest.json"),
                          "evaluation_manifest": relative(directory / "manifest.json"),
                          "training_commit": manifest["git_commit"], "evaluation_commit": eval_manifest["git_commit"],
                          "config_file": manifest["config_file"], "episodes_completed": 400,
                          "interaction_steps": sum(row["steps"] for row in rows),
                          "training_updates": rows[-1]["training_updates_total"],
                          "bo_duplicate_proposals": rows[-1]["bo_duplicate_proposals_total"],
                          "js_float64_evaluations": rows[-1]["js_float64_evaluations_total"],
                          "bo_objective_evaluations": rows[-1]["training_updates_total"] * manifest["agent_defaults"]["attack_optimizing_times"] if config["victim"] == "oarl" else 0,
                          "training_return_mean": statistics.mean(row["episode_return"] for row in rows),
                          "last100_training_return_mean": statistics.mean(row["episode_return"] for row in rows[-100:]),
                          "training_collision_episode_rate": statistics.mean(row["ego_collision_observed"] for row in rows),
                          "last100_training_collision_episode_rate": statistics.mean(row["ego_collision_observed"] for row in rows[-100:]),
                          "last_training_js": rows[-1]["last_training_js"], "last_dual_multiplier": rows[-1]["dual_multiplier"],
                          "runtime": json.loads(manifest["python_runtime"]["stdout"]),
                          "sumo_version": manifest["sumo_version"]["stdout"].splitlines()[0],
                          "checkpoints": checks})
    summaries.sort(key=lambda row: row["run_seed"])
    if [row["run_seed"] for row in summaries] != [0, 1, 2, 3, 4]:
        raise ValueError("Expected run seeds 0 through 4")
    values = {
        "last100_training_return": [row["last100_training_return_mean"] for row in summaries],
        "final_checkpoint_evaluation_return": [row["checkpoints"][-1]["evaluation_mean_return"] for row in summaries],
        "final_checkpoint_evaluation_collision_rate": [row["checkpoints"][-1]["evaluation_collision_episode_rate"] for row in summaries],
    }
    report = {"scope": "Controlled Protocol A reproduction; no Gate; no attack benchmark or claim of paper numerical agreement",
              "summary_commit": git("rev-parse", "HEAD"), "command": [sys.executable, *sys.argv],
              "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "batch": relative(batch), "all_2000_episodes_completed": True, "all_20_checkpoints_verified": True,
              "statistics": "Mean and sample standard deviation of five run-level values; evaluation uses greedy argmax on held-out seeds",
              "aggregate": {name: {"mean": statistics.mean(data), "sample_sd": statistics.stdev(data)} for name, data in values.items()},
              "runs": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
