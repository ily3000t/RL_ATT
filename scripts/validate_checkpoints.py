"""Evaluate local checkpoints in a fresh process and isolated SUMO snapshot."""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from run_baseline import ROOT, SOURCE_PATHS, git, write_json, capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/checkpoint_evaluation.json")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit changes before evaluation")
    run = args.run.resolve()
    training = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if training["status"] != "passed":
        parser.error("Training must complete successfully before validation")
    relative = args.config.resolve().relative_to(ROOT).as_posix()
    config = json.loads(subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT)))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / ".local/runs" / (stamp + "-checkpoint-evaluation-seed" + str(training["config"]["run_seed"]))
    source = output / "source"
    source.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", *SOURCE_PATHS, "rl_att"], cwd=str(ROOT))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            target = (source / member.name).resolve()
            if source.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe archive member")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    # Evaluation may add instrumentation, but its victim/environment must match training.
    for name in ("oarl.py", "Environment/environment/envs/highway_env.py"):
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != training["source_sha256_before"][name]:
            raise ValueError("Evaluation source differs from the training victim/environment")
    env = os.environ.copy()
    env.update(training["environment_overrides"])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    env["PATH"] = os.pathsep.join(training["runtime_path_prepend"] + [env.get("PATH", "")])
    python = training["launch_command"][0]
    manifest_path = output / "manifest.json"
    command = [python, "-m", "rl_att.checkpoint_evaluation", "--manifest", str(manifest_path)]
    manifest = {"kind": "frozen_checkpoint_evaluation", "git_commit": git("rev-parse", "HEAD"),
                "training_run": str(run), "config": config, "config_file": relative,
                "launch_command": command, "wrapper_command": [sys.executable, *sys.argv], "cwd": str(source),
                "environment_overrides": training["environment_overrides"], "SUMO_HOME": env.get("SUMO_HOME"),
                "pip_freeze": capture([python, "-m", "pip", "freeze", "--all"], env),
                "sumo_version": capture(["sumo", "--version"], env), "started_at_utc": stamp, "status": "running"}
    write_json(manifest_path, manifest)
    print("EVALUATION_DIR=" + str(output), flush=True)
    with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
        result = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    manifest.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode,
                    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(manifest_path, manifest)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
