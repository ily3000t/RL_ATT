"""Launch frozen victim evaluations with a committed snapshot and full provenance."""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
from run_baseline import ROOT, SOURCE_PATHS, git, write_json, capture

sys.path.insert(0, str(ROOT))
from rl_att.evaluation.configuration import validate_config
from rl_att.utils.zero_one_dependency import dependency_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-commit", help="Reject a batch child launched after its Git commit changed")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit changes before evaluation")
    if args.expected_commit and git("rev-parse", "HEAD") != args.expected_commit:
        parser.error("Git commit changed since batch dispatch")
    relative = args.config.resolve().relative_to(ROOT).as_posix()
    config = validate_config(json.loads(subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))))
    references = json.loads(subprocess.check_output(
        ["git", "show", "HEAD:configs/frozen_victims.json"], cwd=str(ROOT)))["victims"]
    selected = [r for r in references if r["victim"] in config["victims"] and r["run_seed"] in config["run_seeds"]]
    keys = {(r["victim"], r["run_seed"]) for r in selected}
    if len(keys) != len(selected) or len(keys) != len(config["victims"]) * len(config["run_seeds"]):
        raise ValueError("Frozen registry must have exactly one checkpoint per requested victim and seed")
    trainings = []
    for reference in selected:
        path = (ROOT / reference["checkpoint"]).resolve()
        if ROOT.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != reference["checkpoint_sha256"]:
            raise ValueError("Invalid frozen checkpoint reference")
        training = json.loads((path.parents[2] / "manifest.json").read_text(encoding="utf-8"))
        if training["status"] != "passed" or training["git_commit"] != reference["training_commit"]:
            raise ValueError("Frozen training provenance mismatch")
        trainings.append(training)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output.resolve() if args.output else ROOT / ".local/runs" / (stamp + "-attack-evaluation")
    if (ROOT / ".local/runs").resolve() not in output.resolve().parents:
        parser.error("Output must stay in the ignored .local/runs directory")
    source = output / "source"
    source.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", *SOURCE_PATHS, "rl_att"], cwd=str(ROOT))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            target = (source / member.name).resolve()
            if source.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe source archive member")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    for training in trainings:
        for name in training["source_sha256_before"]:
            if name in SOURCE_PATHS or name.startswith(("Environment/", "Data/")):
                if hashes.get(name) != training["source_sha256_before"][name]:
                    raise ValueError("Victim/environment source differs from training: " + name)
    training = trainings[0]
    env = os.environ.copy()
    env.update(training["environment_overrides"])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    env["PATH"] = os.pathsep.join(training["runtime_path_prepend"] + [env.get("PATH", "")])
    extra_dependencies = []
    if any(a["name"] == "zero_one" for a in config["attacks"]):
        dependency = dependency_record(ROOT / ".local/dependencies/zero-one/zoopt-0.4.2-py3-none-any.whl")
        env["RL_ATT_ZOOPT_WHEEL"] = dependency["path"]
        extra_dependencies.append(dependency)
    python = training["launch_command"][0]
    manifest_path = output / "manifest.json"
    command = [python, "-m", "rl_att.evaluation.run", "--manifest", str(manifest_path)]
    manifest = {"kind": "frozen_attack_evaluation", "git_commit": git("rev-parse", "HEAD"),
                "git_branch": git("branch", "--show-current"), "root": str(ROOT),
                "config_file": relative, "config": config, "victim_references": selected,
                "extra_dependencies": extra_dependencies,
                "launch_command": command, "wrapper_command": [sys.executable, *sys.argv], "cwd": str(source),
                "environment_overrides": {key: env[key] for key in training["environment_overrides"]},
                "runtime_path_prepend": training["runtime_path_prepend"], "host_os": platform.platform(),
                "SUMO_HOME": env.get("SUMO_HOME"), "source_sha256_before": hashes,
                "python_runtime": capture([python, "-c", "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"], env),
                "pip_freeze": capture([python, "-m", "pip", "freeze", "--all"], env),
                "sumo_version": capture(["sumo", "--version"], env), "started_at_utc": stamp, "status": "preparing"}
    write_json(manifest_path, manifest)
    for previous in trainings:
        for key in ("python_runtime", "pip_freeze", "sumo_version"):
            if manifest[key]["returncode"] != 0 or manifest[key]["stdout"] != previous[key]["stdout"]:
                raise ValueError("Runtime differs from frozen victim training: " + key)
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    print("EVALUATION_DIR=" + str(output), flush=True)
    code = 1
    try:
        with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
            code = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).returncode
    finally:
        after = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in hashes}
        changed = [name for name in hashes if after[name] != hashes[name]]
        if any(name != "Data/StraightRoad.sumocfg" for name in changed):
            code = 1
        manifest.update(status="passed" if code == 0 else "failed", returncode=code,
                        source_sha256_after=after, changed_source_files=changed,
                        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(manifest_path, manifest)
    print("EVALUATION_STATUS=" + manifest["status"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
