"""Precommitted short Clean/OARL original-loop and fresh-process resume checks."""

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

from run_baseline import ROOT, SOURCE_PATHS, capture, git, write_json

sys.path.insert(0, str(ROOT))
from rl_att.training.state import require
from rl_att.training.support_audit import audit_run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/defense_training_support.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not git("status", "--porcelain"), "Commit support source, protocol and auditor before experiments")
    commit = git("rev-parse", "HEAD")
    config_path = args.config.resolve()
    relative = config_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))
    require(committed.replace(b"\r\n", b"\n") == config_path.read_bytes().replace(b"\r\n", b"\n"), "Uncommitted support configuration")
    protocol = json.loads(committed)
    require(protocol["engineering_only"] and not protocol["gate_enabled"] and protocol["victims"] == ["clean", "oarl"] and
            11 < protocol["split_after_episode"] < protocol["episodes"], "Only registered short matched support checks")
    output = args.output.resolve()
    require((ROOT / ".local/runs").resolve() in output.parents and not output.exists(), "Use a new ignored run directory")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    frozen_path = ROOT / "configs/frozen_victims.json"
    refs = json.loads(frozen_path.read_text(encoding="utf-8"))["victims"]
    reference = next(r for r in refs if r["victim"] == "clean" and r["run_seed"] == 0)
    training_path = (ROOT / reference["checkpoint"]).parents[2] / "manifest.json"
    frozen = json.loads(training_path.read_text(encoding="utf-8"))
    require(frozen["status"] == "passed" and frozen["git_commit"] == reference["training_commit"], "Invalid frozen runtime provenance")
    env = os.environ.copy()
    env.update(frozen["environment_overrides"])
    env["MPLCONFIGDIR"], env["SUMO_HOME"] = str(output / "matplotlib"), frozen["SUMO_HOME"]
    env["PATH"] = os.pathsep.join(frozen["runtime_path_prepend"] + [env.get("PATH", "")])
    python = frozen["launch_command"][0]
    manifest = dict(kind=protocol["kind"], status="preparing", git_commit=commit,
                    git_branch=git("branch", "--show-current"), root=str(ROOT), protocol=protocol,
                    protocol_sha256=sha(config_path), config_file=relative,
                    wrapper_command=[sys.executable, *sys.argv], host_os=platform.platform(),
                    frozen_runtime_manifest=dict(path=str(training_path), sha256=sha(training_path)),
                    frozen_registry_sha256_before=sha(frozen_path), runtime_path_prepend=frozen["runtime_path_prepend"],
                    environment_overrides={key: env[key] for key in frozen["environment_overrides"]},
                    SUMO_HOME=env["SUMO_HOME"], started_at_utc=datetime.datetime.utcnow().isoformat() + "Z", jobs=[])
    write_json(manifest_path, manifest)
    try:
        commands = dict(python_runtime=[python, "-c", "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"],
                        pip_freeze=[python, "-m", "pip", "freeze", "--all"], sumo_version=["sumo", "--version"])
        for key, cmd in commands.items():
            manifest[key] = capture(cmd, env)
            require(manifest[key]["returncode"] == 0 and manifest[key]["stdout"] == frozen[key]["stdout"], "Frozen runtime mismatch: " + key)
        archive = subprocess.check_output(["git", "archive", "--format=tar", commit, *SOURCE_PATHS, "rl_att"], cwd=str(ROOT))
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        print("TRAINING_SUPPORT_DIR=" + str(output), flush=True)
        for victim in protocol["victims"]:
            training_config = dict(victim=victim, protocol="controlled", run_seed=protocol["run_seed"], sumo_schedule="derived",
                  cpu_threads=1, gate_enabled=False, training=dict(env="highway-v0", algo=victim, seed=protocol["run_seed"],
                  episodes=protocol["episodes"], max_step=protocol["max_steps"], state_dim=16, action_dim=1, action_numb=3,
                  mode="train", save_dir_model="model/", save_dir_data="result/", save_dir_train_data="train/"))
            identity = dict(git_commit=commit, protocol_sha256=manifest["protocol_sha256"], victim=victim,
                            run_seed=protocol["run_seed"], frozen_runtime_manifest_sha256=sha(training_path))
            prefix_checkpoint = None
            for mode in ("reference", "continuous", "prefix", "resumed"):
                require(not git("status", "--porcelain") and git("rev-parse", "HEAD") == commit, "Source changed during support batch")
                directory = output / (victim + "-" + mode)
                source = directory / "source"
                source.mkdir(parents=True)
                with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                    for member in tar.getmembers():
                        target = (source / member.name).resolve()
                        require(source.resolve() in target.parents and (member.isdir() or member.isfile()), "Unsafe source archive member")
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(tar.extractfile(member).read())
                hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*") if p.is_file()}
                for name, digest in frozen["source_sha256_before"].items():
                    if name in SOURCE_PATHS or name.startswith(("Environment/", "Data/")):
                        require(hashes.get(name) == digest, "Original training source changed: " + name)
                child_path = directory / "manifest.json"
                command = [python, "-m", "rl_att.training.support_run", "--manifest", str(child_path)]
                child = dict(kind=protocol["kind"], status="running", mode=mode, config=training_config, identity=identity,
                       git_commit=commit, root=str(ROOT), cwd=str(source), split_after_episode=protocol["split_after_episode"],
                       launch_command=command, source_sha256_before=hashes,
                       started_at_utc=datetime.datetime.utcnow().isoformat() + "Z")
                if mode == "resumed":
                    child.update(resume_path=prefix_checkpoint["path"], resume_sha256=prefix_checkpoint["sha256"])
                write_json(child_path, child)
                with (directory / "stdout.log").open("wb") as out, (directory / "stderr.log").open("wb") as err:
                    code = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).returncode
                child = json.loads(child_path.read_text(encoding="utf-8"))
                after = {name: sha(source / name) for name in hashes}
                changed = [name for name in hashes if hashes[name] != after[name]]
                valid = code == 0 and all(name == "Data/StraightRoad.sumocfg" for name in changed)
                child.update(status="passed" if valid else "failed", returncode=code, source_sha256_after=after,
                             changed_source_files=changed, finished_at_utc=datetime.datetime.utcnow().isoformat() + "Z")
                write_json(child_path, child)
                manifest["jobs"].append(dict(victim=victim, mode=mode, directory=str(directory), identity=identity, returncode=code))
                write_json(manifest_path, manifest)
                require(valid, "Support worker failed: " + victim + "/" + mode + "; inspect stderr.log")
                if mode == "prefix":
                    prefix_checkpoint = json.loads((directory / "support_report.json").read_text(encoding="utf-8"))["boundary_checkpoint"]
                print("TRAINING_SUPPORT_COMPLETED %s %s" % (victim, mode), flush=True)
        report = audit_run(manifest_path)
        require(report["physical_episodes"] == protocol["expected_physical_episodes"], "Support episode count mismatch")
        write_json(output / "audit.json", report)
        require(sha(frozen_path) == manifest["frozen_registry_sha256_before"] and not git("status", "--porcelain") and
                git("rev-parse", "HEAD") == commit, "Source/frozen registry changed during support checks")
        manifest.update(status="passed", frozen_registry_sha256_after=sha(frozen_path), audit_sha256=sha(output / "audit.json"))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        write_json(manifest_path, manifest)
    print("TRAINING_SUPPORT_PASSED physical_episodes=%d" % report["physical_episodes"], flush=True)


if __name__ == "__main__":
    main()
