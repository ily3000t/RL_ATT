"""Launch one fully recorded experiment from a clean Git snapshot."""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

from run_baseline import ROOT, SOURCE_PATHS, capture, git, write_json, agent_defaults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=str(ROOT / ".local/envs/oarl-legacy/python.exe"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=28800)
    parser.add_argument("--expected-commit", help="Reject dispatch after a batch source commit changed")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("A clean, committed workspace is required")
    if args.expected_commit and git("rev-parse", "HEAD") != args.expected_commit:
        parser.error("Git source changed since batch dispatch")
    config_path = args.config.resolve()
    relative = config_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))
    if committed.replace(b"\r\n", b"\n") != config_path.read_bytes().replace(b"\r\n", b"\n"):
        parser.error("Configuration differs from its Git commit")
    config = json.loads(committed)
    if config["training"]["seed"] != config["run_seed"] or config.get("gate_enabled") is not False:
        parser.error("Training seed must match run_seed, and Gate must be disabled")
    if config["victim"] not in ("oarl", "clean", "pgd_consistency"):
        parser.error("Unknown training victim")
    defense_training = config["victim"] == "pgd_consistency"
    if defense_training:
        sys.path.insert(0, str(ROOT))
        from rl_att.training.protocol import validate_config
        validate_config(config)
    for key in ("save_dir_model", "save_dir_data", "save_dir_train_data"):
        path = Path(config["training"][key])
        if path.is_absolute() or path.drive or ".." in path.parts:
            parser.error("All artifacts must stay in the run snapshot")
    executable = shutil.which(args.python)
    if executable is None:
        parser.error("Python executable not found")
    executable = str(Path(executable).resolve())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = args.output.resolve() if args.output else ROOT / ".local/runs" / (stamp + "-" + config["victim"] + "-seed" + str(config["run_seed"]))
    if (ROOT / ".local/runs").resolve() not in run_dir.parents:
        parser.error("Output must be inside the repository's ignored .local/runs directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    source = run_dir / "source"
    source.mkdir()
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
    initial_hashes = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in source.rglob("*") if p.is_file()}
    prefix = Path(executable).parent
    runtime_paths = [str(p) for p in (prefix, prefix / "Library/bin", prefix / "Scripts") if p.is_dir()]
    overrides = {
        "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(run_dir / "matplotlib"),
        "PYTHONHASHSEED": str(config["run_seed"]), "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(config["cpu_threads"]), "MKL_NUM_THREADS": str(config["cpu_threads"]),
        "OPENBLAS_NUM_THREADS": str(config["cpu_threads"]),
    }
    env = os.environ.copy()
    env.update(overrides)
    env["PATH"] = os.pathsep.join(runtime_paths + [env.get("PATH", "")])
    manifest_path = run_dir / "manifest.json"
    command = [executable, "-m", "rl_att.training.experiment" if defense_training else "rl_att.experiment", "--manifest", str(manifest_path)]
    manifest = {
        "kind": "protocol_a_training", "git_commit": git("rev-parse", "HEAD"),
        "git_tree": git("rev-parse", "HEAD^{tree}"), "git_branch": git("branch", "--show-current"),
        "upstream_commit": "29e5c0e2497cd0bd27b6cf83c5bce800a3e2c54a",
        "started_at_utc": stamp, "config_file": relative, "config": config,
        "launch_command": command, "wrapper_command": [sys.executable, *sys.argv],
        "original_main_argv": ["main.py"] + [item for key, value in config["training"].items() for item in ("--" + key, str(value))],
        "cwd": str(source), "host_os": platform.platform(), "SUMO_HOME": env.get("SUMO_HOME"),
        "environment_overrides": overrides, "runtime_path_prepend": runtime_paths,
        "source_sha256_before": initial_hashes, "agent_defaults": agent_defaults((source / "oarl.py").read_text(encoding="utf-8")),
        "git_worktree_clean_before": True, "status": "preparing", "timeout_seconds": args.timeout,
    }
    if defense_training:
        manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        frozen_refs = json.loads((ROOT / "configs/frozen_victims.json").read_text(encoding="utf-8"))["victims"]
        frozen_ref = next(r for r in frozen_refs if r["victim"] == "clean" and r["run_seed"] == config["run_seed"])
        frozen_path = (ROOT / frozen_ref["checkpoint"]).parents[2] / "manifest.json"
        frozen_runtime = json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_runtime["status"] != "passed" or frozen_runtime["git_commit"] != frozen_ref["training_commit"]:
            raise ValueError("Invalid matching Clean runtime provenance")
        manifest["matching_clean_reference"] = frozen_ref
        manifest["frozen_runtime_manifest"] = dict(path=str(frozen_path), sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest())
        for name, expected in frozen_runtime["source_sha256_before"].items():
            if name in SOURCE_PATHS or name.startswith(("Environment/", "Data/")):
                if initial_hashes.get(name) != expected:
                    raise ValueError("Original training source changed: " + name)
    write_json(manifest_path, manifest)
    manifest["pip_freeze"] = capture([executable, "-m", "pip", "freeze", "--all"], env)
    manifest["sumo_version"] = capture(["sumo", "--version"], env)
    manifest["python_runtime"] = capture([executable, "-c", "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"], env)
    if defense_training:
        for key in ("python_runtime", "pip_freeze", "sumo_version"):
            if manifest[key]["returncode"] != 0 or manifest[key]["stdout"] != frozen_runtime[key]["stdout"]:
                raise ValueError("Use the frozen Clean training runtime: " + key)
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    print("RUN_MANIFEST=" + str(manifest_path), flush=True)
    code = 1
    try:
        with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
            result = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                                    timeout=args.timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        code = result.returncode
        if code == 0 and defense_training:
            sys.path.insert(0, str(ROOT))
            from rl_att.training.production_audit import audit_run
            audit = audit_run(manifest_path)
            write_json(run_dir / "training_audit.json", audit)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training_audit_sha256"] = hashlib.sha256((run_dir / "training_audit.json").read_bytes()).hexdigest()
            write_json(manifest_path, manifest)
    except subprocess.TimeoutExpired:
        code = 124
    except Exception:
        code = 1
        raise
    finally:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(returncode=code, status="passed" if code == 0 else "timeout" if code == 124 else "failed",
                        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        git_worktree_clean_after=not bool(git("status", "--porcelain")))
        manifest["source_sha256_after"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                           for name in initial_hashes if (source / name).is_file()}
        if defense_training:
            changed = [name for name in initial_hashes if manifest["source_sha256_after"].get(name) != initial_hashes[name]]
            if any(name != "Data/StraightRoad.sumocfg" for name in changed):
                code = 1
                manifest.update(status="failed", returncode=code, error="Unexpected tracked source mutation")
            if git("rev-parse", "HEAD") != manifest["git_commit"] or git("status", "--porcelain"):
                code = 1
                manifest.update(status="failed", returncode=code, error="Source commit/tree changed during training")
        write_json(manifest_path, manifest)
    print("RUN_STATUS=" + manifest["status"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
