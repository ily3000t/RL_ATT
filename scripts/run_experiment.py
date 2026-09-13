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
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("A clean, committed workspace is required")
    config_path = args.config.resolve()
    relative = config_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))
    if committed.replace(b"\r\n", b"\n") != config_path.read_bytes().replace(b"\r\n", b"\n"):
        parser.error("Configuration differs from its Git commit")
    config = json.loads(committed)
    if config["training"]["seed"] != config["run_seed"] or config.get("gate_enabled") is not False:
        parser.error("Training seed must match run_seed, and Gate must be disabled")
    if config["victim"] not in ("oarl", "clean"):
        parser.error("This stage supports the OARL and matched clean victims")
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
    command = [executable, "-m", "rl_att.experiment", "--manifest", str(manifest_path)]
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
    write_json(manifest_path, manifest)
    manifest["pip_freeze"] = capture([executable, "-m", "pip", "freeze", "--all"], env)
    manifest["sumo_version"] = capture(["sumo", "--version"], env)
    manifest["python_runtime"] = capture([executable, "-c", "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"], env)
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    print("RUN_MANIFEST=" + str(manifest_path), flush=True)
    code = 1
    try:
        with (run_dir / "stdout.log").open("wb") as out, (run_dir / "stderr.log").open("wb") as err:
            result = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                                    timeout=args.timeout, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
    finally:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(returncode=code, status="passed" if code == 0 else "timeout" if code == 124 else "failed",
                        finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        git_worktree_clean_after=not bool(git("status", "--porcelain")))
        manifest["source_sha256_after"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                           for name in initial_hashes if (source / name).is_file()}
        write_json(manifest_path, manifest)
    print("RUN_STATUS=" + manifest["status"], flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
