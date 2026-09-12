"""Run a committed OARL snapshot and record provenance without editing its source.

This records runtime checks as well as experiments. Raw outputs stay in .local/.
The 12 x 2 smoke configuration reaches exactly one train_model call if all steps
succeed. It does not establish convergence, safety or paper reproduction.
"""

import argparse
import ast
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


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ["main.py", "oarl.py", "Environment", "Data", "requirements.txt"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=str(ROOT)).decode("utf-8").strip()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def capture(command, env):
    result = subprocess.run(command, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def agent_defaults(source):
    """Read default values without importing or initializing the algorithm."""
    tree = ast.parse(source)
    agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Agent")
    init = next(node for node in agent.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    values = {}
    for arg, default in zip(init.args.args[-len(init.args.defaults):], init.args.defaults):
        if isinstance(default, ast.Call) and isinstance(default.func, ast.Name) and default.func.id == "int":
            values[arg.arg] = int(ast.literal_eval(default.args[0]))
        else:
            values[arg.arg] = ast.literal_eval(default)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/oarl_smoke.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cpu", action="store_true", help="Hide CUDA to match the upstream CPU setup")
    parser.add_argument("--profile", action="store_true", help="Record call counts without editing upstream source")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit or preserve all workspace changes before running; the snapshot must identify a clean commit.")
    if not args.label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.label):
        parser.error("label must contain only ASCII letters, digits, '-' or '_'")
    config_path = args.config.resolve()
    config_relative = config_path.relative_to(ROOT).as_posix()
    config_bytes = subprocess.check_output(["git", "show", "HEAD:" + config_relative], cwd=str(ROOT))
    if config_bytes.replace(b"\r\n", b"\n") != config_path.read_bytes().replace(b"\r\n", b"\n"):
        parser.error("Configuration must match the committed version")
    config = json.loads(config_bytes)
    required = {"env", "algo", "seed", "episodes", "max_step", "state_dim", "action_dim",
                "action_numb", "mode", "save_dir_model", "save_dir_data", "save_dir_train_data"}
    if set(config) != required or config["mode"] != "train":
        parser.error("Provide all upstream CLI parameters and mode=train for a baseline runtime check")
    for key in ("save_dir_model", "save_dir_data", "save_dir_train_data"):
        output = Path(config[key])
        if output.is_absolute() or output.drive or ".." in output.parts:
            parser.error(key + " must remain inside the isolated run directory")
    executable = shutil.which(args.python)
    if executable is None:
        parser.error("Python executable not found: " + args.python)
    executable = str(Path(executable).resolve())
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = ROOT / ".local" / "runs" / (stamp + "-" + args.label)
    source_dir = run_dir / "source"
    source_dir.mkdir(parents=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD", *SOURCE_PATHS], cwd=str(ROOT))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            target = (source_dir / member.name).resolve()
            if source_dir.resolve() not in target.parents or not (member.isfile() or member.isdir()):
                raise ValueError("Unsafe archive member: " + member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
    source_hashes = {p.relative_to(source_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_dir.rglob("*") if p.is_file()}
    env = os.environ.copy()
    overrides = {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(run_dir / "matplotlib"),
                 "PYTHONHASHSEED": str(config["seed"]), "PYTHONUNBUFFERED": "1",
                 "PYTHONIOENCODING": "utf-8"}
    if args.cpu:
        overrides["CUDA_VISIBLE_DEVICES"] = ""
    env.update(overrides)
    runtime_paths = []
    if os.name == "nt":
        prefix = Path(executable).parent
        runtime_paths = [str(path) for path in (prefix, prefix / "Library/bin", prefix / "Scripts")
                         if path.is_dir()]
        env["PATH"] = os.pathsep.join(runtime_paths + [env.get("PATH", "")])
    command = [executable]
    if args.profile:
        command.extend(["-m", "cProfile", "-o", str(run_dir / "calls.pstats")])
    command.append("main.py")
    for key, value in config.items():
        command.extend(["--" + key, str(value)])
    probe = (
        "import json,sys,platform; result={'python':sys.version,'executable':sys.executable,'os':platform.platform()}; "
        "exec(\"try:\\n import torch\\n result.update(torch=torch.__version__,cuda=torch.version.cuda,"
        "cuda_available=torch.cuda.is_available())\\nexcept Exception as error:\\n result['torch_import_error']=repr(error)\"); "
        "print(json.dumps(result))"
    )
    manifest = {
        "kind": "baseline_runtime_check", "started_at_utc": stamp,
        "git_commit": git("rev-parse", "HEAD"), "git_tree": git("rev-parse", "HEAD^{tree}"),
        "git_branch": git("branch", "--show-current"), "git_worktree_clean_before": True,
        "upstream_commit": "29e5c0e2497cd0bd27b6cf83c5bce800a3e2c54a",
        "wrapper_command": [sys.executable, *sys.argv], "launch_command": command,
        "launch_command_windows": subprocess.list2cmdline(command), "cwd": str(source_dir),
        "config_file": config_relative, "config": config, "seed": config["seed"],
        "sumo_seed_schedule": [2 * (episode // 2) for episode in range(config["episodes"])],
        "sumo_seed_note": "Upstream reset ignores CLI seed; episodes use 0,0,2,2,...",
        "agent_defaults": agent_defaults((source_dir / "oarl.py").read_text(encoding="utf-8")),
        "environment_overrides": overrides, "SUMO_HOME": env.get("SUMO_HOME"),
        "runtime_path_prepend": runtime_paths,
        "sumo_executable": shutil.which("sumo"), "host_os": platform.platform(),
        "timeout_seconds": args.timeout, "source_sha256_before": source_hashes,
        "profile_enabled": args.profile,
        "status": "preparing",
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["python_runtime"] = capture([executable, "-c", probe], env)
    manifest["pip_freeze"] = capture([executable, "-m", "pip", "freeze", "--all"], env)
    manifest["sumo_version"] = capture(["sumo", "--version"], env)
    manifest["status"] = "running"
    write_json(manifest_path, manifest)
    print("Run manifest: " + str(manifest_path), flush=True)
    returncode = 1
    try:
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            result = subprocess.run(command, cwd=str(source_dir), env=env,
                                    stdout=stdout, stderr=stderr, timeout=args.timeout)
        returncode = result.returncode
        manifest["status"] = "passed" if returncode == 0 else "failed"
    except subprocess.TimeoutExpired:
        manifest["status"] = "timeout"
        returncode = 124
    finally:
        manifest["returncode"] = returncode
        manifest["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        manifest["git_worktree_clean_after"] = not bool(git("status", "--porcelain"))
        manifest["source_sha256_after"] = {
            name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
            for name in source_hashes if (source_dir / name).is_file()
        }
        write_json(manifest_path, manifest)
    print("Status: " + manifest["status"] + "; logs: " + str(run_dir), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
