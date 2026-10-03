"""Run the frozen nine-batch validation grid, auditing before advancing each batch."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_baseline import ROOT, git


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / ".local/runs").resolve() not in output.parents or output.exists():
        parser.error("Use a new output directory inside .local/runs; existing runs are never overwritten")
    if git("status", "--porcelain"):
        parser.error("Commit all work before starting validation")
    protocol_path = ROOT / "configs/research/shared_compute_validation.json"
    protocol = json.loads(protocol_path.read_text())
    subprocess.check_call(["git", "diff", "--exit-code", protocol["frozen_method_commit"], "--",
                           "main.py", "oarl.py", "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    for filename, key in (("configs/research_seed_splits.json", "seed_protocol_sha256"),
                          ("configs/frozen_victims.json", "victim_registry_sha256")):
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != protocol[key]:
            raise ValueError("Frozen reference changed: " + filename)
    frozen_reference = protocol["development_references"][0]
    reference_path = ROOT / frozen_reference["path"]
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != frozen_reference["sha256"]:
        raise ValueError("Development runtime reference changed")
    reference = json.loads(reference_path.read_text())
    env = os.environ.copy()
    env.update(reference["environment_overrides"])
    env["PATH"] = os.pathsep.join(reference["runtime_path_prepend"] + [env.get("PATH", "")])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    output.mkdir(parents=True)
    status = dict(kind="shared_compute_validation_pipeline", git_commit=git("rev-parse", "HEAD"),
                  command=[sys.executable] + sys.argv, started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  status="running", groups=[], planned_episodes=protocol["total_episodes"], verified_episodes=0)
    status_path = output / "validation.json"
    save(status_path, status)
    print("VALIDATION_DIR=" + str(output), flush=True)
    clean_reference = None
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for index, group in enumerate(protocol["groups"]):
            if git("status", "--porcelain") or git("rev-parse", "HEAD") != status["git_commit"]:
                raise ValueError("Worktree or commit changed during frozen validation")
            item = dict(gradient_cap=group["gradient_cap"], attack_seed=group["attack_seed"], status="running")
            status["groups"].append(item)
            log = output / ("group-%d-batch.log" % index)
            command = [sys.executable, str(ROOT / "scripts/evaluate_attack_batch.py"), "--configs"] + group["configs"] + ["--jobs", "5"]
            item.update(command=command, log=str(log))
            save(status_path, status)
            with log.open("wb") as stream:
                subprocess.check_call(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, creationflags=creationflags)
            directories = [line.split("=", 1)[1] for line in log.read_text().splitlines() if line.startswith("BATCH_DIR=")]
            if len(directories) != 1:
                raise ValueError("Expected exactly one batch output")
            batch_path = Path(directories[0]) / "batch.json"
            item["batch"] = str(batch_path)
            audit_path = batch_path.parent / "verified-validation-summary.json"
            audit_command = [reference["launch_command"][0], str(ROOT / "scripts/analyze_budget_validation.py"),
                             "--batch", str(batch_path), "--gradient-cap", str(group["gradient_cap"]),
                             "--attack-seed", str(group["attack_seed"]), "--output", str(audit_path)]
            if clean_reference:
                audit_command += ["--clean-reference", str(clean_reference)]
            item.update(status="auditing", audit_command=audit_command, audit_path=str(audit_path))
            save(status_path, status)
            with (output / ("group-%d-audit.log" % index)).open("wb") as stream:
                subprocess.check_call(audit_command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT,
                                      creationflags=creationflags)
            audit = json.loads(audit_path.read_text())
            if not audit["verified"]:
                raise ValueError("Audit did not verify batch")
            item.update(status="passed", audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                        raw_verified_steps=audit["raw_episode_verified_steps"])
            clean_reference = clean_reference or batch_path
            status["verified_episodes"] += sum(r["episodes"] for r in audit["rows"])
            save(status_path, status)
            print("VALIDATION_GROUP_PASSED cap=%d seed=%d verified_episodes=%d" %
                  (group["gradient_cap"], group["attack_seed"], status["verified_episodes"]), flush=True)
        if status["verified_episodes"] != protocol["total_episodes"]:
            raise ValueError("Incomplete validation episode grid")
        status["status"] = "passed"
    except BaseException as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        status["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(status_path, status)
    print("VALIDATION_GRID_PASSED episodes=%d" % status["verified_episodes"], flush=True)


if __name__ == "__main__":
    main()
