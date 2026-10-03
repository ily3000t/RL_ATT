"""Run and audit three frozen simple-control batches; stop on any failure."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from run_baseline import ROOT, git
from run_budget_validation import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or (ROOT / ".local/runs").resolve() not in output.parents or git("status", "--porcelain"):
        parser.error("Use clean Git and a new output directory within .local/runs")
    protocol_path = ROOT / "configs/research/basic_validation_controls.json"
    protocol = json.loads(protocol_path.read_text())
    subprocess.check_call(["git", "diff", "--exit-code", protocol["frozen_method_commit"], "--",
                           "main.py", "oarl.py", "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    for key, digest_key in (("search_pipeline", "search_pipeline_sha256"), ("search_summary", "search_summary_sha256")):
        if hashlib.sha256((ROOT / protocol[key]).read_bytes()).hexdigest() != protocol[digest_key]:
            raise ValueError("Frozen search reference changed")
    proof = protocol["reference_manifests"][0]
    path = ROOT / proof["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != proof["sha256"]:
        raise ValueError("Frozen runtime manifest changed")
    reference = json.loads(path.read_text())
    env = os.environ.copy()
    env.update(reference["environment_overrides"])
    env["PATH"] = os.pathsep.join(reference["runtime_path_prepend"] + [env.get("PATH", "")])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    output.mkdir(parents=True)
    record = dict(kind="basic_validation_pipeline", status="running", git_commit=git("rev-parse", "HEAD"),
                  command=[sys.executable] + sys.argv, protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                  started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), groups=[],
                  planned_episodes=protocol["total_episodes"], verified_episodes=0)
    status_path = output / "basic-validation.json"
    save(status_path, record)
    print("BASIC_VALIDATION_DIR=" + str(output), flush=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        for index, group in enumerate(protocol["groups"]):
            if git("status", "--porcelain") or git("rev-parse", "HEAD") != record["git_commit"]:
                raise ValueError("Frozen workspace changed during controls")
            log = output / ("group-%d-batch.log" % index)
            command = [sys.executable, str(ROOT / "scripts/evaluate_attack_batch.py"), "--configs"] + group["configs"] + ["--jobs", "5"]
            item = dict(attack_seed=group["attack_seed"], status="running", command=command, log=str(log))
            record["groups"].append(item)
            save(status_path, record)
            with log.open("wb") as stream:
                subprocess.check_call(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            paths = [s.split("=", 1)[1] for s in log.read_text().splitlines() if s.startswith("BATCH_DIR=")]
            if len(paths) != 1:
                raise ValueError("Ambiguous batch output")
            batch = Path(paths[0]) / "batch.json"
            audit_path = batch.parent / "verified-basic-validation.json"
            audit_command = [reference["launch_command"][0], str(ROOT / "scripts/analyze_basic_validation.py"),
                             "--batch", str(batch), "--attack-seed", str(group["attack_seed"]), "--output", str(audit_path)]
            item.update(status="auditing", batch=str(batch), audit_path=str(audit_path), audit_command=audit_command)
            save(status_path, record)
            with (output / ("group-%d-audit.log" % index)).open("wb") as stream:
                subprocess.check_call(audit_command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            report = json.loads(audit_path.read_text())
            if not report["verified"]:
                raise ValueError("Audit not verified")
            item.update(status="passed", audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                        raw_verified_steps=report["raw_episode_verified_steps"])
            record["verified_episodes"] += sum(r["episodes"] for r in report["rows"])
            save(status_path, record)
            print("BASIC_GROUP_PASSED seed=%d episodes=%d" % (group["attack_seed"], record["verified_episodes"]), flush=True)
        if record["verified_episodes"] != protocol["total_episodes"]:
            raise ValueError("Incomplete simple-control grid")
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(status_path, record)


if __name__ == "__main__":
    main()
