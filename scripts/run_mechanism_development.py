"""Run all frozen mechanism cells with bounded concurrency and immediate audits."""

import argparse
import concurrent.futures
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from run_baseline import ROOT, git
from run_budget_validation import save
from prepare_mechanism_controls import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or (ROOT / ".local/runs").resolve() not in output.parents or git("status", "--porcelain"):
        parser.error("Use clean Git and a new directory inside .local/runs")
    protocol_path = ROOT / "configs/research/mechanism_development.json"
    protocol = json.loads(protocol_path.read_text())
    subprocess.check_call(["git", "diff", "--exit-code", protocol["frozen_method_commit"], "--",
                           "main.py", "oarl.py", "Environment", "Data", "rl_att", "requirements.txt"], cwd=str(ROOT))
    for filename, key in (("configs/research_seed_splits.json", "seed_splits_sha256"),
                          ("configs/frozen_victims.json", "frozen_victims_sha256")):
        if sha256(ROOT / filename) != protocol[key]:
            raise ValueError("Frozen input changed: " + filename)
    proof = protocol["references"]["original"]["manifests"][0]
    if sha256(ROOT / proof["path"]) != proof["sha256"]:
        raise ValueError("Frozen runtime manifest changed")
    reference = json.loads((ROOT / proof["path"]).read_text())
    env = os.environ.copy()
    env.update(reference["environment_overrides"])
    env["PATH"] = os.pathsep.join(reference["runtime_path_prepend"] + [env.get("PATH", "")])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    output.mkdir(parents=True)
    status_path = output / "mechanism-development.json"
    record = dict(kind="return_mechanism_development_pipeline", git_commit=git("rev-parse", "HEAD"),
                  command=[sys.executable] + sys.argv, protocol_sha256=sha256(protocol_path),
                  status="running", started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  planned_episodes=protocol["total_episodes"], verified_episodes=0,
                  execution=protocol["execution"], groups=[])
    for group in protocol["groups"]:
        item = copy.deepcopy(group)
        item["status"] = "pending"
        record["groups"].append(item)
    save(status_path, record)
    lock, halted = threading.Lock(), threading.Event()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print("MECHANISM_DEVELOPMENT_DIR=" + str(output), flush=True)

    def run(index):
        item = record["groups"][index]
        try:
            with lock:
                if halted.is_set():
                    item["status"] = "not_run_after_failure"
                    save(status_path, record)
                    return
                if git("status", "--porcelain") or git("rev-parse", "HEAD") != record["git_commit"]:
                    raise ValueError("Frozen worktree or source commit changed")
                item.update(status="running", started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                log = output / ("group-%d-batch.log" % index)
                command = [sys.executable, str(ROOT / "scripts/evaluate_attack_batch.py"), "--configs"] + item["configs"] + ["--jobs", str(protocol["execution"]["jobs_per_group"])]
                item.update(command=command, log=str(log))
                save(status_path, record)
            with log.open("wb") as stream:
                code = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT, creationflags=flags).returncode
            paths = [line.split("=", 1)[1] for line in log.read_text().splitlines() if line.startswith("BATCH_DIR=")]
            if len(paths) != 1:
                raise ValueError("Missing or ambiguous batch directory")
            batch_path = Path(paths[0]) / "batch.json"
            with lock:
                item["batch"] = str(batch_path)
                save(status_path, record)
            if code:
                raise ValueError("Evaluation batch failed with code %d" % code)
            audit_path = batch_path.parent / "verified-mechanism-development.json"
            audit_command = [reference["launch_command"][0], str(ROOT / "scripts/analyze_mechanism_controls.py"),
                             "--phase", "development", "--batch", str(batch_path),
                             "--gradient-cap", str(item["gradient_cap"]), "--attack-seed", str(item["attack_seed"]),
                             "--output", str(audit_path)]
            with lock:
                item.update(status="auditing", audit_path=str(audit_path), audit_command=audit_command)
                save(status_path, record)
            with (output / ("group-%d-audit.log" % index)).open("wb") as stream:
                subprocess.check_call(audit_command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            audit = json.loads(audit_path.read_text())
            if not audit["verified"] or audit["git_commit"] != record["git_commit"]:
                raise ValueError("Group audit or source commit mismatch")
            episodes = sum(r["episodes"] for r in audit["rows"])
            if episodes != item["episodes"]:
                raise ValueError("Incomplete audited episode count")
            with lock:
                item.update(status="passed", audit_sha256=sha256(audit_path), raw_verified_steps=audit["raw_episode_verified_steps"],
                            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                record["verified_episodes"] += episodes
                save(status_path, record)
                print("MECHANISM_GROUP_PASSED cap=%d seed=%d verified_episodes=%d" %
                      (item["gradient_cap"], item["attack_seed"], record["verified_episodes"]), flush=True)
        except BaseException as exc:
            halted.set()
            with lock:
                item.update(status="failed", error=repr(exc), finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                save(status_path, record)
                print("MECHANISM_GROUP_FAILED cap=%d seed=%d error=%r" % (item["gradient_cap"], item["attack_seed"], exc), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=protocol["execution"]["parallel_groups"]) as executor:
            futures = [executor.submit(run, index) for index in range(len(record["groups"]))]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        if halted.is_set() or record["verified_episodes"] != protocol["total_episodes"]:
            raise ValueError("Mechanism pipeline incomplete; consult individual group logs")
        record["status"] = "passed"
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(status_path, record)
    print("MECHANISM_GRID_PASSED episodes=%d" % record["verified_episodes"], flush=True)


if __name__ == "__main__":
    main()
