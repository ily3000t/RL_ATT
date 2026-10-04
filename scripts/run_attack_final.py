"""Run the preregistered final matrix once, auditing the anchor before dispatch."""

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
import time

from prepare_attack_final_protocol import ROOT, PROTOCOL
from prepare_mechanism_controls import sha256
from run_baseline import git
from run_budget_validation import save as atomic_save
from check_attack_final_protocol import preflight
from check_attack_final_execution import validate_certificate


def save(path, record):
    # Windows readers may briefly deny replacement of an open status file.
    # Only retry this atomic metadata write, never an experiment or outcome.
    for attempt in range(20):
        try:
            atomic_save(path, record)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.01)


def dispatch(groups, worker, concurrency=2):
    """Anchor synchronously, then bounded submission; no new task after failure."""
    worker(0)
    if groups[0]["status"] != "passed":
        for group in groups[1:]:
            group["status"] = "not_run_after_failure"
        return False
    next_index, halted = 1, False
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        active = {}
        while active or (not halted and next_index < len(groups)):
            halted |= any(g["status"] == "failed" for g in groups)
            while not halted and len(active) < concurrency and next_index < len(groups):
                active[executor.submit(worker, next_index)] = next_index
                next_index += 1
            completed, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in completed:
                index = active.pop(future)
                future.result()
                halted |= groups[index]["status"] != "passed"
    for group in groups[next_index:]:
        group["status"] = "not_run_after_failure"
    return not halted


def run(output, readiness, expected_commit):
    output = output.resolve()
    if output.exists() or (ROOT / ".local/runs").resolve() not in output.parents:
        raise ValueError("Use a new ignored final output directory")
    certificate = json.loads(readiness.read_text())
    validate_certificate(certificate, expected_commit)
    # Initial preflight is required even if the engineering certificate was made
    # on a parent feature commit with identical tooling. No resume or outcome retry.
    inputs = preflight(expected_commit)
    protocol = json.loads(PROTOCOL.read_text())
    training = json.loads((ROOT / protocol["runtime_references"][0]["path"]).read_text())
    env = os.environ.copy()
    env.update(training["environment_overrides"])
    env["PATH"] = os.pathsep.join(training["runtime_path_prepend"] + [env.get("PATH", "")])
    env["MPLCONFIGDIR"] = str(output / "matplotlib")
    output.mkdir(parents=True)
    path = output / "final-attack.json"
    save(output / "preflight.json", inputs)
    record = dict(kind="attack_final_pipeline", git_commit=expected_commit, git_branch=git("branch", "--show-current"),
                  command=[sys.executable] + sys.argv, readiness_path=str(readiness), readiness_sha256=sha256(readiness),
                  protocol_sha256=sha256(PROTOCOL), status="running", planned_episodes=14500, verified_episodes=0,
                  started_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), final_exposure_declared_at_utc=None,
                  execution=protocol["execution"], groups=[dict(copy.deepcopy(g), status="pending") for g in protocol["groups"]])
    save(path, record)
    lock = threading.Lock()
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    print("FINAL_ATTACK_DIR=" + str(output), flush=True)

    def worker(index):
        item = record["groups"][index]
        try:
            with lock:
                if git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain"):
                    raise ValueError("Execution source changed")
                item["status"] = "running"
                item["started_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                # Declare conservatively before first launcher, never after outcome.
                if record["final_exposure_declared_at_utc"] is None:
                    record["final_exposure_declared_at_utc"] = item["started_at_utc"]
                log = output / (item["id"] + "-batch.log")
                command = [sys.executable, str(ROOT / "scripts/evaluate_attack_batch.py"), "--configs"] + item["configs"] + ["--jobs", "5"]
                item.update(command=command, log=str(log))
                save(path, record)
            with log.open("wb") as stream:
                code = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags).returncode
            batches = [line.split("=", 1)[1] for line in log.read_text().splitlines() if line.startswith("BATCH_DIR=")]
            if len(batches) != 1:
                raise ValueError("Missing/ambiguous batch manifest")
            batch = Path(batches[0]) / "batch.json"
            with lock:
                item.update(batch=str(batch), batch_returncode=code)
                save(path, record)
            if code:
                raise ValueError("Batch failed; preserve attempts, no automatic retry")
            audit_path = batch.parent / "verified-final-attack.json"
            audit = [training["launch_command"][0], str(ROOT / "scripts/analyze_attack_final.py"),
                     "--batch", str(batch), "--group", item["id"], "--output", str(audit_path)]
            if index:
                audit += ["--clean-reference", record["groups"][0]["batch"]]
            with lock:
                item.update(status="auditing", audit_path=str(audit_path), audit_command=audit)
                save(path, record)
            with (output / (item["id"] + "-audit.log")).open("wb") as stream:
                subprocess.check_call(audit, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
            proof = json.loads(audit_path.read_text())
            if not proof["verified"] or proof["git_commit"] != expected_commit or proof["episodes"] != item["episodes"]:
                raise ValueError("Group audit incomplete")
            with lock:
                item.update(status="passed", audit_sha256=sha256(audit_path), raw_verified_steps=proof["raw_episode_verified_steps"],
                            finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                record["verified_episodes"] += proof["episodes"]
                save(path, record)
                print("FINAL_GROUP_PASSED group=%s verified_episodes=%d" % (item["id"], record["verified_episodes"]), flush=True)
        except BaseException as exc:
            with lock:
                item.update(status="failed", error=repr(exc), finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                save(path, record)
                print("FINAL_GROUP_FAILED group=%s error=%r" % (item["id"], exc), flush=True)

    try:
        if not dispatch(record["groups"], worker) or record["verified_episodes"] != 14500:
            raise ValueError("Final matrix incomplete; all failed/unfinished attempts retained")
        record["status"] = "groups_passed"
        save(path, record)
        summary = output / "final-comparison.json"
        command = [training["launch_command"][0], str(ROOT / "scripts/summarize_attack_final.py"), "--pipeline", str(path), "--output", str(summary)]
        record["summary_command"] = command
        save(path, record)
        with (output / "summary.log").open("wb") as stream:
            subprocess.check_call(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
        record.update(status="passed", summary_path=str(summary), summary_sha256=sha256(summary))
    except BaseException as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(path, record)
    print("FINAL_MATRIX_PASSED episodes=14500 rows=17", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    run(args.output, args.readiness.resolve(), args.expected_commit)
