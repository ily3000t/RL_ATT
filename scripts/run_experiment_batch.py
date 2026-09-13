"""Run independent committed configs concurrently, each with isolated SUMO files."""

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from run_baseline import ROOT, git, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--name", default="protocol-a-oarl")
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit all changes before launching a batch")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        parser.error("Invalid batch name")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = ROOT / ".local/runs" / (stamp + "-" + args.name)
    batch_dir.mkdir(parents=True, exist_ok=False)
    record = {"git_commit": git("rev-parse", "HEAD"), "command": [sys.executable, *sys.argv],
              "started_at_utc": stamp, "status": "running", "jobs": args.jobs, "runs": []}
    write_json(batch_dir / "batch.json", record)
    print("BATCH_DIR=" + str(batch_dir), flush=True)

    def run(index, config_path):
        run_dir = batch_dir / ("run-%d" % index)
        command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", config_path, "--output", str(run_dir)]
        with (batch_dir / ("runner-%d.log" % index)).open("wb") as log:
            result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"config": config_path, "run_dir": str(run_dir), "returncode": result.returncode}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, i, config) for i, config in enumerate(args.configs)]
        for future in concurrent.futures.as_completed(futures):
            completed = future.result()
            record["runs"].append(completed)
            write_json(batch_dir / "batch.json", record)
            print("COMPLETED=" + json.dumps(completed), flush=True)
    record["status"] = "passed" if all(row["returncode"] == 0 for row in record["runs"]) else "failed"
    record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(batch_dir / "batch.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
