"""Run committed attack evaluation configs concurrently in isolated processes."""

import argparse
import concurrent.futures
import datetime
import os
import subprocess
import sys
from run_baseline import ROOT, git, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    if git("status", "--porcelain") or not 1 <= args.jobs <= 5:
        parser.error("Use a clean Git tree and 1 through 5 jobs")
    if len(set(args.configs)) != len(args.configs):
        parser.error("Duplicate configs are not independent trials")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / ".local/runs" / (stamp + "-attack-benchmark")
    directory.mkdir(parents=True)
    record = {"git_commit": git("rev-parse", "HEAD"), "command": [sys.executable, *sys.argv],
              "configs": args.configs, "jobs": args.jobs, "status": "running", "runs": [],
              "started_at_utc": stamp}
    write_json(directory / "batch.json", record)
    print("BATCH_DIR=" + str(directory), flush=True)

    def run(index, config):
        output = directory / ("run-%d" % index)
        command = [sys.executable, str(ROOT / "scripts/evaluate_attacks.py"), "--config", config,
                   "--output", str(output), "--expected-commit", record["git_commit"]]
        with (directory / ("runner-%d.log" % index)).open("wb") as log:
            code = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).returncode
        return {"config": config, "run_dir": str(output), "returncode": code}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(run, i, c) for i, c in enumerate(args.configs)]
        for future in concurrent.futures.as_completed(futures):
            completed = future.result()
            record["runs"].append(completed)
            write_json(directory / "batch.json", record)
            print("COMPLETED=" + str(completed), flush=True)
    record.update(status="passed" if all(r["returncode"] == 0 for r in record["runs"]) else "failed",
                  finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(directory / "batch.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
