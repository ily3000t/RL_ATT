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
import hashlib


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
    configs = [json.loads((ROOT / path).read_text(encoding="utf-8")) for path in args.configs]
    defense_batch = any(config["victim"] == "pgd_consistency" for config in configs)
    if defense_batch:
        sys.path.insert(0, str(ROOT))
        from rl_att.training.protocol import validate_config
        for config in configs:
            validate_config(config)
        if len({config["training_stage"] for config in configs}) != 1:
            parser.error("Do not mix engineering/full training")
        if configs[0]["training_stage"] == "full" and sorted(config["run_seed"] for config in configs) != list(range(5)):
            parser.error("Full defense batch requires five unique run seeds 0 through 4")
        if len({json.dumps(config["defense"], sort_keys=True) for config in configs}) != 1:
            parser.error("All five runs must use the same defense parameters")
        record["configs_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in args.configs}
    write_json(batch_dir / "batch.json", record)
    print("BATCH_DIR=" + str(batch_dir), flush=True)

    def run(index, config_path):
        run_dir = batch_dir / ("run-%d" % index)
        command = [sys.executable, str(ROOT / "scripts/run_experiment.py"), "--config", config_path, "--output", str(run_dir),
                   "--expected-commit", record["git_commit"]]
        with (batch_dir / ("runner-%d.log" % index)).open("wb") as log:
            result = subprocess.run(command, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return {"config": config_path, "run_dir": str(run_dir), "returncode": result.returncode}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run, i, config) for i, config in enumerate(args.configs)]
            for future in concurrent.futures.as_completed(futures):
                completed = future.result()
                record["runs"].append(completed)
                write_json(batch_dir / "batch.json", record)
                print("COMPLETED=" + json.dumps(completed), flush=True)
        record["status"] = "passed" if all(row["returncode"] == 0 for row in record["runs"]) else "failed"
        if defense_batch and record["status"] == "passed":
            from rl_att.utils.victim_registry import defense_registry
            if git("status", "--porcelain") or git("rev-parse", "HEAD") != record["git_commit"]:
                raise ValueError("Training source changed during batch")
            registry = defense_registry([row["run_dir"] for row in record["runs"]], ROOT,
                                        engineering=configs[0]["training_stage"] == "engineering")
            registry_path = batch_dir / "frozen_defense_victims.json"
            write_json(registry_path, registry)
            record["frozen_registry"] = dict(path=str(registry_path), sha256=hashlib.sha256(registry_path.read_bytes()).hexdigest())
    except Exception as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        record["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(batch_dir / "batch.json", record)
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
