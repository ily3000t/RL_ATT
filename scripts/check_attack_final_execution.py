"""Certify final tooling on a clean commit before any reserved traffic exposure."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from prepare_attack_final_protocol import ROOT, PROTOCOL
from prepare_mechanism_controls import sha256
from check_attack_final_protocol import preflight
from run_baseline import git, write_json

TOOLS = ("scripts/analyze_attack_final.py", "scripts/check_final_auditor_fixtures.py",
         "scripts/check_attack_final_execution.py", "scripts/run_attack_final.py",
         "scripts/summarize_attack_final.py", "scripts/final_statistics.py",
         "scripts/check_attack_final_protocol.py", "scripts/prepare_attack_final_protocol.py",
         "scripts/evaluate_attacks.py", "scripts/evaluate_attack_batch.py",
         "scripts/summarize_proposed_smoke.py", "scripts/summarize_benchmark.py",
         "scripts/analyze_proposed_development.py", "scripts/analyze_progress_retry.py",
         "scripts/analyze_basic_validation.py", "scripts/analyze_mechanism_controls.py",
         "scripts/run_baseline.py", "scripts/run_budget_validation.py",
         "scripts/prepare_mechanism_controls.py", "scripts/prepare_proposed_configs.py")


def tool_hashes():
    tests = subprocess.check_output(["git", "ls-files", "tests"], cwd=str(ROOT)).decode().splitlines()
    return {p: hashlib.sha256(subprocess.check_output(["git", "show", "HEAD:" + p], cwd=str(ROOT))).hexdigest()
            for p in TOOLS + tuple(tests)}


def validate_certificate(record, expected_commit=None):
    if (record["kind"] != "attack_final_execution_readiness" or not record["verified"] or
            not record["final_execution_ready"] or record["new_simulations"] != 0 or
            record["protocol_sha256"] != sha256(PROTOCOL) or record["tools_git_blob_sha256"] != tool_hashes()):
        raise ValueError("Final tooling certificate changed or failed")
    if len(record["inputs"]) != 4 or len({p["path"] for p in record["inputs"]}) != 4 or record["tests_passed"] < 137:
        raise ValueError("Incomplete engineering evidence")
    if expected_commit and (git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain")):
        raise ValueError("Execution source or tracked tree changed")
    for proof in record["inputs"]:
        if sha256(ROOT / proof["path"]) != proof["sha256"]:
            raise ValueError("Engineering certificate input changed")
    preflight_record = json.loads((ROOT / record["inputs"][0]["path"]).read_text())
    fixtures = json.loads((ROOT / record["inputs"][3]["path"]).read_text())
    if (not preflight_record["verified"] or preflight_record["simulations_run"] != 0 or
            preflight_record["git_commit"] != record["git_commit"] or not fixtures["verified"] or
            fixtures["split_id"] != 20 or fixtures["new_simulations"] != 0 or
            fixtures["git_commit"] != record["git_commit"]):
        raise ValueError("Engineering proof provenance mismatch")


def check(output, expected_commit):
    output = output.resolve()
    if output.exists() or (ROOT / ".local/runs").resolve() not in output.parents:
        raise ValueError("Use a new ignored readiness output")
    checks = preflight(expected_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.parent / "protocol-preflight.json", checks)
    reference = json.loads((PROTOCOL).read_text())["runtime_references"][0]["path"]
    training = json.loads((ROOT / reference).read_text())
    env = os.environ.copy()
    env.update(training["environment_overrides"])
    env["PATH"] = os.pathsep.join(training["runtime_path_prepend"] + [env.get("PATH", "")])
    env["MPLCONFIGDIR"] = str(output.parent / "matplotlib")
    python = training["launch_command"][0]
    commands = [[python, "-m", "unittest", "discover", "-s", "tests"],
                [python, str(ROOT / "scripts/check_final_auditor_fixtures.py"), "--output", str(output.parent / "fixtures.json")]]
    inputs = [dict(path=(output.parent / "protocol-preflight.json").relative_to(ROOT).as_posix(),
                   sha256=sha256(output.parent / "protocol-preflight.json"))]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    for index, command in enumerate(commands):
        log = output.parent / ("check-%d.log" % index)
        with log.open("wb") as stream:
            code = subprocess.run(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags).returncode
        if code:
            raise ValueError("Readiness check failed; inspect " + str(log))
        inputs.append(dict(path=log.relative_to(ROOT).as_posix(), sha256=sha256(log)))
    fixture = output.parent / "fixtures.json"
    inputs.append(dict(path=fixture.relative_to(ROOT).as_posix(), sha256=sha256(fixture)))
    proof = json.loads(fixture.read_text())
    if not proof["verified"] or proof["git_commit"] != expected_commit or proof["new_simulations"]:
        raise ValueError("Historical-fixture proof failed")
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != expected_commit:
        raise ValueError("Source changed during readiness check")
    tests = re.search(r"Ran (\d+) tests", (output.parent / "check-0.log").read_text())
    if tests is None:
        raise ValueError("Missing unit test count")
    record = dict(kind="attack_final_execution_readiness", verified=True, final_execution_ready=True,
                  git_commit=expected_commit, git_branch=git("branch", "--show-current"), new_simulations=0,
                  command=[sys.executable] + sys.argv, check_commands=commands, protocol_sha256=sha256(PROTOCOL),
                  tools_git_blob_sha256=tool_hashes(), inputs=inputs, tests_passed=int(tests.group(1)),
                  fixture_episodes=proof["episodes"], fixture_steps=proof["steps"],
                  limitation="Engineering certificate, not final efficacy; preregistration historical readiness remains false")
    write_json(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.output, args.expected_commit)
    print("FINAL_EXECUTION_READY tests=%d fixture_steps=%d new_simulations=0" % (result["tests_passed"], result["fixture_steps"]))
