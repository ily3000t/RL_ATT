"""Read-only final protocol preflight: verify inputs, never run SUMO episodes."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from prepare_attack_final_protocol import (ROOT, PROTOCOL, METHOD_DOCUMENT, make_protocol,
                                           validate_registration, read)
from prepare_mechanism_controls import sha256
from run_baseline import capture, git, write_json
from rl_att.evaluation.seed_control import evaluation_seeds

RUNTIME_PROBE = "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"
CHECKPOINT_PROBE = ("import json,sys,numpy as np; from rl_att.utils.checkpoints import verify_checkpoint; "
                    "refs=json.loads(sys.argv[1]); print(json.dumps([verify_checkpoint(r['checkpoint'], "
                    "np.zeros((1,16),dtype=np.float32),expected_weights=r['weights_sha256']) for r in refs]))")


def validate_runtime(probes, references):
    for reference in references:
        for key in ("python_runtime", "pip_freeze", "sumo_version"):
            if probes[key]["returncode"] != 0 or probes[key]["stdout"] != reference[key]["stdout"]:
                raise ValueError("Runtime differs from frozen Clean training: " + key)


def assert_unexposed():
    # Only filename output: do not parse or inspect any final trajectory outcome.
    result = subprocess.run(["rg", "-l", r'"(?:research_split_id|split_id)"\s*:\s*30|"config_file"\s*:\s*"configs/evaluation/final_',
                             str(ROOT / ".local/runs"), "-g", "manifest.json", "-g", "evaluation.json", "-g", "batch.json"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode == 0:
        raise ValueError("Possible final exposure in recorded run metadata: " + result.stdout.strip())
    if result.returncode != 1:
        raise ValueError("Cannot check reserved split exposure: " + result.stderr)
    return dict(recorded_run_metadata_matches=0,
                scope="Existing launcher manifest/evaluation/batch files under .local/runs; no outcome parsing, no guarantee about unrecorded runs")


def preflight(expected_commit):
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != expected_commit:
        raise ValueError("Require a clean exact execution commit")
    record = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if record != make_protocol():
        raise ValueError("Frozen method, models, templates, or registration changed")
    blob = subprocess.check_output(["git", "show", "HEAD:" + METHOD_DOCUMENT], cwd=str(ROOT))
    if hashlib.sha256(blob).hexdigest() != record["method_document"]["git_blob_sha256"]:
        raise ValueError("Committed method document changed")
    configs = {p: read(p) for p in record["configs"]}
    validate_registration(record, configs)
    exposure = assert_unexposed()
    wheel = record["zero_one_wheel"]
    if sha256(ROOT / wheel["path"]) != wheel["sha256"]:
        raise ValueError("ZOOpt wheel changed")
    trainings = []
    for proof, victim in zip(record["runtime_references"], record["victims"]):
        training = read(proof["path"])
        if training["status"] != "passed" or training["git_commit"] != victim["training_commit"]:
            raise ValueError("Invalid training provenance")
        trainings.append(training)
    env = os.environ.copy()
    env.update(trainings[0]["environment_overrides"])
    env["MPLCONFIGDIR"] = str(ROOT / ".local/matplotlib-preflight")
    env["PATH"] = os.pathsep.join(trainings[0]["runtime_path_prepend"] + [env.get("PATH", "")])
    python = trainings[0]["launch_command"][0]
    probes = dict(python_runtime=capture([python, "-c", RUNTIME_PROBE], env),
                  pip_freeze=capture([python, "-m", "pip", "freeze", "--all"], env),
                  sumo_version=capture(["sumo", "--version"], env))
    validate_runtime(probes, trainings)
    checkpoints = capture([python, "-c", CHECKPOINT_PROBE, json.dumps(record["victims"])], env)
    if checkpoints["returncode"]:
        raise ValueError("Frozen checkpoint reload failed: " + checkpoints["stderr"])
    verified = json.loads(checkpoints["stdout"])
    if len(verified) != 5 or any((v["checkpoint_sha256"], v["weights_sha256"]) !=
                               (r["checkpoint_sha256"], r["weights_sha256"])
                               for v, r in zip(verified, record["victims"])):
        raise ValueError("Frozen checkpoint verification changed")
    if git("status", "--porcelain") or git("rev-parse", "HEAD") != expected_commit:
        raise ValueError("Tracked tree or source commit changed during preflight")
    return dict(kind="attack_final_protocol_preflight", verified=True, git_commit=expected_commit,
                git_branch=git("branch", "--show-current"), git_tree=git("rev-parse", "HEAD^{tree}"),
                protocol_sha256=sha256(PROTOCOL), host_os=platform.platform(),
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                wrapper_command=[sys.executable] + sys.argv,
                config_raw_sha256={p: sha256(ROOT / p) for p in configs},
                role_seeds={p: evaluation_seeds(c["run_seeds"][0], 50, research_seeds=c["research_seeds"])
                            for p, c in configs.items()}, environment_overrides=trainings[0]["environment_overrides"],
                runtime_path_prepend=trainings[0]["runtime_path_prepend"], probes=probes,
                checkpoint_probe=checkpoints, checkpoint_verification=verified, exposure_check=exposure,
                totals=record["totals"], simulations_run=0, final_outcomes_seen=False,
                final_execution_ready=False, next_requirements=record["readiness_requirements"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / ".local/runs").resolve() not in output.parents or output.exists():
        parser.error("Use a new output file inside ignored .local/runs")
    record = preflight(args.expected_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    print("FINAL_PREFLIGHT verified=True configs=60 model_hashes=5 simulations=0 execution_ready=False")
    print("PREFLIGHT=" + str(output))


if __name__ == "__main__":
    main()
