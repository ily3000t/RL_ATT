"""Launch committed, audited historical defense controls in a private SUMO copy."""

import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile

from run_baseline import ROOT, SOURCE_PATHS, capture, git, write_json
from diagnose_defense_pilot import AuditedReader, sha
from analyze_attack_final import expected_seeds

sys.path.insert(0, str(ROOT))
from rl_att.evaluation.controlled_replay import TRACE_KEYS, audit_run, fingerprint
from rl_att.evaluation.defense_diagnostics import index_grid, require, verify_trace


def bound_traces(reader, path, episodes, max_steps):
    """Stream large search metadata away while retaining historically bound bytes."""
    path = Path(path).resolve()
    require(reader.root in path.parents, "Historical traces outside source root")
    key = path.relative_to(reader.root).as_posix()
    require(key in reader.expected, "Historical trace absent from audit")
    digest = hashlib.sha256()
    traces = {e["episode"]: [] for e in episodes}
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line.decode("utf-8"))
            require(row["episode"] in traces, "Unexpected historical episode")
            traces[row["episode"]].append(dict({k: row[k] for k in TRACE_KEYS}, attack_metadata={}, attack_cost={}))
    require(digest.hexdigest() == reader.expected[key], "Historical trace bytes changed")
    reader.inputs[str(path)] = digest.hexdigest()
    for episode in episodes:
        verify_trace(traces[episode["episode"]], episode, max_steps)
    return traces


def prepare(config, group, output):
    source_summary = ROOT / config["source_summary"]
    require(sha(source_summary) == config["source_summary_sha256"], "Pilot summary changed")
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    audit_path, batch_path = (Path(summary[k]["path"]).resolve() for k in ("raw_audit", "batch_manifest"))
    require(sha(audit_path) == config["expected_audit_sha256"] == summary["raw_audit"]["sha256"], "Historical audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit["verified"] and audit["group_id"] == "development_pilot" and audit["actual_episodes"] == 350 and
            audit["real_steps"] == 57195 and audit["git_commit"] == config["expected_execution_commit"] == summary["experiment_commit"],
            "Wrong or incomplete inherited pilot")
    reader = AuditedReader(batch_path.parents[3], audit["source_sha256"])
    batch = reader.json(batch_path)
    require(batch["status"] == "passed" and batch["git_commit"] == config["expected_execution_commit"] and
            sha(batch_path) == summary["batch_manifest"]["sha256"], "Pilot batch provenance mismatch")
    protocol = reader.json(reader.root / "configs/research/defense_baseline.json")
    frozen = json.loads(subprocess.check_output(["git", "show", "HEAD:configs/frozen_victims.json"], cwd=str(ROOT)))["victims"]
    registered = {(r["victim"], r["run_seed"]): r for r in frozen}
    refs = {(r["victim"], r["run_seed"]): r for r in protocol["victim_references"]}
    entries, directories, manifests = [], {}, {}
    methods = ["none"] + config["attacks"]
    seeds = config["groups"][group]["checkpoint_seeds"]
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        original = reader.json(directory / "manifest.json")
        require(original["status"] == "passed" and original["git_commit"] == config["expected_execution_commit"] and run["returncode"] == 0,
                "Incomplete original evaluation")
        require(not original["config"]["gate_enabled"] and original["config"]["research_seeds"]["split_id"] == 10,
                "Historical traffic/Gate protocol mismatch")
        evaluation = reader.json(directory / "evaluation.json")
        for entry in evaluation["runs"]:
            key = entry["victim"], entry["run_seed"], entry["attack"]["name"]
            if key[0] not in config["victims"] or key[1] not in seeds or key[2] not in methods:
                continue
            require(entry["effective_seeds"] == expected_seeds(original["config"], key[1], key[2]) and
                    entry["effective_seeds"]["episode_sumo_seeds"] == config["traffic_seeds"], "Historical role/traffic seeds changed")
            reference = refs[key[:2]]
            require(reference == registered[key[:2]] and entry["frozen_unchanged"] and
                    all(entry[k] == reference[k] for k in ("checkpoint_sha256", "weights_sha256")), "Unregistered victim")
            require(key not in directories, "Duplicate historical entry")
            entries.append(entry)
            directories[key], manifests[key] = directory / entry["results_directory"], original
    indexed = index_grid(entries, config["victims"], seeds, methods)
    jobs = []
    inputs = output / "inputs"
    inputs.mkdir()
    for victim in config["victims"]:
        for seed in seeds:
            reference = refs[(victim, seed)]
            checkpoint = (ROOT / reference["checkpoint"]).resolve()
            require(ROOT in checkpoint.parents and sha(checkpoint) == reference["checkpoint_sha256"], "Frozen checkpoint changed")
            job = dict(job_id="%s-seed%d" % (victim, seed), reference=reference,
                       source_effective_seeds={}, replay_effective_seeds={}, traces={}, episodes={})
            for method in methods:
                key = victim, seed, method
                directory = directories[key]
                episodes = reader.json(directory / "episodes.json")
                require(len(episodes) == config["episodes_per_arm"] and
                        [e["episode"] for e in episodes] == list(range(1, config["episodes_per_arm"] + 1)) and
                        [e["sumo_seed"] for e in episodes] == config["traffic_seeds"], "Incomplete historical episode grid")
                job["episodes"][method] = episodes
                job["traces"][method] = bound_traces(reader, directory / "steps.jsonl", episodes, config["max_steps"])
                effective = indexed[key]["effective_seeds"]
                job["source_effective_seeds"][method] = effective
                job["replay_effective_seeds"][method] = dict(effective, attack_rng="fixed_historical_inputs_no_new_attack_rng")
            path = inputs / (job["job_id"] + ".json")
            # JSON keys normalize episode integers before content hashing.
            job = json.loads(json.dumps(job))
            write_json(path, job)
            jobs.append(dict(job_id=job["job_id"], input_path=str(path), input_content_sha256=fingerprint(job),
                             checkpoint_sha256=reference["checkpoint_sha256"], weights_sha256=reference["weights_sha256"],
                             effective_seeds=job["replay_effective_seeds"]))
    reader.inputs[str(source_summary.resolve())], reader.inputs[str(audit_path)] = sha(source_summary), sha(audit_path)
    return jobs, reader.inputs, manifests[next(iter(manifests))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/defense_controlled_replay.json")
    parser.add_argument("--group", choices=("engineering_smoke", "development_diagnostics"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not git("status", "--porcelain"), "Commit source/config/auditor before replay")
    commit = git("rev-parse", "HEAD")
    config_path = args.config.resolve()
    relative = config_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))
    require(committed.replace(b"\r\n", b"\n") == config_path.read_bytes().replace(b"\r\n", b"\n"), "Uncommitted replay configuration")
    config = json.loads(committed)
    require(config["gate_enabled"] is False and config["traffic_split"] == 10, "Only registered development traffic without Gate")
    output = args.output.resolve()
    require((ROOT / ".local/runs").resolve() in output.parents and not output.exists(), "Use a new ignored run directory")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest = dict(kind=config["kind"], git_commit=commit, git_branch=git("branch", "--show-current"),
                    root=str(ROOT), config=config, config_file=relative, config_sha256=sha(config_path), group=args.group,
                    wrapper_command=[sys.executable, *sys.argv], host_os=platform.platform(), status="preparing",
                    started_at_utc=datetime.datetime.utcnow().isoformat() + "Z")
    write_json(manifest_path, manifest)
    try:
        jobs, inherited, original = prepare(config, args.group, output)
        source = output / "source"
        source.mkdir()
        archive = subprocess.check_output(["git", "archive", "--format=tar", commit, *SOURCE_PATHS, "rl_att"], cwd=str(ROOT))
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                target = (source / member.name).resolve()
                require(source.resolve() in target.parents and (member.isdir() or member.isfile()), "Unsafe source archive member")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(tar.extractfile(member).read())
        hashes = {p.relative_to(source).as_posix(): sha(p) for p in source.rglob("*") if p.is_file()}
        for name, digest in original["source_sha256_before"].items():
            if name in SOURCE_PATHS or name.startswith(("Environment/", "Data/")):
                require(hashes.get(name) == digest, "Victim/environment differs from audited pilot: " + name)
        env = os.environ.copy()
        env.update(original["environment_overrides"])
        env["MPLCONFIGDIR"], env["SUMO_HOME"] = str(output / "matplotlib"), original["SUMO_HOME"]
        env["PATH"] = os.pathsep.join(original["runtime_path_prepend"] + [env.get("PATH", "")])
        python = original["launch_command"][0]
        command = [python, "-m", "rl_att.evaluation.controlled_replay_run", "--manifest", str(manifest_path)]
        manifest.update(jobs=jobs, historical_inputs_sha256=inherited, cwd=str(source), launch_command=command,
                        environment_overrides={k: env[k] for k in original["environment_overrides"]}, SUMO_HOME=env["SUMO_HOME"],
                        runtime_path_prepend=original["runtime_path_prepend"], source_sha256_before=hashes,
                        expected_cases=len(jobs) * len(config["attacks"]) * config["episodes_per_arm"],
                        expected_physical_episodes=len(jobs) * (1 + 2 * len(config["attacks"])) * config["episodes_per_arm"])
        for key, cmd in (("python_runtime", [python, "-c", "import json,sys,torch,numpy,scipy,sklearn; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_threads=torch.get_num_threads(),numpy=numpy.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__)))"]),
                         ("pip_freeze", [python, "-m", "pip", "freeze", "--all"]), ("sumo_version", ["sumo", "--version"])):
            manifest[key] = capture(cmd, env)
            require(manifest[key]["returncode"] == 0 and manifest[key]["stdout"] == original[key]["stdout"], "Frozen runtime mismatch: " + key)
        require(not git("status", "--porcelain") and git("rev-parse", "HEAD") == commit, "Source changed during preparation")
        manifest["status"] = "running"
        write_json(manifest_path, manifest)
        print("CONTROLLED_REPLAY_DIR=" + str(output), flush=True)
        with (output / "stdout.log").open("wb") as out, (output / "stderr.log").open("wb") as err:
            code = subprocess.run(command, cwd=str(source), env=env, stdout=out, stderr=err,
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0).returncode
        manifest["returncode"] = code
        require(code == 0, "Controlled worker failed; inspect stderr.log")
        report = audit_run(manifest_path)
        write_json(output / "audit.json", report)
        after = {name: sha(source / name) for name in hashes}
        changed = [name for name in hashes if hashes[name] != after[name]]
        require(all(name == "Data/StraightRoad.sumocfg" for name in changed), "Worker changed algorithm source")
        require(not git("status", "--porcelain") and git("rev-parse", "HEAD") == commit, "Source changed during replay")
        manifest.update(status="passed", source_sha256_after=after, changed_source_files=changed, audit_sha256=sha(output / "audit.json"))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_at_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        write_json(manifest_path, manifest)
    print("CONTROLLED_REPLAY_PASSED cases=%d physical_episodes=%d" % (report["paired_cases"], report["physical_episodes"]), flush=True)


if __name__ == "__main__":
    main()
