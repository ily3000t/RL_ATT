"""Diagnose the complete registered, audited defense pilot without new rollouts."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from run_baseline import ROOT, capture, git, write_json

sys.path.insert(0, str(ROOT))
from rl_att.evaluation.defense_diagnostics import diagnose_pair, index_grid, require, summarize_cases
from analyze_attack_final import expected_seeds


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedReader:
    """Use only bytes already bound by the registered historical audit."""
    def __init__(self, root, expected):
        self.root, self.expected, self.inputs = Path(root).resolve(), expected, {}

    def bytes(self, path):
        path = Path(path).resolve()
        require(self.root in path.parents, "Input outside original source root")
        key = path.relative_to(self.root).as_posix()
        require(key in self.expected, "Input absent from historical audit")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        require(digest == self.expected[key], "Audited input bytes changed: " + key)
        self.inputs[str(path)] = digest
        return content

    def json(self, path):
        return json.loads(self.bytes(path).decode("utf-8"))

    def traces(self, path, episodes):
        traces = {e["episode"]: [] for e in episodes}
        require(len(traces) == len(episodes), "Duplicate episode metadata")
        for line in self.bytes(path).decode("utf-8").splitlines():
            row = json.loads(line)
            require(row["episode"] in traces, "Unexpected raw episode")
            traces[row["episode"]].append(row)
        return traces


def analyze(config, summary):
    audit_path = Path(summary["raw_audit"]["path"]).resolve()
    batch_path = Path(summary["batch_manifest"]["path"]).resolve()
    require(sha(audit_path) == config["expected_audit_sha256"] == summary["raw_audit"]["sha256"],
            "Registered historical audit changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit["verified"] and audit["group_id"] == "development_pilot" and
            audit["git_commit"] == config["expected_execution_commit"] == summary["experiment_commit"] and
            audit["actual_episodes"] == config["expected_episodes"] and audit["real_steps"] == config["expected_real_steps"],
            "Historical audit does not identify the complete pilot")
    source_root = batch_path.parents[3]
    reader = AuditedReader(source_root, audit["source_sha256"])
    batch = reader.json(batch_path)
    require(sha(batch_path) == summary["batch_manifest"]["sha256"] and batch["status"] == "passed" and
            batch["git_commit"] == config["expected_execution_commit"], "Pilot batch provenance mismatch")
    protocol = reader.json(source_root / "configs/research/defense_baseline.json")
    refs = {(r["victim"], r["run_seed"]): r for r in protocol["victim_references"]}
    entries, directories = [], {}
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest = reader.json(directory / "manifest.json")
        require(run["returncode"] == 0 and manifest["status"] == "passed" and
                manifest["git_commit"] == config["expected_execution_commit"], "Incomplete/mixed pilot run")
        require(manifest["config"]["research_seeds"]["split_id"] == config["traffic_split"] and
                manifest["config"]["research_seeds"]["attack_seed"] == config["attack_seed"] and
                not manifest["config"]["gate_enabled"], "Unexpected traffic/attack protocol")
        evaluation = reader.json(directory / "evaluation.json")
        for entry in evaluation["runs"]:
            key = entry["victim"], entry["run_seed"], entry["attack"]["name"]
            require(key not in directories, "Duplicate diagnostic entry")
            reference = refs[(entry["victim"], entry["run_seed"])]
            require(entry["frozen_unchanged"] and all(entry[k] == reference[k] for k in
                    ("checkpoint_sha256", "weights_sha256")), "Unregistered victim checkpoint")
            effective = expected_seeds(manifest["config"], entry["run_seed"], entry["attack"]["name"])
            require(entry["effective_seeds"] == effective,
                    "Effective role seeds changed")
            directories[key] = directory / entry["results_directory"]
            entries.append(entry)
    indexed = index_grid(entries, config["victims"], config["checkpoint_seeds"], config["attacks"])
    cases = []
    for victim in config["victims"]:
        for seed in config["checkpoint_seeds"]:
            baseline_dir = directories[(victim, seed, "none")]
            baseline = reader.json(baseline_dir / "episodes.json")
            clean_traces = reader.traces(baseline_dir / "steps.jsonl", baseline)
            for name in config["attacks"]:
                directory = directories[(victim, seed, name)]
                episodes = reader.json(directory / "episodes.json")
                entry = indexed[(victim, seed, name)]
                require(len(episodes) == len(baseline) and [e["episode"] for e in episodes] == list(range(1, len(baseline) + 1)) and
                        [e["sumo_seed"] for e in episodes] == entry["effective_seeds"]["episode_sumo_seeds"], "Unpaired episode grid")
                traces = reader.traces(directory / "steps.jsonl", episodes)
                for reference, episode in zip(baseline, episodes):
                    detail = diagnose_pair(clean_traces[reference["episode"]], traces[episode["episode"]], reference, episode,
                                           config["max_steps"], config["early_collision_real_steps"],
                                           config["late_collision_first_real_step"])
                    detail.update(victim=victim, checkpoint_seed=seed, attack=name,
                                  case_id="%s-seed%d-%s-episode%d" % (victim, seed, name, episode["episode"]))
                    cases.append(detail)
    require(len(cases) == config["expected_episodes"] and sum(c["steps"] for c in cases) == config["expected_real_steps"],
            "Diagnostic coverage differs from complete pilot")
    groups, checkpoint_groups, traffic_groups = [], [], []
    for victim in config["victims"]:
        for name in config["attacks"]:
            group = [c for c in cases if c["victim"] == victim and c["attack"] == name]
            groups.append(dict(victim=victim, attack=name, **summarize_cases(group)))
            for seed in config["checkpoint_seeds"]:
                checkpoint_groups.append(dict(victim=victim, attack=name, checkpoint_seed=seed,
                                              **summarize_cases([c for c in group if c["checkpoint_seed"] == seed])))
            for seed in sorted({c["sumo_seed"] for c in group}):
                traffic_groups.append(dict(victim=victim, attack=name, sumo_seed=seed,
                                           **summarize_cases([c for c in group if c["sumo_seed"] == seed])))
    return dict(kind=config["kind"], verified=True, execution_commit=audit["git_commit"],
                inherited_audit_sha256=sha(audit_path), input_sha256=reader.inputs,
                actual_episodes=len(cases), real_steps=sum(c["steps"] for c in cases),
                independent_traffic_clusters=len({c["sumo_seed"] for c in cases}),
                checkpoint_pairs=len(config["checkpoint_seeds"]), new_policy_forward_calls=0, new_sumo_steps=0,
                episode_cases=cases, summary=groups, per_checkpoint=checkpoint_groups, per_traffic=traffic_groups,
                limitation=config["interpretation"],
                collision_scope="SUMO ego colliding IDs sampled after each real interaction; not heuristic reward flag or physical-contact certification",
                prefix_scope="Cross-episode states compared only up to first action divergence; local action counts use clean_action_at_visited_state",
                reset_collision_status="Not logged by old evaluator; first-step attribution requires a new controlled reset/branch protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/defense_diagnostics.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not git("status", "--porcelain"), "Commit diagnostic source before analyzing")
    config_path = args.config.resolve()
    require(ROOT in config_path.parents, "Configuration outside repository")
    relative = config_path.relative_to(ROOT).as_posix()
    import subprocess
    committed = subprocess.check_output(["git", "show", "HEAD:" + relative], cwd=str(ROOT))
    require(committed.replace(b"\r\n", b"\n") == config_path.read_bytes().replace(b"\r\n", b"\n"), "Uncommitted diagnostic configuration")
    config = json.loads(committed)
    output = args.output.resolve()
    require((ROOT / ".local/runs").resolve() in output.parents and not output.exists(), "Use a new ignored run directory")
    source = ROOT / config["source_summary"]
    require(sha(source) == config["source_summary_sha256"], "Pilot summary fingerprint changed")
    summary = json.loads(source.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    manifest = dict(kind=config["kind"], status="running", analysis_commit=git("rev-parse", "HEAD"),
                    execution_commit=config["expected_execution_commit"], command=[sys.executable, *sys.argv],
                    started_utc=datetime.datetime.utcnow().isoformat() + "Z", config=config,
                    config_sha256=sha(config_path), input_summary_sha256=sha(source),
                    python=sys.version, torch=torch.__version__, numpy=np.__version__, os=platform.platform(),
                    pip_freeze=capture([sys.executable, "-m", "pip", "freeze", "--all"], os.environ.copy()),
                    sumo_version=capture(["sumo", "--version"], os.environ.copy()))
    write_json(output / "manifest.json", manifest)
    try:
        report = analyze(config, summary)
        report["analysis_commit"] = manifest["analysis_commit"]
        write_json(output / "diagnostics.json", report)
        require(not git("status", "--porcelain") and git("rev-parse", "HEAD") == manifest["analysis_commit"],
                "Diagnostic source changed during analysis")
        manifest.update(status="passed", output_sha256=sha(output / "diagnostics.json"))
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest["finished_utc"] = datetime.datetime.utcnow().isoformat() + "Z"
        write_json(output / "manifest.json", manifest)
    print("DEFENSE_DIAGNOSIS_PASSED episodes=%d real_steps=%d output=%s" %
          (report["actual_episodes"], report["real_steps"], output))


if __name__ == "__main__":
    main()
