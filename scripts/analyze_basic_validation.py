"""Audit fixed simple attacks against the frozen shared-validation Clean reference."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from summarize_proposed_smoke import ROOT, require
from summarize_benchmark import verify_steps
from analyze_proposed_development import audit_episode_records
from analyze_progress_retry import verify_v0_reference
from prepare_basic_validation import make_config
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.evaluation.results import summarize_episodes

PROTOCOL = ROOT / "configs/research/basic_validation_controls.json"


def expected_seeds(config, checkpoint, name):
    seeds = evaluation_seeds(checkpoint, 20, research_seeds=config["research_seeds"])
    mechanisms = dict(none="unused_no_attack", fgsm="unused_deterministic_gradient",
                      random="seedsequence_attack_phase3_episode_step_local_numpy",
                      pgd="seedsequence_attack_phase3_episode_step_local_numpy",
                      oarl_bo="optimizer_and_discarded_torch_draws_reinitialized_each_attack")
    seeds.update(policy_rng="unused_greedy_argmax", attack_rng=mechanisms[name])
    return seeds


def validate_reference(manifest, reference, specifications, seed):
    for key in ("source_sha256_before", "python_runtime", "pip_freeze", "sumo_version", "victim_references"):
        require(manifest[key] == reference[key], "Shared validation provenance differs: " + key)
    require(manifest["config"] == make_config(reference["config"], dict(attacks=specifications), seed),
            "Simple controls differ beyond registered methods and attack seed")
    require(not manifest["extra_dependencies"], "Simple controls unexpectedly use simulation-search dependency")


def verify_random_seed(row, seed, name):
    metadata = row["attack_metadata"]
    if name in ("random", "pgd"):
        expected = int(np.random.SeedSequence([seed, 3, row["episode"] - 1, row["step"]]).generate_state(1)[0])
        require(metadata["state_seed"] == expected, "Raw stochastic state seed mismatch")
    if name == "oarl_bo":
        require(metadata["optimizer_seed"] == seed and metadata["online_adaptation"] == "obs2=obs1; no future observation",
                "BO seed or online adaptation changed")


def analyze(batch_path, seed):
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text(encoding="utf-8"))
    protocol = read(PROTOCOL)
    pipeline_path = ROOT / protocol["search_pipeline"]
    require(hashlib.sha256(pipeline_path.read_bytes()).hexdigest() == protocol["search_pipeline_sha256"], "Search pipeline changed")
    require(hashlib.sha256((ROOT / protocol["search_summary"]).read_bytes()).hexdigest() == protocol["search_summary_sha256"], "Search summary changed")
    references = {}
    for proof in protocol["reference_manifests"]:
        path = ROOT / proof["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest() == proof["sha256"], "Frozen reference manifest changed")
        references[proof["checkpoint_seed"]] = read(path)
    batch = read(batch_path)
    group = next(g for g in protocol["groups"] if g["attack_seed"] == seed)
    require(batch["status"] == "passed" and batch["configs"] == group["configs"] and len(batch["runs"]) == 5,
            "Incomplete or unregistered simple-control batch")
    names = [a["name"] for a in protocol["basic_specs"] if seed == 0 or a["name"] != "fgsm"]
    rows, conditions, raw_steps = [], set(), 0
    for run in batch["runs"]:
        directory = Path(run["run_dir"])
        manifest, evaluation = read(directory / "manifest.json"), read(directory / "evaluation.json")
        require(run["returncode"] == 0 and manifest["status"] == "passed" and
                manifest["git_commit"] == evaluation["git_commit"] == batch["git_commit"], "Run provenance mismatch")
        checkpoint = manifest["config"]["run_seeds"][0]
        require(manifest["config"] == read(ROOT / run["config"]), "Config file differs from manifest")
        validate_reference(manifest, references[checkpoint], protocol["basic_specs"], seed)
        require([e["attack"]["name"] for e in evaluation["runs"]] == names, "Missing or reordered method")
        clean, clean_steps = None, None
        for entry in evaluation["runs"]:
            name = entry["attack"]["name"]
            key = checkpoint, name
            require(key not in conditions, "Duplicate checkpoint/method")
            conditions.add(key)
            require(entry["frozen_unchanged"] and entry["run_seed"] == checkpoint and entry["victim"] == "clean", "Victim changed")
            for field in ("checkpoint_sha256", "weights_sha256"):
                require(entry[field] == manifest["victim_references"][0][field], "Frozen weights changed")
            require(entry["attack"] == next(a for a in manifest["config"]["attacks"] if a["name"] == name), "Attack config changed")
            require(entry["effective_seeds"] == expected_seeds(manifest["config"], checkpoint, name), "Effective seed or RNG mismatch")
            result_dir = directory / entry["results_directory"]
            episodes = read(result_dir / "episodes.json")
            require(len(episodes) == 20 and [e["episode"] for e in episodes] == list(range(1, 21)) and
                    [e["sumo_seed"] for e in episodes] == entry["effective_seeds"]["episode_sumo_seeds"], "Episode pairing mismatch")
            path = result_dir / "steps.jsonl"
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open() as stream:
                steps = [json.loads(line) for line in stream]
            require(verify_steps(result_dir, name, episodes) == audit_episode_records(steps, episodes, 200), "Raw accounting differs")
            raw_steps += len(steps)
            if name == "none":
                clean = episodes
                clean_steps = {(s["episode"], s["step"]): s for s in steps}
            else:
                diverged = set()
                for step in steps:
                    verify_random_seed(step, seed, name)
                    ep = step["episode"]
                    if ep in diverged:
                        continue
                    baseline = clean_steps.get((ep, step["step"]))
                    require(baseline is not None and baseline["observation"] == step["observation"], "Unpaired initial transition prefix")
                    if baseline["action"] != step["action"]:
                        diverged.add(ep)
                    else:
                        require(all(baseline[k] == step[k] for k in ("reward", "terminated", "next_observation", "safety")),
                                "Same action prefix produced different transition")
            summary = entry["summary"]
            require({k: v for k, v in summary.items() if k != "safety"} == summarize_episodes(episodes, None if name == "none" else clean),
                    "Outcome or conversion denominator differs from episode records")
            costs = {k: sum(s["attack_cost"].get(k, 0) for s in steps)
                     for k in ("objective_evaluations", "gradient_evaluations", "policy_forward_calls")}
            costs.update(new_shadow_transitions=0, shadow_steps=0, replay_steps=0, warmup_steps=0,
                         evaluator_policy_forward_calls=sum(e["research_audit"]["evaluator_policy_forward_calls"] for e in episodes))
            rows.append(dict(checkpoint_seed=checkpoint, attack=name, episodes=20, checkpoint_sha256=entry["checkpoint_sha256"],
                             summary=summary, collisions=sum(e["ego_collision_observed"] for e in episodes), costs=costs, episode_rows=episodes))
    require(conditions == {(s, n) for s in range(5) for n in names}, "Incomplete checkpoint grid")
    reference_batch = ROOT / protocol["clean_reference_batch"]
    regression = verify_v0_reference(batch_path, reference_batch, clean_only=True)
    read(reference_batch)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), universal_newlines=True).strip()
    require(commit == batch["git_commit"] and not subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT)),
            "Use one clean commit for simple runs and audit")
    return dict(kind="basic_validation_controls", verified=True, git_commit=commit, analysis_commit=commit,
                attack_seed=seed, batch=str(batch_path.relative_to(ROOT)), rows=rows, source_sha256=sources,
                raw_episode_verified_steps=raw_steps, clean_regression=regression)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--attack-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.batch.resolve(), args.attack_seed)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("BASIC_VALIDATION_AUDIT_PASSED seed=%d steps=%d" % (args.attack_seed, result["raw_episode_verified_steps"]))
