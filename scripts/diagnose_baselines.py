"""Audit frozen policy actions, paired failures and crossed traffic controls."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
from run_baseline import ROOT, git, capture
import os

sys.path.insert(0, str(ROOT))
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.evaluation.trace_diagnostics import paired_trace
from rl_att.evaluation.results import summarize_episodes
from rl_att.evaluation.sumo_metrics import summarize_safety
from rl_att.evaluation.seed_control import evaluation_seeds
from summarize_benchmark import verify_steps


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def same(actual, expected):
    if isinstance(expected, dict):
        return all(k in actual and same(actual[k], v) for k,v in expected.items())
    if isinstance(expected, float):
        return bool(np.isclose(actual, expected, atol=1e-9, rtol=1e-12))
    return actual == expected


def load_steps(directory):
    episodes = [[] for _ in range(20)]
    with (directory / "steps.jsonl").open(encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            episode = episodes[row["episode"]-1]
            if row["step"] != len(episode):
                raise ValueError("Nonconsecutive raw steps")
            episode.append(row)
    if not all(episodes):
        raise ValueError("Missing diagnostic episodes")
    return episodes


def policy_audit(victim, rows):
    clean_margins, target_hits, target_total, roundoff_checks = [], 0, 0, 0
    for begin in range(0, len(rows), 512):
        batch = rows[begin:begin+512]
        clean = np.array([r["observation"] for r in batch])
        adv = np.array([r["adversarial_observation"] for r in batch])
        delta = np.array([r["perturbation"] for r in batch])
        if not np.isfinite(adv).all() or not np.allclose(adv-clean, delta, atol=1e-7, rtol=0):
            raise ValueError("Invalid perturbation record")
        if np.any(abs(delta) > .2*abs(clean)+.05+2e-7):
            raise ValueError("Perturbation exceeded frozen budget")
        logits = victim.logits(clean).numpy()
        clean_actions = victim.probabilities(clean).argmax(-1).numpy()
        adv_actions = victim.probabilities(adv).argmax(-1).numpy()
        margins = np.sort(logits, axis=1)[:,-1] - np.sort(logits, axis=1)[:,-2]
        clean_margins.extend(float(v) for v in margins)
        for i,r in enumerate(batch):
            for observed, expected, state in ((clean_actions[i], r["clean_action_at_visited_state"], clean[i]),
                                               (adv_actions[i], r["action"], adv[i])):
                if observed != expected:
                    # Batch GEMM may round differently from the original single-state call.
                    roundoff_checks += 1
                    if victim.action(state) != expected:
                        raise ValueError("Recorded action does not match the frozen victim")
            if "target_action" in r["attack_metadata"]:
                target_total += 1
                target_hits += int(r["attack_metadata"]["target_action"] == r["action"])
    return dict(recorded_actions_verified=2*len(rows), batch_roundoff_rechecks=roundoff_checks,
                clean_logit_margin_p05=float(np.percentile(clean_margins,5)),
                clean_logit_margin_median=float(np.median(clean_margins)),
                target_hits=target_hits, target_total=target_total,
                target_hit_rate=target_hits/target_total if target_total else None)


def detail(victim, baseline, attacked, diagnosis):
    first = diagnosis["first_action_divergence"]
    if first is None:
        return None
    r = attacked[first]
    return dict(step=first, clean_action=baseline[first]["action"], attacked_action=r["action"],
                observation=r["observation"], perturbation=r["perturbation"],
                adversarial_observation=r["adversarial_observation"],
                clean_logits=victim.logits(r["observation"]).tolist(),
                adversarial_logits=victim.logits(r["adversarial_observation"]).tolist(),
                clean_probabilities=victim.probabilities(r["observation"]).tolist(),
                adversarial_probabilities=victim.probabilities(r["adversarial_observation"]).tolist(),
                events=[dict(step=x["step"], action=x["action"], lane_before=x["observation"][13]*10,
                             lane_after=x["next_observation"][13]*10 if x["safety"]["ego_present"] else None,
                             reward=x["reward"], terminated=x["terminated"], collision=x["safety"]["ego_collision_observed"])
                        for x in attacked if x["step"] <= first+3 or x["step"] >= len(attacked)-3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain"):
        parser.error("Commit diagnostic code before analysis")
    if read(args.cross_batch / "batch.json")["status"] != "passed":
        parser.error("Wait for the crossed diagnostic batch to complete before analysis")
    output = args.output.resolve()
    if (ROOT / ".local").resolve() not in output.parents:
        parser.error("Raw diagnostic output must stay under .local")
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "docs/STAGE4_BENCHMARK_RESULTS.json"
    previous = read(source)
    references = {r["run_seed"]:r for r in read(ROOT / "configs/frozen_victims.json")["victims"] if r["victim"] == "clean"}
    manifest = dict(git_commit=git("rev-parse", "HEAD"), command=[sys.executable,*sys.argv],
                    python=sys.version, torch=torch.__version__, numpy=np.__version__, os=platform.platform(),
                    input_summary_sha256=sha(source), inputs={}, status="running",
                    pip_freeze=capture([sys.executable,"-m","pip","freeze","--all"], os.environ.copy()),
                    sumo_version=capture(["sumo","--version"], os.environ.copy()))
    write(output / "manifest.json", manifest)
    audits, all_cases, selected_cases = [], [], []
    old_episode_records = {}
    for seed in range(5):
        victim = VictimAdapter.from_reference(references[seed], ROOT)
        entries = [r for r in previous["runs"] if r["run_seed"] == seed]
        clean_dir = ROOT / next(r for r in entries if r["attack"] == "none")["result_directory"]
        baseline_steps = load_steps(clean_dir)
        baseline = read(clean_dir / "episodes.json")
        for entry in entries:
            name, directory = entry["attack"], ROOT / entry["result_directory"]
            manifest["inputs"][str(directory / "steps.jsonl")] = sha(directory / "steps.jsonl")
            traces, episodes = load_steps(directory), read(directory / "episodes.json")
            old_episode_records[(seed,name)] = episodes
            rows = [r for episode in traces for r in episode]
            audit = dict(run_seed=seed, attack=name, **policy_audit(victim, rows))
            cases = []
            for index, (base, trace, episode) in enumerate(zip(baseline_steps,traces,episodes)):
                diagnosis = paired_trace(base,trace)
                if diagnosis["steps"] != episode["steps"] or bool(diagnosis["collision_steps"]) != episode["ego_collision_observed"]:
                    raise ValueError("Episode outcome disagrees with collision trace")
                for sample in trace:
                    for pair in sample["safety"]["pairs"]:
                        gap, closing = pair["gap_m"], pair["closing_speed_mps"]
                        if gap > 0 and closing > 0 and not (np.isclose(pair["ttc_s"], gap/closing) and np.isclose(pair["drac_mps2"], closing**2/(2*gap))):
                            raise ValueError("Safety metric arithmetic mismatch")
                case = dict(run_seed=seed, attack=name, episode=index+1, **diagnosis,
                            clean_return=baseline[index]["episode_return"], attacked_return=episode["episode_return"],
                            return_drop=baseline[index]["episode_return"]-episode["episode_return"])
                cases.append(case)
                if seed == 2 and name in ("fgsm","zero_one") and index in (0,1,2):
                    selected_cases.append(dict(case, first_divergence=detail(victim,base,trace,diagnosis)))
            summary = summarize_episodes(episodes, baseline)
            summary["safety"] = summarize_safety([r["safety"] for r in rows])
            for key in ("episode_return_mean", "collision_rate", "attack_rate", "safety"):
                if not same(summary[key], entry["summary"][key]):
                    raise ValueError("Recomputed metric differs from published summary: "+key)
            firsts = [c["first_action_divergence"] for c in cases if c["first_action_divergence"] is not None]
            delays = [c["divergence_to_collision"] for c in cases if c["divergence_to_collision"] is not None]
            audit.update(episodes_with_divergence=len(firsts), first_divergence_median=float(np.median(firsts)) if firsts else None,
                         divergence_to_collision_median=float(np.median(delays)) if delays else None,
                         first_step_divergences=sum(v==0 for v in firsts), collision_episodes=sum(bool(c["collision_steps"]) for c in cases),
                         terminated_episodes=sum(c["terminated"] for c in cases),
                         lane_index_transitions=sum(len(c["lane_index_transition_steps"]) for c in cases),
                         steps=len(rows), local_action_change_rate=sum(c["local_action_changes"] for c in cases)/len(rows))
            audits.append(audit)
            all_cases.extend(cases)
        victim.assert_frozen()
        print("TRACE_DIAGNOSED="+str(seed), flush=True)
    batch = read(args.cross_batch / "batch.json")
    if batch["status"] != "passed" or len(batch["runs"]) != 5:
        raise ValueError("Crossed diagnostic batch is incomplete")
    cells, seen, diagonal = [], set(), 0
    for child in batch["runs"]:
        directory = Path(child["run_dir"])
        run, evaluation = read(directory / "manifest.json"), read(directory / "evaluation.json")
        if run["status"] != "passed" or child["returncode"] != 0 or run["git_commit"] != batch["git_commit"]:
            raise ValueError("Crossed run provenance mismatch")
        for key in ("pip_freeze", "sumo_version"):
            if manifest[key]["stdout"] != run[key]["stdout"]:
                raise ValueError("Diagnostic runtime changed: "+key)
        group = run["config"]["traffic_seed"]
        manifest["inputs"][str(directory / "manifest.json")] = sha(directory / "manifest.json")
        for seed in range(5):
            if (seed,group) in seen:
                raise ValueError("Repeated crossed cell")
            seen.add((seed,group))
            pair = [e for e in evaluation["runs"] if e["run_seed"] == seed]
            if [e["attack"]["name"] for e in pair] != ["none","fgsm"]:
                raise ValueError("Crossed control methods changed")
            baseline = read(directory / pair[0]["results_directory"] / "episodes.json")
            outcomes = {}
            for entry in pair:
                name = entry["attack"]["name"]
                if not entry["frozen_unchanged"] or entry["checkpoint_sha256"] != references[seed]["checkpoint_sha256"]:
                    raise ValueError("Crossed checkpoint changed")
                if not same(entry["effective_seeds"], {k:v for k,v in evaluation_seeds(seed,20,group).items() if k not in ("policy_rng","attack_rng")}):
                    raise ValueError("Crossed seed allocation differs from protocol")
                path = directory / entry["results_directory"]
                episodes = read(path / "episodes.json")
                verify_steps(path,name,episodes)
                manifest["inputs"][str(path / "steps.jsonl")] = sha(path / "steps.jsonl")
                if seed == group:
                    old = old_episode_records[(seed,name)]
                    if any({k:v for k,v in a.items() if k!='attack_wall_seconds'} != {k:v for k,v in b.items() if k!='attack_wall_seconds'} for a,b in zip(old,episodes)):
                        raise ValueError("Diagonal differs from original baseline")
                    diagonal += len(episodes)
                outcomes[name] = summarize_episodes(episodes,baseline)
            cells.append(dict(checkpoint_seed=seed, traffic_group=group, clean_return=outcomes['none']['episode_return_mean'],
                              fgsm_return=outcomes['fgsm']['episode_return_mean'], return_drop=outcomes['fgsm']['return_drop_mean'],
                              clean_collision_rate=outcomes['none']['collision_rate'], fgsm_collision_rate=outcomes['fgsm']['collision_rate']))
    if len(seen) != 25 or diagonal != 200:
        raise ValueError("Incomplete crossed matrix or diagonal proof")
    result = dict(kind="baseline_diagnosis_not_ablation", analysis_commit=manifest['git_commit'],
                  cross_experiment_commit=batch['git_commit'], cross_batch=str(args.cross_batch),
                  original_trace_steps=sum(r['steps'] for r in audits), recorded_actions_verified=sum(r['recorded_actions_verified'] for r in audits),
                  diagonal_episodes_exactly_matched=diagonal, cross_episodes=1000, audits=audits,
                  crossed_cells=sorted(cells,key=lambda c:(c['checkpoint_seed'],c['traffic_group'])),
                  selected_seed2_cases=selected_cases,
                  seed2_zero_one_episodes=[c for c in all_cases if c['run_seed']==2 and c['attack']=='zero_one'])
    write(output / "all_episode_cases.json",all_cases)
    write(output / "summary.json",result)
    manifest.update(status="passed", summary_sha256=sha(output/'summary.json'),
                    all_episode_cases_sha256=sha(output/'all_episode_cases.json'))
    write(output / "manifest.json",manifest)
    print("DIAGNOSIS_PASSED="+str(output))


if __name__ == "__main__":
    main()
