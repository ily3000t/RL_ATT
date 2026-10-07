"""Controlled historical interventions, not newly generated adaptive attacks."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from .defense_diagnostics import require, verify_trace


ARMS = ("no_attack", "single_then_clean", "sustained_reference")
TRACE_KEYS = ("episode", "step", "observation", "adversarial_observation", "perturbation",
              "next_observation", "action", "clean_action_at_visited_state", "reward",
              "terminated", "attacked", "safety")


def fingerprint(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def first_divergence(history):
    return next((r["step"] for r in history if r["action"] != r["clean_action_at_visited_state"]), None)


def intervention(arm, step, observation, history, divergence):
    require(arm in ARMS, "Unknown intervention arm")
    observation = np.asarray(observation, dtype=np.float64)
    active = arm == "sustained_reference" or (arm == "single_then_clean" and step == divergence)
    if active:
        require(step < len(history), "Replay outlasted historical episode")
        row = history[step]
        require(np.array_equal(observation, row["observation"]), "Cannot replay perturbation at an off-path state")
        return np.asarray(row["adversarial_observation"], dtype=np.float64).copy(), True
    return observation.copy(), False


def assert_historical_step(actual, expected):
    for key in ("observation", "next_observation"):
        require(np.array_equal(actual[key], expected[key]), "Historical observation mismatch: " + key)
    for key in ("action", "clean_action_at_visited_state", "reward", "terminated", "safety"):
        require(actual[key] == expected[key], "Historical transition mismatch: " + key)


def episode_summary(rows, sumo_seed, reset, warmup_steps, forwards, wall_seconds):
    require(bool(rows), "Empty controlled episode")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray(row["next_observation"], dtype=np.float64).tobytes())
        digest.update(np.asarray([row["action"], row["reward"], row["terminated"]], dtype=np.float64).tobytes())
    result = dict(episode=rows[0]["episode"], sumo_seed=sumo_seed, steps=len(rows),
                  episode_return=sum(r["reward"] for r in rows),
                  action_counts=[sum(r["action"] == a for r in rows) for a in range(3)],
                  trajectory_sha256=digest.hexdigest(), terminated=rows[-1]["terminated"],
                  truncated=not rows[-1]["terminated"],
                  ego_collision_observed=any(r["safety"]["ego_collision_observed"] for r in rows),
                  first_collision_real_step=next((r["step"] + 1 for r in rows if r["safety"]["ego_collision_observed"]), None),
                  action_changed_steps=sum(r["action"] != r["clean_action_at_visited_state"] for r in rows),
                  attacked_steps=sum(r["attacked"] for r in rows),
                  changed_steps=sum(any(r["perturbation"]) for r in rows), reset=reset,
                  resources=dict(real_interaction_steps=len(rows), reset_warmup_steps=warmup_steps,
                                 policy_forward_calls=forwards, gradient_evaluations=0,
                                 new_attack_search_transitions=0, wall_seconds=wall_seconds))
    return result


def audit_case(baseline, single, sustained, history, historical_episode, max_steps):
    """Recompute all summaries and check the shared pre-intervention prefix."""
    divergence = first_divergence(history)
    for arm in (baseline, single, sustained):
        verify_trace(arm["rows"], arm["episode"], max_steps)
    for actual, expected in zip(sustained["rows"], history):
        assert_historical_step(actual, expected)
    require(len(sustained["rows"]) == len(history) and
            sustained["episode"]["trajectory_sha256"] == historical_episode["trajectory_sha256"],
            "Sustained replay did not reproduce historical trajectory")
    require(baseline["episode"]["reset"] == single["episode"]["reset"] == sustained["episode"]["reset"],
            "Arms have different reset states or warmup events")
    expected_interventions = [] if divergence is None else [divergence]
    require([r["step"] for r in single["rows"] if r["replayed_input"]] == expected_interventions,
            "Single arm must intervene exactly once at first action divergence")
    for row in single["rows"]:
        if row["step"] != divergence:
            require(row["adversarial_observation"] == row["observation"] and
                    row["action"] == row["clean_action_at_visited_state"], "Single arm did not return to normal input")
    prefix = len(history) if divergence is None else divergence
    require(len(baseline["rows"]) >= prefix and len(single["rows"]) >= prefix, "Incomplete shared prefix")
    for index in range(prefix):
        assert_historical_step(single["rows"][index], baseline["rows"][index])
        assert_historical_step(sustained["rows"][index], baseline["rows"][index])
    if divergence is not None:
        branch = [arm["rows"][divergence] for arm in (baseline, single, sustained)]
        require(all(r["pre_state_sha256"] is not None for r in branch) and
                len({r["pre_state_sha256"] for r in branch}) == 1, "Branch-point observable SUMO states differ")
        require(branch[1]["action"] == branch[2]["action"] != branch[0]["action"], "First intervention did not change action")
        assert_historical_step(branch[1], branch[2])
    else:
        require(single["episode"]["trajectory_sha256"] == baseline["episode"]["trajectory_sha256"] ==
                sustained["episode"]["trajectory_sha256"], "No-divergence case changed trajectory")
    b, s, a = (arm["episode"] for arm in (baseline, single, sustained))
    eligible = not b["ego_collision_observed"]
    reset_ego_collision = bool(b["reset"]["warmup_ego_collision_steps"])
    return dict(episode=b["episode"], sumo_seed=b["sumo_seed"], first_action_divergence=divergence,
                no_intervention=divergence is None, clean_return=b["episode_return"],
                single_return=s["episode_return"], sustained_return=a["episode_return"],
                single_return_drop=b["episode_return"] - s["episode_return"],
                sustained_return_drop=b["episode_return"] - a["episode_return"],
                sustained_minus_single_drop=s["episode_return"] - a["episode_return"],
                baseline_collision=b["ego_collision_observed"], single_collision=s["ego_collision_observed"],
                sustained_collision=a["ego_collision_observed"], asr_eligible=eligible,
                reset_ego_collision=reset_ego_collision,
                reset_clear_eligible=eligible and not reset_ego_collision,
                single_conversion=eligible and s["ego_collision_observed"],
                sustained_conversion=eligible and a["ego_collision_observed"],
                single_reset_clear_conversion=eligible and not reset_ego_collision and s["ego_collision_observed"],
                sustained_reset_clear_conversion=eligible and not reset_ego_collision and a["ego_collision_observed"],
                single_first_collision_real_step=s["first_collision_real_step"],
                sustained_first_collision_real_step=a["first_collision_real_step"],
                single_early_conversion=eligible and s["first_collision_real_step"] is not None and s["first_collision_real_step"] <= 3,
                sustained_early_conversion=eligible and a["first_collision_real_step"] is not None and a["first_collision_real_step"] <= 3,
                single_late_conversion=eligible and s["first_collision_real_step"] is not None and s["first_collision_real_step"] >= 21,
                sustained_late_conversion=eligible and a["first_collision_real_step"] is not None and a["first_collision_real_step"] >= 21)


def summarize(cases):
    require(bool(cases), "Empty controlled cohort")
    keys = ("clean_return", "single_return", "sustained_return", "single_return_drop",
            "sustained_return_drop", "sustained_minus_single_drop")
    result = {k + "_mean": statistics.mean(c[k] for c in cases) for k in keys}
    for key in ("no_intervention", "baseline_collision", "single_collision", "sustained_collision",
                "asr_eligible", "reset_ego_collision", "reset_clear_eligible", "single_conversion", "sustained_conversion",
                "single_reset_clear_conversion", "sustained_reset_clear_conversion",
                "single_early_conversion", "sustained_early_conversion", "single_late_conversion", "sustained_late_conversion"):
        result[key + "_episodes"] = sum(c[key] for c in cases)
    result.update(episodes=len(cases), independent_traffic_clusters=len({c["sumo_seed"] for c in cases}),
                  collision_pairs=dict(both=sum(c["single_collision"] and c["sustained_collision"] for c in cases),
                                       single_only=sum(c["single_collision"] and not c["sustained_collision"] for c in cases),
                                       sustained_only=sum(not c["single_collision"] and c["sustained_collision"] for c in cases),
                                       neither=sum(not c["single_collision"] and not c["sustained_collision"] for c in cases)))
    return result


def audit_run(manifest_path):
    """Independent raw-file audit; imports no SUMO or policy implementation."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output, config = manifest_path.parent, manifest["config"]
    hashes, cases, resources = {}, [], []

    def read(path):
        content = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(content).hexdigest()
        return content.decode("utf-8")

    def load_arm(job, method, arm):
        directory = output / job["job_id"] / method / arm
        episodes = json.loads(read(directory / "episodes.json"))
        require([e["episode"] for e in episodes] == list(range(1, config["episodes_per_arm"] + 1)), "Incomplete episode grid")
        indexed = {e["episode"]: dict(episode=e, rows=[]) for e in episodes}
        for line in read(directory / "steps.jsonl").splitlines():
            row = json.loads(line)
            require(row["episode"] in indexed, "Unexpected controlled episode")
            indexed[row["episode"]]["rows"].append(row)
        for item in indexed.values():
            ep, rows = item["episode"], item["rows"]
            require(ep["sumo_seed"] == config["traffic_seeds"][ep["episode"] - 1], "Unexpected controlled traffic")
            reset = ep["reset"]
            require(reset["observation"] == rows[0]["observation"] and len(reset["warmup"]) == ep["resources"]["reset_warmup_steps"] and
                    reset["warmup_ego_collision_steps"] == [r["warmup_step"] for r in reset["warmup"] if "Auto" in r["colliding_ids"]],
                    "Invalid reset boundary accounting")
            require(reset["observable_state"]["colliding_ids"] == reset["warmup"][-1]["colliding_ids"], "Final reset collision inconsistent")
            for r in rows:
                require(r["pre_state_sha256"] == (fingerprint(r["pre_state"]) if r["pre_state"] is not None else None), "Corrupted branch fingerprint")
                delta = np.asarray(r["adversarial_observation"]) - r["observation"]
                require(r["attacked"] == bool(np.any(delta != 0)), "Inconsistent controlled intervention flag")
            resource = ep["resources"]
            require(resource["real_interaction_steps"] == len(rows) and resource["policy_forward_calls"] == 2 * len(rows) and
                    resource["gradient_evaluations"] == resource["new_attack_search_transitions"] == 0, "Invalid diagnostic cost ledger")
            recomputed = episode_summary(rows, ep["sumo_seed"], reset, resource["reset_warmup_steps"],
                                         resource["policy_forward_calls"], resource["wall_seconds"])
            require(recomputed == ep, "Controlled episode summary differs from raw steps")
            verify_trace(rows, ep, config["max_steps"])
            history = job["traces"][method][str(ep["episode"])]
            divergence = first_divergence(history)
            for index, row in enumerate(rows):
                supplied, replayed = intervention(arm, index, row["observation"], history, divergence)
                require(np.array_equal(supplied, row["adversarial_observation"]) and replayed == row["replayed_input"], "Wrong intervention policy")
            if arm == "no_attack":
                require(len(rows) == len(history), "Baseline length differs from historical episode")
                for row, old in zip(rows, history):
                    assert_historical_step(row, old)
                require(ep["trajectory_sha256"] == job["episodes"][method][ep["episode"] - 1]["trajectory_sha256"], "Baseline digest mismatch")
        total = json.loads(read(directory / "resources.json"))
        require(total["sumo_simulation_step_calls"] == sum(e["resources"]["real_interaction_steps"] + e["resources"]["reset_warmup_steps"] for e in episodes) and
                total["policy_forward_calls"] == sum(e["resources"]["policy_forward_calls"] for e in episodes), "Worker resource totals mismatch")
        resources.append(dict(job_id=job["job_id"], method=method, arm=arm, **total))
        return indexed

    for entry in manifest["jobs"]:
        job = json.loads(read(Path(entry["input_path"])))
        require(fingerprint(job) == entry["input_content_sha256"], "Input fingerprint mismatch")
        baseline = load_arm(job, "none", "no_attack")
        for method in config["attacks"]:
            single = load_arm(job, method, "single_then_clean")
            sustained = load_arm(job, method, "sustained_reference")
            for episode in baseline:
                case = audit_case(baseline[episode], single[episode], sustained[episode],
                                  job["traces"][method][str(episode)], job["episodes"][method][episode - 1], config["max_steps"])
                case.update(victim=job["reference"]["victim"], checkpoint_seed=job["reference"]["run_seed"], attack=method)
                cases.append(case)
    require(len(cases) == manifest["expected_cases"], "Incomplete controlled diagnostic grid")
    require(len(resources) * config["episodes_per_arm"] == manifest["expected_physical_episodes"], "Wrong physical episode count")
    groups, traffic = [], []
    for victim in config["victims"]:
        for method in config["attacks"]:
            group = [c for c in cases if c["victim"] == victim and c["attack"] == method]
            groups.append(dict(victim=victim, attack=method, **summarize(group)))
            for seed in config["traffic_seeds"]:
                traffic.append(dict(victim=victim, attack=method, sumo_seed=seed,
                                    **summarize([c for c in group if c["sumo_seed"] == seed])))
    return dict(kind=config["kind"], verified=True, git_commit=manifest["git_commit"], group=manifest["group"],
                inherited_execution_commit=config["expected_execution_commit"], inherited_audit_sha256=config["expected_audit_sha256"],
                source_inputs_sha256=manifest["historical_inputs_sha256"], output_inputs_sha256=hashes,
                physical_episodes=manifest["expected_physical_episodes"], paired_cases=len(cases),
                independent_traffic_clusters=len(config["traffic_seeds"]), summary=groups, per_traffic=traffic,
                episode_cases=cases, resources=resources,
                total_policy_forward_calls=sum(r["policy_forward_calls"] for r in resources),
                total_sumo_simulation_steps=sum(r["sumo_simulation_step_calls"] for r in resources),
                interpretation=config["interpretation"])
