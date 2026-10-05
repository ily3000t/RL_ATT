"""Read-only defense trace diagnostics; timing is not a recovery experiment."""

import hashlib
import statistics

import numpy as np

from .trace_diagnostics import paired_trace


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_trace(trace, episode, max_steps):
    """Validate the complete trace, including steps after policy divergence."""
    require(bool(trace) and len(trace) <= max_steps, "Empty/overlong episode")
    digest, reward, counts = hashlib.sha256(), 0.0, [0, 0, 0]
    changes = attacked = changed = 0
    previous = None
    for index, row in enumerate(trace):
        require(row["step"] == index and row["episode"] == episode["episode"],
                "Nonconsecutive/mixed episode trace")
        vectors = {}
        for key in ("observation", "adversarial_observation", "perturbation", "next_observation"):
            vector = np.asarray(row[key], dtype=np.float64)
            require(vector.shape == (16,) and np.isfinite(vector).all(), "Invalid observation vector")
            vectors[key] = vector
        obs, adv, delta = (vectors[k] for k in ("observation", "adversarial_observation", "perturbation"))
        require(np.allclose(adv - obs, delta, atol=2e-7, rtol=0), "Inconsistent perturbation")
        require(np.all(abs(delta) <= .2 * abs(obs) + .05 + 2e-7), "Out-of-box perturbation")
        if previous is not None:
            require(np.array_equal(previous, obs), "Discontinuous episode observations")
        previous = vectors["next_observation"]
        for key in ("action", "clean_action_at_visited_state"):
            require(type(row[key]) is int and row[key] in (0, 1, 2), "Invalid discrete action")
        for value in (row["terminated"], row["attacked"], row["safety"]["ego_collision_observed"],
                      row["safety"]["ego_present"]):
            require(type(value) is bool, "Invalid boolean trace field")
        require(not row["terminated"] or index == len(trace) - 1, "Steps after termination")
        require(np.isfinite(row["reward"]), "Nonfinite reward")
        reward += row["reward"]
        counts[row["action"]] += 1
        changes += int(row["action"] != row["clean_action_at_visited_state"])
        attacked += int(row["attacked"])
        changed += int(np.any(delta != 0))
        digest.update(vectors["next_observation"].tobytes())
        digest.update(np.asarray([row["action"], row["reward"], row["terminated"]], dtype=np.float64).tobytes())
    require(episode["steps"] == len(trace) and episode["action_counts"] == counts,
            "Episode steps/actions disagree with trace")
    require(abs(reward - episode["episode_return"]) <= 1e-9, "Episode return disagrees with trace")
    require(episode["trajectory_sha256"] == digest.hexdigest(), "Trajectory digest mismatch")
    require(episode["terminated"] == trace[-1]["terminated"] and
            episode["truncated"] == (not trace[-1]["terminated"]), "Termination summary mismatch")
    require(trace[-1]["terminated"] or len(trace) == max_steps, "Premature unrecorded truncation")
    require(episode["ego_collision_observed"] == any(r["safety"]["ego_collision_observed"] for r in trace),
            "Collision summary mismatch")
    require((episode["action_changed_steps"], episode["attacked_steps"], episode["changed_steps"]) ==
            (changes, attacked, changed), "Attack/action counters disagree with trace")


def diagnose_pair(clean, attacked, reference, episode, max_steps=200, early_steps=3, late_first_step=21):
    """Compare one policy to its own no-attack reference on the same traffic."""
    require(0 < early_steps < late_first_step <= max_steps, "Invalid timing thresholds")
    require((reference["episode"], reference["sumo_seed"]) == (episode["episode"], episode["sumo_seed"]),
            "Unpaired reference traffic")
    verify_trace(clean, reference, max_steps)
    verify_trace(attacked, episode, max_steps)
    diagnosis = paired_trace(clean, attacked)
    changes = [r["step"] for r in attacked if r["action"] != r["clean_action_at_visited_state"]]
    first_change = changes[0] if changes else None
    require(first_change == diagnosis["first_action_divergence"],
            "First local action change disagrees with paired divergence")
    first = diagnosis["first_collision_step"]
    collision = first is not None
    eligible = not reference["ego_collision_observed"]
    if not collision:
        timing = "no_collision"
    elif first < early_steps:
        timing = "first_3_real_steps"
    elif first + 1 >= late_first_step:
        timing = "real_step_21_or_later"
    else:
        timing = "real_steps_4_to_20"
    local_counts = [[0] * 3 for _ in range(3)]
    target_hits = target_queries = witness_queries = alternate_witnesses = 0
    costs = {k: 0 for k in ("objective_evaluations", "gradient_evaluations", "policy_forward_calls",
                            "new_shadow_transitions", "shadow_steps", "replay_steps", "warmup_steps")}
    for row in attacked:
        local_counts[row["clean_action_at_visited_state"]][row["action"]] += 1
        target = row["attack_metadata"].get("target_action")
        if target is not None:
            require(type(target) is int and target in (0, 1, 2), "Invalid target action")
            target_queries += 1
            target_hits += int(row["action"] == target)
        witness = row["attack_metadata"].get("planned_action")
        if witness is not None:
            require(type(witness) is int and witness in (0, 1, 2) and witness == row["action"],
                    "Execution differs from planned witness action")
            witness_queries += 1
            alternate_witnesses += int(target is not None and target != witness)
        for key in costs:
            value = row["attack_cost"].get(key, 0)
            require(type(value) is int and value >= 0, "Invalid resource count")
            costs[key] += value
    return dict(episode=episode["episode"], sumo_seed=episode["sumo_seed"], **diagnosis,
                first_local_action_change=first_change, collision_timing=timing,
                clean_collision=reference["ego_collision_observed"], asr_eligible=eligible,
                collision_conversion=eligible and collision, clean_return=reference["episode_return"],
                attacked_return=episode["episode_return"],
                return_drop=reference["episode_return"] - episode["episode_return"],
                first_collision_real_step=first + 1 if collision else None,
                ego_missing_steps=sum(not r["safety"]["ego_present"] for r in attacked),
                local_action_counts=local_counts, target_queries=target_queries, target_hits=target_hits,
                witness_verified_steps=witness_queries, requested_target_mismatches_with_valid_witness=alternate_witnesses,
                costs=costs)


def summarize_cases(cases):
    require(bool(cases), "Empty diagnostic cohort")
    steps = sum(c["steps"] for c in cases)
    eligible = sum(c["asr_eligible"] for c in cases)
    conversions = sum(c["collision_conversion"] for c in cases)
    collisions = [c for c in cases if c["first_collision_step"] is not None]
    delays = [c["divergence_to_collision"] for c in collisions
              if c["divergence_to_collision"] is not None and c["divergence_to_collision"] >= 0]
    timing = {k: sum(c["collision_timing"] == k for c in cases)
              for k in ("first_3_real_steps", "real_steps_4_to_20", "real_step_21_or_later", "no_collision")}
    local = [[sum(c["local_action_counts"][i][j] for c in cases) for j in range(3)] for i in range(3)]
    queries = sum(c["target_queries"] for c in cases)
    hits = sum(c["target_hits"] for c in cases)
    costs = {k: sum(c["costs"][k] for c in cases) for k in cases[0]["costs"]}
    return dict(episodes=len(cases), real_steps=steps,
                clean_return_mean=statistics.mean(c["clean_return"] for c in cases),
                attacked_return_mean=statistics.mean(c["attacked_return"] for c in cases),
                return_drop_mean=statistics.mean(c["return_drop"] for c in cases),
                baseline_collision_episodes=sum(c["clean_collision"] for c in cases),
                collision_episodes=len(collisions), collision_timing=timing,
                asr_eligible=eligible, collision_conversions=conversions,
                asr=conversions / eligible if eligible else None,
                early_collision_conversions=sum(c["collision_conversion"] and c["collision_timing"] == "first_3_real_steps" for c in cases),
                late_collision_conversions=sum(c["collision_conversion"] and c["collision_timing"] == "real_step_21_or_later" for c in cases),
                action_changed_steps=sum(c["local_action_changes"] for c in cases),
                action_change_rate=sum(c["local_action_changes"] for c in cases) / steps,
                episodes_with_action_divergence=sum(c["first_action_divergence"] is not None for c in cases),
                divergence_to_collision_median=statistics.median(delays) if delays else None,
                collision_delays_with_prior_divergence=len(delays),
                lane_index_transitions=sum(len(c["lane_index_transition_steps"]) for c in cases),
                terminated_episodes=sum(c["terminated"] for c in cases),
                ego_missing_steps=sum(c["ego_missing_steps"] for c in cases),
                local_action_counts=local, target_queries=queries, target_hits=hits,
                witness_verified_steps=sum(c["witness_verified_steps"] for c in cases),
                requested_target_mismatches_with_valid_witness=sum(c["requested_target_mismatches_with_valid_witness"] for c in cases),
                target_hit_rate=hits / queries if queries else None, costs=costs)


def index_grid(entries, victims, seeds, names):
    indexed = {(e["victim"], e["run_seed"], e["attack"]["name"]): e for e in entries}
    expected = {(v, s, a) for v in victims for s in seeds for a in names}
    require(len(indexed) == len(entries) and set(indexed) == expected, "Incomplete/duplicate diagnostic grid")
    return indexed
