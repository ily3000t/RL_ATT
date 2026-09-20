"""Descriptive paired-trace diagnostics; no counterfactual dynamics claims."""

import numpy as np


def paired_trace(clean, attacked):
    first = None
    for index, (base, row) in enumerate(zip(clean, attacked)):
        if base["step"] != index or row["step"] != index:
            raise ValueError("Trace steps must be consecutive")
        if not np.array_equal(base["observation"], row["observation"]):
            raise ValueError("Environment diverged before the first action difference")
        if base["action"] != row["action"]:
            first = index
            break
        for key in ("next_observation", "reward", "terminated", "safety"):
            if base[key] != row[key]:
                raise ValueError("Matched action prefix has inconsistent transitions")
    if first is None and len(clean) != len(attacked):
        raise ValueError("Identical action prefix has different termination lengths")
    collisions = [r["step"] for r in attacked if r["safety"]["ego_collision_observed"]]
    lane_transitions = [r["step"] for r in attacked if r["safety"]["ego_present"]
                        and r["observation"][13] != r["next_observation"][13]]
    local_changes = sum(r["action"] != r["clean_action_at_visited_state"] for r in attacked)
    return dict(first_action_divergence=first,
                first_collision_step=collisions[0] if collisions else None,
                collision_steps=collisions, lane_index_transition_steps=lane_transitions,
                steps=len(attacked), terminated=attacked[-1]["terminated"],
                local_action_changes=local_changes,
                divergence_to_collision=(collisions[0]-first if collisions and first is not None else None))
