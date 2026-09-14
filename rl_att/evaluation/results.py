"""Explicit denominators and paired outcome summaries; no invented safety values."""

import json
import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize_episodes(rows, reference=None):
    if not rows:
        raise ValueError("Cannot summarize empty evaluation")
    steps = sum(r["steps"] for r in rows)
    result = {"episodes": len(rows), "steps": steps,
              "episode_return_mean": float(np.mean([r["episode_return"] for r in rows])),
              "collision_rate": sum(r["ego_collision_observed"] for r in rows) / len(rows),
              "attack_rate": sum(r["attacked_steps"] for r in rows) / steps,
              "observation_changed_rate": sum(r["changed_steps"] for r in rows) / steps,
              "action_change_rate": sum(r["action_changed_steps"] for r in rows) / steps,
              "linf_max": max(r["linf_max"] for r in rows), "l2_max": max(r["l2_max"] for r in rows),
              "objective_evaluations": sum(r["objective_evaluations"] for r in rows),
              "attack_policy_forward_calls": sum(r["attack_policy_forward_calls"] for r in rows),
              "gradient_evaluations": sum(r.get("gradient_evaluations", 0) for r in rows),
              "scaled_linf_max": max(r["scaled_linf_max"] for r in rows) if all(r.get("scaled_linf_max") is not None for r in rows) else None,
              "attack_wall_seconds": sum(r["attack_wall_seconds"] for r in rows),
              "return_drop_mean": None, "attack_success_rate": None,
              "attack_success_definition": "collision conversion among paired non-collision clean episodes"}
    if reference is not None:
        if len(reference) != len(rows) or any((r["episode"], r["sumo_seed"]) != (b["episode"], b["sumo_seed"])
                                               for r, b in zip(rows, reference)):
            raise ValueError("Return drop requires matching episode and SUMO seeds")
        drops = [b["episode_return"] - r["episode_return"] for r, b in zip(rows, reference)]
        eligible = [(r, b) for r, b in zip(rows, reference) if not b["ego_collision_observed"]]
        successes = sum(r["ego_collision_observed"] for r, _ in eligible)
        result.update(return_drop_mean=float(np.mean(drops)),
                      attack_success_rate=successes / len(eligible) if eligible else None,
                      attack_successes=successes, attack_success_eligible_episodes=len(eligible))
    return result
