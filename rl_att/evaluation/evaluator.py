"""Environment-independent rollout loop; attacks cannot replace the frozen victim."""

import hashlib
import json
import numpy as np
from rl_att.attacks.base import AttackContext
from rl_att.attacks.box import box_scales
from .sumo_metrics import summarize_safety
from .results import write_json, summarize_episodes


def validate_budget(result, observation, budget):
    result.validate(observation)
    delta = np.asarray(result.perturbation)
    norm = budget["norm"]
    if norm == "none":
        if result.attacked or np.any(delta != 0):
            raise ValueError("No-attack budget violated")
    elif norm == "affine_box":
        if result.attacked:
            u1, u2 = (result.metadata["parameters"][key] for key in ("u1", "u2"))
            lo1, hi1 = budget["multiplicative_bounds"]
            lo2, hi2 = budget["additive_bounds"]
            if not lo1 <= u1 <= hi1 or not lo2 <= u2 <= hi2:
                raise ValueError("Affine parameter bounds violated")
            expected = u1 * np.asarray(observation, dtype=np.float32) + u2
            if not np.allclose(expected, result.adversarial_observation, rtol=0, atol=2e-7):
                raise ValueError("Affine observation does not match declared parameters")
    elif norm in ("linf", "l2"):
        value = np.linalg.norm(delta, ord=np.inf if norm == "linf" else 2)
        if value > budget["epsilon"] + 1e-7:
            raise ValueError("Perturbation norm budget violated")
    elif norm == "observation_scaled_linf":
        scales = box_scales(observation, budget["relative_scale"], budget["absolute_scale"])
        if np.any(np.abs(delta) > budget["epsilon"] * scales + 2e-7):
            raise ValueError("Observation-scaled perturbation envelope violated")
    else:
        raise ValueError("Unsupported perturbation budget")


class AttackEvaluator:
    def __init__(self, env, victim, attack, metrics, config, seeds, budget):
        self.env, self.victim, self.attack, self.metrics = env, victim, attack, metrics
        self.config, self.seeds, self.budget = config, seeds, budget

    def run(self, output, reference=None):
        output.mkdir(parents=True, exist_ok=False)
        rows, all_samples = [], []
        self.victim.assert_frozen()
        try:
            with (output / "steps.jsonl").open("w", encoding="utf-8") as raw:
                for index in range(self.config["episodes"]):
                    obs = self.env.reset()
                    score, collision, done = 0.0, False, False
                    actions, samples, digest = [0, 0, 0], [], hashlib.sha256()
                    attacked = changed = action_changed = evaluations = forward_calls = gradients = 0
                    wall_seconds = linf = l2 = scaled_linf = 0.0
                    for step in range(self.config["max_steps"]):
                        clean_obs = np.asarray(obs).copy()
                        clean_action = self.victim.action(clean_obs)
                        context = AttackContext(self.seeds["run_seed"], self.seeds["attack_seed"], index, step)
                        supplied = clean_obs.copy()
                        result = self.attack(supplied, self.victim, context)
                        if not np.array_equal(supplied, clean_obs):
                            raise ValueError("Attack mutated its input observation")
                        validate_budget(result, clean_obs, self.budget)
                        scaled_norm = None
                        if self.budget["norm"] == "observation_scaled_linf":
                            scales = box_scales(clean_obs, self.budget["relative_scale"], self.budget["absolute_scale"])
                            scaled_norm = float(np.max(np.abs(result.perturbation) / scales))
                            scaled_linf = max(scaled_linf, scaled_norm)
                        action = self.victim.action(result.adversarial_observation)
                        obs, reward, done = self.env.step(action)[:3]
                        if not np.isfinite(reward) or not np.isfinite(obs).all():
                            raise FloatingPointError("Non-finite environment transition")
                        score += float(reward)
                        actions[action] += 1
                        safety = self.metrics.sample()
                        samples.append(safety)
                        collision = collision or safety["ego_collision_observed"]
                        attacked += int(result.attacked)
                        changed += int(np.any(result.perturbation != 0))
                        action_changed += int(action != clean_action)
                        linf = max(linf, float(np.linalg.norm(result.perturbation, ord=np.inf)))
                        l2 = max(l2, float(np.linalg.norm(result.perturbation)))
                        evaluations += result.attack_cost["objective_evaluations"]
                        forward_calls += result.attack_cost["policy_forward_calls"]
                        gradients += result.attack_cost.get("gradient_evaluations", 0)
                        wall_seconds += result.attack_cost["wall_seconds"]
                        digest.update(np.asarray(obs, dtype=np.float64).tobytes())
                        digest.update(np.asarray([action, reward, done], dtype=np.float64).tobytes())
                        raw.write(json.dumps({"episode": index + 1, "step": step,
                                              "observation": clean_obs.tolist(),
                                              "adversarial_observation": result.adversarial_observation.tolist(),
                                              "perturbation": result.perturbation.tolist(), "attacked": result.attacked,
                                              "clean_action_at_visited_state": clean_action, "action": action,
                                              "reward": float(reward), "terminated": bool(done),
                                              "next_observation": np.asarray(obs).tolist(),
                                              "attack_cost": result.attack_cost, "attack_metadata": result.metadata,
                                              "scaled_linf": scaled_norm,
                                              "safety": safety}, allow_nan=False) + "\n")
                        if done:
                            break
                    row = {"episode": index + 1, "sumo_seed": self.seeds["episode_sumo_seeds"][index],
                           "episode_return": score, "steps": step + 1, "terminated": bool(done),
                           "truncated": not bool(done), "ego_collision_observed": collision,
                           "action_counts": actions, "trajectory_sha256": digest.hexdigest(),
                           "attacked_steps": attacked, "changed_steps": changed, "action_changed_steps": action_changed,
                           "linf_max": linf, "l2_max": l2, "objective_evaluations": evaluations,
                           "scaled_linf_max": scaled_linf if self.budget["norm"] == "observation_scaled_linf" else None,
                           "gradient_evaluations": gradients,
                           "attack_policy_forward_calls": forward_calls, "attack_wall_seconds": wall_seconds,
                           "safety": summarize_safety(samples, **self.config["metric_percentiles"])}
                    rows.append(row)
                    all_samples.extend(samples)
                    write_json(output / "episodes.json", rows)
                    raw.flush()
            summary = summarize_episodes(rows, reference)
            summary["safety"] = summarize_safety(all_samples, **self.config["metric_percentiles"])
            write_json(output / "summary.json", summary)
            return rows, summary
        finally:
            self.victim.assert_frozen()
