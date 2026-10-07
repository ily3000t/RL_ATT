"""Recompute defense KL, clean-anchored bounds and costs from raw numeric probes."""

import numpy as np
from .state import require


def audit_update(row, config, batch_size):
    arrays = {name: np.asarray(row[name], dtype=np.float64) for name in
              ("clean_observation", "adversarial_observation", "clean_probability", "adversarial_probability")}
    require(all(np.isfinite(value).all() for value in arrays.values()), "Non-finite raw consistency update")
    clean, adversarial = arrays["clean_observation"], arrays["adversarial_observation"]
    require(clean.shape == adversarial.shape == (batch_size, 16), "Wrong consistency observation shape")
    radius = config["epsilon"] * (config["multiplicative"] * np.abs(clean) + config["additive"])
    violation = float(np.max(np.abs(adversarial - clean) - radius))
    require(violation <= 2e-6, "Raw consistency perturbation exceeds box")
    p, q = arrays["clean_probability"], arrays["adversarial_probability"]
    require(p.shape == q.shape == (batch_size, 3) and (p >= 0).all() and (q >= 0).all() and
            np.allclose(p.sum(axis=1), 1., atol=1e-6, rtol=0) and np.allclose(q.sum(axis=1), 1., atol=1e-6, rtol=0),
            "Invalid raw categorical probabilities")
    p, q = np.maximum(p, config["probability_floor"]), np.maximum(q, config["probability_floor"])
    p, q = p / p.sum(axis=1, keepdims=True), q / q.sum(axis=1, keepdims=True)
    kl = float(np.maximum((p * (np.log(p) - np.log(q))).sum(axis=1), 0.).mean())
    metrics = row["metrics"]
    require(all(np.isfinite(value) for value in metrics.values()) and metrics["batch_size"] == batch_size,
            "Invalid raw consistency metrics")
    require(abs(kl - metrics["consistency_loss"]) <= 2e-6 and abs(kl - metrics["selected_kl_mean"]) <= 2e-6,
            "Raw KL differs from selected/training objective")
    require(kl + 2e-6 >= metrics["initial_kl_mean"] and metrics["coefficient"] == config["coefficient"],
            "Consistency best candidate or coefficient drift")
    require(abs(metrics["total_actor_loss"] - metrics["base_actor_loss"] - config["coefficient"] * kl) <=
            2e-6 * max(1., abs(metrics["total_actor_loss"])), "Consistency total actor loss mismatch")
    for key, expected in dict(gradient_evaluations=config["steps"], search_actor_forward_calls=config["steps"] + 1,
                             search_observation_rows=(config["steps"] + 1) * batch_size,
                             candidate_observations=(config["steps"] + 1) * batch_size).items():
        require(metrics[key] == expected, "Consistency search cost mismatch: " + key)
    require(abs(max(0., violation) - metrics["max_box_violation"]) <= 2e-6 and
            metrics["selected_perturbed_rows"] == int((np.abs(adversarial - clean).sum(axis=1) > 0).sum()),
            "Consistency perturbation summary mismatch")
    return dict(kl=kl, max_box_violation=max(0., violation))


def audit_training(rows, config, state, ledger, segment_updates):
    counts = state["counts"]
    updates, batch_size = ledger["counts"]["primary_updates"], 128
    require(counts["updates"] == updates and len(rows) == segment_updates, "Missing consistency update probes")
    require([row["training_update"] for row in rows] == list(range(updates - segment_updates + 1, updates + 1)),
            "Consistency updates skipped/duplicated")
    checked = [audit_update(row, config, batch_size) for row in rows]
    for key, per_update in dict(updates=1, gradient_evaluations=config["steps"],
            search_actor_forward_calls=config["steps"] + 1, search_observation_rows=(config["steps"] + 1) * batch_size,
            candidate_observations=(config["steps"] + 1) * batch_size,
            regularizer_actor_forward_calls=1, regularizer_observation_rows=batch_size).items():
        require(counts[key] == per_update * updates, "Cumulative consistency count drift: " + key)
    require(ledger["counts"]["candidate_observations"] == counts["candidate_observations"], "Candidate ledger drift")
    phase = ledger["phases"]["primary_update"]
    calls = (config["steps"] + 4) * updates
    require(phase["network_forward_calls"]["actor"] == calls and phase["network_observation_rows"]["actor"] == calls * batch_size,
            "Actual actor hooks differ from declared PGD costs")
    require(all(ledger["counts"][name] == 0 for name in ("bo_objective_evaluations", "auxiliary_updates", "attacked_interaction_steps",
                "privileged_branch_steps", "privileged_replay_steps")), "Undeclared defense information or training path")
    require(rows and rows[-1]["metrics"] == state["last"], "Last consistency probe differs from snapshot")
    return dict(verified=True, segment_updates=segment_updates, cumulative_counts=counts,
                raw_kl_min=min(value["kl"] for value in checked), raw_kl_max=max(value["kl"] for value in checked),
                max_box_violation=max(value["max_box_violation"] for value in checked))
