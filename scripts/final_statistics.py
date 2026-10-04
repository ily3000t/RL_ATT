"""Registered paired outcome statistics, conditional on five frozen checkpoints.

Input rows are actual episodes: checkpoint_seed, sumo_seed, attack_seed, value.
Deterministic FGSM outcomes may be broadcast for pairing. This function must not
be used to expand resource records or raw sample counts. No simulation is run.
"""

import numpy as np
from prepare_attack_final_protocol import STATISTICS


def outcome_cube(rows, traffic_seeds, deterministic=False):
    traffic = list(traffic_seeds)
    if len(traffic) < 2 or len(set(traffic)) != len(traffic):
        raise ValueError("At least two distinct ordered traffic clusters required")
    attack_seeds = [0] if deterministic else [0, 1, 2]
    expected = {(model, seed, attack) for model in range(5) for seed in traffic for attack in attack_seeds}
    actual = {}
    for row in rows:
        key = (row["checkpoint_seed"], row["sumo_seed"], row["attack_seed"])
        if key in actual or key not in expected:
            raise ValueError("Duplicate or unexpected actual episode identity")
        value = float(row["value"])
        if not np.isfinite(value):
            raise ValueError("Nonfinite paired outcome")
        actual[key] = value
    if set(actual) != expected:
        raise ValueError("Missing paired model/traffic/attack episode")
    return np.array([[[actual[(model, seed, 0 if deterministic else attack)] for attack in range(3)]
                      for model in range(5)] for seed in traffic], dtype=np.float64)


def paired_summary(first, second, traffic_seeds, first_deterministic=False, second_deterministic=False):
    """Average repeated attacks before resampling paired traffic blocks.

The same seed and traffic order produce the same bootstrap indices for every
contrast and budget, preserving joint pairing. Five checkpoints stay fixed.
"""
    traffic = list(traffic_seeds)
    delta = outcome_cube(first, traffic, first_deterministic) - outcome_cube(second, traffic, second_deterministic)
    model_traffic = delta.mean(axis=2)
    cluster = model_traffic.mean(axis=1)
    bootstrap = STATISTICS["bootstrap"]
    indices = np.random.RandomState(bootstrap["seed"]).randint(0, len(traffic),
                                                               (bootstrap["draws"], len(traffic)))
    means = cluster[indices].mean(axis=1)
    def interval(confidence):
        tail = (1 - confidence) * 50
        return [float(v) for v in np.percentile(means, [tail, 100 - tail])]
    tol = 1e-9
    return dict(mean_delta=float(cluster.mean()), actual_episodes=dict(first=len(first), second=len(second)),
                traffic_clusters=len(traffic), fixed_model_traffic_units=5 * len(traffic),
                attack_repeats_per_model_traffic=3,
                descriptive_interval=interval(bootstrap["descriptive_confidence"]),
                primary_family_adjusted_interval=interval(bootstrap["primary_adjusted_confidence"]),
                bootstrap_seed=bootstrap["seed"], bootstrap_draws=bootstrap["draws"],
                per_checkpoint=[dict(checkpoint_seed=m, mean_delta=float(model_traffic[:, m].mean())) for m in range(5)],
                per_traffic=[dict(sumo_seed=s, mean_delta=float(cluster[i])) for i, s in enumerate(traffic)],
                leave_one_traffic_out=[dict(removed_sumo_seed=s,
                                           mean_delta=float(np.delete(cluster, i).mean())) for i, s in enumerate(traffic)],
                model_traffic_wins=int((model_traffic < -tol).sum()),
                model_traffic_ties=int((np.abs(model_traffic) <= tol).sum()),
                model_traffic_losses=int((model_traffic > tol).sum()),
                interpretation="First minus second. Conditional on five fixed models; correlated attack repeats are averaged, not independent traffic samples.")
