"""Separate frozen checkpoint identity from diagnostic evaluation traffic."""

from rl_att.utils.seeding import seed_manifest


def evaluation_seeds(checkpoint_seed, episodes, traffic_seed=None):
    seeds = seed_manifest(checkpoint_seed, "controlled", episodes, phase="evaluation")
    if traffic_seed is not None:
        if type(traffic_seed) is not int or not 0 <= traffic_seed < 5:
            raise ValueError("Diagnostic traffic_seed must identify held-out group 0 through 4")
        traffic = seed_manifest(traffic_seed, "controlled", episodes, phase="evaluation")
        for key in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed", "episode_sumo_seeds"):
            seeds[key] = traffic[key]
        seeds["diagnostic_traffic_group"] = traffic_seed
        seeds["checkpoint_training_seed"] = checkpoint_seed
    return seeds
