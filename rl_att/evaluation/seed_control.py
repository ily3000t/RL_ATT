"""Separate frozen checkpoint identity from diagnostic evaluation traffic."""

from rl_att.utils.seeding import seed_manifest
import numpy as np


def evaluation_seeds(checkpoint_seed, episodes, traffic_seed=None, research_seeds=None):
    seeds = seed_manifest(checkpoint_seed, "controlled", episodes, phase="evaluation")
    if research_seeds is not None:
        if traffic_seed is not None:
            raise ValueError("Cannot combine research and legacy traffic")
        root, split = research_seeds["root_seed"], research_seeds["split_id"]
        expected = [int(np.random.SeedSequence([root, 1, split, i]).generate_state(1)[0]) % (2 ** 31 - 1)
                    for i in range(episodes)]
        if expected != research_seeds["episode_sumo_seeds"]:
            raise ValueError("Research traffic list differs from frozen seed derivation")
        environment_seed = int(np.random.SeedSequence([root, 1, split, 2 ** 31 - 2]).generate_state(1)[0]) % (2 ** 31 - 1)
        for key in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed"):
            seeds[key] = environment_seed
        seeds.update(protocol="controlled_research_v1", sumo_schedule="research_split_v1",
                     episode_sumo_seeds=expected, attack_seed=research_seeds["attack_seed"],
                     research_root_seed=root, research_split_id=split, checkpoint_training_seed=checkpoint_seed)
    if traffic_seed is not None:
        if type(traffic_seed) is not int or not 0 <= traffic_seed < 5:
            raise ValueError("Diagnostic traffic_seed must identify held-out group 0 through 4")
        traffic = seed_manifest(traffic_seed, "controlled", episodes, phase="evaluation")
        for key in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed", "episode_sumo_seeds"):
            seeds[key] = traffic[key]
        seeds["diagnostic_traffic_group"] = traffic_seed
        seeds["checkpoint_training_seed"] = checkpoint_seed
    return seeds
