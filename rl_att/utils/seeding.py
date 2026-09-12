"""Explicit seed protocols with an unchanged legacy mode."""

from contextlib import contextmanager
import numpy as np
import torch


def seed_manifest(run_seed, protocol, episodes, sumo_schedule="derived", phase="train"):
    if protocol not in ("legacy", "controlled"):
        raise ValueError("Unknown seed protocol: " + protocol)
    if sumo_schedule not in ("derived", "constant"):
        raise ValueError("Unknown SUMO seed schedule")
    if not isinstance(run_seed, int) or not 0 <= run_seed < 2 ** 31:
        raise ValueError("run_seed must be a nonnegative 31-bit integer")
    if phase not in ("train", "evaluation"):
        raise ValueError("Unknown phase")
    values = {name: run_seed for name in ("run_seed", "policy_seed", "numpy_seed",
                                         "python_seed", "torch_seed", "sumo_seed", "attack_seed")}
    values.update(protocol=protocol, policy_rng="isolated_cpu" if protocol == "controlled" else "shared_torch",
                  attack_rng="optimizer_reinitialized_each_update", phase=phase)
    if protocol == "legacy" and phase == "train":
        values.update(sumo_seed=0, attack_seed=0, sumo_schedule="upstream_paired")
        values["episode_sumo_seeds"] = [2 * (episode // 2) for episode in range(episodes)]
    elif sumo_schedule == "constant":
        values["sumo_schedule"] = "constant"
        values["episode_sumo_seeds"] = [run_seed] * episodes
    else:
        # Phase separation prevents evaluation traffic from reusing training seeds.
        phase_id = 0 if phase == "train" else 1
        values["sumo_schedule"] = "seedsequence_v1"
        values["episode_sumo_seeds"] = [
            int(np.random.SeedSequence([run_seed, phase_id, episode]).generate_state(1)[0]) % (2 ** 31 - 1)
            for episode in range(episodes)
        ]
    return values


class TorchPolicyStream:
    """Separate Categorical action sampling from Torch initialization/training RNG.

    OARL's original Categorical API does not accept a generator. Swapping CPU RNG
    state around just the action call preserves its implementation and restores
    the training RNG even if an exception is raised. Experiments are CPU-only.
    """

    def __init__(self, seed):
        self.state = torch.Generator().manual_seed(seed).get_state()

    @contextmanager
    def activate(self):
        training_state = torch.get_rng_state()
        torch.set_rng_state(self.state)
        try:
            yield
        finally:
            self.state = torch.get_rng_state()
            torch.set_rng_state(training_state)
