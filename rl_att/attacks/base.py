"""Single-observation attack contract in the environment's normalized units."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class AttackContext:
    run_seed: int
    attack_seed: int
    episode: int
    step: int  # zero based; schedule restarts each episode

    def __post_init__(self):
        for value in (self.run_seed, self.attack_seed, self.episode, self.step):
            if type(value) is not int or not 0 <= value < 2 ** 31:
                raise ValueError("Context indices/seeds must be nonnegative 31-bit integers")


@dataclass
class AttackResult:
    adversarial_observation: np.ndarray
    perturbation: np.ndarray
    attacked: bool
    attack_cost: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self, observation):
        clean = np.asarray(observation)
        adv, delta = np.asarray(self.adversarial_observation), np.asarray(self.perturbation)
        if clean.shape != (16,) or adv.shape != clean.shape or delta.shape != clean.shape:
            raise ValueError("Attack result must retain the 16D observation shape")
        if not all(np.isfinite(x).all() for x in (clean, adv, delta)):
            raise ValueError("Attack result must be finite")
        if not np.allclose(adv.astype(float) - clean.astype(float), delta, atol=1e-7, rtol=0):
            raise ValueError("Perturbation does not match adversarial observation")
        if not self.attacked and (np.any(delta != 0) or not np.array_equal(adv, clean)):
            raise ValueError("An unapplied attack must preserve the observation exactly")
        return self


class BaseAttack(ABC):
    @abstractmethod
    def __call__(self, observation, victim, context):
        """Return AttackResult without mutating observation or victim."""
        raise NotImplementedError
