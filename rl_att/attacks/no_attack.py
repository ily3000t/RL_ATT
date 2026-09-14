import numpy as np
from .base import BaseAttack, AttackResult


class NoAttack(BaseAttack):
    def __call__(self, observation, victim, context):
        obs = np.asarray(observation).copy()
        return AttackResult(obs, np.zeros_like(obs), False,
                            {"objective_evaluations": 0, "policy_forward_calls": 0, "wall_seconds": 0.0},
                            {"name": "none"}).validate(observation)
