"""Shared observation-scaled L-infinity envelope and isolated per-state randomness."""

import math
import numpy as np
from .base import BaseAttack, AttackResult
from .no_attack import NoAttack


def box_scales(observation, relative_scale, absolute_scale):
    obs = np.asarray(observation, dtype=np.float64)
    if obs.shape != (16,) or not np.isfinite(obs).all():
        raise ValueError("Expected finite 16D observation")
    return relative_scale * np.abs(obs) + absolute_scale


def state_rng(context):
    # The same step has the same draw even when earlier episodes terminate early.
    words = [context.attack_seed, 3, context.episode, context.step]
    seed = int(np.random.SeedSequence(words).generate_state(1)[0])
    return np.random.RandomState(seed), seed


class BoxAttack(BaseAttack):
    def __init__(self, epsilon=1.0, relative_scale=0.2, absolute_scale=0.05, every_n_steps=1):
        if any(not math.isfinite(v) for v in (epsilon, relative_scale, absolute_scale)):
            raise ValueError("Attack budget must be finite")
        if epsilon < 0 or relative_scale < 0 or absolute_scale <= 0:
            raise ValueError("Nonnegative epsilon/relative scale and positive absolute scale required")
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("every_n_steps must be positive")
        self.epsilon, self.relative_scale, self.absolute_scale = epsilon, relative_scale, absolute_scale
        self.every_n_steps = every_n_steps

    def skipped(self, observation, victim, context, name):
        if context.step % self.every_n_steps or self.epsilon == 0:
            result = NoAttack()(observation, victim, context)
            result.metadata.update(name=name, skipped="zero_budget" if self.epsilon == 0 else "fixed_frequency")
            return result
        return None

    def scales(self, observation):
        return box_scales(observation, self.relative_scale, self.absolute_scale)

    def result(self, observation, adversarial, cost, metadata):
        obs = np.asarray(observation, dtype=np.float64)
        delta = np.asarray(adversarial, dtype=np.float64) - obs
        scales = self.scales(obs)
        if np.any(np.abs(delta) > self.epsilon * scales + 1e-7):
            raise ValueError("Attack escaped its observation-scaled envelope")
        metadata.update(norm="observation_scaled_linf", epsilon=self.epsilon,
                        relative_scale=self.relative_scale, absolute_scale=self.absolute_scale,
                        scaled_linf=float(np.max(np.abs(delta) / scales)), physical_clipping=False)
        return AttackResult(np.asarray(adversarial, dtype=np.float64), delta, True, cost, metadata).validate(obs)
