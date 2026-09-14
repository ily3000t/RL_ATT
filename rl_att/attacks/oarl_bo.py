"""Delegate BO to the original Agent, with distinct batch and online contracts."""

import time
import numpy as np
import torch
from oarl import Agent
from rl_att.utils.seeding import TorchPolicyStream
from .base import BaseAttack, AttackResult
from .no_attack import NoAttack


class _BOBridge(Agent):
    def __init__(self, actor, seed, evaluations):
        # No networks, replay, optimizers, or random initialization are allocated.
        self.actor = actor
        self.attack_seed = seed
        self.attack_optimizing_times = evaluations
        self.js_float64_evaluations = 0
        self.trace = []

    def js_d_loss(self, u1, u2):
        value = super().js_d_loss(u1, u2)
        self.trace.append(({"u1": float(u1), "u2": float(u2)}, value))
        return value


class OARLBOAttack(BaseAttack):
    def __init__(self, evaluations=5, every_n_steps=1):
        if type(evaluations) is not int or evaluations < 1:
            raise ValueError("evaluations must be a positive integer")
        if type(every_n_steps) is not int or every_n_steps < 1:
            raise ValueError("every_n_steps must be a positive integer")
        self.evaluations = evaluations
        self.every_n_steps = every_n_steps

    def search_batch(self, actor, prob, prob_next, obs1, obs2, attack_seed):
        """Training-equivalent loss/gradient/RNG path; caller owns RNG and tensors.

        This is an optional wrapper entry, not a change to original train_model.
        The online __call__ below does not read a future environment observation.
        """
        bridge = _BOBridge(actor, attack_seed, self.evaluations)
        loss = bridge.get_optimal_perturb_Bayes(prob, prob_next, obs1, obs2)
        parameters = next(parameters for parameters, value in bridge.trace if value is loss)
        return loss, parameters, bridge

    def __call__(self, observation, victim, context):
        if context.step % self.every_n_steps:
            result = NoAttack()(observation, victim, context)
            result.metadata.update(name="oarl_bo", skipped="fixed_frequency")
            return result
        start = time.perf_counter()
        obs = np.asarray(observation)
        if obs.shape != (16,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite 16D observation")
        state = torch.as_tensor(obs, dtype=torch.float32).reshape(1, 16)
        # Discarded Categorical draws in original BO cannot advance policy RNG.
        with TorchPolicyStream(context.attack_seed).activate(), torch.no_grad():
            prob = victim.probabilities(state)
            loss, parameters, bridge = self.search_batch(
                victim.actor, prob, prob, state, state, context.attack_seed)
            adv = (parameters["u1"] * state + parameters["u2"]).reshape(16).cpu().numpy()
        trace = [{"parameters": p, "objective": float(v.item())} for p, v in bridge.trace]
        return AttackResult(adv, adv.astype(float) - obs.astype(float), True,
                            {"objective_evaluations": len(trace), "policy_forward_calls": 1 + 2 * len(trace),
                             "wall_seconds": time.perf_counter() - start},
                            {"name": "oarl_bo", "objective": "2*JS(pi(obs),pi(u1*obs+u2))",
                             "online_adaptation": "obs2=obs1; no future observation",
                             "parameters": parameters, "objective_value": float(loss.item()), "trace": trace,
                             "js_float64_evaluations": bridge.js_float64_evaluations,
                             "norm": "affine_box", "multiplicative_bounds": [0.8, 1.2],
                             "additive_bounds": [-0.05, 0.05], "clipping": False,
                             "optimizer_seed": context.attack_seed}).validate(observation)
