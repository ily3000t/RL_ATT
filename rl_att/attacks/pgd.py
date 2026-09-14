"""Random-start logit-margin PGD with fixed clean-state projection bounds."""

import math
import time
import numpy as np
import torch
from .box import BoxAttack, state_rng
from .objectives import margin


class PGDAttack(BoxAttack):
    def __init__(self, steps=10, step_size=0.2, objective="untargeted_logit_margin", **kwargs):
        super().__init__(**kwargs)
        if type(steps) is not int or steps < 1 or not math.isfinite(step_size) or step_size <= 0:
            raise ValueError("Positive PGD steps and normalized step_size required")
        self.steps, self.step_size = steps, step_size
        if objective != "untargeted_logit_margin":
            raise ValueError("PGD requires the declared logit-margin objective")

    def __call__(self, observation, victim, context):
        skipped = self.skipped(observation, victim, context, "pgd")
        if skipped is not None:
            return skipped
        start = time.perf_counter()
        obs = np.asarray(observation, dtype=np.float64)
        scales = self.scales(obs)
        label = int(torch.softmax(victim.logits(obs), dim=0).argmax().item())
        rng, seed = state_rng(context)
        # Project normalized coordinates, keeping the box anchored to CLEAN obs.
        offset = rng.uniform(-self.epsilon, self.epsilon, 16)
        objectives, zero_gradients = [], 0
        for _ in range(self.steps):
            with torch.enable_grad():
                state = torch.tensor(obs + scales * offset, dtype=torch.float32, requires_grad=True)
                loss = margin(victim.logits(state, input_grad=True), label)
                grad = torch.autograd.grad(loss, state)[0]
            if not bool(torch.isfinite(grad).all()):
                raise FloatingPointError("Non-finite PGD gradient")
            objectives.append(float(loss.item()))
            zero_gradients += int(not bool(grad.abs().sum()))
            offset = np.clip(offset + self.step_size * grad.sign().numpy(), -self.epsilon, self.epsilon)
        adv = obs + scales * offset
        final = float(margin(victim.logits(adv), label).item())
        return self.result(obs, adv,
                           {"objective_evaluations": self.steps + 1, "policy_forward_calls": self.steps + 2,
                            "gradient_evaluations": self.steps, "wall_seconds": time.perf_counter() - start},
                           {"name": "pgd", "objective": "untargeted_logit_margin", "clean_action": label,
                            "final_objective": final, "objective_trace": objectives, "steps": self.steps,
                            "step_size": self.step_size, "random_start": True, "restarts": 1,
                            "state_seed": seed, "zero_gradient_steps": zero_gradients})
