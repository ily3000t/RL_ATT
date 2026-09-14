"""One-step logit-margin FGSM in the declared observation envelope."""

import time
import numpy as np
import torch
from .box import BoxAttack, state_rng
from .objectives import margin


class FGSMAttack(BoxAttack):
    def __init__(self, objective="untargeted_logit_margin", **kwargs):
        super().__init__(**kwargs)
        if objective != "untargeted_logit_margin":
            raise ValueError("FGSM requires the declared logit-margin objective")

    def __call__(self, observation, victim, context):
        skipped = self.skipped(observation, victim, context, "fgsm")
        if skipped is not None:
            return skipped
        start = time.perf_counter()
        obs = np.asarray(observation, dtype=np.float64)
        with torch.enable_grad():
            state = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
            logits = victim.logits(state, input_grad=True)
            label = int(torch.softmax(logits.detach(), dim=0).argmax().item())
            loss = margin(logits, label)
            grad = torch.autograd.grad(loss, state)[0]
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError("Non-finite FGSM gradient")
        adv = obs + self.epsilon * self.scales(obs) * grad.sign().numpy()
        final = float(margin(victim.logits(adv), label).item())
        return self.result(obs, adv,
                           {"objective_evaluations": 2, "policy_forward_calls": 2, "gradient_evaluations": 1,
                            "wall_seconds": time.perf_counter() - start},
                           {"name": "fgsm", "objective": "untargeted_logit_margin", "clean_action": label,
                            "initial_objective": float(loss.item()), "final_objective": final,
                            "zero_gradient": not bool(grad.abs().sum()), "steps": 1})


