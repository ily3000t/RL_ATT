import time
from .box import BoxAttack, state_rng


class RandomNoiseAttack(BoxAttack):
    def __call__(self, observation, victim, context):
        skipped = self.skipped(observation, victim, context, "random")
        if skipped is not None:
            return skipped
        start = time.perf_counter()
        rng, seed = state_rng(context)
        delta = rng.uniform(-self.epsilon, self.epsilon, 16) * self.scales(observation)
        return self.result(observation, observation + delta,
                           {"objective_evaluations": 0, "policy_forward_calls": 0, "gradient_evaluations": 0,
                            "wall_seconds": time.perf_counter() - start},
                           {"name": "random", "distribution": "independent_uniform_box", "state_seed": seed})
