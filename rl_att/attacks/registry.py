from .base import BaseAttack


class AttackRegistry:
    def __init__(self):
        self._types = {}

    def register(self, name, attack_type):
        if not name or name in self._types or not issubclass(attack_type, BaseAttack):
            raise ValueError("Unique name and BaseAttack implementation required")
        self._types[name] = attack_type

    def create(self, name, **parameters):
        if name not in self._types:
            raise ValueError("Unknown attack: " + name)
        if name in ("ours_return", "ours_safety", "zero_one_budgeted_return", "zero_one_budgeted_safety",
                    "ours_progress_return", "ours_progress_safety"):
            objective = name.rsplit("_", 1)[-1]
            if parameters.get("objective", objective) != objective:
                raise ValueError("Attack name and objective disagree")
            parameters["objective"] = objective
        return self._types[name](**parameters)

    @classmethod
    def defaults(cls):
        from .no_attack import NoAttack
        from .oarl_bo import OARLBOAttack
        from .random_noise import RandomNoiseAttack
        from .fgsm import FGSMAttack
        from .pgd import PGDAttack
        from .zero_one import ZeroOneAttack
        from .proposed import ProposedAttack
        from .zero_one_controls import BudgetedZeroOneAttack
        from .progress_retry import ProgressRetryAttack
        registry = cls()
        registry.register("none", NoAttack)
        registry.register("oarl_bo", OARLBOAttack)
        registry.register("random", RandomNoiseAttack)
        registry.register("fgsm", FGSMAttack)
        registry.register("pgd", PGDAttack)
        registry.register("zero_one", ZeroOneAttack)
        registry.register("ours_return", ProposedAttack)
        registry.register("ours_safety", ProposedAttack)
        registry.register("zero_one_budgeted_return", BudgetedZeroOneAttack)
        registry.register("zero_one_budgeted_safety", BudgetedZeroOneAttack)
        registry.register("ours_progress_return", ProgressRetryAttack)
        registry.register("ours_progress_safety", ProgressRetryAttack)
        return registry
