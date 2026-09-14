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
        return self._types[name](**parameters)

    @classmethod
    def defaults(cls):
        from .no_attack import NoAttack
        from .oarl_bo import OARLBOAttack
        from .random_noise import RandomNoiseAttack
        from .fgsm import FGSMAttack
        from .pgd import PGDAttack
        from .zero_one import ZeroOneAttack
        registry = cls()
        registry.register("none", NoAttack)
        registry.register("oarl_bo", OARLBOAttack)
        registry.register("random", RandomNoiseAttack)
        registry.register("fgsm", FGSMAttack)
        registry.register("pgd", PGDAttack)
        registry.register("zero_one", ZeroOneAttack)
        return registry
