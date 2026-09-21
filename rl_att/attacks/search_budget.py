"""Atomic multi-resource accounting for simulator-assisted search."""

RESOURCES = ("gradient_evaluations", "policy_forward_calls", "new_shadow_transitions", "shadow_steps")


class BudgetExhausted(Exception):
    pass


class SearchBudget:
    def __init__(self, limits):
        if set(limits) != set(RESOURCES) or any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError("Explicit nonnegative integer resource limits required")
        self.limits = dict(limits)
        self.used = dict.fromkeys(RESOURCES, 0)
        self.reserved = dict.fromkeys(RESOURCES, 0)

    def remaining(self):
        return {k: self.limits[k] - self.used[k] - self.reserved[k] for k in RESOURCES}

    def _check(self, costs):
        if set(costs) - set(RESOURCES) or any(type(v) is not int or v < 0 for v in costs.values()):
            raise ValueError("Invalid resource charge")
        if any(v > self.remaining()[k] for k, v in costs.items()):
            raise BudgetExhausted("Insufficient search resources: " + str(costs))

    def charge(self, **costs):
        self._check(costs)
        for k, v in costs.items():
            self.used[k] += v

    def reserve(self, **costs):
        self._check(costs)
        for k, v in costs.items():
            self.reserved[k] += v

    def consume_reserved(self, **costs):
        if set(costs) - set(RESOURCES) or any(type(v) is not int or v < 0 or v > self.reserved[k]
                                               for k, v in costs.items()):
            raise ValueError("Invalid reservation consumption")
        for k, v in costs.items():
            self.reserved[k] -= v
            self.used[k] += v

    def snapshot(self):
        return dict(limits=dict(self.limits), used=dict(self.used), reserved=dict(self.reserved))
