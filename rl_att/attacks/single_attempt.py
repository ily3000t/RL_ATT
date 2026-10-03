"""Mechanism control: the existing behavior search with no independent retry."""

from .proposed import ProposedAttack


class SingleAttemptAttack(ProposedAttack):
    method_prefix = "ours_single_"

    def __init__(self, max_attempts=1, objective="return", **kwargs):
        if type(max_attempts) is not int or max_attempts != 1:
            raise ValueError("Single-attempt control requires max_attempts=1")
        if objective != "return":
            raise ValueError("This mechanism control uses the Return objective")
        super().__init__(max_attempts=1, objective=objective, **kwargs)
