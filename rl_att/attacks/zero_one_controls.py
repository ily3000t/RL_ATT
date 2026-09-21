"""New budgeted controls; the original ZeroOneAttack remains unchanged."""

from contextlib import redirect_stdout
import io
import itertools
from .proposed import ProposedAttack
from .zero_one import derived_seed, isolated_rng
from rl_att.utils.zero_one_dependency import load_zoopt


class BudgetedZeroOneAttack(ProposedAttack):
    search_kind = "target_sequence"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.evaluations < 4:
            raise ValueError("ZOOpt needs at least four candidates")

    def search(self, engine):
        seed = derived_seed(engine.context.attack_seed, 5, engine.context.episode, engine.context.step)
        def objective(solution):
            if len(engine.trace) - 1 >= self.evaluations:
                raise ValueError("Outer optimizer exceeded complete candidate cap")
            targets = [int(a) for a in (solution.get_x() if hasattr(solution, "get_x") else solution)]
            return engine.rollout("target_sequence", targets)
        with isolated_rng(seed), redirect_stdout(io.StringIO()):
            if 3 ** engine.horizon <= self.evaluations:
                for targets in itertools.product(range(3), repeat=engine.horizon):
                    objective(targets)
            else:
                zoopt = load_zoopt()
                dimension = zoopt.Dimension(engine.horizon, [[0, 2]] * engine.horizon, [False] * engine.horizon)
                zoopt.Opt.min(zoopt.Objective(objective, dimension), zoopt.Parameter(budget=self.evaluations, seed=seed))
