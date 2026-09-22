import unittest
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.proposed import ProposedAttack, DEFAULT_LIMITS
from rl_att.attacks.progress_retry import ProgressRetryAttack, ProgressRetrySearch
from rl_att.attacks.search_budget import SearchBudget
from rl_att.attacks.registry import AttackRegistry


class ProgressRetryTests(unittest.TestCase):
    def setUp(self):
        self.obs = np.full(16, .2)
        self.obs[0] = .01
        self.victim = linear_victim()

    def test_first_retry_allowed_but_flat_retry_does_not_get_another_restart(self):
        oracle = BudgetOracle(self.obs)
        engine = ProgressRetrySearch(ProgressRetryAttack(), self.victim,
                                      AttackContext(0, 0, 0, 0, oracle, 2), self.obs, [],
                                      SearchBudget(DEFAULT_LIMITS), 2)
        node = engine.node((), self.obs)
        # An analytic linear policy cannot make its very negative action 2 win.
        engine.invert(node, (), 2, 0)
        self.assertIn(2, engine.retry_targets(node, (), [2]))
        engine.invert(node, (), 2, 1)
        self.assertNotIn(2, engine.retry_targets(node, (), [2]))
        self.assertNotIn(2, node["branches"])
        self.assertEqual(len(engine.suppressed), 1)
        engine.retry_targets(node, (), [2])
        self.assertEqual(len(engine.suppressed), 1)
        # A cached query must not turn the same final margin into new progress.
        before = engine.last_retry.copy()
        engine.invert(node, (), 2, 1)
        self.assertEqual(before, engine.last_retry)

    def test_progress_allows_next_retry_until_original_cap(self):
        engine = ProgressRetrySearch(ProgressRetryAttack(), self.victim,
                                      AttackContext(0, 0, 0, 0, BudgetOracle(self.obs), 2), self.obs, [],
                                      SearchBudget(DEFAULT_LIMITS), 2)
        node = engine.node((), self.obs)
        node["attempts"][2] = 2
        engine.last_retry[((), 2)] = dict(previous_best=-3., margin=-2., improved=True)
        self.assertEqual(engine.retry_targets(node, (), [2]), [2])
        node["attempts"][2] = 3
        self.assertEqual(engine.retry_targets(node, (), [2]), [])

    def test_known_optimum_frozen_model_and_every_forward_still_counted(self):
        results = []
        for cls in (ProposedAttack, ProgressRetryAttack):
            calls = []
            hook = self.victim.actor.register_forward_hook(lambda *args: calls.append(1))
            try:
                result = cls(horizon=2)(self.obs, self.victim,
                                        AttackContext(0, 0, 0, 0, BudgetOracle(self.obs), 2))
            finally:
                hook.remove()
            self.assertEqual(len(calls), result.attack_cost["policy_forward_calls"])
            self.assertEqual(result.metadata["selected_return"], -2.)
            results.append(result)
        self.assertLess(results[1].attack_cost["gradient_evaluations"], results[0].attack_cost["gradient_evaluations"])
        self.assertGreater(results[1].metadata["suppressed_retry_targets"], 0)
        self.victim.assert_frozen()
        self.assertEqual(AttackRegistry.defaults().create("ours_progress_safety").objective, "safety")
