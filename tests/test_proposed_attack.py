import random
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_basic_attacks import linear_victim
from test_zero_one import ToyOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.proposed import ProposedAttack, DEFAULT_LIMITS
from rl_att.attacks.behavior_search import BehaviorSearch, witness_seed
from rl_att.attacks.search_budget import SearchBudget
from rl_att.attacks.rollout_objectives import rollout_key, optimizer_value
from rl_att.attacks.zero_one_controls import BudgetedZeroOneAttack


class BudgetOracle(ToyOracle):
    def budgeted_step(self, action, remaining):
        return self.core.budgeted_step(action, remaining)


class ProposedTests(unittest.TestCase):
    def setUp(self):
        self.victim = linear_victim()
        self.obs = np.full(16, .2)
        self.obs[0] = .01

    def test_known_branch_action_and_every_forward_accounted(self):
        oracle = BudgetOracle(self.obs)
        attack = ProposedAttack(horizon=2)
        calls = []
        handle = self.victim.actor.register_forward_hook(lambda *args: calls.append(1))
        state = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        original = self.obs.copy()
        try:
            first = attack(self.obs, self.victim, AttackContext(0, 0, 0, 0, oracle, 2))
            self.assertEqual(first.metadata["selected_return"], -2.)
            self.assertEqual(first.metadata["planned_action"], 1)
            oracle.core.observe(1, oracle.step_fn(1))
            second = attack(self.obs, self.victim, AttackContext(0, 0, 0, 1, oracle, 1))
        finally:
            handle.remove()
        self.assertEqual(len(calls), sum(r.attack_cost["policy_forward_calls"] for r in (first, second)))
        self.assertEqual(second.attack_cost["objective_evaluations"], 0)
        self.assertTrue(first.metadata["failed_target_attempts"] > 0)
        self.assertTrue(first.metadata["retry_attempts"] > 0)
        for result in (first, second):
            b = result.metadata["budget"]
            self.assertTrue(all(b["used"][k] + b["reserved"][k] <= b["limits"][k] for k in b["limits"]))
            self.assertTrue(np.all(np.abs(result.perturbation) <= .2 * abs(self.obs) + .05 + 1e-7))
        self.assertTrue(np.array_equal(original, self.obs))
        self.assertEqual(state[0], random.getstate())
        self.assertTrue(np.array_equal(state[1][1], np.random.get_state()[1]))
        self.assertEqual(state[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(state[2], torch.get_rng_state()))
        self.victim.assert_frozen()
        self.assertTrue(all(p.grad is None for p in self.victim.actor.parameters()))

    def test_same_observation_different_history_and_failed_target(self):
        oracle = BudgetOracle(self.obs)
        attack = ProposedAttack(horizon=2)
        engine = BehaviorSearch(attack, self.victim, AttackContext(0, 0, 0, 0, oracle, 2),
                                self.obs, [], SearchBudget(DEFAULT_LIMITS), 2)
        left, right = engine.node((0,), self.obs), engine.node((1,), self.obs)
        self.assertIsNot(left, right)
        witness = engine.invert(left, (0,), 2, 0)
        self.assertNotEqual(witness["action"], 2)
        self.assertNotIn(2, left["branches"])
        self.assertEqual(left["attempts"][2], 1)
        before = engine.budget.snapshot()
        engine.invert(left, (0,), 2, 0)
        self.assertEqual(engine.budget.snapshot(), before)
        self.assertEqual(engine.inner_cache_hits, 1)
        self.assertNotEqual(witness_seed(0, 0, (0,), 2, 0), witness_seed(0, 0, (1,), 2, 0))

    def test_exhaustion_keeps_complete_fallback_and_never_scores_partial(self):
        limits = dict(DEFAULT_LIMITS, gradient_evaluations=0)
        oracle = BudgetOracle(self.obs)
        result = ProposedAttack(horizon=2, resource_limits=limits)(
            self.obs, self.victim, AttackContext(0, 0, 0, 0, oracle, 2))
        self.assertFalse(result.attacked)
        self.assertTrue(result.metadata["budget_exhausted"])
        self.assertEqual(result.metadata["selected_return"], 2.)
        self.assertEqual(len(result.metadata["candidate_trace"]), 1)
        self.assertEqual(len(result.metadata["incomplete_candidates"]), 1)
        self.assertFalse(result.metadata["oracle_unplanned_fallback"])

    def test_zero_simulation_budget_returns_explicit_unverified_clean(self):
        limits = dict(DEFAULT_LIMITS, shadow_steps=0, new_shadow_transitions=0)
        oracle = BudgetOracle(self.obs)
        result = ProposedAttack(horizon=2, resource_limits=limits)(
            self.obs, self.victim, AttackContext(0, 0, 0, 0, oracle, 2))
        self.assertFalse(result.attacked)
        self.assertTrue(result.metadata["oracle_unplanned_fallback"])
        self.assertIsNone(result.metadata["selected_candidate"])
        self.assertEqual(result.attack_cost["shadow_steps"], 0)
        self.assertEqual(oracle.core.counts["shadow_steps"], 0)

    def test_early_terminal_is_complete_not_fake_low_return(self):
        oracle = BudgetOracle(self.obs)
        oracle.core.step_fn = lambda action: dict(oracle.reset(), done=True, collision=action == 1, reward=1.)
        result = ProposedAttack(horizon=20, objective="safety")(
            self.obs, self.victim, AttackContext(0, 0, 0, 0, oracle, 200))
        self.assertEqual(result.metadata["selected_score"], [-1, 1.])
        self.assertTrue(all(t["terminated"] and len(t["actions"]) == 1 for t in result.metadata["candidate_trace"]))
        self.assertTrue(all(not t["horizon_truncated"] for t in result.metadata["candidate_trace"]))

    def test_objective_collision_priority_and_sequential_sum(self):
        collision = rollout_key([1000.], [True], "safety")
        safe = rollout_key([-1000.], [False], "safety")
        self.assertLess(collision, safe)
        self.assertLess(optimizer_value(collision), optimizer_value(safe))
        self.assertLess(rollout_key([-1000.], [False], "return"), rollout_key([1000.], [True], "return"))

    def test_tail_limits_and_invalid_context(self):
        attack = ProposedAttack()
        result = attack(self.obs, self.victim, AttackContext(0, 0, 0, 0, BudgetOracle(self.obs), 1))
        self.assertEqual(result.metadata["budget"]["limits"]["gradient_evaluations"], 20)
        self.assertEqual(result.metadata["budget"]["limits"]["new_shadow_transitions"], 10)
        with self.assertRaises(ValueError):
            attack(self.obs, self.victim, AttackContext(0, 0, 0, 0))

    def test_control_uses_shared_witnesses_and_complete_rollouts(self):
        traces = []
        for cls in (ProposedAttack, BudgetedZeroOneAttack):
            result = cls(horizon=2)(self.obs, self.victim, AttackContext(0, 0, 0, 0, BudgetOracle(self.obs), 2))
            self.assertEqual(result.metadata["selected_return"], -2.)
            traces.append({(tuple(t["history"]), t["target"], t["attempt"]): t
                           for t in result.metadata["inner_attempt_trace"]})
        common = set(traces[0]) & set(traces[1])
        self.assertTrue(common)
        self.assertTrue(all(traces[0][k] == traces[1][k] for k in common))
