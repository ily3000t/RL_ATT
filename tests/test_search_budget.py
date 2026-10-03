import copy
import unittest
from rl_att.attacks.search_budget import SearchBudget, BudgetExhausted
from rl_att.evaluation.replay_oracle import ReplayOracle


class BudgetTests(unittest.TestCase):
    def test_atomic_charge_and_reserved_execution(self):
        budget = SearchBudget(dict(gradient_evaluations=4, policy_forward_calls=6,
                                   new_shadow_transitions=2, shadow_steps=5))
        budget.reserve(policy_forward_calls=2)
        budget.charge(gradient_evaluations=2, policy_forward_calls=3)
        before = budget.snapshot()
        with self.assertRaises(BudgetExhausted):
            budget.charge(gradient_evaluations=1, policy_forward_calls=2)
        self.assertEqual(before, budget.snapshot())
        budget.consume_reserved(policy_forward_calls=2)
        self.assertEqual(budget.used["policy_forward_calls"], 5)

    def test_oracle_replay_preflight_rejection_has_no_side_effect(self):
        calls = []
        state = dict(observation=[0.] * 16, reward=0., done=False, collision=False)
        def reset():
            calls.append("reset")
            return state
        def step(action):
            calls.append(action)
            return state
        oracle = ReplayOracle(reset, step)
        oracle.start_episode(state)
        oracle.step(0)
        oracle.step(0)
        oracle.begin()
        oracle.step(1)
        oracle.begin()
        oracle.step(0)  # cached cursor differs from physical (1,)
        before = copy.deepcopy((oracle.__dict__, calls))
        rejected = oracle.budgeted_step(2, dict(new_shadow_transitions=1, shadow_steps=1))
        self.assertFalse(rejected["accepted"])
        self.assertEqual((oracle.__dict__, calls), before)
        accepted = oracle.budgeted_step(2, dict(new_shadow_transitions=1, shadow_steps=2))
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["costs"], dict(new_shadow_transitions=1, shadow_steps=2))
        oracle.begin()
        cached = oracle.budgeted_step(0, dict(new_shadow_transitions=0, shadow_steps=0))
        self.assertTrue(cached["accepted"])

    def test_live_fallback_is_not_claimed_as_verified(self):
        state = dict(observation=[0.] * 16, reward=0., done=False, collision=False)
        oracle = ReplayOracle(lambda: state, lambda action: state)
        oracle.start_episode(state)
        with self.assertRaises(ValueError):
            oracle.observe(2, state)
        oracle.observe_fallback(2, state)
        self.assertEqual(oracle.counts["live_verified_steps"], 0)
        self.assertEqual(oracle.counts["live_unverified_fallback_steps"], 1)
        self.assertEqual(oracle.live, (2,))

    def test_warmup_is_included_in_atomic_replay_cost(self):
        state = dict(observation=[0.] * 16, reward=0., done=False, collision=False)
        oracle = ReplayOracle(lambda: state, lambda action: state, reset_step_cost=lambda: 81)
        oracle.start_episode(state)
        self.assertEqual(oracle.counts["shadow_steps"], 81)
        oracle.step(0)
        oracle.begin()
        before = copy.deepcopy(oracle.__dict__)
        reply = oracle.budgeted_step(1, dict(new_shadow_transitions=1, shadow_steps=81))
        self.assertFalse(reply["accepted"])
        self.assertEqual(before, oracle.__dict__)
        reply = oracle.budgeted_step(1, dict(new_shadow_transitions=1, shadow_steps=82))
        self.assertTrue(reply["accepted"])
        self.assertEqual(reply["costs"]["shadow_steps"], 82)
        self.assertEqual(oracle.counts["warmup_steps"], 162)
