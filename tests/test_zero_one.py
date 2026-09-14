import os
from pathlib import Path
import random
import unittest
import numpy as np
import torch
from test_basic_attacks import linear_victim
from rl_att.attacks.base import AttackContext
from rl_att.attacks.zero_one import ZeroOneAttack
from rl_att.evaluation.replay_oracle import ReplayOracle
from rl_att.evaluation.evaluator import validate_budget


class ToyOracle:
    def __init__(self, obs):
        self.obs = obs.tolist()
        self.core = ReplayOracle(self.reset, self.step_fn)
        self.core.start_episode(self.reset())

    def reset(self):
        return dict(observation=self.obs, reward=0., done=False, collision=False)

    def step_fn(self, action):
        return dict(self.reset(), reward=-1. if action == 1 else 1.)

    def begin(self):
        return self.core.begin()

    def step(self, action):
        return self.core.step(action)

    def counts(self):
        return dict(self.core.counts)


class ZeroOneTests(unittest.TestCase):
    def setUp(self):
        self.victim = linear_victim()
        self.obs = np.full(16, .2)
        self.obs[0] = .01
        self.oracle = ToyOracle(self.obs)
        self.budget = dict(norm="observation_scaled_linf", epsilon=1., relative_scale=.2, absolute_scale=.05)

    def test_targeted_inversion_bounds_frozen_and_correct_action(self):
        attack = ZeroOneAttack()
        for target in (0, 1):
            adv, action = attack.invert(self.obs, target, self.victim, 0)
            self.assertEqual(action, target)
            self.assertTrue(np.all(np.abs(adv - self.obs) <= .2 * abs(self.obs) + .05 + 1e-7))
        self.victim.assert_frozen()
        self.assertTrue(all(p.grad is None for p in self.victim.actor.parameters()))

    def test_rollout_minimum_block_execution_and_rng_isolation(self):
        attack = ZeroOneAttack(horizon=2)
        py, npstate, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        first = attack(self.obs, self.victim, AttackContext(0, 0, 0, 0, self.oracle, 2))
        self.assertEqual(first.metadata["selected_return"], -2)
        self.assertEqual(first.attack_cost["objective_evaluations"], 9)
        self.assertEqual(first.attack_cost["candidate_rollouts"], 9)
        self.oracle.core.observe(1, self.oracle.step_fn(1))
        second = attack(self.obs, self.victim, AttackContext(0, 0, 0, 1, self.oracle, 1))
        self.assertEqual(second.attack_cost["objective_evaluations"], 0)
        self.assertFalse(second.metadata["planned"])
        validate_budget(first, self.obs, self.budget)
        validate_budget(second, self.obs, self.budget)
        self.assertEqual(py, random.getstate())
        self.assertTrue(np.array_equal(npstate[1], np.random.get_state()[1]))
        self.assertEqual(npstate[2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(ts, torch.get_rng_state()))
        with self.assertRaises(ValueError):
            attack(self.obs, self.victim, AttackContext(0, 0, 0, 1, self.oracle, 1))

    def test_missing_oracle_and_mismatched_plan_fail(self):
        attack = ZeroOneAttack(horizon=2)
        with self.assertRaises(ValueError):
            attack(self.obs, self.victim, AttackContext(0, 0, 0, 0))
        attack(self.obs, self.victim, AttackContext(0, 0, 0, 0, self.oracle, 2))
        with self.assertRaises(ValueError):
            attack(self.obs + .001, self.victim, AttackContext(0, 0, 0, 1, self.oracle, 1))

    def test_pinned_zoopt_budget_and_repeatability(self):
        wheel = Path(__file__).resolve().parents[1] / ".local/dependencies/zero-one/zoopt-0.4.2-py3-none-any.whl"
        if not wheel.exists():
            self.skipTest("Optional pinned ZOOpt wheel unavailable")
        old = os.environ.get("RL_ATT_ZOOPT_WHEEL")
        os.environ["RL_ATT_ZOOPT_WHEEL"] = str(wheel)
        try:
            results = [ZeroOneAttack(horizon=3)(self.obs, self.victim,
                       AttackContext(0, 0, 0, 0, ToyOracle(self.obs), 3)) for _ in range(2)]
            self.assertEqual(results[0].attack_cost["objective_evaluations"], 10)
            self.assertEqual(results[0].metadata["candidate_trace"], results[1].metadata["candidate_trace"])
            self.assertTrue(np.array_equal(results[0].adversarial_observation, results[1].adversarial_observation))
        finally:
            if old is None:
                del os.environ["RL_ATT_ZOOPT_WHEEL"]
            else:
                os.environ["RL_ATT_ZOOPT_WHEEL"] = old


if __name__ == "__main__":
    unittest.main()
