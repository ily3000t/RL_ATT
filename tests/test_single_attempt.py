import copy
import json
from pathlib import Path
import random
import unittest
import numpy as np
import torch
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.proposed import ProposedAttack
from rl_att.attacks.registry import AttackRegistry
from rl_att.attacks.single_attempt import SingleAttemptAttack
from rl_att.evaluation.configuration import validate_config


class SingleAttemptTests(unittest.TestCase):
    def test_matches_existing_search_with_one_attempt_and_counts_every_forward(self):
        victim = linear_victim()
        obs = np.full(16, .2)
        obs[0] = .01
        state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        outputs = []
        for attack in (ProposedAttack(horizon=2, max_attempts=1), SingleAttemptAttack(horizon=2)):
            oracle = BudgetOracle(obs)
            calls, results = [], []
            handle = victim.actor.register_forward_hook(lambda *args: calls.append(1))
            try:
                for step in range(2):
                    result = attack(obs, victim, AttackContext(0, 0, 0, step, oracle, 2 - step))
                    oracle.core.observe(result.metadata["planned_action"],
                                        oracle.step_fn(result.metadata["planned_action"]))
                    results.append(result)
            finally:
                handle.remove()
            self.assertEqual(len(calls), sum(r.attack_cost["policy_forward_calls"] for r in results))
            self.assertEqual(results[0].metadata["selected_return"], -2.)
            self.assertGreater(results[0].metadata["failed_target_attempts"], 0)
            self.assertEqual(results[0].metadata["retry_attempts"], 0)
            self.assertTrue(all(a["attempt"] == 0 for a in results[0].metadata["inner_attempt_trace"]))
            outputs.append(results)
        for old, new in zip(*outputs):
            self.assertTrue(np.array_equal(old.adversarial_observation, new.adversarial_observation))
            self.assertTrue(np.array_equal(old.perturbation, new.perturbation))
            self.assertEqual(old.attacked, new.attacked)
            self.assertEqual({k: v for k, v in old.attack_cost.items() if k != "wall_seconds"},
                             {k: v for k, v in new.attack_cost.items() if k != "wall_seconds"})
            self.assertEqual({k: v for k, v in old.metadata.items() if k != "name"},
                             {k: v for k, v in new.metadata.items() if k != "name"})
            self.assertEqual(new.metadata["name"], "ours_single_return")
        self.assertEqual(state[0], random.getstate())
        self.assertTrue(np.array_equal(state[1][1], np.random.get_state()[1]))
        self.assertEqual(state[1][2:], np.random.get_state()[2:])
        self.assertTrue(torch.equal(state[2], torch.get_rng_state()))
        victim.assert_frozen()

    def test_constructor_registry_and_config_refuse_hidden_retries(self):
        self.assertIsInstance(AttackRegistry.defaults().create("ours_single_return"), SingleAttemptAttack)
        for attempts in (0, 2, 3, True, 1.):
            with self.assertRaises(ValueError):
                SingleAttemptAttack(max_attempts=attempts)
        with self.assertRaises(ValueError):
            AttackRegistry.defaults().create("ours_single_return", objective="safety")
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/evaluation/proposed_smoke_seed0.json").read_text())
        control = copy.deepcopy(config["attacks"][3])
        control["name"] = "ours_single_return"
        control["parameters"]["max_attempts"] = 1
        config["attacks"] = [config["attacks"][0], control]
        validate_config(config)
        control["parameters"]["max_attempts"] = 3
        with self.assertRaisesRegex(ValueError, "Single-attempt"):
            validate_config(config)
