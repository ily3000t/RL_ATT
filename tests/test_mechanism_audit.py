import copy
from pathlib import Path
import sys
import unittest
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.single_attempt import SingleAttemptAttack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_mechanism_controls import audit_attempts, verify_shared_witnesses
from prepare_mechanism_controls import CONTROL_NAMES
sys.path.pop(0)


class MechanismAuditTests(unittest.TestCase):
    def setUp(self):
        obs = np.full(16, .2)
        obs[0] = .01
        attack = SingleAttemptAttack(horizon=2)
        result = attack(obs, linear_victim(), AttackContext(0, 0, 0, 0, BudgetOracle(obs), 2))
        self.rows = [dict(episode=1, attack_metadata=result.metadata, attack_cost=result.attack_cost)]
        self.params = dict(max_attempts=1, inner_steps=2)

    def test_single_control_trace_and_retry_seed_counter_tampering(self):
        witnesses, counts = audit_attempts(self.rows, "ours_single_return", 0, self.params)
        self.assertGreater(len(witnesses), 0)
        self.assertEqual(counts["retries"], 0)
        for field, value in (("attempt", 1), ("seed", 0), ("succeeded", True)):
            rows = copy.deepcopy(self.rows)
            attempt = rows[0]["attack_metadata"]["inner_attempt_trace"][-1]
            attempt[field] = value if field != "succeeded" else not attempt[field]
            with self.assertRaises(ValueError):
                audit_attempts(rows, "ours_single_return", 0, self.params)
        rows = copy.deepcopy(self.rows)
        rows[0]["attack_cost"]["gradient_evaluations"] += 1
        with self.assertRaisesRegex(ValueError, "Gradient"):
            audit_attempts(rows, "ours_single_return", 0, self.params)

    def test_shared_first_attempt_requires_same_outcome_but_allows_different_visited_histories(self):
        key = (1, (), 1, 0)
        witnesses = {name: {key: (123, 1, .5)} for name in CONTROL_NAMES[1:]}
        witnesses["ours_return"][(1, (1,), 2, 1)] = (234, 0, -.5)
        proof = verify_shared_witnesses(witnesses)
        self.assertEqual(len(proof), 6)
        self.assertTrue(all(p["shared_first_attempts"] == 1 for p in proof))
        witnesses["ours_single_return"][key] = (123, 0, -.5)
        with self.assertRaisesRegex(ValueError, "primitive differs"):
            verify_shared_witnesses(witnesses)
