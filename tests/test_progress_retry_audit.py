from pathlib import Path
import sys
import unittest
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.progress_retry import ProgressRetryAttack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_progress_retry import audit_stops
sys.path.pop(0)


class ProgressRetryAuditTests(unittest.TestCase):
    def test_actual_stops_match_trace_and_tampering_is_rejected(self):
        obs = np.full(16, .2)
        obs[0] = .01
        result = ProgressRetryAttack(horizon=2)(obs, linear_victim(),
                                                AttackContext(0, 0, 0, 0, BudgetOracle(obs), 2))
        rows = [dict(attack_metadata=result.metadata)]
        self.assertGreater(audit_stops(rows), 0)
        stop = result.metadata["retry_stop_trace"][0]
        stop["margin"] += .1
        with self.assertRaisesRegex(ValueError, "Stop margin"):
            audit_stops(rows)
