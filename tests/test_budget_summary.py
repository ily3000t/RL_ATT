import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_budget_sweep import aggregate, check_grid
sys.path.pop(0)


class BudgetSummaryTests(unittest.TestCase):
    def test_duplicate_seed_cannot_replace_missing_grid_cell(self):
        reports = [dict(gradient_cap=c, attack_seed=s) for c in (100, 200) for s in range(3)]
        check_grid(reports)
        reports[-1] = reports[0]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            check_grid(reports)

    def test_partial_or_unregistered_grid_is_rejected(self):
        reports = [dict(gradient_cap=c, attack_seed=s) for c in (100, 200) for s in range(3)]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            check_grid(reports[:-1])
        reports[-1]["gradient_cap"] = 50
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            check_grid(reports)

    def test_outcomes_use_episode_weights_and_clean_eligible_denominator(self):
        row = dict(episodes=2, collisions=1,
                   summary=dict(attack_success_eligible_episodes=1, attack_successes=0,
                                episode_return_mean=10., steps=300),
                   audit=dict(costs=dict(gradient_evaluations=20), complete_search_candidates=3,
                              live_unverified_fallback_steps=0),
                   candidate_availability=dict(blocks=4, selected_complete_fallback_blocks=2),
                   episode_rows=[dict(research_audit=dict(oracle_episode_setup_cost=dict(shadow_steps=82))) for _ in range(2)])
        other = copy.deepcopy(row)
        other.update(episodes=1, collisions=1, episode_rows=other["episode_rows"][:1])
        other["summary"].update(attack_successes=1, episode_return_mean=40., steps=100)
        other["audit"]["costs"]["gradient_evaluations"] = 10
        combined = aggregate([row, other])
        self.assertEqual(combined["mean_return"], 20.)
        self.assertEqual(combined["conversion_rate"], .5)
        self.assertEqual(combined["collisions"], 2)
        self.assertEqual(combined["episodes"], 3)
        self.assertEqual(combined["costs"]["gradient_evaluations"], 30)
        self.assertEqual(combined["oracle_setup_shadow_steps"], 246)
        self.assertEqual(combined["availability"]["blocks"], 8)
