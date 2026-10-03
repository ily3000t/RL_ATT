import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_budget_validation import concentration, validate_grid
sys.path.pop(0)


class ValidationSummaryTests(unittest.TestCase):
    def pairs(self):
        first = dict(attack_seed=0, checkpoint_seed=0, episodes=[
            dict(episode=1, sumo_seed=101, clean_collision=False, first_collision=True, second_collision=False, first_steps=2, second_steps=200),
            dict(episode=2, sumo_seed=102, clean_collision=False, first_collision=False, second_collision=True, first_steps=200, second_steps=1),
            dict(episode=3, sumo_seed=103, clean_collision=True, first_collision=True, second_collision=False, first_steps=1, second_steps=200)])
        second = copy.deepcopy(first)
        second["attack_seed"] = 1
        second["episodes"][1].update(second_collision=False, second_steps=200)
        return [first, second]

    def test_cluster_omission_keeps_losses_and_excludes_clean_collisions(self):
        result = concentration(self.pairs(), [101, 102, 103])
        self.assertEqual(result["net_conversions"], 1)
        self.assertEqual(result["eligible"], 4)
        self.assertEqual(result["leave_one_traffic_out_net_range"], [-1, 2])
        rows = result["traffic"]
        self.assertEqual([r["net_conversions"] for r in rows], [2, -1, 0])
        self.assertEqual(rows[0]["rate_difference_without_this_traffic"], -.5)
        self.assertEqual(rows[1]["second_only_one_step_conversion"], 1)
        self.assertEqual(rows[2]["first_one_step_collision"], 2)
        self.assertEqual(rows[2]["first_only_one_step_conversion"], 0)
        self.assertEqual(rows[2]["net_without_this_traffic"], 1)

    def test_duplicate_replicates_and_mispaired_traffic_are_rejected(self):
        pairs = self.pairs()
        with self.assertRaisesRegex(ValueError, "Duplicate paired"):
            concentration(pairs + [pairs[0]], [101, 102, 103])
        pairs[0]["episodes"][0]["sumo_seed"] = 102
        with self.assertRaisesRegex(ValueError, "Traffic order"):
            concentration(pairs, [101, 102, 103])

    def test_zero_discordance_traffic_still_required(self):
        pairs = self.pairs()
        for p in pairs:
            p["episodes"].pop()
        with self.assertRaisesRegex(ValueError, "Missing traffic"):
            concentration(pairs, [101, 102, 103])

    def test_validation_grid_rejects_duplicate_or_missing_high_budget(self):
        reports = [dict(gradient_cap=c, attack_seed=s) for c in (100, 200, 400) for s in range(3)]
        validate_grid(reports)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_grid(reports[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_grid(reports + [reports[0]])
