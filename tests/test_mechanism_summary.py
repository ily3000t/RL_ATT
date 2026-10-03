import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_mechanism_development import validate_grid, aggregate_contrast, win_tie_loss
sys.path.pop(0)


class MechanismSummaryTests(unittest.TestCase):
    def test_complete_grid_rejects_missing_duplicate_and_unregistered_cells(self):
        grid = [dict(gradient_cap=c, attack_seed=s) for c in (100, 200, 400) for s in (0, 1, 2)]
        validate_grid(grid)
        for wrong in (grid[:-1], grid + [grid[0]], grid[:-1] + [dict(gradient_cap=800, attack_seed=2)]):
            with self.assertRaises(ValueError):
                validate_grid(wrong)

    def test_correlated_replicates_cluster_by_ten_traffic_and_five_models(self):
        pairs = []
        for seed in (0, 1, 2):
            for checkpoint in range(5):
                episodes = [dict(episode=e, sumo_seed=1000 + e, clean_collision=False,
                                 first_collision=e == 1, second_collision=False,
                                 return_first_minus_second=-10. if e == 1 else 1.,
                                 gradient_first_minus_second=2, forward_first_minus_second=4)
                            for e in range(1, 11)]
                pairs.append(dict(attack_seed=seed, checkpoint_seed=checkpoint, episodes=episodes, identical_trajectories=0))
        result, raw = aggregate_contrast(pairs)
        self.assertEqual(len(raw), 150)
        self.assertEqual(len(result["per_model"]), 5)
        self.assertEqual(len(result["per_traffic"]), 10)
        self.assertAlmostEqual(result["mean_return_delta"], -.1)
        self.assertEqual(result["net_conversions"], 15)
        self.assertEqual(result["leave_one_traffic_out_net_conversion_range"], [0, 15])
        self.assertAlmostEqual(result["leave_one_traffic_out_return_delta_range"][1], 1.)
        self.assertEqual(result["per_model_return_win_tie_loss"], dict(lower=5, tied=0, higher=0))
        self.assertEqual(result["per_traffic_return_win_tie_loss"], dict(lower=1, tied=0, higher=9))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_contrast(pairs + [pairs[0]])
        with self.assertRaisesRegex(ValueError, "Missing"):
            aggregate_contrast(pairs[:-1])
        wrong = copy.deepcopy(pairs)
        wrong[0]["episodes"][0]["sumo_seed"] = 999
        with self.assertRaisesRegex(ValueError, "cluster"):
            aggregate_contrast(wrong)

    def test_fixed_numerical_tie_tolerance_does_not_change_mean_deltas(self):
        self.assertEqual(win_tie_loss([-1., 0., 1e-12, 1.]), dict(lower=1, tied=2, higher=1))
