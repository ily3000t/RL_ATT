import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from final_statistics import paired_summary, outcome_cube


def rows(traffic, values, deterministic=False):
    return [dict(checkpoint_seed=m, sumo_seed=s, attack_seed=a, value=values[i] + m + a)
            for i, s in enumerate(traffic) for m in range(5)
            for a in ([0] if deterministic else [0, 1, 2])]


class FinalStatisticsTests(unittest.TestCase):
    def test_fifty_clusters_instead_of_750_independent_samples(self):
        traffic = list(range(100, 150))
        first = rows(traffic, [-7.] * 50)
        second = rows(traffic, [0.] * 50)
        report = paired_summary(first, second, traffic)
        self.assertEqual(report["actual_episodes"], dict(first=750, second=750))
        self.assertEqual(report["traffic_clusters"], 50)
        self.assertEqual(report["fixed_model_traffic_units"], 250)
        self.assertEqual(report["mean_delta"], -7.)
        self.assertEqual(report["descriptive_interval"], [-7., -7.])
        self.assertEqual(report["primary_family_adjusted_interval"], [-7., -7.])

    def test_common_traffic_effect_cancels_and_bootstrap_is_reproducible(self):
        traffic = [17, 31, 55]
        first = rows(traffic, [-4., 3., -2.])
        second = rows(traffic, [0., 0., 0.])
        expected = paired_summary(first, second, traffic)
        for r in first + second:
            r["value"] += {17: 1000., 31: -30., 55: 300.}[r["sumo_seed"]]
        self.assertEqual(expected, paired_summary(first, second, traffic))
        self.assertAlmostEqual(expected["mean_delta"], -1.)
        self.assertEqual([r["mean_delta"] for r in expected["leave_one_traffic_out"]], [.5, -3., -.5])

    def test_deterministic_outcome_pairing_does_not_inflate_actual_count(self):
        traffic = [17, 31]
        fgsm = rows(traffic, [-2., -2.], deterministic=True)
        stochastic = rows(traffic, [0., 0.])
        report = paired_summary(fgsm, stochastic, traffic, first_deterministic=True)
        self.assertEqual(report["actual_episodes"], dict(first=10, second=30))
        self.assertEqual(report["mean_delta"], -3.)
        self.assertEqual(report["traffic_clusters"], 2)

    def test_missing_duplicate_unpaired_nonfinite_rows_rejected(self):
        traffic = [17, 31]
        original = rows(traffic, [0., 0.])
        for change in ("missing", "duplicate", "unknown_traffic", "nonfinite", "wrong_model"):
            data = copy.deepcopy(original)
            if change == "missing":
                data.pop()
            elif change == "duplicate":
                data.append(data[0])
            elif change == "unknown_traffic":
                data[0]["sumo_seed"] = 99
            elif change == "nonfinite":
                data[0]["value"] = float("nan")
            else:
                data[0]["checkpoint_seed"] = 5
            with self.assertRaises(ValueError):
                outcome_cube(data, traffic)
        with self.assertRaises(ValueError):
            outcome_cube(original, [17, 17])


if __name__ == "__main__":
    unittest.main()
