import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_single_validation import validate_reports, COMPARISON_NAMES
from summarize_mechanism_development import aggregate_contrast
sys.path.pop(0)


class SingleValidationSummaryTests(unittest.TestCase):
    def test_twenty_traffic_clusters_preserve_correlated_replicates(self):
        pairs = []
        for seed in (0, 1, 2):
            for checkpoint in range(5):
                episodes = [dict(episode=e, sumo_seed=2000 + e, clean_collision=False,
                                 first_collision=e == 1, second_collision=False,
                                 return_first_minus_second=-20. if e == 1 else 1.,
                                 gradient_first_minus_second=2, forward_first_minus_second=4)
                            for e in range(1, 21)]
                pairs.append(dict(attack_seed=seed, checkpoint_seed=checkpoint, episodes=episodes, identical_trajectories=0))
        result, raw = aggregate_contrast(pairs, expected_episodes=20)
        self.assertEqual(len(raw), 300)
        self.assertEqual(len(result["per_model"]), 5)
        self.assertEqual(len(result["per_traffic"]), 20)
        self.assertAlmostEqual(result["mean_return_delta"], -.05)
        self.assertEqual(result["net_conversions"], 15)
        self.assertEqual(result["leave_one_traffic_out_net_conversion_range"], [0, 15])
        self.assertAlmostEqual(result["leave_one_traffic_out_return_delta_range"][1], 1.)
        self.assertEqual(result["per_traffic_return_win_tie_loss"], dict(lower=1, tied=0, higher=19))
        for data, count in ((pairs, 10), (pairs, 30), (pairs[:-1], 20), (pairs + [pairs[0]], 20)):
            with self.assertRaises(ValueError):
                aggregate_contrast(data, expected_episodes=count)

    def test_reused_and_new_rows_require_complete_paired_provenance(self):
        protocol = dict(new_methods=["none", "ours_single_return"], baseline_experiment_commit="old")
        traffic = dict(episode_sumo_seeds=list(range(2001, 2021)))
        reports = []
        for cap in (100, 200, 400):
            for seed in (0, 1, 2):
                rows = []
                for checkpoint in range(5):
                    for name in COMPARISON_NAMES:
                        new = name in protocol["new_methods"]
                        rows.append(dict(checkpoint_seed=checkpoint, attack=name, episodes=20,
                                         checkpoint_sha256="model%d" % checkpoint,
                                         origin="new_run" if new else "reused_validation", source_commit="new" if new else "old",
                                         episode_rows=[dict(episode=e, sumo_seed=s) for e, s in enumerate(traffic["episode_sumo_seeds"], 1)]))
                reports.append(dict(gradient_cap=cap, attack_seed=seed, verified=True,
                                    kind="single_candidate_validation", git_commit="new", research_split_id=20,
                                    new_episodes=200, clean_regression=dict(passed=True), traffic=copy.deepcopy(traffic), rows=rows))
        observed_traffic, models = validate_reports(reports, protocol, "new")
        self.assertEqual(observed_traffic, traffic)
        self.assertEqual(len(models), 5)
        variants = []
        wrong = copy.deepcopy(reports)
        wrong[0]["rows"][0]["source_commit"] = "old"
        variants.append(wrong)
        wrong = copy.deepcopy(reports)
        wrong[0]["rows"][1]["origin"] = "new_run"
        variants.append(wrong)
        wrong = copy.deepcopy(reports)
        wrong[1]["rows"][0]["checkpoint_sha256"] = "changed"
        variants.append(wrong)
        wrong = copy.deepcopy(reports)
        wrong[0]["rows"][0]["episode_rows"][0]["sumo_seed"] = 999
        variants.append(wrong)
        wrong = copy.deepcopy(reports)
        wrong[0]["rows"][-1] = copy.deepcopy(wrong[0]["rows"][0])
        variants.append(wrong)
        for wrong in variants:
            with self.assertRaises(ValueError):
                validate_reports(wrong, protocol, "new")
