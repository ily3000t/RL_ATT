import unittest

from scripts.analyze_pgd_training import episode_windows, probe_profile, paired_training


def cohorts():
    return {name: {seed: [dict(episode=i+1, sumo_seed=1000*seed+i, episode_return=value,
                              steps=200, ego_collision_observed=(i == 399), terminated=(i == 399))
                         for i in range(400)] for seed in range(5)}
            for name, value in (("clean", 10.), ("oarl", 11.), ("pgd_consistency", 8.))}


class PGDTrainingAnalysisTests(unittest.TestCase):
    def test_paired_training_does_not_mix_evaluation_or_episode_denominators(self):
        result = paired_training(cohorts())
        self.assertEqual(result["cohorts"]["pgd_consistency"]["episodes"], 500)
        self.assertEqual(result["cohorts"]["pgd_consistency"]["collisions"], 5)
        self.assertEqual(result["cohorts"]["pgd_consistency"]["collision_episode_fraction"], .01)
        self.assertEqual(result["per_seed"][0]["pgd_minus_clean_return"], -2.)

    def test_mismatched_traffic_is_not_a_paired_comparison(self):
        value = cohorts()
        value["clean"][2][250]["sumo_seed"] += 1
        with self.assertRaisesRegex(ValueError, "not paired"):
            paired_training(value)

    def test_missing_seed_or_duplicate_episode_is_rejected(self):
        value = cohorts()
        del value["clean"][4]
        with self.assertRaisesRegex(ValueError, "five seeds"):
            paired_training(value)
        value = cohorts()
        value["pgd_consistency"][1][-1]["episode"] = 399
        with self.assertRaisesRegex(ValueError, "grid"):
            paired_training(value)

    def test_partial_window_cannot_silently_shrink_last100(self):
        rows = cohorts()["clean"][0]
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            episode_windows(rows[:-1])
        self.assertEqual(episode_windows(rows)[-1]["first_episode"], 301)

    def test_deterministic_constant_policy_and_uncertain_policy_are_distinct(self):
        constant = probe_profile([[0., 0., 1.], [0., 0., 1.]])
        uncertain = probe_profile([[1./3]*3, [1./3]*3])
        self.assertEqual(constant["greedy_action_counts"], [0, 0, 2])
        self.assertEqual(constant["entropy_nats_mean"], 0.)
        self.assertGreater(uncertain["entropy_nats_mean"], 1.)

    def test_non_probability_arrays_are_rejected(self):
        for value in ([[0., 0., 0.]], [[-.1, .1, 1.]], [[0., 1.]], [[float("nan"), 0., 1.]]):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                probe_profile(value)


if __name__ == "__main__":
    unittest.main()
