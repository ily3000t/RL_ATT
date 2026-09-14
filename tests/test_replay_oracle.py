import unittest
from rl_att.evaluation.replay_oracle import ReplayOracle


class ReplayOracleTests(unittest.TestCase):
    def setUp(self):
        self.actions = []
        self.drift = False

        def transition():
            return dict(observation=[sum(self.actions) + int(self.drift)] * 16,
                        reward=float(len(self.actions)), done=len(self.actions) == 3, collision=False)

        def reset():
            self.actions = []
            return transition()

        def step(action):
            self.actions.append(action)
            return transition()

        self.oracle = ReplayOracle(reset, step)
        self.oracle.start_episode(transition())

    def test_prefix_cache_replay_and_live_confirmation(self):
        oracle = self.oracle
        oracle.begin()
        first = oracle.step(1)
        oracle.step(2)
        oracle.begin()
        self.assertEqual(oracle.step(1), first)
        oracle.step(0)  # reset and replay prefix (1,), then branch
        self.assertEqual(oracle.counts["replay_steps"], 1)
        self.assertEqual(oracle.counts["cache_hits"], 1)
        oracle.observe(1, first)
        self.assertEqual(oracle.begin(), first)
        oracle.step(2)
        oracle.step(0)
        with self.assertRaises(ValueError):
            oracle.step(0)  # terminal is not a replayable continuation

    def test_live_mismatch_and_replay_drift_fail_closed(self):
        oracle = self.oracle
        oracle.begin()
        first = oracle.step(1)
        with self.assertRaises(ValueError):
            oracle.observe(2, first)
        with self.assertRaises(ValueError):
            oracle.observe(1, dict(first, reward=99))
        oracle.begin()
        self.drift = True
        with self.assertRaises(ValueError):
            oracle.step(0)


if __name__ == "__main__":
    unittest.main()
