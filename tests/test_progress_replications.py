import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_progress_replications import first_divergence
sys.path.pop(0)


class ReplicationDivergenceTests(unittest.TestCase):
    def trajectories(self):
        old = [dict(step=i, action=0, reward=0., terminated=i == 2,
                    observation=[0.], next_observation=[0.],
                    attack_metadata=dict(planned=i == 0), attack_cost=dict(wall_seconds=1.)) for i in range(3)]
        old[0]["attack_metadata"].update(selected_candidate=0, candidate_trace=[dict(actions=[0, 0, 1])],
                                         inner_attempt_trace=[dict(history=[0, 0], target=1, attempt=2, margin=.1)])
        new = copy.deepcopy(old)
        new[0]["attack_metadata"].update(candidate_trace=[dict(actions=[0, 0, 0])],
                                         inner_attempt_trace=[dict(history=[0, 0], target=1, attempt=1, margin=-.1)],
                                         retry_stop_trace=[dict(history=[0, 0], target=1, margin=-.1)])
        old[2].update(action=1, reward=-1., next_observation=[1.])
        return old, new

    def test_first_difference_retains_attempt_evidence_and_does_not_mutate_input(self):
        old, new = self.trajectories()
        original = copy.deepcopy((old, new))
        result = first_divergence(old, new)
        self.assertEqual(result["step_zero_based"], 2)
        self.assertEqual((result["v0_action"], result["progress_action"]), (1, 0))
        self.assertEqual(result["v0"]["attempts_at_divergence"][0]["attempt"], 2)
        self.assertEqual(result["progress"]["stops_at_divergence"][0]["target"], 1)
        self.assertNotIn("wall_seconds", result["v0"]["block_start_cost"])
        self.assertEqual((old, new), original)
        self.assertIsNone(first_divergence(old, old))

    def test_unpaired_observation_or_missing_terminal_record_is_rejected(self):
        old, new = self.trajectories()
        new[2]["observation"] = [2.]
        with self.assertRaisesRegex(ValueError, "Different input"):
            first_divergence(old, new)
        with self.assertRaisesRegex(ValueError, "Unequal episode length"):
            first_divergence(old, old[:2])
