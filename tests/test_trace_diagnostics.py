import copy
import unittest
from rl_att.evaluation.trace_diagnostics import paired_trace


def row(step, action=2):
    return dict(step=step, action=action, clean_action_at_visited_state=2,
                observation=[0.]*16, next_observation=[0.]*16, reward=1., terminated=False,
                safety=dict(ego_present=True, ego_collision_observed=False, pairs=[]))


class TraceDiagnosticTests(unittest.TestCase):
    def test_divergence_and_terminal_stale_observation(self):
        clean = [row(i) for i in range(3)]
        attacked = copy.deepcopy(clean[:2])
        attacked[0]["action"] = 1
        attacked[0]["next_observation"][13] = .3
        attacked[1]["observation"][13] = .3
        attacked[1]["safety"].update(ego_present=False, ego_collision_observed=True)
        attacked[1]["terminated"] = True
        result = paired_trace(clean, attacked)
        self.assertEqual(result["first_action_divergence"], 0)
        self.assertEqual(result["lane_index_transition_steps"], [0])
        self.assertEqual(result["divergence_to_collision"], 1)

    def test_environment_divergence_without_changed_action_is_rejected(self):
        clean = [row(0), row(1)]
        attacked = copy.deepcopy(clean)
        attacked[0]["reward"] = 2.
        with self.assertRaises(ValueError):
            paired_trace(clean, attacked)
        with self.assertRaises(ValueError):
            paired_trace(clean, clean[:1])
