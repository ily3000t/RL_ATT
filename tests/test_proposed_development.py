import copy
from pathlib import Path
import sys
import unittest
import hashlib
import numpy as np
import math
from rl_att.attacks.rollout_objectives import optimizer_value

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_proposed_development import pair_outcomes, retry_yield, audit_episode_records, objective_opportunities
sys.path.pop(0)


class DevelopmentAnalysisTests(unittest.TestCase):
    def test_safety_scalar_keeps_collision_priority_at_finite_extremes(self):
        self.assertLess(optimizer_value((-1, 1e308)), optimizer_value((0, -1e308)))
        for reward in (-200., -10., 0., 10., 200.):
            for collision in (-1, 0):
                self.assertEqual(optimizer_value((collision, reward)),
                                 2. * collision + 2. / math.pi * math.atan(reward))

    def test_objective_opportunity_is_measured_on_same_candidate_set(self):
        def block(candidates):
            return dict(attack_metadata=dict(planned=True, candidate_trace=[dict(collisions=[c], value=v, actions=[a])
                                                                           for c, v, a in candidates]))
        result = objective_opportunities([block([(False, -1., 0), (True, 10., 1)]),
                                          block([(False, 1., 0), (True, 0., 1)]),
                                          block([(False, 1., 0)])])
        self.assertEqual(result["mixed_collision_candidates"], 2)
        self.assertEqual(result["no_collision_candidate"], 1)
        self.assertEqual(result["return_safety_selection_differs"], 1)
        self.assertEqual(result["return_safety_actual_sequence_differs"], 1)

    def test_raw_episode_integrity_checks_return_costs_digest_and_completion(self):
        obs = [0.] * 16
        raw = [dict(episode=1, step=0, action=2, reward=-1., terminated=True, next_observation=obs,
                    safety=dict(ego_collision_observed=True),
                    attack_cost=dict(policy_forward_calls=3, objective_evaluations=1))]
        digest = hashlib.sha256(np.asarray(obs, dtype=np.float64).tobytes())
        digest.update(np.asarray([2, -1., True], dtype=np.float64).tobytes())
        episode = dict(episode=1, steps=1, episode_return=-1., terminated=True, truncated=False,
                        ego_collision_observed=True, gradient_evaluations=0, attack_policy_forward_calls=3,
                        objective_evaluations=1, action_counts=[0, 0, 1], trajectory_sha256=digest.hexdigest())
        self.assertEqual(audit_episode_records(raw, [episode], 200), 1)
        episode["episode_return"] = 1.
        with self.assertRaisesRegex(ValueError, "summary/raw mismatch"):
            audit_episode_records(raw, [episode], 200)
        raw[0]["terminated"] = False
        with self.assertRaisesRegex(ValueError, "Incomplete raw episode"):
            audit_episode_records(raw, [episode], 200)

    @staticmethod
    def episode(index, collision, reward, digest):
        return dict(episode=index, sumo_seed=100 + index, ego_collision_observed=collision,
                    episode_return=reward, trajectory_sha256=digest, steps=10,
                    gradient_evaluations=2, attack_policy_forward_calls=4)

    def test_paired_denominator_excludes_preexisting_collision(self):
        clean = [self.episode(i, i == 3, 10., "c") for i in (1, 2, 3)]
        first = [self.episode(i, i != 2, float(i), "a") for i in (1, 2, 3)]
        second = [self.episode(i, i == 2, 0., "b") for i in (1, 2, 3)]
        result = pair_outcomes(clean, first, second)
        self.assertEqual(result["first_only_collision"], 2)
        self.assertEqual(result["second_only_collision"], 1)
        self.assertEqual(result["eligible_clean_noncollision"], 2)
        self.assertEqual(result["first_only_conversion"], 1)
        self.assertEqual(result["second_only_conversion"], 1)
        self.assertEqual(result["return_first_minus_second_mean"], 2.)
        second[0]["sumo_seed"] += 1
        with self.assertRaisesRegex(ValueError, "Pair keys"):
            pair_outcomes(clean, first, second)

    @staticmethod
    def row():
        attempts = [dict(history=[], target=target, actual=actual, attempt=index, seed=index, margin=margin)
                    for target, actual, index, margin in ((0, 2, 0, -.5), (1, 2, 0, -.5),
                                                         (0, 2, 1, -.5), (0, 0, 2, .1))]
        return dict(episode=1, step=0, action=0, attack_metadata=dict(planned=True, search_kind="behavior",
                    inner_attempt_trace=attempts, selected_candidate=0, candidate_trace=[dict(actions=[0])]))

    def test_retry_new_branch_and_selection_counted_once(self):
        row = self.row()
        result = retry_yield([row])
        self.assertEqual(result["attempts"], 4)
        self.assertEqual(result["retries"], 2)
        self.assertEqual(result["retry_new_branches"], 1)
        self.assertEqual(result["retry_new_branches_on_selected_plan"], 1)
        self.assertEqual(result["retry_repeated_margin_and_action"], 1)
        second = copy.deepcopy(row)
        second["episode"] = 2
        self.assertEqual(retry_yield([row, second])["retry_new_branches"], 2)

    def test_failed_target_can_discover_another_valid_branch(self):
        row = self.row()
        row["attack_metadata"]["inner_attempt_trace"][2]["actual"] = 1
        result = retry_yield([row])
        self.assertEqual(result["retry_target_hits"], 1)
        self.assertEqual(result["retry_new_branches"], 2)
        self.assertEqual(result["retry_new_branches_on_selected_plan"], 1)
        row["attack_metadata"]["inner_attempt_trace"][2]["attempt"] = 2
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            retry_yield([row])
