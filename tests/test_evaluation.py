import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import Mock
import numpy as np
import torch
import oarl
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.attacks.base import AttackResult
from rl_att.attacks.no_attack import NoAttack
from rl_att.evaluation.configuration import validate_config
from rl_att.evaluation.evaluator import AttackEvaluator, validate_budget
from rl_att.evaluation.results import summarize_episodes
from rl_att.evaluation.sumo_metrics import following_measures, SUMOMetrics, summarize_safety
from rl_att.utils.seeding import seed_manifest


class EvaluationTests(unittest.TestCase):
    def test_following_formulas_and_missing_denominators(self):
        row = following_measures(20, 15, 10)
        self.assertEqual(row["ttc_s"], 4)
        self.assertEqual(row["drac_mps2"], 0.625)
        not_closing = following_measures(20, 8, 10)
        overlap = following_measures(0, 15, 10)
        self.assertIsNone(not_closing["ttc_s"])
        self.assertEqual(not_closing["drac_mps2"], 0)
        self.assertEqual(overlap["ttc_s"], 0)
        self.assertIsNone(overlap["drac_mps2"])
        summary = summarize_safety([{"ego_present": True, "pairs": [row, not_closing, overlap]}])
        self.assertEqual(summary["finite_ttc_samples"], 2)
        self.assertEqual(summary["finite_drac_samples"], 2)
        self.assertEqual(summary["overlap_samples"], 1)
        self.assertIsNone(summarize_safety([])["minimum_ttc_s"])
        json.dumps(summary, allow_nan=False)
        with self.assertRaises(ValueError):
            following_measures(float("nan"), 15, 10)

    def test_traci_gap_correction_and_removed_ego_collision(self):
        vehicle, simulation = Mock(), Mock()
        vehicle.getIDList.return_value = ["Auto", "front", "rear"]
        vehicle.getLaneID.return_value = "lane0"
        vehicle.getLeader.return_value = ("front", 17.5)
        vehicle.getFollower.return_value = ("rear", 8)
        vehicle.getMinGap.side_effect = lambda v: {"Auto": 2.5, "rear": 2}[v]
        vehicle.getSpeed.side_effect = lambda v: {"Auto": 15, "front": 10, "rear": 17}[v]
        simulation.getCollidingVehiclesIDList.return_value = []
        metrics = SUMOMetrics(types.SimpleNamespace(vehicle=vehicle, simulation=simulation))
        row = metrics.sample()
        self.assertEqual(row["pairs"][0]["gap_m"], 20)
        self.assertEqual(row["pairs"][1]["gap_m"], 10)
        self.assertEqual(row["pairs"][1]["ttc_s"], 5)
        vehicle.getIDList.return_value = []
        simulation.getCollidingVehiclesIDList.return_value = ["Auto"]
        vehicle.getLeader.reset_mock()
        row = metrics.sample()
        self.assertTrue(row["ego_collision_observed"])
        self.assertFalse(row["ego_present"])
        self.assertEqual(row["pairs"], [])
        vehicle.getLeader.assert_not_called()

    def test_missing_and_other_lane_neighbors_are_not_safety_samples(self):
        vehicle, simulation = Mock(), Mock()
        vehicle.getIDList.return_value = ["Auto", "other"]
        vehicle.getLaneID.side_effect = lambda v: v
        vehicle.getLeader.return_value = None
        vehicle.getFollower.return_value = ("other", 10)
        simulation.getCollidingVehiclesIDList.return_value = []
        row = SUMOMetrics(types.SimpleNamespace(vehicle=vehicle, simulation=simulation)).sample()
        self.assertEqual(row["pairs"], [])

    def test_configuration_and_budget_reject_silent_changes(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/evaluation/oarl_bo_adapter_smoke.json").read_text())
        validate_config(config)
        for changed in (dict(config, gate_enabled=True), dict(config, max_steps=0),
                        dict(config, action_selection="sample")):
            with self.assertRaises(ValueError):
                validate_config(changed)
        changed = copy.deepcopy(config)
        changed["attacks"][1]["budget"]["epsilon"] = 0.2
        with self.assertRaises(ValueError):
            validate_config(changed)
        obs = np.ones(16)
        result = AttackResult(obs + 0.3, np.ones(16) * 0.3, True)
        with self.assertRaises(ValueError):
            validate_budget(result, obs, {"norm": "linf", "epsilon": 0.2})
        with self.assertRaises(ValueError):
            validate_budget(result, obs, {"norm": "none"})

    def test_rollout_termination_freezing_and_paired_summary(self):
        class Env:
            def reset(self):
                self.steps = 0
                return np.zeros(16)

            def step(self, action):
                self.steps += 1
                return np.ones(16) * self.steps, 0.5, self.steps == 2

        metrics = Mock()
        metrics.sample.return_value = {"ego_collision_observed": False, "ego_present": True, "pairs": []}
        victim = VictimAdapter(oarl.ActorNet(16, 3, 128))
        config = {"episodes": 2, "max_steps": 3, "metric_percentiles": {"ttc_percentile": 5, "drac_percentile": 95}}
        seeds = seed_manifest(0, "controlled", 2, phase="evaluation")
        with TemporaryDirectory() as temp:
            rows, summary = AttackEvaluator(Env(), victim, NoAttack(), metrics, config, seeds,
                                            {"norm": "none"}).run(Path(temp) / "evaluation")
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(summary["episode_return_mean"], 1)
            self.assertEqual(summary["attack_rate"], 0)
            self.assertTrue(all(r["terminated"] and not r["truncated"] for r in rows))
            self.assertEqual(summarize_episodes(rows, rows)["return_drop_mean"], 0)
            attacked = copy.deepcopy(rows)
            attacked[0]["ego_collision_observed"] = True
            attacked[0]["episode_return"] = 0
            comparison = summarize_episodes(attacked, rows)
            self.assertEqual(comparison["attack_success_rate"], 0.5)
            self.assertEqual(comparison["return_drop_mean"], 0.5)
            all_collision = copy.deepcopy(rows)
            for r in all_collision:
                r["ego_collision_observed"] = True
            self.assertIsNone(summarize_episodes(attacked, all_collision)["attack_success_rate"])
            with self.assertRaises(ValueError):
                summarize_episodes(attacked, rows[:1])
        victim.assert_frozen()
