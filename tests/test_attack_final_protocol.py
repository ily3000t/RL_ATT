"""Protect the unexposed final matrix and its predeclared scientific controls."""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_attack_final_protocol import (matrix, validate_registration, PROTOCOL, STATISTICS,
                                           FAILURE_POLICY, canonical_sha256)
from rl_att.evaluation.seed_control import evaluation_seeds


class FinalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs, cls.groups, _ = matrix()
        cls.record = json.loads(PROTOCOL.read_text(encoding="utf-8"))

    def test_complete_unexposed_grid_and_actual_episode_counts(self):
        validate_registration(self.record, self.configs)
        self.assertEqual(len(self.configs), 60)
        self.assertEqual(len(self.groups), 12)
        self.assertEqual(sum(g["episodes"] for g in self.groups), 14500)
        methods = {}
        for config in self.configs.values():
            self.assertEqual(config["attacks"][0]["name"], "none")
            self.assertEqual(config["episodes"], 50)
            for attack in config["attacks"]:
                methods[attack["name"]] = methods.get(attack["name"], 0) + 50
        self.assertEqual(methods["none"], 3000)
        self.assertEqual(methods["fgsm"], 250)
        self.assertEqual(methods["pgd"], 750)
        self.assertEqual(methods["ours_single_return"], 2250)
        self.assertFalse(self.record["final_execution_ready"])

    def test_all_fifty_traffic_seeds_paired_across_models_and_attacks(self):
        traffic, environments = set(), set()
        for config in self.configs.values():
            roles = evaluation_seeds(config["run_seeds"][0], 50, research_seeds=config["research_seeds"])
            traffic.add(tuple(roles["episode_sumo_seeds"]))
            environments.add(tuple(roles[k] for k in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed")))
            self.assertEqual(roles["checkpoint_training_seed"], config["run_seeds"][0])
        self.assertEqual(len(traffic), 1)
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(next(iter(traffic))), 50)
        registry = self.record["seed_protocol"]["splits"]
        self.assertFalse(set(registry["30"]) & set(registry["10"] + registry["20"]))

    def test_missing_duplicate_or_added_conditions_rejected(self):
        for change in ("missing", "extra", "duplicate_group"):
            configs, record = copy.deepcopy(self.configs), copy.deepcopy(self.record)
            path = next(iter(configs))
            if change == "missing":
                configs.pop(path)
            elif change == "extra":
                configs["unexpected.json"] = configs[path]
            else:
                record["groups"][1] = record["groups"][0]
            with self.assertRaises(ValueError):
                validate_registration(record, configs)

    def test_configuration_changes_rejected_even_if_digest_is_recomputed(self):
        path = "configs/evaluation/final_search_g100_attack0_seed0.json"
        for change in ("gate", "checkpoint", "traffic", "budget", "objective", "retry"):
            configs, record = copy.deepcopy(self.configs), copy.deepcopy(self.record)
            config = configs[path]
            if change == "gate":
                config["gate_enabled"] = True
            elif change == "checkpoint":
                config["run_seeds"] = [1]
            elif change == "traffic":
                config["research_seeds"]["episode_sumo_seeds"].reverse()
            else:
                parameters = config["attacks"][3]["parameters"]
                if change == "budget":
                    parameters["resource_limits"]["gradient_evaluations"] = 101
                elif change == "objective":
                    parameters["objective"] = "safety"
                else:
                    parameters["max_attempts"] = 3
            record["configs"][path] = canonical_sha256(config)
            with self.assertRaises(ValueError):
                validate_registration(record, configs)

    def test_no_post_test_analysis_policy_change(self):
        record = copy.deepcopy(self.record)
        record["statistics"]["bootstrap"]["resampling_unit"] = "episode"
        with self.assertRaises(ValueError):
            validate_registration(record, self.configs)
        self.assertEqual(STATISTICS["traffic_clusters"], 50)
        self.assertEqual(FAILURE_POLICY["infrastructure_retries"], 1)


if __name__ == "__main__":
    unittest.main()
