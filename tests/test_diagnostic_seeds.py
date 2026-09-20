import copy
import json
from pathlib import Path
import unittest
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.evaluation.configuration import validate_config
from rl_att.utils.seeding import seed_manifest


class DiagnosticSeedTests(unittest.TestCase):
    def test_existing_protocol_is_exactly_preserved(self):
        for seed in range(5):
            self.assertEqual(evaluation_seeds(seed, 20), seed_manifest(seed, "controlled", 20, phase="evaluation"))

    def test_checkpoint_and_traffic_are_independent(self):
        first, second = evaluation_seeds(2, 20, 1), evaluation_seeds(4, 20, 1)
        self.assertEqual(first["episode_sumo_seeds"], second["episode_sumo_seeds"])
        self.assertEqual(first["numpy_seed"], 1)
        self.assertEqual(first["run_seed"], 2)
        self.assertEqual(first["policy_seed"], 2)
        self.assertEqual(first["attack_seed"], 2)  # deterministic FGSM does not consume it
        self.assertNotEqual(first["episode_sumo_seeds"], evaluation_seeds(2, 20, 2)["episode_sumo_seeds"])

    def test_misleading_legacy_or_stochastic_cross_comparisons_rejected(self):
        path = Path(__file__).resolve().parents[1] / "configs/evaluation/benchmark_stage3_seed0.json"
        config = json.loads(path.read_text())
        config["attacks"] = [config["attacks"][0], config["attacks"][2]]
        config.update(traffic_seed=2, verify_legacy_no_attack=False)
        validate_config(config)
        for key, value in (("traffic_seed", True), ("traffic_seed", 5), ("verify_legacy_no_attack", True)):
            invalid = copy.deepcopy(config)
            invalid[key] = value
            with self.assertRaises(ValueError):
                validate_config(invalid)
        config["attacks"][1]["name"] = "pgd"
        with self.assertRaises(ValueError):
            validate_config(config)
