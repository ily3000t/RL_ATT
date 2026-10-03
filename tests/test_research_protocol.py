import copy
import json
from pathlib import Path
import sys
import unittest
from rl_att.evaluation.configuration import validate_config
from rl_att.evaluation.seed_control import evaluation_seeds

ROOT = Path(__file__).resolve().parents[1]


class ResearchProtocolTests(unittest.TestCase):
    def test_all_old_and_new_configs_validate(self):
        for path in (ROOT / "configs/evaluation").glob("*.json"):
            validate_config(json.loads(path.read_text(encoding="utf-8")))

    def test_checkpoint_and_attack_rng_do_not_change_traffic(self):
        config = json.loads((ROOT / "configs/evaluation/proposed_smoke_seed0.json").read_text())
        r = config["research_seeds"]
        first = evaluation_seeds(0, 2, research_seeds=r)
        second = evaluation_seeds(2, 2, research_seeds=dict(r, attack_seed=1))
        for key in ("python_seed", "numpy_seed", "torch_seed", "sumo_seed", "episode_sumo_seeds"):
            self.assertEqual(first[key], second[key])
        self.assertEqual(second["attack_seed"], 1)
        self.assertEqual(second["checkpoint_training_seed"], 2)
        broken = copy.deepcopy(r)
        broken["episode_sumo_seeds"][0] += 1
        with self.assertRaises(ValueError):
            evaluation_seeds(0, 2, research_seeds=broken)
        config["attacks"][1]["parameters"]["objective"] = "safety"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_frozen_split_record_reproducible_and_disjoint(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        try:
            from prepare_proposed_configs import protocol
            expected = protocol()
        finally:
            sys.path.pop(0)
        self.assertEqual(expected, json.loads((ROOT / "configs/research_seed_splits.json").read_text()))

    def test_progress_replications_change_only_attack_seed(self):
        base = ROOT / "configs/evaluation"
        for checkpoint in range(5):
            reference = json.loads((base / ("proposed_progress_development_seed%d.json" % checkpoint)).read_text())
            for attack_seed in (1, 2):
                actual = json.loads((base / ("proposed_progress_development_attack%d_seed%d.json" % (attack_seed, checkpoint))).read_text())
                self.assertEqual(actual["research_seeds"]["attack_seed"], attack_seed)
                actual["research_seeds"]["attack_seed"] = 0
                self.assertEqual(actual, reference)
