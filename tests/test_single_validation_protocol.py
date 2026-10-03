import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_single_validation import candidate_config, NEW_NAMES
sys.path.pop(0)


class SingleValidationProtocolTests(unittest.TestCase):
    def test_all_cells_change_only_method_list_and_single_retry_parameter(self):
        protocol = json.loads((ROOT / "configs/research/single_candidate_validation.json").read_text())
        self.assertEqual({(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]},
                         {(c, s) for c in (100, 200, 400) for s in range(3)})
        count = 0
        for group in protocol["groups"]:
            for checkpoint, path in enumerate(group["configs"]):
                config = json.loads((ROOT / path).read_text())
                self.assertEqual(config, candidate_config(checkpoint, group["gradient_cap"], group["attack_seed"]))
                old = json.loads((ROOT / ("configs/evaluation/budget_validation_g%d_attack%d_seed%d.json" %
                                          (group["gradient_cap"], group["attack_seed"], checkpoint))).read_text())
                self.assertEqual({k: v for k, v in config.items() if k != "attacks"},
                                 {k: v for k, v in old.items() if k != "attacks"})
                expected = copy.deepcopy(next(a for a in old["attacks"] if a["name"] == "ours_progress_return"))
                expected["name"] = "ours_single_return"
                expected["parameters"].pop("retry_rule")
                expected["parameters"]["max_attempts"] = 1
                self.assertEqual(config["attacks"][1], expected)
                self.assertEqual(tuple(a["name"] for a in config["attacks"]), NEW_NAMES)
                count += config["episodes"] * len(config["attacks"])
        self.assertEqual(count, protocol["total_episodes"])
        self.assertEqual(count, 1800)
        self.assertEqual(protocol["split_id"], 20)
        self.assertEqual(protocol["comparison_episodes"], 3600)

    def test_unregistered_seed_budget_or_model_is_rejected(self):
        for values in ((5, 100, 0), (0, 800, 0), (0, 100, 3)):
            with self.assertRaises(ValueError):
                candidate_config(*values)
