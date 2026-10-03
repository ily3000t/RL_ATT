import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_mechanism_controls import make_config, CONTROL_NAMES
sys.path.pop(0)


class MechanismDevelopmentProtocolTests(unittest.TestCase):
    def test_complete_nine_cell_grid_preserves_frozen_search_and_traffic(self):
        protocol = json.loads((ROOT / "configs/research/mechanism_development.json").read_text())
        self.assertEqual(len(protocol["groups"]), 9)
        self.assertEqual({(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]},
                         {(c, s) for c in (100, 200, 400) for s in (0, 1, 2)})
        total, paths = 0, set()
        for group in protocol["groups"]:
            self.assertEqual(len(group["configs"]), 5)
            for checkpoint, path in enumerate(group["configs"]):
                config = json.loads((ROOT / path).read_text())
                self.assertEqual(config, make_config(checkpoint, 10, group["gradient_cap"], group["attack_seed"]))
                self.assertEqual([a["name"] for a in config["attacks"]], list(CONTROL_NAMES))
                self.assertNotIn(path, paths)
                paths.add(path)
                total += config["episodes"] * len(config["attacks"])
        self.assertEqual(total, protocol["total_episodes"])
        self.assertEqual(total, 2250)
        self.assertEqual(protocol["unique_clean_model_traffic_pairs"], 50)
        self.assertEqual(protocol["split_id"], 10)
