import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_mechanism_controls import CONTROL_NAMES, make_config, sha256, source_sha256
sys.path.pop(0)


class MechanismProtocolTests(unittest.TestCase):
    def test_entire_planned_grid_preserves_shared_budgets_and_separates_seeds(self):
        traffic = json.loads((ROOT / "configs/research_seed_splits.json").read_text())["splits"]
        for seed in range(5):
            for cap in (100, 200, 400):
                for attack_seed in (0, 1, 2):
                    config = make_config(seed, 10, cap, attack_seed)
                    self.assertEqual([a["name"] for a in config["attacks"]], list(CONTROL_NAMES))
                    self.assertFalse(config["gate_enabled"])
                    self.assertEqual(config["research_seeds"]["episode_sumo_seeds"], traffic["10"])
                    self.assertEqual(config["research_seeds"]["attack_seed"], attack_seed)
                    base = copy.deepcopy(config["attacks"][1]["parameters"])
                    for spec in config["attacks"][1:]:
                        p = copy.deepcopy(spec["parameters"])
                        self.assertEqual(p.pop("max_attempts"), 1 if spec["name"] == "ours_single_return" else 3)
                        if spec["name"] == "ours_progress_return":
                            self.assertEqual(p.pop("retry_rule"), "strict_margin_progress")
                        expected = copy.deepcopy(base)
                        expected.pop("max_attempts")
                        self.assertEqual(p, expected)
                        self.assertEqual(p["resource_limits"], dict(gradient_evaluations=cap, policy_forward_calls=2 * cap,
                                                                  new_shadow_transitions=200, shadow_steps=4000))

    def test_frozen_smoke_and_provenance_match_generated_configs(self):
        protocol = json.loads((ROOT / "configs/research/mechanism_controls.json").read_text())
        self.assertEqual(protocol["smoke"]["total_episodes"], 50)
        self.assertEqual(protocol["planned_development"]["total_episodes"], 2250)
        for seed, path in enumerate(protocol["smoke"]["configs"]):
            self.assertEqual(json.loads((ROOT / path).read_text()), make_config(seed))
        for path, digest in protocol["frozen_source_sha256"].items():
            self.assertEqual(source_sha256(path), digest)
        self.assertEqual(sha256(ROOT / "configs/frozen_victims.json"), protocol["frozen_victims_sha256"])
