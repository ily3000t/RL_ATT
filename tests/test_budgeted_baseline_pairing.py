import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_budgeted_zero_one_replications import validate_pairing
sys.path.pop(0)


class BudgetedBaselinePairingTests(unittest.TestCase):
    def fixture(self):
        base = ROOT / "configs/evaluation"
        configs = [json.loads((base / name).read_text()) for name in (
            "zero_one_budgeted_development_attack1_seed0.json", "proposed_progress_development_attack1_seed0.json")]
        manifests = [dict(config=config, python_runtime="runtime", pip_freeze="freeze", sumo_version="sumo",
                          source_sha256_before={"actor": "abc"}, victim_references=["same"]) for config in configs]
        entries = [[dict(attack=a, run_seed=0, effective_seeds={"attack_seed": 1, "sumo_seed": 42,
                         "attack_rng": "unused_no_attack" if a["name"] == "none" else "sha256_attack_episode_full_history_target_attempt_local_numpy"},
                         checkpoint_sha256="checkpoint", weights_sha256="weights") for a in config["attacks"]] for config in configs]
        return manifests[0], manifests[1], entries[0], entries[1], 1

    def test_matched_batches_accept_retry_rule_but_reject_confounders(self):
        fixture = self.fixture()
        validate_pairing(*fixture)
        for field in ("python_runtime", "pip_freeze", "sumo_version", "source_sha256_before", "victim_references"):
            changed = copy.deepcopy(fixture)
            changed[0][field] = "different"
            with self.assertRaisesRegex(ValueError, "provenance differs"):
                validate_pairing(*changed)

    def test_rng_labels_are_checked_separately_from_actual_seed_values(self):
        fixture = self.fixture()
        validate_pairing(*fixture)
        for key, value, message in (("attack_rng", "unused_no_attack", "RNG mechanism"),
                                    ("attack_seed", 2, "victim/seed differs"),
                                    ("sumo_seed", 43, "victim/seed differs")):
            changed = copy.deepcopy(fixture)
            changed[2][1]["effective_seeds"][key] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_pairing(*changed)
        for field in ("run_seed", "effective_seeds", "checkpoint_sha256", "weights_sha256"):
            changed = copy.deepcopy(fixture)
            changed[2][1][field] = "different"
            with self.assertRaisesRegex(ValueError, "victim/seed differs"):
                validate_pairing(*changed)
        for key in ("budget", "parameters"):
            changed = copy.deepcopy(fixture)
            changed[2][1]["attack"][key]["epsilon"] = .5
            with self.assertRaisesRegex(ValueError, "budget"):
                validate_pairing(*changed)

    def test_configs_only_add_missing_baseline_conditions_and_attack_seed(self):
        base = ROOT / "configs/evaluation"
        for seed in range(5):
            original = json.loads((base / ("proposed_development_seed%d.json" % seed)).read_text())
            original["attacks"] = [a for a in original["attacks"] if a["name"] in ("none", "zero_one_budgeted_return", "zero_one_budgeted_safety")]
            for attack_seed in (1, 2):
                actual = json.loads((base / ("zero_one_budgeted_development_attack%d_seed%d.json" % (attack_seed, seed))).read_text())
                self.assertEqual(actual["research_seeds"]["attack_seed"], attack_seed)
                actual["research_seeds"]["attack_seed"] = 0
                self.assertEqual(actual, original)
