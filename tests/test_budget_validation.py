import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_proposed_configs import protocol as seed_protocol
from prepare_budget_validation import validation_config
from analyze_budget_validation import validate_effective_seeds, validate_frozen_run
from summarize_proposed_smoke import validate_audit_protocol
from rl_att.evaluation.seed_control import evaluation_seeds
from rl_att.evaluation.configuration import validate_config
sys.path.pop(0)


class BudgetValidationTests(unittest.TestCase):
    def setUp(self):
        self.base = json.loads((ROOT / "configs/evaluation/budget_sweep_g200_attack0_seed0.json").read_text())
        self.traffic = json.loads((ROOT / "configs/research_seed_splits.json").read_text())["splits"]["20"]

    def test_development_default_cannot_silently_accept_validation_or_final_test(self):
        validation = validation_config(self.base, 100, 1, self.traffic)
        validate_audit_protocol(validation, 20, 20)
        with self.assertRaisesRegex(ValueError, "requested research protocol"):
            validate_audit_protocol(validation, 10, 10)
        with self.assertRaisesRegex(ValueError, "supported"):
            validate_audit_protocol(validation, 20, 10)
        validation["research_seeds"]["split_id"] = 30
        with self.assertRaisesRegex(ValueError, "supported"):
            validate_audit_protocol(validation, 50, 30)

    def test_validation_rejects_algorithm_or_runtime_drift(self):
        reference = dict(config=self.base, source_sha256_before={"attack.py": "frozen"}, python_runtime={},
                         pip_freeze={}, sumo_version={}, victim_references=[])
        current = copy.deepcopy(reference)
        current["config"] = validation_config(self.base, 100, 1, self.traffic)
        validate_frozen_run(current, reference, 100, 1, self.traffic)
        changed = copy.deepcopy(current)
        changed["config"]["attacks"][3]["parameters"]["max_attempts"] = 4
        with self.assertRaisesRegex(ValueError, "extend beyond"):
            validate_frozen_run(changed, reference, 100, 1, self.traffic)
        current["source_sha256_before"]["attack.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            validate_frozen_run(current, reference, 100, 1, self.traffic)

    def test_numeric_seeds_and_rng_mechanisms_are_both_verified(self):
        config = validation_config(self.base, 100, 1, self.traffic)
        seeds = evaluation_seeds(0, 20, research_seeds=config["research_seeds"])
        seeds.update(policy_rng="unused_greedy_argmax", attack_rng="unused_no_attack")
        entry = dict(run_seed=0, attack=dict(name="none"), effective_seeds=seeds)
        validate_effective_seeds(entry, config)
        for key, value in (("numpy_seed", 0), ("attack_seed", 2), ("attack_rng", "wrong_rng")):
            changed = copy.deepcopy(entry)
            changed["effective_seeds"][key] = value
            with self.assertRaisesRegex(ValueError, "seed values or RNG"):
                validate_effective_seeds(changed, config)

    def test_full_grid_preserves_algorithms_and_disjoint_seed_lists(self):
        self.assertEqual(seed_protocol(), json.loads((ROOT / "configs/research_seed_splits.json").read_text()))
        protocol = json.loads((ROOT / "configs/research/shared_compute_validation.json").read_text())
        self.assertEqual({(g["gradient_cap"], g["attack_seed"]) for g in protocol["groups"]},
                         {(c, s) for c in (100, 200, 400) for s in range(3)})
        self.assertEqual(len(protocol["groups"]), 9)
        total = 0
        for group in protocol["groups"]:
            self.assertEqual(len(group["configs"]), 5)
            for checkpoint, filename in enumerate(group["configs"]):
                config = json.loads((ROOT / filename).read_text())
                validate_config(config)
                base = copy.deepcopy(self.base)
                base["run_seeds"] = [checkpoint]
                self.assertEqual(config, validation_config(base, group["gradient_cap"], group["attack_seed"], self.traffic))
                self.assertEqual(len(set(config["research_seeds"]["episode_sumo_seeds"])), 20)
                total += config["episodes"] * len(config["attacks"])
        self.assertEqual(total, protocol["total_episodes"])
        self.assertEqual(total, 4500)
