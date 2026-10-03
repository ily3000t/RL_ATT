import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_basic_validation import make_config
from analyze_basic_validation import expected_seeds, validate_reference, verify_random_seed
from rl_att.evaluation.configuration import validate_config
sys.path.pop(0)


class BasicValidationTests(unittest.TestCase):
    def test_frozen_grid_runs_deterministic_fgsm_once_and_reuses_traffic(self):
        p = json.loads((ROOT / "configs/research/basic_validation_controls.json").read_text())
        total, attacks, fgsm = 0, 0, 0
        for group in p["groups"]:
            for checkpoint, path in enumerate(group["configs"]):
                c = json.loads((ROOT / path).read_text())
                validate_config(c)
                reference = json.loads((ROOT / ("configs/evaluation/budget_validation_g400_attack0_seed%d.json" % checkpoint)).read_text())
                self.assertEqual(c, make_config(reference, dict(attacks=p["basic_specs"]), group["attack_seed"]))
                total += 20 * len(c["attacks"])
                attacks += 20 * (len(c["attacks"]) - 1)
                fgsm += 20 * any(a["name"] == "fgsm" for a in c["attacks"])
        self.assertEqual((total, attacks, fgsm), (1300, 1000, 100))

    def test_changed_runtime_or_search_caps_cannot_enter_simple_controls(self):
        c = json.loads((ROOT / "configs/evaluation/budget_validation_g400_attack0_seed0.json").read_text())
        specs = json.loads((ROOT / "configs/evaluation/benchmark_stage3_seed0.json").read_text())["attacks"]
        reference = dict(config=c, source_sha256_before={}, python_runtime={}, pip_freeze={}, sumo_version={}, victim_references=[])
        current = copy.deepcopy(reference)
        current.update(config=make_config(c, dict(attacks=specs), 1), extra_dependencies=[])
        validate_reference(current, reference, specs, 1)
        changed = copy.deepcopy(current)
        changed["config"]["attacks"][2]["parameters"]["steps"] = 2
        with self.assertRaises(ValueError):
            validate_reference(changed, reference, specs, 1)
        current["source_sha256_before"]["attack.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            validate_reference(current, reference, specs, 1)

    def test_stochastic_seed_uses_zero_based_episode_and_rejects_shift(self):
        import numpy as np
        value = int(np.random.SeedSequence([2, 3, 0, 5]).generate_state(1)[0])
        row = dict(episode=1, step=5, attack_metadata=dict(state_seed=value))
        verify_random_seed(row, 2, "pgd")
        with self.assertRaisesRegex(ValueError, "state seed"):
            verify_random_seed(row, 1, "pgd")
        row["episode"] = 2
        with self.assertRaisesRegex(ValueError, "state seed"):
            verify_random_seed(row, 2, "random")

    def test_rng_labels_differ_while_traffic_values_remain_paired(self):
        c = json.loads((ROOT / "configs/evaluation/basic_validation_attack0_seed0.json").read_text())
        a, b = expected_seeds(c, 0, "fgsm"), expected_seeds(c, 0, "oarl_bo")
        self.assertNotEqual(a.pop("attack_rng"), b.pop("attack_rng"))
        self.assertEqual(a, b)
