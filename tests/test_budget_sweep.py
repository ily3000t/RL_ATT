import copy
import json
from pathlib import Path
import sys
import unittest
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.progress_retry import ProgressRetryAttack
from rl_att.attacks.zero_one_controls import BudgetedZeroOneAttack
from rl_att.attacks.proposed import DEFAULT_LIMITS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_budget_sweep import candidate_availability, validate_budget_reference
sys.path.pop(0)


class BudgetSweepTests(unittest.TestCase):
    def test_exhausted_search_keeps_verified_complete_fallback_for_both_methods(self):
        obs = np.full(16, .2)
        obs[0] = .01
        victim = linear_victim()
        for cls in (ProgressRetryAttack, BudgetedZeroOneAttack):
            result = cls(horizon=2, resource_limits=dict(DEFAULT_LIMITS, gradient_evaluations=0))(
                obs, victim, AttackContext(0, 0, 0, 0, BudgetOracle(obs), 2))
            counts = candidate_availability([dict(attack_metadata=result.metadata)])
            self.assertEqual(counts["no_complete_search_candidate_blocks"], 1)
            self.assertEqual(counts["selected_complete_fallback_blocks"], 1)
            self.assertEqual(counts["incomplete_candidates"], 1)
            self.assertEqual(counts["unplanned_fallback_blocks"], 0)
            self.assertFalse(result.attacked)

    def test_unplanned_and_complete_fallback_cannot_be_conflated(self):
        metadata = dict(planned=True, candidate_trace=[], incomplete_candidates=[{}], selected_candidate=None,
                        oracle_unplanned_fallback=True)
        counts = candidate_availability([dict(attack_metadata=metadata)])
        self.assertEqual(counts["unplanned_fallback_blocks"], 1)
        self.assertEqual(counts["selected_complete_fallback_blocks"], 0)
        metadata["candidate_trace"] = [dict(kind="fallback")]
        with self.assertRaisesRegex(ValueError, "complete plan"):
            candidate_availability([dict(attack_metadata=metadata)])

    def test_frozen_grid_changes_only_model_compute_caps(self):
        protocol = json.loads((ROOT / "configs/research/shared_compute_budget_sweep.json").read_text())
        self.assertEqual(protocol["gradient_forward_caps"], [[400, 800], [200, 400], [100, 200]])
        self.assertEqual(len(protocol["groups"]), 7)
        total = 0
        for group in protocol["groups"]:
            self.assertEqual(len(group["configs"]), 5)
            for seed, path in enumerate(group["configs"]):
                actual = json.loads((ROOT / path).read_text())
                self.assertEqual(actual["research_seeds"]["attack_seed"], group["attack_seed"])
                self.assertEqual(actual["run_seeds"], [seed])
                total += actual["episodes"] * len(actual["attacks"])
                if group["gradient_cap"] == 400:
                    names = {"none", "zero_one_budgeted_return", "zero_one_budgeted_safety"}
                else:
                    names = {"none", "zero_one_budgeted_return", "zero_one_budgeted_safety", "ours_progress_return", "ours_progress_safety"}
                self.assertEqual({a["name"] for a in actual["attacks"]}, names)
                for attack in actual["attacks"]:
                    reference_name = ("proposed_progress_development_seed%d.json" if attack["name"].startswith("ours_progress_")
                                      else "proposed_development_seed%d.json") % seed
                    config = json.loads((ROOT / "configs/evaluation" / reference_name).read_text())
                    config["research_seeds"]["attack_seed"] = group["attack_seed"]
                    reference = next(a for a in config["attacks"] if a["name"] == attack["name"])
                    expected = copy.deepcopy(reference)
                    if attack["name"] != "none":
                        expected["parameters"]["resource_limits"].update(gradient_evaluations=group["gradient_cap"],
                                                                         policy_forward_calls=group["forward_cap"])
                    self.assertEqual(attack, expected)
                    self.assertEqual({k: v for k, v in actual.items() if k != "attacks"},
                                     {k: v for k, v in config.items() if k != "attacks"})
        self.assertEqual(total, protocol["additional_episodes"])

    def test_reference_check_rejects_simulation_budget_and_rng_changes(self):
        directory = ROOT / "configs/evaluation"
        low = json.loads((directory / "budget_sweep_g100_attack0_seed0.json").read_text())
        high = json.loads((directory / "proposed_progress_development_seed0.json").read_text())
        manifests = [dict(config=c, source_sha256_before={}, python_runtime={}, pip_freeze={}, sumo_version={}, victim_references=[])
                     for c in (low, high)]
        entries = [dict(attack=next(a for a in c["attacks"] if a["name"] == "ours_progress_return"), run_seed=0,
                        effective_seeds=dict(attack_seed=0), checkpoint_sha256="a", weights_sha256="b") for c in (low, high)]
        args = (manifests[0], entries[0], manifests[1], entries[1], 100)
        validate_budget_reference(*args)
        modified = copy.deepcopy(args)
        modified[1]["attack"]["parameters"]["resource_limits"]["shadow_steps"] = 2000
        with self.assertRaises(ValueError):
            validate_budget_reference(*modified)
        modified = copy.deepcopy(args)
        modified[1]["effective_seeds"]["attack_seed"] = 1
        with self.assertRaisesRegex(ValueError, "victim/seed differs"):
            validate_budget_reference(*modified)
