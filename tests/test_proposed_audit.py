import copy
from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.proposed import ProposedAttack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_proposed_smoke import audit_steps
import summarize_proposed_smoke
sys.path.pop(0)


class ProposedAuditTests(unittest.TestCase):
    def setUp(self):
        victim = linear_victim()
        obs = np.full(16, .2)
        obs[0] = .01
        oracle = BudgetOracle(obs)
        attack = ProposedAttack(horizon=2)
        self.parameters = dict(evaluations=10, objective="return", horizon=2, epsilon=1.,
                               resource_limits=attack.resource_limits)
        self.rows = []
        for step in range(2):
            result = attack(obs, victim, AttackContext(0, 0, 0, step, oracle, 2 - step))
            action = victim.action(result.adversarial_observation)
            transition = oracle.step_fn(action)
            oracle.core.observe(action, transition)
            self.rows.append(dict(episode=1, step=step, attack_metadata=result.metadata,
                                  attack_cost=result.attack_cost, action=action, reward=transition["reward"],
                                  safety=dict(ego_collision_observed=False), terminated=False,
                                  perturbation=result.perturbation.tolist(), attacked=result.attacked,
                                  scaled_linf=result.metadata["scaled_linf"], clean_action_at_visited_state=0))

    def test_valid_trace_and_tampered_budget_rejected(self):
        result = audit_steps(self.rows, self.parameters)
        self.assertEqual(result["blocks"], 1)
        self.rows[1]["attack_metadata"]["budget"]["used"]["policy_forward_calls"] += 1
        with self.assertRaisesRegex(ValueError, "Ledger"):
            audit_steps(self.rows, self.parameters)

    def test_execution_mismatch_and_incomplete_candidate_rejected(self):
        corrupted = copy.deepcopy(self.rows)
        corrupted[1]["reward"] += 1
        with self.assertRaisesRegex(ValueError, "execution"):
            audit_steps(corrupted, self.parameters)
        trace = self.rows[0]["attack_metadata"]["candidate_trace"][0]
        for key in ("actions", "rewards", "collisions"):
            trace[key].pop()
        with self.assertRaisesRegex(ValueError, "Partial"):
            audit_steps(self.rows, self.parameters)

    def test_repeat_ignores_only_timer_and_rejects_reward_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batches = []
            manifest = dict(status="passed", config={}, victim_references=[], python_runtime={}, pip_freeze={}, sumo_version={})
            for label, timer in (("original", 1.), ("repeat", 2.)):
                directory = root / label
                result = directory / "clean-seed0-none"
                result.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps(manifest))
                (result / "steps.jsonl").write_text(json.dumps(dict(reward=3., attack_cost=dict(wall_seconds=timer))) + "\n")
                batch = root / (label + ".json")
                batch.write_text(json.dumps(dict(status="passed", git_commit=label,
                                                  runs=[dict(config="same.json", run_dir=str(directory))])))
                batches.append(batch)
            with patch.object(summarize_proposed_smoke, "ROOT", root):
                report = summarize_proposed_smoke.verify_repeat(*batches)
                self.assertEqual(report["verified_steps"], 1)
                (root / "repeat/clean-seed0-none/steps.jsonl").write_text(
                    json.dumps(dict(reward=4., attack_cost=dict(wall_seconds=2.))) + "\n")
                with self.assertRaisesRegex(ValueError, "Repeat step differs"):
                    summarize_proposed_smoke.verify_repeat(*batches)
