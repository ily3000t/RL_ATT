import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_attack_final import audit_raw, audit_witnesses, audit_oracle, verify_prefix
from prepare_attack_final_protocol import final_config
from rl_att.attacks.no_attack import NoAttack
from rl_att.evaluation.evaluator import AttackEvaluator
from rl_att.attacks.behavior_search import witness_seed


class Env:
    def reset(self):
        return np.zeros(16)

    def step(self, action):
        return np.ones(16), 1., True


class Victim:
    def assert_frozen(self):
        pass

    def action(self, observation):
        return 0


class Metrics:
    def sample(self):
        return dict(ego_collision_observed=False, ego_present=True, pairs=[])


class FinalRawTests(unittest.TestCase):
    def setUp(self):
        self.config = final_config(0, None, 0)
        self.temp = TemporaryDirectory()
        directory = Path(self.temp.name) / "mock"
        seeds = dict(run_seed=0, attack_seed=0, episode_sumo_seeds=self.config["research_seeds"]["episode_sumo_seeds"])
        self.episodes, _ = AttackEvaluator(Env(), Victim(), NoAttack(), Metrics(), self.config, seeds,
                                          dict(norm="none")).run(directory)
        self.steps = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        self.spec = self.config["attacks"][0]

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_fifty_episode_raw_loop(self):
        count, result = audit_raw(self.steps, self.episodes, self.config, self.spec)
        self.assertEqual(count, 50)
        self.assertEqual(result["episode_return_mean"], 1.)

    def test_corrupt_counts_norms_evaluator_cost_and_traffic_rejected(self):
        for change in ("missing", "duplicate", "norm", "cost", "seed"):
            steps, episodes = copy.deepcopy(self.steps), copy.deepcopy(self.episodes)
            if change == "missing":
                steps.pop()
            elif change == "duplicate":
                steps.append(steps[0])
            elif change == "norm":
                episodes[0]["l2_max"] = 1.
            elif change == "cost":
                episodes[0]["research_audit"]["evaluator_policy_forward_calls"] = 1
            else:
                episodes[0]["sumo_seed"] = 123
            with self.assertRaises(ValueError):
                audit_raw(steps, episodes, self.config, self.spec)

    def test_safety_witness_and_unintended_retries(self):
        spec = next(a for a in final_config(0, 100, 0)["attacks"] if a["name"] == "zero_one_budgeted_safety")
        attempt = dict(attempt=0, seed=witness_seed(0, 0, [], 1, 0), history=[], target=1, actual=2, margin=-1., succeeded=False)
        step = dict(episode=1, attack_cost=dict(gradient_evaluations=2),
                    attack_metadata=dict(name=spec["name"], objective="safety", gate_enabled=False, planned=True,
                                         search_kind="target_sequence", inner_attempt_trace=[attempt], retry_attempts=0, failed_target_attempts=1))
        self.assertEqual(len(audit_witnesses([step], spec["name"], 0, spec["parameters"])), 1)
        attempt["attempt"] = 1
        with self.assertRaises(ValueError):
            audit_witnesses([step], spec["name"], 0, spec["parameters"])

    def test_unpaired_transition_before_first_action_divergence_rejected(self):
        changed = copy.deepcopy(self.steps)
        changed[0]["next_observation"][0] = 9.
        with self.assertRaises(ValueError):
            verify_prefix(changed, self.steps)

    def test_oracle_setup_costs_included_and_ledger_drift_rejected(self):
        from unittest.mock import Mock
        oracle = dict(status="passed", returncode=0, account_reset_warmup=True,
                      source_sha256_before={"Data/StraightRoad.sumocfg": "private"},
                      counts=dict(live_verified_steps=1, shadow_steps=4, replay_steps=0, warmup_steps=2,
                                  shadow_resets=1, cache_hits=0))
        raw = [dict(attack_cost=dict(new_shadow_transitions=2, shadow_steps=2))]
        episodes = [dict(research_audit=dict(oracle_episode_setup_cost=dict(shadow_steps=2, warmup_steps=2, shadow_resets=1)))]
        reader = Mock()
        reader.json.return_value = oracle
        entry = dict(attack=dict(parameters={}), results_directory="method")
        with patch("analyze_attack_final.audit_steps", return_value=dict(unplanned_fallback_steps=0)), \
             patch("analyze_attack_final.verify_sources"):
            result = audit_oracle(reader, ROOT, entry, raw, episodes, {})
            self.assertEqual(result["shadow_steps"], 4)
            self.assertEqual(result["warmup_steps"], 2)
            oracle["counts"]["shadow_steps"] = 3
            with self.assertRaises(ValueError):
                audit_oracle(reader, ROOT, entry, raw, episodes, {})



if __name__ == "__main__":
    unittest.main()
