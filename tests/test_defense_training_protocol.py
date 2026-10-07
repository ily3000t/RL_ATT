import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from rl_att.training.production_audit import audit_episodes
from rl_att.training.protocol import validate_config
from rl_att.training.session import training_seeds
from rl_att.utils.victim_registry import registry_references
from rl_att.evaluation.configuration import validate_config as validate_evaluation


ROOT = Path(__file__).resolve().parents[1]


def short_config():
    return json.loads((ROOT / "configs/experiments/pgd_consistency_training_smoke.json").read_text(encoding="utf-8"))


def raw_example():
    config = short_config()
    seeds = training_seeds(config)
    episodes, transitions, updates, interaction = [], [], 0, 0
    for index in range(16):
        digest, reward_sum = hashlib.sha256(), 0.
        for step in range(2):
            observation = np.full(16, index + step / 10., dtype=np.float64)
            following = np.full(16, index + (step + 1) / 10., dtype=np.float64)
            done, reward = step == 1, -.25 + step
            interaction += 1
            updates += int(index > 10 and interaction % 2 == 0)
            transitions.append(dict(episode=index + 1, step=step + 1, observation=observation.tolist(), next_observation=following.tolist(),
                action=2, reward=reward, done=done, ego_collision_ids=[], training_updates_total=updates))
            digest.update(following.tobytes())
            digest.update(np.asarray([2, reward, int(done)], dtype=np.float64).tobytes())
            reward_sum += reward
        episodes.append(dict(episode=index + 1, steps=2, episode_return=reward_sum, sumo_seed=seeds["episode_sumo_seeds"][index],
            trajectory_sha256=digest.hexdigest(), terminated=True, truncated=False, ego_collision_observed=False,
            sumo_collision_vehicle_ids=[], warmup_steps=81, training_updates_total=updates))
    return episodes, transitions, config, seeds


def registry(engineering=False):
    return dict(kind="defense_victim_registry", verified=True, engineering_only=engineering, benchmark_eligible=not engineering,
                checkpoint_type="inference_actor_only", checkpoint_selection="episode_400_predeclared", victims=[
                    dict(victim="pgd_consistency", run_seed=seed, training_episode=16 if engineering else 400,
                         engineering_only=engineering, benchmark_eligible=not engineering) for seed in range(5)])


class DefenseTrainingProtocolTests(unittest.TestCase):
    def test_five_full_configs_match_original_training_and_each_other(self):
        for seed in range(5):
            path = ROOT / ("configs/experiments/pgd_consistency_protocol_a_seed%d.json" % seed)
            config = validate_config(json.loads(path.read_text(encoding="utf-8")))
            clean = json.loads((ROOT / ("configs/experiments/clean_protocol_a_seed%d.json" % seed)).read_text(encoding="utf-8"))
            self.assertEqual(config["run_seed"], seed)
            for key, value in clean["training"].items():
                if key != "algo":
                    self.assertEqual(config["training"][key], value)
            self.assertEqual(config["defense"], short_config()["defense"])

    def test_full_checkpoint_selection_cannot_silently_change(self):
        config = json.loads((ROOT / "configs/experiments/pgd_consistency_protocol_a_seed0.json").read_text(encoding="utf-8"))
        for change in (dict(checkpoint_episodes=[100, 400]), dict(checkpoint_selection="best_return")):
            with self.assertRaises(ValueError):
                validate_config(dict(config, **change))

    def test_invalid_seed_gate_or_unbounded_engineering_is_rejected(self):
        for change in (dict(run_seed=5), dict(gate_enabled=True), dict(cpu_threads=2)):
            with self.assertRaises(ValueError):
                validate_config(dict(short_config(), **change))
        config = short_config()
        config["training"]["max_step"] = 200
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_streamed_raw_episodes_recompute_return_digest_and_update_cadence(self):
        result = audit_episodes(*raw_example())
        self.assertEqual(result, dict(interactions=32, updates=5, warmup_steps=1296))

    def test_missing_transition_or_wrong_update_cadence_is_rejected(self):
        episodes, transitions, config, seeds = raw_example()
        with self.assertRaises(ValueError):
            audit_episodes(episodes, transitions[:-1], config, seeds)
        transitions[-1]["training_updates_total"] += 1
        with self.assertRaisesRegex(ValueError, "cadence"):
            audit_episodes(episodes, transitions, config, seeds)

    def test_reward_and_observation_chain_cannot_hide_in_summary(self):
        episodes, transitions, config, seeds = raw_example()
        transitions[0]["reward"] += .1
        with self.assertRaisesRegex(ValueError, "summary"):
            audit_episodes(episodes, transitions, config, seeds)
        episodes, transitions, config, seeds = raw_example()
        transitions[1]["observation"][0] += .1
        with self.assertRaisesRegex(ValueError, "consecutive"):
            audit_episodes(episodes, transitions, config, seeds)

    def test_engineering_checkpoint_cannot_become_benchmark_by_default(self):
        with self.assertRaisesRegex(ValueError, "Engineering"):
            registry_references(registry(True))
        self.assertEqual(len(registry_references(registry(True), allow_engineering=True)), 5)

    def test_full_registry_requires_all_seeds_and_matching_eligibility(self):
        self.assertEqual(len(registry_references(registry())), 5)
        value = registry()
        value["victims"].pop()
        with self.assertRaisesRegex(ValueError, "five-seed"):
            registry_references(value)
        value = registry()
        value["victims"][2]["benchmark_eligible"] = False
        with self.assertRaisesRegex(ValueError, "eligibility"):
            registry_references(value)

    def test_new_policy_has_no_upstream_no_attack_reference(self):
        config = json.loads((ROOT / "configs/evaluation/pgd_consistency_integration_smoke.json").read_text(encoding="utf-8"))
        validate_evaluation(config)
        config["verify_legacy_no_attack"] = True
        with self.assertRaisesRegex(ValueError, "upstream"):
            validate_evaluation(config)


if __name__ == "__main__":
    unittest.main()
