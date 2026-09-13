import unittest
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from unittest.mock import Mock, patch
import numpy as np
import torch

from rl_att.utils.seeding import seed_manifest, TorchPolicyStream
from Environment.environment.envs import highway_env


class SeedProtocolTests(unittest.TestCase):
    def test_legacy_keeps_paired_sumo_and_fixed_attack_seed(self):
        seeds = seed_manifest(4, "legacy", 6)
        self.assertEqual(seeds["episode_sumo_seeds"], [0, 0, 2, 2, 4, 4])
        self.assertEqual(seeds["attack_seed"], 0)
        self.assertEqual(seeds["torch_seed"], 4)
        self.assertEqual(seeds["policy_rng"], "shared_torch")

    def test_controlled_schedule_is_repeatable_disjoint_and_rng_neutral(self):
        np.random.seed(91)
        state = np.random.get_state()
        first = seed_manifest(2, "controlled", 400)
        second = seed_manifest(2, "controlled", 400)
        other_run = seed_manifest(3, "controlled", 400)
        evaluation = seed_manifest(2, "controlled", 400, phase="evaluation")
        self.assertEqual(first, second)
        self.assertEqual(len(set(first["episode_sumo_seeds"])), 400)
        self.assertFalse(set(first["episode_sumo_seeds"]) & set(other_run["episode_sumo_seeds"]))
        self.assertFalse(set(first["episode_sumo_seeds"]) & set(evaluation["episode_sumo_seeds"]))
        self.assertTrue(np.array_equal(state[1], np.random.get_state()[1]))
        self.assertEqual(first["attack_seed"], 2)

    def test_policy_sampling_does_not_advance_training_rng(self):
        torch.manual_seed(9)
        state = torch.get_rng_state().clone()
        stream = TorchPolicyStream(2)
        with stream.activate():
            first = torch.rand(8)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        torch.rand(100)
        with stream.activate():
            second = torch.rand(8)
        replay = TorchPolicyStream(2)
        with replay.activate():
            self.assertTrue(torch.equal(first, torch.rand(8)))
        with replay.activate():
            self.assertTrue(torch.equal(second, torch.rand(8)))

    def test_policy_rng_restored_after_exception(self):
        state = torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with TorchPolicyStream(0).activate():
                torch.rand(1)
                raise RuntimeError("test")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_environment_reset_uses_explicit_schedule_or_legacy_pairing(self):
        for schedule, expected in (([17, 31, 42], [17, 31, 42]), (None, [0, 0, 2])):
            with tempfile.TemporaryDirectory() as temp:
                config = Path(temp) / "test.sumocfg"
                config.write_text('<configuration><random_numberType><seed value="70"/></random_numberType></configuration>')
                env = highway_env.HighwayEnv(sumo_seed_schedule=schedule)
                env.obs_to_state = Mock(return_value=[0.0] * 16)
                with patch.object(highway_env, "config_path", str(config)), patch.object(highway_env, "traci") as traci:
                    traci.vehicle.getIDList.return_value = ["Auto"]
                    traci.vehicle.getSpeedMode.return_value = 31
                    observed = []
                    for _ in expected:
                        env.reset()
                        observed.append(int(ET.parse(str(config)).find('.//seed').get('value')))
                self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
