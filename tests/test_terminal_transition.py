"""Regression coverage for the original environment's unreachable terminal branch.

No simulator is started and no SUMO config is rewritten by these tests.
"""

import unittest
from unittest.mock import Mock, patch

from Environment.environment.envs import highway_env


class TerminalTransitionTests(unittest.TestCase):
    def make_environment(self):
        env = highway_env.HighwayEnv()
        env.state = [0.25] * 16
        env.VehicleIds = ["Auto"]
        env._findstate = Mock(return_value=[0.0] * 16)
        return env

    def test_removed_ego_returns_last_valid_state_and_existing_terminal_reward(self):
        env = self.make_environment()
        last_valid_state = list(env.state)
        env.TotalReward = 3.5
        env.obs_to_state = Mock(side_effect=AssertionError("Removed ego must not be queried"))
        with patch.object(highway_env, "traci") as traci:
            traci.vehicle.getAngle.return_value = 90.0
            traci.vehicle.getIDList.return_value = []
            result = env.step(2)
            traci.simulationStep.assert_called_once_with()
            traci.vehicle.subscribe.assert_not_called()
        self.assertEqual(result, (last_valid_state, 0, True, 0, 0, 0, {}))
        self.assertEqual(env.TotalReward, 3.5)
        env._findstate.assert_called_once_with()
        env.obs_to_state.assert_not_called()

    def test_surviving_ego_keeps_observation_reward_and_update_order(self):
        env = self.make_environment()
        next_state = [0.5] * 16
        events = []
        env._findstate.side_effect = lambda: events.append("pre_observation") or [0.0] * 16
        env.obs_to_state = Mock(side_effect=lambda: events.append("next_observation") or next_state)
        env.get_reward_v = Mock(side_effect=lambda action: events.append("reward") or 0.75)
        with patch.object(highway_env, "traci") as traci:
            traci.vehicle.getAngle.return_value = 90.0
            traci.vehicle.getIDList.return_value = ["Auto"]
            traci.vehicle.getSpeedMode.return_value = 31
            traci.simulationStep.side_effect = lambda: events.append("simulation_step")
            traci.vehicle.getAllSubscriptionResults.return_value = {
                "Auto": {highway_env.tc.VAR_SPEED: 12.0, highway_env.tc.VAR_DISTANCE: 123.0}
            }
            result = env.step(2)
            traci.vehicle.changeLane.assert_not_called()
        self.assertEqual(events, ["pre_observation", "simulation_step", "next_observation", "reward"])
        self.assertEqual(result, (next_state, 0.75, False, 123.0, 0, 0, {}))
        self.assertEqual(env.TotalReward, 0.75)
        env.get_reward_v.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
