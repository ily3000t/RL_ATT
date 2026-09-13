import unittest
from unittest.mock import Mock
import numpy as np
import torch
from oarl import Agent
from rl_att.agents.clean_victim import CleanVictimAgent


class CleanVictimTests(unittest.TestCase):
    def test_clean_matches_zero_robust_terms_without_bo_or_dual_updates(self):
        observations = np.random.RandomState(13).uniform(size=(32, 16)).astype(np.float32)

        def update(agent_type):
            torch.manual_seed(2)
            np.random.seed(2)
            agent = agent_type(16, 1, 3, buffer_size=64)
            initial = [p.detach().clone() for net in (agent.actor, agent.qf1, agent.qf2) for p in net.parameters()]
            for i in range(31):
                agent.replay_buffer.add(observations[i], i % 3, 0.3, observations[i + 1], i % 9 == 0)
            dual_before = agent.dual_cst.detach().clone()
            if agent_type is CleanVictimAgent:
                agent.get_optimal_perturb_Bayes = Mock(side_effect=AssertionError("Clean victim called BO"))
                agent.dual_cst_optimizer.step = Mock(side_effect=AssertionError("Clean victim updated dual"))
            else:
                agent.get_optimal_perturb_Bayes = lambda prob, *args: prob.sum() * 0
            agent.train_model()
            parameters = [p.detach().clone() for net in (agent.actor, agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target) for p in net.parameters()]
            if agent_type is CleanVictimAgent:
                self.assertTrue(torch.equal(dual_before, agent.dual_cst))
                self.assertEqual(agent.js_float64_evaluations, 0)
                agent.get_optimal_perturb_Bayes.assert_not_called()
                agent.dual_cst_optimizer.step.assert_not_called()
                self.assertTrue(any(not torch.equal(a, b) for a, b in zip(initial, parameters)))
            return parameters, torch.get_rng_state().clone(), np.random.get_state()

        expected, expected_torch, expected_numpy = update(Agent)
        actual, actual_torch, actual_numpy = update(CleanVictimAgent)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(expected_torch, actual_torch))
        self.assertTrue(np.array_equal(expected_numpy[1], actual_numpy[1]))
        self.assertEqual(expected_numpy[2:], actual_numpy[2:])
