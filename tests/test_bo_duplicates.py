import unittest
from unittest.mock import patch
import numpy as np
import torch
import oarl
from bayes_opt import BayesianOptimization


class DuplicateProposalTests(unittest.TestCase):
    def test_duplicate_proposals_keep_all_evaluations_and_gradients(self):
        torch.manual_seed(0)
        agent = oarl.Agent(16, 1, 3, buffer_size=32)
        observations = torch.from_numpy(np.random.RandomState(1).uniform(size=(8, 16)).astype(np.float32))
        _, prob = agent.select_action_batch(observations)
        _, prob_next = agent.select_action_batch(observations + 0.02)
        created = []

        def optimizer(*args, **kwargs):
            instance = BayesianOptimization(*args, **kwargs)
            instance.suggest = lambda utility: {"u1": 0.8, "u2": -0.05}
            created.append(instance)
            return instance

        with patch.object(oarl, "BayesianOptimization", side_effect=optimizer), patch.object(agent, "js_d_loss", wraps=agent.js_d_loss) as loss:
            result = agent.get_optimal_perturb_Bayes(prob, prob_next, observations, observations + 0.02)
        self.assertEqual(loss.call_count, 5)
        self.assertEqual(len(created[0].space), 1)
        self.assertTrue(result.requires_grad)
        result.backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in agent.actor.parameters()))
        self.assertTrue(any(bool(p.grad.abs().sum() > 0) for p in agent.actor.parameters()))
