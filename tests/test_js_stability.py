import types
import subprocess
from pathlib import Path
import unittest
import numpy as np
import torch
import oarl


class JSStabilityTests(unittest.TestCase):
    def test_identical_tiny_distributions_have_finite_zero_gradient(self):
        p = torch.tensor([[1e-25, 1.0]], requires_grad=True)
        q = torch.tensor([[1e-25, 1.0]], requires_grad=True)
        value = oarl.Agent._js_divergence_float64(p, q).mean()
        value.backward()
        self.assertEqual(value.item(), 0.0)
        self.assertTrue(torch.equal(p.grad, torch.zeros_like(p)))
        self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))

    def test_zero_and_subnormal_probabilities_keep_finite_softmax_gradients(self):
        logits = torch.tensor([[0.0, -110.0, -90.0]], requires_grad=True)
        other_logits = torch.tensor([[0.0, -90.0, -110.0]], requires_grad=True)
        p, q = torch.softmax(logits, 1), torch.softmax(other_logits, 1)
        value = oarl.Agent._js_divergence_float64(p, q).mean()
        value.backward()
        self.assertTrue(bool(torch.isfinite(value)))
        self.assertGreaterEqual(value.item(), 0)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(other_logits.grad).all()))

    def test_shared_zero_probability_has_finite_zero_gradient(self):
        logits = torch.tensor([[0.0, -110.0, -120.0]], requires_grad=True)
        p = torch.softmax(logits, 1)
        value = oarl.Agent._js_divergence_float64(p, p).mean()
        value.backward()
        self.assertEqual(value.item(), 0)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_full_saturated_actor_update_recovers_from_old_nan(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(["git", "show", "9d0086c:oarl.py"], cwd=str(root)).decode("utf-8")
        before = types.ModuleType("before_js_stability")
        exec(compile(source, "before_js_stability.py", "exec"), before.__dict__)

        def train(agent_type):
            torch.manual_seed(2)
            np.random.seed(2)
            agent = agent_type(16, 1, 3, buffer_size=32)
            agent.actor.pi.weight.data.zero_()
            agent.actor.pi.bias.data.copy_(torch.tensor([0., -60., -70.]))
            for index in range(16):
                obs = np.full(16, index / 16, dtype=np.float32)
                agent.replay_buffer.add(obs, index % 3, 0.5, obs, False)
            agent.train_model()
            return agent

        old = train(before.Agent)
        current = train(oarl.Agent)
        self.assertTrue(any(not bool(torch.isfinite(p).all()) for p in old.actor.parameters()))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in current.actor.parameters()))
        self.assertEqual(current.js_float64_evaluations, 5)
