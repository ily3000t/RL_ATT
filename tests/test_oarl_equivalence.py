"""Check the modified default Agent against the frozen upstream implementation."""

from pathlib import Path
import subprocess
import types
import unittest
import numpy as np
import torch
import oarl


class OARLEquivalenceTests(unittest.TestCase):
    def test_default_seed_preserves_upstream_bo_and_full_update(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            ["git", "show", "upstream-oarl-29e5c0e2497c:oarl.py"], cwd=str(root)
        ).decode("utf-8")
        upstream = types.ModuleType("upstream_equivalence")
        exec(compile(source, "frozen_upstream_oarl.py", "exec"), upstream.__dict__)
        observations = np.random.RandomState(44).uniform(0.05, 0.95, (32, 16))

        def update(agent_type):
            torch.manual_seed(4)
            np.random.seed(4)
            agent = agent_type(16, 1, 3, buffer_size=64)
            for index in range(31):
                agent.replay_buffer.add(observations[index], index % 3,
                                        index / 31.0, observations[index + 1], index % 7 == 0)
            agent.train_model()
            tensors = [p.detach().cpu().clone() for net in
                       (agent.actor, agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target)
                       for p in net.parameters()]
            tensors.append(agent.dual_cst.detach().cpu().clone())
            return tensors, torch.get_rng_state().clone(), np.random.get_state()

        expected, expected_torch, expected_numpy = update(upstream.Agent)
        actual, actual_torch, actual_numpy = update(oarl.Agent)
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(expected_torch, actual_torch))
        self.assertTrue(np.array_equal(expected_numpy[1], actual_numpy[1]))
        self.assertEqual(expected_numpy[2:], actual_numpy[2:])
