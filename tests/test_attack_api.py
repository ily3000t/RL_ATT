import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
import oarl
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.attacks import AttackContext, AttackRegistry, AttackResult
from rl_att.attacks.no_attack import NoAttack
from rl_att.attacks.oarl_bo import OARLBOAttack


class AttackAPITests(unittest.TestCase):
    def test_no_attack_and_frozen_input_gradients(self):
        victim = VictimAdapter(oarl.ActorNet(16, 3, 128))
        obs = np.random.RandomState(3).uniform(size=16)
        before = obs.copy()
        result = NoAttack()(obs, victim, AttackContext(0, 0, 0, 0))
        self.assertTrue(np.array_equal(obs, result.adversarial_observation))
        result.adversarial_observation[0] = -1
        self.assertTrue(np.array_equal(obs, before))
        state = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        victim.probabilities(state, input_grad=True)[0].backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(all(p.grad is None for p in victim.actor.parameters()))
        victim.assert_frozen()
        with torch.no_grad():
            next(victim.actor.parameters()).add_(1)
        with self.assertRaises(ValueError):
            victim.assert_frozen()

    def test_contract_registry_and_frequency(self):
        registry = AttackRegistry.defaults()
        with self.assertRaises(ValueError):
            registry.register("none", NoAttack)
        with self.assertRaises(ValueError):
            registry.create("unknown")
        with self.assertRaises(ValueError):
            AttackResult(np.ones(16), np.zeros(16), False).validate(np.zeros(16))
        with self.assertRaises(ValueError):
            OARLBOAttack(every_n_steps=0)
        result = OARLBOAttack(every_n_steps=2)(np.ones(16), None, AttackContext(0, 0, 0, 1))
        self.assertFalse(result.attacked)
        self.assertEqual(result.attack_cost["objective_evaluations"], 0)

    def test_batch_wrapper_preserves_full_update_and_rng(self):
        observations = np.random.RandomState(44).uniform(0.05, 0.95, (32, 16))

        def update(wrapped):
            torch.manual_seed(4)
            np.random.seed(4)
            agent = oarl.Agent(16, 1, 3, buffer_size=64, attack_seed=2)
            for index in range(31):
                agent.replay_buffer.add(observations[index], index % 3,
                                        index / 31.0, observations[index + 1], index % 7 == 0)
            if wrapped:
                attack = OARLBOAttack()
                agent.get_optimal_perturb_Bayes = lambda p, q, x, y: attack.search_batch(
                    agent.actor, p, q, x, y, agent.attack_seed)[0]
            agent.train_model()
            values = [p.detach().clone() for net in
                      (agent.actor, agent.qf1, agent.qf2, agent.qf1_target, agent.qf2_target)
                      for p in net.parameters()] + [agent.dual_cst.detach().clone()]
            return values, torch.get_rng_state().clone(), np.random.get_state()

        expected, et, en = update(False)
        actual, at, an = update(True)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(expected, actual)))
        self.assertTrue(torch.equal(et, at))
        self.assertTrue(np.array_equal(en[1], an[1]))
        self.assertEqual(en[2:], an[2:])

    def test_online_bo_bounds_queries_and_rng_isolation(self):
        torch.manual_seed(2)
        victim = VictimAdapter(oarl.ActorNet(16, 3, 128))
        obs = np.random.RandomState(8).uniform(-0.1, 1, 16)
        before = obs.copy()
        rng = torch.get_rng_state().clone()
        attack = OARLBOAttack()
        first = attack(obs, victim, AttackContext(2, 2, 0, 0))
        second = attack(obs, victim, AttackContext(2, 2, 0, 0))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(np.array_equal(obs, before))
        self.assertTrue(np.array_equal(first.adversarial_observation, second.adversarial_observation))
        self.assertEqual(first.metadata["trace"], second.metadata["trace"])
        self.assertEqual(first.attack_cost["objective_evaluations"], 5)
        self.assertEqual(first.attack_cost["policy_forward_calls"], 11)
        self.assertTrue(np.all(np.abs(first.perturbation) <= 0.2 * np.abs(obs) + 0.05 + 1e-7))
        victim.assert_frozen()

    def test_wrapper_duplicate_and_saturated_probabilities(self):
        actor = oarl.ActorNet(16, 3, 128)
        with torch.no_grad():
            actor.pi.weight.zero_()
            actor.pi.bias.copy_(torch.tensor([0., -60., -70.]))
        state = torch.ones(4, 16)
        prob = actor(state, softmax_dim=1)
        with patch.object(oarl.BayesianOptimization, "suggest", return_value={"u1": 0.8, "u2": -0.05}):
            loss, params, bridge = OARLBOAttack().search_batch(actor, prob, prob, state, state, 0)
        self.assertEqual(len(bridge.trace), 5)
        self.assertEqual(bridge.js_float64_evaluations, 5)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in actor.parameters()))
