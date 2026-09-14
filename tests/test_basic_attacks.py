import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
import oarl
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.attacks.base import AttackContext
from rl_att.attacks.random_noise import RandomNoiseAttack
from rl_att.attacks.fgsm import FGSMAttack
from rl_att.attacks.pgd import PGDAttack
from rl_att.attacks.box import box_scales
from rl_att.evaluation.configuration import validate_config
from rl_att.evaluation.evaluator import validate_budget


def linear_victim():
    actor = oarl.ActorNet(16, 3, 128)
    with torch.no_grad():
        actor.fc1.weight.zero_()
        actor.fc1.bias.fill_(2)
        actor.fc1.weight[0, 0] = 1
        actor.pi.weight.zero_()
        actor.pi.weight[0, 0] = 2
        actor.pi.weight[1, 0] = -2
        actor.pi.bias.copy_(torch.tensor([-4., 4., -100.]))
    return VictimAdapter(actor)


class BasicAttackTests(unittest.TestCase):
    def setUp(self):
        self.victim = linear_victim()
        self.obs = np.full(16, 0.2)
        self.obs[0] = 0.01
        self.context = AttackContext(2, 2, 3, 4)
        self.budget = {"norm": "observation_scaled_linf", "epsilon": 1.0,
                       "relative_scale": 0.2, "absolute_scale": 0.05}

    def test_logits_capture_matches_probabilities_and_removes_hook(self):
        before = len(self.victim.actor.pi._forward_hooks)
        prob = self.victim.probabilities(self.obs)
        self.assertTrue(torch.equal(prob, torch.softmax(self.victim.logits(self.obs), dim=0)))
        with self.assertRaises(ValueError):
            self.victim.logits(np.zeros(15))
        self.assertEqual(len(self.victim.actor.pi._forward_hooks), before)

    def test_fgsm_has_analytic_sign_and_changes_action_without_training(self):
        result = FGSMAttack()(self.obs, self.victim, self.context)
        expected = self.obs.copy()
        expected[0] -= 0.052
        self.assertTrue(np.allclose(result.adversarial_observation, expected, atol=1e-12))
        self.assertEqual(self.victim.action(self.obs), 0)
        self.assertEqual(self.victim.action(result.adversarial_observation), 1)
        self.assertGreater(result.metadata["final_objective"], result.metadata["initial_objective"])
        self.assertEqual(result.attack_cost["gradient_evaluations"], 1)
        self.assertTrue(all(p.grad is None for p in self.victim.actor.parameters()))
        self.victim.assert_frozen()

    def test_pgd_projects_every_iterate_and_uses_declared_gradient_budget(self):
        points = []
        logits = self.victim.logits

        def capture(obs, input_grad=False):
            points.append(obs.detach().numpy().copy() if torch.is_tensor(obs) else np.asarray(obs).copy())
            return logits(obs, input_grad=input_grad)

        with patch.object(self.victim, "logits", side_effect=capture):
            result = PGDAttack()(self.obs, self.victim, self.context)
        self.assertEqual(len(points), 12)
        for point in points:
            self.assertTrue(np.all(np.abs(point - self.obs) <= 0.2 * np.abs(self.obs) + 0.05 + 1e-7))
        self.assertEqual(result.attack_cost["gradient_evaluations"], 10)
        self.assertEqual(result.attack_cost["policy_forward_calls"], 12)
        self.assertEqual(self.victim.action(result.adversarial_observation), 1)
        self.victim.assert_frozen()

    def test_random_and_pgd_repeat_per_context_without_global_rng_changes(self):
        for attack in (RandomNoiseAttack(), PGDAttack(), FGSMAttack()):
            numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
            first = attack(self.obs, self.victim, self.context)
            attack(self.obs, self.victim, AttackContext(2, 2, 3, 5))
            second = attack(self.obs, self.victim, self.context)
            self.assertTrue(np.array_equal(first.adversarial_observation, second.adversarial_observation))
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            self.assertTrue(np.array_equal(numpy_state[1], np.random.get_state()[1]))
            self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
            validate_budget(first, self.obs, self.budget)
        self.assertFalse(np.array_equal(RandomNoiseAttack()(self.obs, self.victim, self.context).perturbation,
                                       RandomNoiseAttack()(self.obs, self.victim, AttackContext(2, 2, 4, 4)).perturbation))

    def test_zero_budget_and_frequency_preserve_observation_exactly(self):
        for attack_type in (RandomNoiseAttack, FGSMAttack, PGDAttack):
            for attack, context in ((attack_type(epsilon=0), self.context),
                                    (attack_type(every_n_steps=3), self.context)):
                result = attack(self.obs, self.victim, context)
                self.assertFalse(result.attacked)
                self.assertTrue(np.array_equal(self.obs, result.adversarial_observation))
                self.assertEqual(result.attack_cost["objective_evaluations"], 0)

    def test_common_budget_checks_large_negative_and_zero_features(self):
        obs = np.linspace(-5, 5, 16)
        obs[0] = 0
        result = RandomNoiseAttack()(obs, self.victim, self.context)
        validate_budget(result, obs, self.budget)
        result.adversarial_observation[0] = 0.06
        result.perturbation[0] = 0.06
        with self.assertRaises(ValueError):
            validate_budget(result, obs, self.budget)

    def test_margin_gradient_stays_informative_for_saturated_softmax(self):
        with torch.no_grad():
            self.victim.actor.pi.bias[0] = 1000
        # A new adapter freezes the deliberately constructed saturated test model.
        victim = VictimAdapter(self.victim.actor)
        self.assertEqual(float(victim.probabilities(self.obs)[0]), 1.)
        result = FGSMAttack()(self.obs, victim, self.context)
        self.assertFalse(result.metadata["zero_gradient"])
        self.assertTrue(np.isfinite(result.perturbation).all())
