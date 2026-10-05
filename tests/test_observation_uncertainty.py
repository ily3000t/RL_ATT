import unittest

import numpy as np
import torch

from rl_att.utils.observation_uncertainty import inverse_observation_box, sample_inverse_observations


class ObservationUncertaintyTests(unittest.TestCase):
    def test_signed_zero_and_cross_zero_bounds(self):
        lower, upper = inverse_observation_box([1., -1., 0., .025, -.025])
        np.testing.assert_allclose(lower, [.95 / 1.2, -1.05 / .8, -.0625, -.025 / .8, -.075 / .8])
        np.testing.assert_allclose(upper, [1.05 / .8, -.95 / 1.2, .0625, .075 / .8, .025 / .8])
        wrong_lower, wrong_upper = .75, 1.25
        self.assertNotEqual(lower[0], wrong_lower)
        self.assertNotEqual(upper[0], wrong_upper)

    def test_identity_and_additive_only_limits(self):
        observed = np.array([-1., 0., 1.])
        lower, upper = inverse_observation_box(observed, 0, 0)
        np.testing.assert_array_equal(lower, observed)
        np.testing.assert_array_equal(upper, observed)
        lower, upper = inverse_observation_box(observed, 0, .05)
        np.testing.assert_allclose(lower, observed - .05)
        np.testing.assert_allclose(upper, observed + .05)

    def test_all_legal_forward_perturbations_include_original(self):
        rng = np.random.RandomState(91)
        original = rng.uniform(-10, 10, size=(2000, 16))
        scales = .2 * abs(original) + .05
        for multiplier in (-np.ones_like(original), np.ones_like(original), rng.uniform(-1, 1, original.shape)):
            observed = original + scales * multiplier
            lower, upper = inverse_observation_box(observed)
            self.assertTrue(np.all(original >= lower - 1e-12))
            self.assertTrue(np.all(original <= upper + 1e-12))

    def test_inverse_candidates_obey_forward_constraint_and_edges_are_tight(self):
        observed = np.random.RandomState(3).uniform(-5, 5, (8, 16))
        candidates = sample_inverse_observations(observed, 100, np.random.RandomState(4))
        self.assertEqual(candidates.shape, (8, 100, 16))
        self.assertTrue(np.all(abs(observed[:, None, :] - candidates) <= .2 * abs(candidates) + .05 + 1e-12))
        lower, upper = inverse_observation_box(observed)
        for endpoint in (lower, upper):
            np.testing.assert_allclose(abs(observed - endpoint), .2 * abs(endpoint) + .05, atol=1e-12)

    def test_explicit_rng_is_repeatable_and_does_not_consume_global_stream(self):
        before = np.random.get_state()
        a = sample_inverse_observations(np.zeros(16), 10, np.random.RandomState(5))
        after = np.random.get_state()
        b = sample_inverse_observations(np.zeros(16), 10, np.random.RandomState(5))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (10, 16))
        self.assertGreater(np.count_nonzero(a), 0)
        self.assertTrue((a < 0).any() and (a > 0).any())
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_inputs_and_unsupported_relative_scale_are_rejected(self):
        for rho, beta in ((1, .05), (-.1, .05), (.2, -.1), (np.nan, .05), (.2, np.inf)):
            with self.assertRaises(ValueError):
                inverse_observation_box([0.], rho, beta)
        for observed in ([np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                inverse_observation_box(observed)
        for count, rng in ((0, np.random.RandomState(0)), (True, np.random.RandomState(0)), (2, None)):
            with self.assertRaises(ValueError):
                sample_inverse_observations([0.], count, rng)

    def test_action_independent_detached_penalty_has_no_actor_signal(self):
        logits = torch.tensor([.3, -.2, .8], requires_grad=True)
        reward_q = torch.tensor([.1, .7, .5])
        pi = torch.softmax(logits, dim=0)
        baseline = -(pi * reward_q).sum()
        base_grad = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        scalar_error = torch.tensor(2.).detach()
        wrong_loss = baseline + .7 * (pi * scalar_error).sum()
        wrong_grad = torch.autograd.grad(wrong_loss, logits, retain_graph=True)[0]
        self.assertTrue(torch.allclose(base_grad, wrong_grad, atol=1e-6))
        action_error = torch.tensor([0., 1., 3.]).detach()
        conditioned_grad = torch.autograd.grad(baseline + .7 * (pi * action_error).sum(), logits)[0]
        self.assertGreater(float(torch.norm(conditioned_grad - base_grad)), .1)


if __name__ == "__main__":
    unittest.main()
