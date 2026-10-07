import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from oarl import ActorNet
from rl_att.defenses.pgd_consistency import DEFAULTS, categorical_kl, search, validate_config
from rl_att.training.session import TrainingSession
from rl_att.training.state import agent_state, global_rng_state, tree_digest
from test_training_support import config, populate


def defense_config(**kwargs):
    value = config("pgd_consistency")
    value["defense"] = dict(DEFAULTS, **kwargs)
    return value


class PGDConsistencyTests(unittest.TestCase):
    def session(self, **kwargs):
        return TrainingSession(defense_config(**kwargs), dict(git_commit="test-pgd"),
                               agent_kwargs=dict(buffer_size=32, batch_size=16))

    def test_zero_coefficient_and_zero_budget_reduce_exactly_to_clean(self):
        clean = TrainingSession(config(), dict(git_commit="test-clean"), agent_kwargs=dict(buffer_size=32, batch_size=16))
        populate(clean.agent)
        clean.update()
        expected = tree_digest(dict(agent=agent_state(clean.agent), rng=global_rng_state()))
        for disabled in (dict(coefficient=0), dict(epsilon=0), dict(multiplicative=0, additive=0)):
            session = self.session(**disabled)
            before_aux = tree_digest(session.auxiliary_streams.state_dict())
            populate(session.agent)
            session.update()
            state = agent_state(session.agent)
            state.pop("defense_training")
            self.assertEqual(expected, tree_digest(dict(agent=state, rng=global_rng_state())))
            self.assertEqual(before_aux, tree_digest(session.auxiliary_streams.state_dict()))
            self.assertEqual(session.resources.phases["primary_update"]["network_forward_calls"]["actor"], 2)

    def test_enabled_update_preserves_critic_targets_and_primary_rng(self):
        clean = TrainingSession(config(), dict(git_commit="test-clean"), agent_kwargs=dict(buffer_size=32, batch_size=16))
        populate(clean.agent)
        clean.update()
        expected, expected_rng = agent_state(clean.agent), tree_digest(global_rng_state())
        session = self.session(coefficient=100.)
        populate(session.agent)
        session.update()
        actual = agent_state(session.agent)
        self.assertEqual(expected_rng, tree_digest(global_rng_state()))
        for name in ("qf1", "qf2", "qf1_target", "qf2_target"):
            self.assertEqual(tree_digest(expected["networks"][name]), tree_digest(actual["networks"][name]))
        for name in ("qf1_optimizer", "qf2_optimizer", "dual_cst_optimizer"):
            self.assertEqual(tree_digest(expected["optimizers"][name]), tree_digest(actual["optimizers"][name]))
        self.assertNotEqual(tree_digest(expected["networks"]["actor"]), tree_digest(actual["networks"]["actor"]))
        self.assertEqual(session.resources.counts["bo_objective_evaluations"], 0)
        self.assertEqual(session.resources.counts["attacked_interaction_steps"], 0)
        self.assertEqual(session.resources.phases["primary_update"]["optimizer_steps"],
                         dict(actor_optimizer=1, qf1_optimizer=1, qf2_optimizer=1))

    def test_actual_search_and_training_costs_are_not_batch_calls_confused(self):
        session = self.session()
        populate(session.agent)
        session.update()
        counts = session.agent.consistency_counts
        self.assertEqual(counts["gradient_evaluations"], 5)
        self.assertEqual(counts["search_actor_forward_calls"], 6)
        self.assertEqual(counts["candidate_observations"], 96)
        self.assertEqual(counts["regularizer_observation_rows"], 16)
        phase = session.resources.phases["primary_update"]
        self.assertEqual(phase["network_forward_calls"]["actor"], 9)
        self.assertEqual(phase["network_observation_rows"]["actor"], 144)
        self.assertAlmostEqual(session.agent.last_consistency["total_actor_loss"],
                               session.agent.last_consistency["base_actor_loss"] + .1 * session.agent.last_consistency["consistency_loss"], places=6)

    def test_search_is_bounded_and_best_candidate_does_not_regress(self):
        torch.manual_seed(67)
        actor = ActorNet(16, 3, 128)
        observation = torch.tensor(np.random.RandomState(71).normal(size=(8, 16)), dtype=torch.float32)
        observation[0].zero_()
        observation[1, 0] = -9.
        probability = actor(observation, softmax_dim=-1)
        auxiliary = np.random.RandomState(113)
        before = tree_digest(global_rng_state())
        adversarial, measured = search(actor, observation, probability, auxiliary, DEFAULTS)
        self.assertFalse(adversarial.requires_grad)
        self.assertTrue(bool(((adversarial - observation).abs() <= .2 * observation.abs() + .05 + 1e-6).all()))
        self.assertGreaterEqual(measured["selected_kl_mean"] + 1e-8, measured["initial_kl_mean"])
        self.assertGreater(measured["selected_perturbed_rows"], 0)
        self.assertEqual(before, tree_digest(global_rng_state()))
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))

    def test_constant_policy_preserves_zero_candidate_on_ties(self):
        actor = ActorNet(16, 3, 128)
        for parameter in actor.parameters():
            parameter.data.zero_()
        observation = torch.ones(4, 16)
        adversarial, measured = search(actor, observation, actor(observation, softmax_dim=-1), np.random.RandomState(4), DEFAULTS)
        self.assertTrue(torch.equal(observation, adversarial))
        self.assertEqual(measured["selected_kl_mean"], 0)

    def test_outer_kl_has_declared_direction_and_detached_clean_target(self):
        p = torch.tensor([[.7, .2, .1]], requires_grad=True)
        q = torch.tensor([[.2, .3, .5]], requires_grad=True)
        loss = categorical_kl(p.detach(), q, 1e-8).mean()
        self.assertAlmostEqual(float(loss.item()), float((p.detach() * (p.detach().log() - q.detach().log())).sum().item()), places=6)
        loss.backward()
        self.assertIsNone(p.grad)
        self.assertTrue(bool(torch.isfinite(q.grad).all()))

    def test_resume_restores_nonempty_optimizer_search_rng_and_counters(self):
        session = self.session()
        populate(session.agent)
        session.update()
        session.loop.update(completed_episodes=13, interactions=41)
        session.resources.counts["real_interaction_steps"] = 41
        tests_root = Path(__file__).resolve().parents[1] / ".local/tests"
        tests_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(tests_root)) as directory:
            path = Path(directory) / "pgd.pt"
            digest = session.save(path, dict(reset_times=13))
            session.update()
            action = session.action(np.ones(16))
            expected = tree_digest(dict(agent=agent_state(session.agent), rng=global_rng_state(),
                                  auxiliary=session.auxiliary_streams.state_dict(), policy=session.policy_stream.state))
            other = self.session()
            other.restore(path, digest, SimpleNamespace())
            other.update()
            self.assertEqual(action, other.action(np.ones(16)))
            self.assertEqual(expected, tree_digest(dict(agent=agent_state(other.agent), rng=global_rng_state(),
                             auxiliary=other.auxiliary_streams.state_dict(), policy=other.policy_stream.state)))

    def test_invalid_configuration_and_nonfinite_input_are_rejected(self):
        for wrong in (dict(coefficient=-1), dict(steps=True), dict(step_size=0), dict(epsilon=float('nan')),
                      dict(probability_floor=1), dict(random_start=False)):
            with self.assertRaises(ValueError):
                validate_config(dict(DEFAULTS, **wrong))
        actor = ActorNet(16, 3, 128)
        with self.assertRaisesRegex(ValueError, "Finite"):
            search(actor, torch.full((2, 16), float('nan')), torch.ones(2, 3) / 3, np.random.RandomState(4), DEFAULTS)


if __name__ == "__main__":
    unittest.main()
