import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from oarl import Agent
from rl_att.agents.clean_victim import CleanVictimAgent
from rl_att.training.session import TrainingSession, initialize_global_rng, training_seeds
from rl_att.training.state import (agent_state, global_rng_state, load_snapshot, optimizer_state, replay_state,
                                   restore_agent, tree_digest)
from rl_att.training.streams import AuxiliaryStreams


def config(victim="clean"):
    return dict(victim=victim, protocol="controlled", run_seed=0, sumo_schedule="derived", gate_enabled=False,
                training=dict(state_dim=16, action_dim=1, action_numb=3, episodes=16, max_step=16))


def populate(agent, count=41):
    observations = np.random.RandomState(71).normal(size=(count + 1, 16)).astype(np.float32)
    for i in range(count):
        agent.replay_buffer.add(observations[i], i % 3, float(i) / 30., observations[i + 1], i % 8 == 0)


class TrainingSupportTests(unittest.TestCase):
    def setUp(self):
        self.temporary_root = Path(__file__).resolve().parents[1] / ".local/tests"
        self.temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(self.temporary_root))
        self.path = Path(self.temporary.name) / "training.pt"
        self.identity = dict(git_commit="test", config_sha256="test", source="test")

    def tearDown(self):
        self.temporary.cleanup()

    def session(self, victim="clean"):
        return TrainingSession(config(victim), self.identity, agent_kwargs=dict(buffer_size=32, batch_size=16))

    def test_clean_and_oarl_fixed_batch_updates_match_unwrapped_algorithms(self):
        for victim, cls in (("clean", CleanVictimAgent), ("oarl", Agent)):
            initialize_global_rng(training_seeds(config(victim)))
            original = cls(16, 1, 3, buffer_size=32, batch_size=16, attack_seed=0)
            original.train()
            populate(original)
            original.train_model()
            expected = tree_digest(dict(agent=agent_state(original), rng=global_rng_state()))
            session = self.session(victim)
            populate(session.agent)
            session.update()
            self.assertEqual(expected, tree_digest(dict(agent=agent_state(session.agent), rng=global_rng_state())))
            self.assertEqual(session.resources.counts["primary_updates"], 1)
            self.assertEqual(session.resources.counts["bo_objective_evaluations"], 0 if victim == "clean" else 5)
            phase = session.resources.phases["primary_update"]
            self.assertEqual(phase["network_forward_calls"]["actor"], 2 if victim == "clean" else 12)
            self.assertEqual(phase["network_observation_rows"]["actor"], 32 if victim == "clean" else 192)
            self.assertEqual(sum(phase["optimizer_steps"].values()), 3 if victim == "clean" else 4)

    def test_resume_preserves_nonempty_adam_wrapped_fifo_and_all_rng(self):
        for victim in ("clean", "oarl"):
            session = self.session(victim)
            populate(session.agent)
            session.update()
            session.action(np.ones(16))
            session.loop.update(completed_episodes=13, interactions=41)
            session.resources.counts["real_interaction_steps"] = 41
            session.auxiliary_streams.numpy["belief"].normal(size=20)
            with session.auxiliary_streams.torch["sampling"].activate():
                torch.rand(9)
            digest = session.save(self.path, dict(reset_times=13))
            checkpoint_state = tree_digest(dict(agent=agent_state(session.agent), rng=global_rng_state(),
                               policy=session.policy_stream.state, auxiliary=session.auxiliary_streams.state_dict()))
            session.update()
            action = session.action(np.zeros(16))
            expected = tree_digest(dict(agent=agent_state(session.agent), rng=global_rng_state(),
                                  policy=session.policy_stream.state, auxiliary=session.auxiliary_streams.state_dict()))
            resumed = self.session(victim)
            env = SimpleNamespace(reset_times=0, extra="constructor-only")
            resumed.restore(self.path, digest, env)
            self.assertFalse(hasattr(env, "extra"))
            self.assertEqual(env.reset_times, 13)
            self.assertEqual(checkpoint_state, tree_digest(dict(agent=agent_state(resumed.agent), rng=global_rng_state(),
                             policy=resumed.policy_stream.state, auxiliary=resumed.auxiliary_streams.state_dict())))
            resumed.update()
            self.assertEqual(resumed.action(np.zeros(16)), action)
            self.assertEqual(expected, tree_digest(dict(agent=agent_state(resumed.agent), rng=global_rng_state(),
                             policy=resumed.policy_stream.state, auxiliary=resumed.auxiliary_streams.state_dict())))
            self.assertEqual(resumed.agent.replay_buffer.ptr, 9)
            self.path.unlink()

    def test_auxiliary_sampling_and_initialization_do_not_change_base_streams(self):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        before = tree_digest(global_rng_state())
        streams = AuxiliaryStreams(0)
        streams.numpy["belief"].uniform(size=(7, 16))
        with streams.torch["initialization"].activate():
            torch.nn.Linear(16, 3)
        self.assertEqual(before, tree_digest(global_rng_state()))
        saved = streams.state_dict()
        expected = streams.numpy["belief"].rand(8)
        streams.load_state_dict(saved)
        np.testing.assert_array_equal(expected, streams.numpy["belief"].rand(8))
        with self.assertRaises(RuntimeError):
            with streams.torch["sampling"].activate():
                torch.rand(3)
                raise RuntimeError("expected")
        self.assertEqual(before, tree_digest(global_rng_state()))

    def test_extra_network_and_optimizer_resume_with_matching_component_schema(self):
        session = self.session()
        with session.auxiliary_streams.torch["initialization"].activate():
            network = torch.nn.Linear(16, 3)
        optimizer = torch.optim.Adam(network.parameters(), lr=.001)
        optimizer.zero_grad()
        network(torch.ones(2, 16)).sum().backward()
        optimizer.step()
        session.register_component("test_cost", network, optimizer)
        digest = session.save(self.path, dict(reset_times=0))
        other = self.session()
        with self.assertRaisesRegex(ValueError, "component schema"):
            other.restore(self.path, digest, SimpleNamespace())
        restored = torch.nn.Linear(16, 3)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
        other.register_component("test_cost", restored, restored_optimizer)
        other.restore(self.path, digest, SimpleNamespace())
        self.assertEqual(tree_digest(optimizer_state(optimizer)), tree_digest(optimizer_state(restored_optimizer)))
        self.assertEqual(tree_digest(network.state_dict()), tree_digest(restored.state_dict()))

    def test_mid_episode_save_and_inference_only_restore_are_rejected(self):
        session = self.session()
        session.begin_episode()
        with self.assertRaisesRegex(ValueError, "episode-boundary"):
            session.save(self.path, dict(reset_times=0))
        session.finish_episode()
        torch.save(session.agent.actor, str(self.path))
        import hashlib
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError, "Not a full"):
            load_snapshot(self.path, digest, self.identity)

    def test_hash_and_identity_mismatch_fail_before_restore(self):
        session = self.session()
        digest = session.save(self.path, dict(reset_times=0))
        with self.assertRaisesRegex(ValueError, "file hash"):
            session.restore(self.path, "wrong", SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "identity"):
            load_snapshot(self.path, digest, dict(git_commit="different"))
        with self.assertRaisesRegex(ValueError, "already exists"):
            session.save(self.path, dict(reset_times=0))

    def test_compact_replay_restores_unused_zero_slots_and_cursor(self):
        session = self.session()
        populate(session.agent, 10)
        state = agent_state(session.agent)
        session.agent.replay_buffer.obs1_buf.fill(99)
        restore_agent(session.agent, state)
        self.assertTrue(np.all(session.agent.replay_buffer.obs1_buf[10:] == 0))
        self.assertEqual(tree_digest(replay_state(session.agent.replay_buffer)), tree_digest(state["replay"]))

    def test_export_is_new_actor_artifact_without_benchmark_eligibility(self):
        session = self.session()
        session.action(np.ones(16))
        record = session.export_actor(self.path.with_suffix(".pkl"))
        self.assertEqual(record["checkpoint_type"], "inference_actor_only")
        self.assertTrue(record["engineering_only"])
        self.assertFalse(record["benchmark_eligible"])
        self.assertFalse(record["gate_enabled"])


if __name__ == "__main__":
    unittest.main()
