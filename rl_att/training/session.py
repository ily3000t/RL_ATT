"""OARL-matched CPU training facade with complete episode-boundary state."""

import copy
import json
from pathlib import Path
import random

import numpy as np
import torch

from oarl import Agent
from rl_att.agents.clean_victim import CleanVictimAgent
from rl_att.utils.checkpoints import predictions, verify_checkpoint, weights_sha256
from rl_att.utils.seeding import seed_manifest, TorchPolicyStream
from .resources import TrainingResources
from .state import (agent_state, global_rng_state, load_snapshot, network_state, optimizer_state, require,
                    restore_agent, restore_global_rng, restore_network, save_snapshot)
from .streams import AuxiliaryStreams


def training_seeds(config):
    seeds = seed_manifest(config["run_seed"], "controlled", config["training"]["episodes"], config["sumo_schedule"])
    if config["victim"] == "clean":
        seeds["attack_rng"] = "unused_clean_training"
    seeds["numpy_rng"] = "upstream_shared_environment_and_replay_preserved"
    return seeds


def initialize_global_rng(seeds):
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])


class TrainingSession:
    def __init__(self, config, identity, initialize_rng=True, agent_kwargs=None):
        require(config["victim"] in ("clean", "oarl") and config["protocol"] == "controlled" and
                config["gate_enabled"] is False, "Only matched controlled Clean/OARL training without Gate")
        require(not torch.cuda.is_available(), "Use the frozen CPU environment")
        self.config, self.identity = copy.deepcopy(config), copy.deepcopy(identity)
        self.seeds = training_seeds(config)
        if initialize_rng:
            initialize_global_rng(self.seeds)
        cli = config["training"]
        cls = CleanVictimAgent if config["victim"] == "clean" else Agent
        self.agent = cls(cli["state_dim"], cli["action_dim"], cli["action_numb"],
                         attack_seed=self.seeds["attack_seed"], **(agent_kwargs or {}))
        self.agent.train()
        self.policy_stream = TorchPolicyStream(self.seeds["policy_seed"])
        self.auxiliary_streams = AuxiliaryStreams(config["run_seed"])
        self.resources = TrainingResources()
        self.components = {}
        self.loop = dict(completed_episodes=0, interactions=0, updates=0, policy_version=0)
        self.probes = []
        self.in_episode = False

    def register_component(self, name, network, optimizer):
        require(name and name not in self.components, "Duplicate/empty training component")
        self.components[name] = network, optimizer

    def action(self, observation):
        if len(self.probes) < 64:
            self.probes.append(np.asarray(observation, dtype=np.float32).copy())
        with self.resources.measure(self.agent, "interaction_policy"), self.policy_stream.activate():
            return int(self.agent.select_action_single(observation, "train"))

    def update(self):
        with self.resources.measure(self.agent, "primary_update"):
            self.agent.train_model()
        self.loop["updates"] += 1
        self.loop["policy_version"] += 1
        self.resources.add("primary_updates")

    def transition(self, observation, action, reward, next_observation, done):
        require(self.in_episode, "Transition outside episode")
        self.agent.replay_buffer.add(observation, action, reward, next_observation, done)
        self.loop["interactions"] += 1
        self.resources.add("real_interaction_steps")
        # Preserve main.py: zero-based episode > 10 and cumulative interactions % 2.
        if self.loop["completed_episodes"] > 10 and self.loop["interactions"] % 2 == 0:
            self.update()

    def begin_episode(self):
        require(not self.in_episode, "Already in episode")
        self.in_episode = True

    def finish_episode(self):
        require(self.in_episode, "No active episode")
        self.in_episode = False
        self.loop["completed_episodes"] += 1

    def payload(self, environment_state):
        require(not self.in_episode, "Only episode-boundary snapshots are supported")
        return dict(schema_version=1, kind="episode_boundary_training_state", boundary="before_next_episode_reset",
                    identity=copy.deepcopy(self.identity), config=copy.deepcopy(self.config), effective_seeds=copy.deepcopy(self.seeds),
                    agent=agent_state(self.agent), global_rng=global_rng_state(), policy_rng=self.policy_stream.state.clone(),
                    auxiliary_rng=self.auxiliary_streams.state_dict(), loop=copy.deepcopy(self.loop),
                    environment=copy.deepcopy(environment_state), resources=self.resources.state_dict(),
                    probes=copy.deepcopy(self.probes),
                    components={name: dict(network=network_state(network), optimizer=optimizer_state(optimizer))
                                for name, (network, optimizer) in self.components.items()})

    def save(self, path, environment_state):
        return save_snapshot(path, self.payload(environment_state))

    def restore(self, path, expected_hash, environment):
        require(not self.in_episode, "Cannot restore during an active episode")
        payload = load_snapshot(path, expected_hash, self.identity)
        require(payload["config"] == self.config and payload["effective_seeds"] == self.seeds, "Resume configuration/seeds mismatch")
        require(set(payload["components"]) == set(self.components), "Training component schema mismatch")
        loop = payload["loop"]
        require(payload["environment"]["reset_times"] == loop["completed_episodes"] and
                loop["updates"] == loop["policy_version"] and
                loop["interactions"] == payload["resources"]["counts"]["real_interaction_steps"], "Resume loop/ledger inconsistent")
        restore_agent(self.agent, payload["agent"])
        self.policy_stream.state = payload["policy_rng"].clone()
        self.auxiliary_streams.load_state_dict(payload["auxiliary_rng"])
        self.loop, self.probes = copy.deepcopy(payload["loop"]), copy.deepcopy(payload["probes"])
        environment.__dict__.clear()
        environment.__dict__.update(copy.deepcopy(payload["environment"]))
        self.resources.load_state_dict(payload["resources"])
        for name, (network, optimizer) in self.components.items():
            saved = payload["components"][name]
            restore_network(network, saved["network"])
            optimizer.load_state_dict(saved["optimizer"])
        # Constructors/loaders may consume RNG: restore the global streams last.
        restore_global_rng(payload["global_rng"])
        return payload

    def export_actor(self, path, engineering_only=True):
        require(not self.in_episode and self.probes, "Export actor at a populated episode boundary")
        path = Path(path)
        require(not path.exists(), "Actor artifact already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        probes = np.stack(self.probes)
        expected = predictions(self.agent.actor, probes).numpy()
        digest = weights_sha256(self.agent.actor)
        torch.save(self.agent.actor, str(path))
        record = verify_checkpoint(path, probes, expected, digest)
        record.update(kind="new_training_actor_artifact", checkpoint=str(path), identity=self.identity,
                      victim=self.config["victim"], training_episode=self.loop["completed_episodes"],
                      training_updates=self.loop["updates"], effective_seeds=self.seeds,
                      auxiliary_role_seeds=self.auxiliary_streams.seeds, gate_enabled=False,
                      engineering_only=engineering_only, benchmark_eligible=False,
                      provenance_scope="New training artifact; does not replace or append old frozen registry automatically")
        record_path = path.with_suffix(".json")
        require(not record_path.exists(), "Actor record already exists")
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record


def canonical_training_state(agent, policy_state, interactions, updates, completed_episodes):
    """Common digest used for original-loop and support/resume comparisons."""
    return dict(agent=agent_state(agent), policy_rng=policy_state.clone(), global_rng=global_rng_state(),
                interactions=interactions, updates=updates, completed_episodes=completed_episodes)
