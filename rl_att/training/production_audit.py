"""Streaming audit of recorded PGD training, raw trajectories and all checkpoints."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from rl_att.utils.checkpoints import verify_checkpoint
from .consistency_audit import audit_update, audit_training
from .protocol import validate_config
from .session import training_seeds
from .state import require, tree_digest


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path):
    with gzip.open(str(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def audit_episodes(episodes, transitions, config, seeds):
    require(len(episodes) == config["training"]["episodes"], "Incomplete training episode grid")
    iterator = iter(transitions)
    interaction, updates, warmup = 0, 0, 0
    for index, episode in enumerate(episodes):
        require(episode["episode"] == index + 1 and episode["sumo_seed"] == seeds["episode_sumo_seeds"][index], "Wrong episode/traffic seed")
        require(type(episode["steps"]) is int and 1 <= episode["steps"] <= config["training"]["max_step"], "Invalid episode length")
        require(type(episode["warmup_steps"]) is int and episode["warmup_steps"] >= 0, "Invalid SUMO reset costs")
        warmup += episode["warmup_steps"]
        reward_sum, collisions, digest, previous, ended = 0., set(), hashlib.sha256(), None, False
        for step in range(episode["steps"]):
            row = next(iterator, None)
            require(row is not None and not ended and row["episode"] == index + 1 and row["step"] == step + 1, "Missing/duplicate raw transition")
            observation, next_observation = np.asarray(row["observation"], dtype=np.float64), np.asarray(row["next_observation"], dtype=np.float64)
            require(observation.shape == next_observation.shape == (16,) and np.isfinite(observation).all() and np.isfinite(next_observation).all(),
                    "Invalid raw transition observation")
            require(previous is None or np.array_equal(previous, observation), "Broken consecutive observations")
            require(type(row["action"]) is int and row["action"] in (0, 1, 2) and math.isfinite(row["reward"]) and type(row["done"]) is bool,
                    "Invalid raw transition outcome")
            require(not row["ego_collision_ids"] or "Auto" in row["ego_collision_ids"], "Wrong ego collision label")
            interaction += 1
            updates += int(index > 10 and interaction % 2 == 0)
            require(row["training_updates_total"] == updates, "Training update cadence changed")
            reward_sum += row["reward"]
            collisions.update(row["ego_collision_ids"])
            digest.update(next_observation.tobytes())
            digest.update(np.asarray([row["action"], row["reward"], int(row["done"])], dtype=np.float64).tobytes())
            previous, ended = next_observation, row["done"]
        require(reward_sum == episode["episode_return"] and digest.hexdigest() == episode["trajectory_sha256"] and
                bool(collisions) == episode["ego_collision_observed"] and sorted(collisions) == episode["sumo_collision_vehicle_ids"],
                "Raw trajectory/reward/collision summary differs")
        require(ended == episode["terminated"] and (not ended) == episode["truncated"] and
                (ended or episode["steps"] == config["training"]["max_step"]) and episode["training_updates_total"] == updates,
                "Episode boundary or update count mismatch")
    require(next(iterator, None) is None, "Extra raw training transitions")
    return dict(interactions=interaction, updates=updates, warmup_steps=warmup)


def audit_run(path):
    path = Path(path)
    directory = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = validate_config(manifest["config"])
    report_path = directory / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes_path, transitions_path, updates_path = (directory / name for name in
                                                    ("episodes.jsonl", "transitions.jsonl.gz", "defense_updates.jsonl.gz"))
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    seeds = training_seeds(config)
    require(report["effective_seeds"] == manifest["effective_seeds"] == seeds, "Training seed identity drift")
    observed = audit_episodes(episodes, jsonl(transitions_path), config, seeds)
    require(report["completed_episodes"] == config["training"]["episodes"] and report["interactions"] == observed["interactions"] and
            report["updates"] == observed["updates"] and observed["updates"] > 0, "Training report incomplete")
    require([c["episode"] for c in report["checkpoints"]] == config["checkpoint_episodes"], "Missing/duplicate checkpoint grid")
    files = {str(file): sha(file) for file in (report_path, episodes_path, transitions_path, updates_path)}
    checkpoints, final = [], None
    for checkpoint in report["checkpoints"]:
        full_path = Path(checkpoint["full_training_path"])
        require(sha(full_path) == checkpoint["full_training_sha256"], "Full training checkpoint changed")
        files[str(full_path)] = checkpoint["full_training_sha256"]
        payload = torch.load(str(full_path), map_location="cpu")
        require(payload["kind"] == "episode_boundary_training_state" and payload["boundary"] == "before_next_episode_reset" and
                payload["identity"] == manifest["training_identity"] and payload["config"] == config and payload["effective_seeds"] == seeds and
                payload["loop"]["completed_episodes"] == checkpoint["episode"] and payload["environment"]["reset_times"] == checkpoint["episode"] and
                tree_digest(payload["agent"]) == checkpoint["numeric_agent_sha256"], "Invalid full checkpoint identity/state")
        prefix = episodes[:checkpoint["episode"]]
        require(payload["loop"]["updates"] == prefix[-1]["training_updates_total"] and
                payload["loop"]["interactions"] == sum(row["steps"] for row in prefix) and
                payload["resources"]["counts"]["sumo_reset_warmup_steps"] == sum(row["warmup_steps"] for row in prefix) and
                set(payload["agent"]["networks"]) == {"actor", "qf1", "qf2", "qf1_target", "qf2_target"} and
                set(payload["agent"]["optimizers"]) == {"actor_optimizer", "qf1_optimizer", "qf2_optimizer", "dual_cst_optimizer"},
                "Incomplete intermediate training checkpoint")
        for key, value in manifest["agent_defaults"].items():
            expected = seeds["attack_seed"] if key == "attack_seed" else value
            require(payload["agent"]["hyperparameters"].get(key) == expected, "Agent hyperparameters drifted: " + key)
        actor = checkpoint["actor"]
        actor_path = Path(actor["checkpoint"])
        require(sha(actor_path) == actor["checkpoint_sha256"] and actor["training_episode"] == checkpoint["episode"] and
                actor["engineering_only"] == (config["training_stage"] != "full") and actor["benchmark_eligible"] is False and
                actor["defense_config"] == config["defense"], "Checkpoint incorrectly registered")
        expected = np.asarray(actor["probabilities"], dtype=np.float32)
        verify_checkpoint(actor_path, np.stack(payload["probes"]), expected, actor["weights_sha256"])
        require(tree_digest(torch.load(str(actor_path), map_location="cpu").state_dict()) ==
                tree_digest(payload["agent"]["networks"]["actor"]["weights"]), "Actor does not match full checkpoint")
        require(tree_digest(payload["agent"]["defense_training"]["attack_rng"]) == tree_digest(payload["auxiliary_rng"]["numpy"]["attack"]),
                "Search RNG missing from full checkpoint")
        files[str(actor_path)] = actor["checkpoint_sha256"]
        checkpoints.append(dict(episode=checkpoint["episode"], checkpoint=str(actor_path),
                                checkpoint_sha256=actor["checkpoint_sha256"], weights_sha256=actor["weights_sha256"],
                                full_training_checkpoint=str(full_path), full_training_sha256=checkpoint["full_training_sha256"]))
        final = payload
    ledger = report["cumulative_resources"]
    require(ledger == final["resources"] and report["updates"] == final["loop"]["updates"] and
            report["defense_training"] == final["agent"]["defense_training"]["counts"] and
            ledger["counts"]["real_interaction_steps"] == observed["interactions"] and
            ledger["counts"]["sumo_reset_warmup_steps"] == observed["warmup_steps"] and
            ledger["counts"]["primary_updates"] == observed["updates"], "Final raw/ledger/full-state counts differ")
    count, low, high, violation, last = 0, float("inf"), 0., 0., None
    for row in jsonl(updates_path):
        count += 1
        require(row["training_update"] == count, "Raw updates skipped/duplicated")
        value = audit_update(row, config["defense"], 128)
        low, high, violation = min(low, value["kl"]), max(high, value["kl"]), max(violation, value["max_box_violation"])
        last = row
    require(count == observed["updates"], "Missing raw PGD update probes")
    # Validate cumulative counts without retaining every raw observation in memory.
    audit_training([last], config["defense"], final["agent"]["defense_training"], ledger, 1)
    require(ledger["phases"]["primary_update"]["optimizer_steps"] ==
            {name: count for name in ("actor_optimizer", "qf1_optimizer", "qf2_optimizer")}, "Actual optimizer cost differs")
    return dict(schema_version=1, kind="pgd_consistency_training_audit", verified=True,
        engineering_only=config["training_stage"] != "full", benchmark_eligible=config["training_stage"] == "full",
        git_commit=manifest["git_commit"], config_sha256=manifest["config_sha256"], run_seed=config["run_seed"],
        victim="pgd_consistency", episodes=len(episodes), counts=observed, checkpoints=checkpoints,
        resources=ledger, raw_kl_min=low, raw_kl_max=high, max_box_violation=violation, source_sha256=files,
        interpretation="Complete recorded training/checkpoint audit; no convergence, defense efficacy or safety claim")
