"""Recorded PGD baseline training; uses the tested TrainingSession update path."""

import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import time

import gym
import numpy as np
import torch
import Environment.environment
from Environment.environment.envs.highway_env import traci
from .protocol import validate_config
from .session import TrainingSession, initialize_global_rng, training_seeds
from .state import require, tree_digest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = validate_config(manifest["config"])
    cli, directory = config["training"], path.parent
    identity = dict(git_commit=manifest["git_commit"], config_sha256=manifest["config_sha256"],
                    victim=config["victim"], run_seed=config["run_seed"])
    env = gym.make(cli["env"])
    env.seed(config["run_seed"])
    seeds = training_seeds(config)
    initialize_global_rng(seeds)
    resetting, warmup = [False], [0]
    original_step = traci.simulationStep

    def simulation_step(*args, **kwargs):
        value = original_step(*args, **kwargs)
        warmup[0] += int(resetting[0])
        return value

    traci.simulationStep = simulation_step
    started = False
    records = []
    start_time = time.monotonic()
    try:
        env.start(gui=False)
        started = True
        session = TrainingSession(config, identity, initialize_rng=False)
        env.unwrapped.sumo_seed_schedule = seeds["episode_sumo_seeds"]
        manifest.update(effective_seeds=seeds, auxiliary_role_seeds=session.auxiliary_streams.seeds,
                        training_identity=identity, initial_agent_sha256=tree_digest(session.payload(env.unwrapped.__dict__)["agent"]))
        write_json(path, manifest)
        (directory / "checkpoints").mkdir()
        with gzip.open(str(directory / "defense_updates.jsonl.gz"), "wt", encoding="utf-8", compresslevel=1) as updates, \
             gzip.open(str(directory / "transitions.jsonl.gz"), "wt", encoding="utf-8", compresslevel=1) as transitions, \
             (directory / "episodes.jsonl").open("w", encoding="utf-8") as episodes:
            original_update = session.update

            def recorded_update():
                original_update()
                record = dict(training_update=session.loop["updates"], metrics=session.agent.last_consistency,
                              **{name: value.tolist() for name, value in session.agent.last_consistency_trace.items()})
                updates.write(json.dumps(record, separators=(",", ":")) + "\n")

            session.update = recorded_update
            for index in range(cli["episodes"]):
                session.begin_episode()
                before = warmup[0]
                resetting[0] = True
                try:
                    observation = env.reset()
                finally:
                    resetting[0] = False
                episode_warmup = warmup[0] - before
                session.resources.add("sumo_reset_warmup_steps", episode_warmup)
                reward_sum, collisions, digest, done = 0., set(), hashlib.sha256(), False
                for step_index in range(cli["max_step"]):
                    observation = np.asarray(observation, dtype=float)
                    action = session.action(observation)
                    next_observation, reward, done = env.step(action)[:3]
                    session.transition(observation, action, reward, next_observation, done)
                    ids = list(traci.simulation.getCollidingVehiclesIDList())
                    ego_ids = ids if "Auto" in ids else []
                    collisions.update(ego_ids)
                    next_values = np.asarray(next_observation, dtype=np.float64)
                    digest.update(next_values.tobytes())
                    digest.update(np.asarray([action, float(reward), int(done)], dtype=np.float64).tobytes())
                    transitions.write(json.dumps(dict(episode=index + 1, step=step_index + 1,
                        observation=np.asarray(observation).tolist(), next_observation=next_values.tolist(), action=action,
                        reward=float(reward), done=bool(done), ego_collision_ids=ego_ids,
                        training_updates_total=session.loop["updates"]), separators=(",", ":")) + "\n")
                    reward_sum += float(reward)
                    observation = next_observation
                    if done:
                        break
                session.finish_episode()
                row = dict(episode=index + 1, steps=step_index + 1, episode_return=reward_sum,
                    terminated=bool(done), truncated=not done, sumo_seed=seeds["episode_sumo_seeds"][index],
                    ego_collision_observed=bool(collisions), sumo_collision_vehicle_ids=sorted(collisions),
                    trajectory_sha256=digest.hexdigest(), training_updates_total=session.loop["updates"], warmup_steps=episode_warmup)
                episodes.write(json.dumps(row) + "\n")
                episodes.flush()
                updates.flush()
                transitions.flush()
                write_json(directory / "progress.json", dict(row, elapsed_seconds=time.monotonic() - start_time))
                print("DEFENSE_TRAIN_PROGRESS " + json.dumps(row), flush=True)
                if index + 1 in config["checkpoint_episodes"]:
                    full_path = directory / "checkpoints" / ("training%d.pt" % (index + 1))
                    full_hash = session.save(full_path, env.unwrapped.__dict__)
                    actor_path = Path.cwd() / Path(cli["save_dir_model"]) / ("policy%d.pkl" % (index + 1))
                    actor = session.export_actor(actor_path, engineering_only=config["training_stage"] != "full")
                    records.append(dict(episode=index + 1, full_training_path=str(full_path), full_training_sha256=full_hash,
                                        actor=actor, numeric_agent_sha256=tree_digest(session.payload(env.unwrapped.__dict__)["agent"])))
                    write_json(directory / "checkpoint_index.json", records)
        report = dict(kind="pgd_consistency_training", engineering_only=config["training_stage"] != "full",
            completed_episodes=session.loop["completed_episodes"], interactions=session.loop["interactions"],
            updates=session.loop["updates"], cumulative_resources=session.resources.state_dict(),
            defense_training=session.agent.training_state_dict()["counts"], checkpoints=records,
            effective_seeds=seeds, auxiliary_role_seeds=session.auxiliary_streams.seeds,
            training_wall_seconds=time.monotonic() - start_time, finished_at_utc=datetime.datetime.utcnow().isoformat() + "Z")
        write_json(directory / "training_report.json", report)
    finally:
        if started:
            env.close()
        traci.simulationStep = original_step


if __name__ == "__main__":
    main()
