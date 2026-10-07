"""Short original-loop/continuous/fresh-process-resume engineering worker."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import gym
import numpy as np
import torch

import Environment.environment
from Environment.environment.envs.highway_env import traci
from rl_att.experiment import main as recorded_original
from .session import TrainingSession, canonical_training_state, initialize_global_rng, training_seeds
from .state import require, save_snapshot, tree_digest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def reference(manifest_path, manifest):
    captured = {}
    previous = sys.getprofile()
    original_step = traci.simulationStep
    resetting, calls = [False], dict(sumo_simulation_steps=0, sumo_reset_warmup_steps=0)

    def simulation_step(*args, **kwargs):
        value = original_step(*args, **kwargs)
        calls["sumo_simulation_steps"] += 1
        calls["sumo_reset_warmup_steps"] += int(resetting[0])
        return value

    def profile(frame, event, value):
        if frame.f_code.co_name == "reset" and Path(frame.f_code.co_filename).name == "highway_env.py":
            if event == "call":
                resetting[0] = True
            elif event == "return":
                resetting[0] = False
        # Capture before CSV/plot postprocessing, which is outside the training loop.
        parent = frame.f_back
        at_output = (event == "call" and frame.f_code.co_name == "__init__" and
                     "pandas" in frame.f_code.co_filename and parent is not None and
                     parent.f_code.co_name == "train" and Path(parent.f_code.co_filename).name == "main.py" and
                     "model_l" in parent.f_locals and "df" not in parent.f_locals)
        at_return = (event == "return" and frame.f_code.co_name == "train" and
                     Path(frame.f_code.co_filename).name == "main.py" and "model_l" in frame.f_locals)
        if not captured and (at_output or at_return):
            source = parent if at_output else frame
            agent = source.f_locals["model_l"]
            captured["canonical"] = canonical_training_state(agent, agent.policy_stream.state,
                  source.f_locals["interaction_times"], agent.update_count, source.f_locals["n_epi"] + 1)
    original_argv = sys.argv
    try:
        traci.simulationStep = simulation_step
        sys.setprofile(profile)
        sys.argv = ["rl_att.experiment", "--manifest", str(manifest_path)]
        recorded_original()
    finally:
        sys.setprofile(previous)
        traci.simulationStep = original_step
        sys.argv = original_argv
    require("canonical" in captured, "Original loop did not finish")
    digest = save_snapshot(manifest_path.parent / "canonical.pt", captured["canonical"])
    write_json(manifest_path.parent / "support_report.json", dict(mode="reference", canonical_sha256=digest,
               canonical_state_sha256=tree_digest(captured["canonical"]), completed_episodes=manifest["config"]["training"]["episodes"],
               segment_counts=dict(real_interaction_steps=captured["canonical"]["interactions"],
                                   sumo_reset_warmup_steps=calls["sumo_reset_warmup_steps"])))
    require(calls["sumo_simulation_steps"] == calls["sumo_reset_warmup_steps"] + captured["canonical"]["interactions"],
            "Original reference contains unaccounted simulation steps")


def train(manifest_path, manifest):
    config, mode = manifest["config"], manifest["mode"]
    directory, cli = manifest_path.parent, config["training"]
    env = gym.make(cli["env"])
    env.seed(config["run_seed"])
    seeds = training_seeds(config)
    initialize_global_rng(seeds)
    original_step = traci.simulationStep
    resetting, warmup_calls = [False], [0]

    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if resetting[0]:
            warmup_calls[0] += 1
        return result
    traci.simulationStep = step
    started = False
    try:
        env.start(gui=False)
        started = True
        session = TrainingSession(config, manifest["identity"], initialize_rng=False)
        env.unwrapped.sumo_seed_schedule = seeds["episode_sumo_seeds"]
        if mode == "resumed":
            session.restore(manifest["resume_path"], manifest["resume_sha256"], env.unwrapped)
            require(session.loop["completed_episodes"] == manifest["split_after_episode"], "Wrong resume boundary")
        begin_counts = copy.deepcopy(session.resources.counts)
        begin_episode = session.loop["completed_episodes"]
        if config["victim"] == "pgd_consistency":
            update_path = directory / "defense_updates.jsonl"
            update_path.touch()
            original_update = session.update
            def recorded_update():
                original_update()
                row = dict(training_update=session.loop["updates"], metrics=session.agent.last_consistency,
                           **{name: value.tolist() for name, value in session.agent.last_consistency_trace.items()})
                with update_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
            session.update = recorded_update
        stop = manifest["split_after_episode"] if mode == "prefix" else cli["episodes"]
        checkpoint = None
        with (directory / "episodes.jsonl").open("w", encoding="utf-8") as raw:
            while session.loop["completed_episodes"] < stop:
                index = session.loop["completed_episodes"]
                session.begin_episode()
                before_warmup = warmup_calls[0]
                resetting[0] = True
                try:
                    observation = env.reset()
                finally:
                    resetting[0] = False
                session.resources.add("sumo_reset_warmup_steps", warmup_calls[0] - before_warmup)
                reward_sum, collisions, digest, done = 0., set(), hashlib.sha256(), False
                for step_index in range(cli["max_step"]):
                    observation = np.asarray(observation, dtype=float)
                    action = session.action(observation)
                    next_observation, reward, done = env.step(action)[:3]
                    session.transition(observation, action, reward, next_observation, done)
                    observation = next_observation
                    reward_sum += float(reward)
                    ids = traci.simulation.getCollidingVehiclesIDList()
                    if "Auto" in ids:
                        collisions.update(ids)
                    digest.update(np.asarray(observation, dtype=np.float64).tobytes())
                    digest.update(np.asarray([action, float(reward), int(done)], dtype=np.float64).tobytes())
                    if done:
                        break
                session.finish_episode()
                row = dict(episode=index + 1, sumo_seed=seeds["episode_sumo_seeds"][index],
                           steps=step_index + 1, episode_return=reward_sum, terminated=bool(done),
                           truncated=not done, ego_collision_observed=bool(collisions),
                           sumo_collision_vehicle_ids=sorted(collisions), trajectory_sha256=digest.hexdigest(),
                           training_updates_total=session.loop["updates"])
                raw.write(json.dumps(row) + "\n")
                raw.flush()
                print("SUPPORT_PROGRESS %s %s episode=%d updates=%d" % (config["victim"], mode, index + 1, session.loop["updates"]), flush=True)
                if session.loop["completed_episodes"] == manifest["split_after_episode"]:
                    path = directory / "training-boundary.pt"
                    checkpoint = dict(path=str(path), sha256=session.save(path, env.unwrapped.__dict__),
                         canonical_state_sha256=tree_digest(canonical_training_state(session.agent, session.policy_stream.state,
                         session.loop["interactions"], session.loop["updates"], session.loop["completed_episodes"])))
        canonical = canonical_training_state(session.agent, session.policy_stream.state,
                      session.loop["interactions"], session.loop["updates"], session.loop["completed_episodes"])
        canonical_sha = save_snapshot(directory / "canonical.pt", canonical)
        final_path = directory / "training-final.pt"
        final_sha = session.save(final_path, env.unwrapped.__dict__)
        actor = session.export_actor(directory / "actor.pkl")
        segment_counts = {key: value - begin_counts[key] for key, value in session.resources.counts.items()}
        require(segment_counts["sumo_reset_warmup_steps"] == warmup_calls[0], "Warmup costs lost on resume")
        report = dict(mode=mode, canonical_sha256=canonical_sha, canonical_state_sha256=tree_digest(canonical),
                      completed_episodes=session.loop["completed_episodes"], begin_episode=begin_episode,
                      boundary_checkpoint=checkpoint, final_training_checkpoint=dict(path=str(final_path), sha256=final_sha),
                      actor_artifact=actor, cumulative_resources=session.resources.state_dict(),
                      segment_counts=segment_counts, effective_seeds=session.seeds,
                      auxiliary_role_seeds=session.auxiliary_streams.seeds,
                      comparison_reference=manifest.get("comparison_reference", "original_loop"),
                      interpretation=("Enabled PGD engineering repeat/resume only; no efficacy or benchmark model" if config["victim"] == "pgd_consistency" else
                                      "Engineering compatibility only; no defense enabled, convergence or frozen benchmark model"))
        write_json(directory / "support_report.json", report)
    finally:
        if started:
            env.close()
        traci.simulationStep = original_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    path = args.manifest.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    require(manifest["mode"] in ("reference", "continuous", "prefix", "resumed"), "Unknown support check mode")
    if manifest["mode"] == "reference" and manifest["config"]["victim"] != "pgd_consistency":
        reference(path, manifest)
    else:
        train(path, manifest)


if __name__ == "__main__":
    main()
