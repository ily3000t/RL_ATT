"""Instrument the existing main.py loop without replacing its algorithm.

Run only from a committed, isolated source snapshot via scripts/run_experiment.py.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import time
import xml.etree.ElementTree as ET

import gym
import numpy as np
import torch
import oarl
from rl_att.utils.checkpoints import predictions, verify_checkpoint, weights_sha256
from rl_att.utils.seeding import seed_manifest, TorchPolicyStream


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    cli = config["training"]
    seeds = seed_manifest(config["run_seed"], config["protocol"], cli["episodes"], config["sumo_schedule"])
    manifest["effective_seeds"] = seeds
    manifest["gate_enabled"] = False
    write_json(manifest_path, manifest)
    artifact_dir = manifest_path.parent
    (artifact_dir / "checkpoints").mkdir(exist_ok=True)
    (artifact_dir / "episodes.jsonl").touch()
    state = {"episode": -1, "active": False, "steps": 0, "agent": None, "env": None}
    original_agent = oarl.Agent
    original_make = gym.make

    def finish_episode():
        if not state["active"]:
            return
        agent = state["agent"]
        row = {
            "episode": state["episode"] + 1, "episode_return": state["return"],
            "steps": state["steps"], "terminated": state["done"],
            "truncated": not state["done"] and state["steps"] == cli["max_step"],
            "sumo_seed": seeds["episode_sumo_seeds"][state["episode"]],
            "ego_collision_observed": bool(state["collision_ids"]),
            "sumo_collision_vehicle_ids": sorted(state["collision_ids"]),
            "mean_observed_speed": state["speed_sum"] / max(1, state["steps"]),
            "training_updates_total": agent.update_count,
            "last_training_js": agent.last_js,
            "bo_duplicate_proposals_total": agent.bo_duplicate_proposals,
            "dual_multiplier": float(agent.dual_cst.detach().exp().item()),
            "trajectory_sha256": state["trace"].hexdigest(),
            "elapsed_seconds": time.monotonic() - state["started"],
        }
        with (artifact_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        write_json(artifact_dir / "progress.json", row)
        print("RUN_PROGRESS " + json.dumps({key: row[key] for key in ("episode", "episode_return", "steps", "training_updates_total")}), flush=True)
        state["active"] = False

    class RecordedAgent(original_agent):
        def __init__(self, *agent_args, **kwargs):
            kwargs["attack_seed"] = seeds["attack_seed"]
            # Upstream uses super(Agent, self), resolving Agent in its module.
            # Keep that original class binding while its constructor executes.
            patched_agent = oarl.Agent
            oarl.Agent = original_agent
            try:
                super(RecordedAgent, self).__init__(*agent_args, **kwargs)
            finally:
                oarl.Agent = patched_agent
            self.update_count = 0
            self.last_js = None
            self.bo_duplicate_proposals = 0
            self.current_bo_proposals = None
            self.probes = []
            self.policy_stream = TorchPolicyStream(seeds["policy_seed"]) if config["protocol"] == "controlled" else None
            state["agent"] = self

        def select_action_single(self, observation, mode="train"):
            if len(self.probes) < 64:
                self.probes.append(np.asarray(observation, dtype=np.float32).copy())
            if self.policy_stream is None:
                return super(RecordedAgent, self).select_action_single(observation, mode)
            with self.policy_stream.activate():
                return super(RecordedAgent, self).select_action_single(observation, mode)

        def get_optimal_perturb_Bayes(self, *values):
            self.current_bo_proposals = []
            result = super(RecordedAgent, self).get_optimal_perturb_Bayes(*values)
            if len(self.current_bo_proposals) != self.attack_optimizing_times:
                raise ValueError("BO evaluation count differs from configuration")
            self.bo_duplicate_proposals += len(self.current_bo_proposals) - len(set(self.current_bo_proposals))
            self.current_bo_proposals = None
            self.last_js = float(result.detach().item())
            return result

        def js_d_loss(self, u1, u2):
            if self.current_bo_proposals is not None:
                self.current_bo_proposals.append((float(u1), float(u2)))
            return super(RecordedAgent, self).js_d_loss(u1, u2)

        def train_model(self):
            super(RecordedAgent, self).train_model()
            self.update_count += 1
            # An error diagnostic, not an action filter or an update gate.
            for network in (self.actor, self.qf1, self.qf2, self.qf1_target, self.qf2_target):
                if any(not bool(torch.isfinite(p).all()) for p in network.parameters()):
                    raise FloatingPointError("Non-finite trained parameters")
            if not bool(torch.isfinite(self.dual_cst.exp()).all()):
                raise FloatingPointError("Non-finite robust multiplier")

        def save_model(self, episode, model_path):
            finish_episode()
            observations = np.stack(self.probes)
            expected = predictions(self.actor, observations)
            weight_hash = weights_sha256(self.actor)
            super(RecordedAgent, self).save_model(episode, model_path)
            checkpoint = Path(model_path) / ("policy%d.pkl" % episode)
            checked = verify_checkpoint(checkpoint, observations, expected.numpy(), weight_hash)
            checked.update(episode=episode, run_seed=config["run_seed"], protocol=config["protocol"],
                           victim=config["victim"], git_commit=manifest["git_commit"],
                           training_updates=self.update_count, effective_seeds=seeds,
                           checkpoint_path=str(checkpoint.resolve()), gate_enabled=False)
            probe_path = artifact_dir / "probe_observations.npy"
            if not probe_path.exists():
                np.save(str(probe_path), observations)
            checked["probe_sha256"] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
            write_json(artifact_dir / "checkpoints" / ("policy%d.json" % episode), checked)

    def make_environment(*make_args, **kwargs):
        env = original_make(*make_args, **kwargs)
        state["env"] = env
        if config["protocol"] == "controlled":
            env.unwrapped.sumo_seed_schedule = seeds["episode_sumo_seeds"]
        old_reset, old_step = env.reset, env.step

        def reset():
            finish_episode()
            observation = old_reset()
            state.update(episode=state["episode"] + 1, active=True, steps=0, done=False,
                         collision_ids=set(), speed_sum=0.0, trace=hashlib.sha256(), started=time.monotonic())
            state["return"] = 0.0
            from Environment.environment.envs.highway_env import config_path
            actual_seed = int(ET.parse(config_path).find(".//seed").get("value"))
            if actual_seed != seeds["episode_sumo_seeds"][state["episode"]]:
                raise ValueError("SUMO configuration seed differs from the recorded schedule")
            return observation

        def step(action):
            result = old_step(action)
            observation, reward, done = result[:3]
            state["steps"] += 1
            state["return"] += float(reward)
            state["done"] = bool(done)
            state["speed_sum"] += float(observation[0]) * 35
            state["trace"].update(np.asarray(observation, dtype=np.float64).tobytes())
            state["trace"].update(np.asarray([int(action), float(reward), int(done)], dtype=np.float64).tobytes())
            from Environment.environment.envs.highway_env import traci
            collided = traci.simulation.getCollidingVehiclesIDList()
            if "Auto" in collided:
                state["collision_ids"].update(collided)
            return result

        env.reset, env.step = reset, step
        return env

    oarl.Agent, gym.make = RecordedAgent, make_environment
    sys.argv = ["main.py"]
    for key, value in cli.items():
        sys.argv.extend(["--" + key, str(value)])
    try:
        runpy.run_path("main.py", run_name="__main__")
        finish_episode()
    finally:
        oarl.Agent, gym.make = original_agent, original_make
        # main closes on success; also release TraCI on failure.
        if state["env"] is not None:
            from Environment.environment.envs.highway_env import traci
            if traci.isLoaded():
                state["env"].close()


if __name__ == "__main__":
    main()
