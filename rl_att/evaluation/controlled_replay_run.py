"""Private SUMO worker for registered defense branch diagnostics."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from Environment.environment.envs.highway_env import HighwayEnv
import traci

from rl_att.agents.victim_adapter import VictimAdapter
from .controlled_replay import (assert_historical_step, episode_summary,
                                fingerprint, first_divergence, intervention)
from .defense_diagnostics import require
from .results import write_json
from .sumo_metrics import SUMOMetrics


def observable_state():
    """All vehicles' queried kinematics/control modes; not a latent SUMO snapshot."""
    vehicles = []
    for vehicle in sorted(traci.vehicle.getIDList()):
        vehicles.append(dict(id=vehicle, position=list(traci.vehicle.getPosition(vehicle)),
                             speed=traci.vehicle.getSpeed(vehicle), acceleration=traci.vehicle.getAcceleration(vehicle),
                             angle=traci.vehicle.getAngle(vehicle), lane=traci.vehicle.getLaneID(vehicle),
                             lane_index=traci.vehicle.getLaneIndex(vehicle), lane_position=traci.vehicle.getLanePosition(vehicle),
                             route=list(traci.vehicle.getRoute(vehicle)), route_index=traci.vehicle.getRouteIndex(vehicle),
                             speed_mode=traci.vehicle.getSpeedMode(vehicle),
                             lane_change_mode=traci.vehicle.getLaneChangeMode(vehicle),
                             length_m=traci.vehicle.getLength(vehicle), min_gap_m=traci.vehicle.getMinGap(vehicle)))
    return dict(time_s=traci.simulation.getTime(), vehicles=vehicles,
                colliding_ids=sorted(traci.simulation.getCollidingVehiclesIDList()),
                starting_teleport_ids=sorted(traci.simulation.getStartingTeleportIDList()),
                ending_teleport_ids=sorted(traci.simulation.getEndingTeleportIDList()))


class Instrumentation:
    def __init__(self):
        self.original_step = traci.simulationStep
        self.original_change = traci.vehicle.changeLane
        self.steps, self.resetting, self.warmup, self.commands = 0, False, [], []

    def step(self, *args, **kwargs):
        value = self.original_step(*args, **kwargs)
        self.steps += 1
        if self.resetting:
            self.warmup.append(dict(warmup_step=len(self.warmup) + 1, time_s=traci.simulation.getTime(),
                                    ego_present="Auto" in traci.vehicle.getIDList(),
                                    colliding_ids=sorted(traci.simulation.getCollidingVehiclesIDList())))
        return value

    def change(self, vehicle, target_lane, duration, *args, **kwargs):
        self.commands.append(dict(vehicle=vehicle, from_lane=traci.vehicle.getLaneIndex(vehicle),
                                  target_lane=target_lane, duration_s=duration,
                                  time_s=traci.simulation.getTime()))
        return self.original_change(vehicle, target_lane, duration, *args, **kwargs)

    def install(self):
        traci.simulationStep = self.step
        traci.vehicle.changeLane = self.change

    def restore(self):
        traci.simulationStep = self.original_step
        traci.vehicle.changeLane = self.original_change


def run_arm(job, method, arm, victim, output, config):
    seeds = job["source_effective_seeds"][method]
    random.seed(seeds["python_seed"])
    np.random.seed(seeds["numpy_seed"])
    torch.manual_seed(seeds["torch_seed"])
    env = HighwayEnv(sumo_seed_schedule=seeds["episode_sumo_seeds"])
    require(env.ErrorPropability == 0 and env.DisableFaultSimulation, "Replay assumes unchanged disabled fault simulation")
    instrumentation = Instrumentation()
    instrumentation.install()
    forwards = [0]
    hook = victim.actor.register_forward_hook(lambda module, args, value: forwards.__setitem__(0, forwards[0] + 1))
    directory = output / job["job_id"] / method / arm
    directory.mkdir(parents=True, exist_ok=False)
    episodes = []
    total_start = time.monotonic()
    started = False
    try:
        env.start(gui=False)
        started = True
        metrics = SUMOMetrics(traci)
        with (directory / "steps.jsonl").open("w", encoding="utf-8") as raw:
            for index, seed in enumerate(seeds["episode_sumo_seeds"]):
                episode = index + 1
                history = job["traces"][method][str(episode)]
                divergence = first_divergence(history)
                branch_steps = ({first_divergence(job["traces"][name][str(episode)]) for name in config["attacks"]}
                                if method == "none" else {divergence}) - {None}
                start, forward_start, step_start = time.monotonic(), forwards[0], instrumentation.steps
                instrumentation.warmup = []
                instrumentation.resetting = True
                try:
                    obs = np.asarray(env.reset(), dtype=np.float64).copy()
                finally:
                    instrumentation.resetting = False
                reset = dict(observation=obs.tolist(), observable_state=observable_state(), safety=metrics.sample(),
                             warmup=instrumentation.warmup,
                             warmup_ego_collision_steps=[r["warmup_step"] for r in instrumentation.warmup if "Auto" in r["colliding_ids"]])
                warmup_steps = instrumentation.steps - step_start
                rows = []
                for step in range(config["max_steps"]):
                    before = obs.copy()
                    state = observable_state() if step in branch_steps else None
                    supplied, replayed = intervention(arm, step, before, history, divergence)
                    clean_action, action = victim.action(before), victim.action(supplied)
                    instrumentation.commands = []
                    obs, reward, done = env.step(action)[:3]
                    obs = np.asarray(obs, dtype=np.float64).copy()
                    delta = supplied - before
                    row = dict(episode=episode, step=step, observation=before.tolist(),
                               next_observation=obs.tolist(), adversarial_observation=supplied.tolist(),
                               perturbation=delta.tolist(), action=action, clean_action_at_visited_state=clean_action,
                               reward=float(reward), terminated=bool(done), attacked=bool(np.any(delta != 0)),
                               safety=metrics.sample(), attack_cost={}, attack_metadata={},
                               replayed_input=replayed, lane_change_commands=instrumentation.commands,
                               pre_state=state, pre_state_sha256=fingerprint(state) if state is not None else None)
                    if arm in ("no_attack", "sustained_reference") or (divergence is not None and step <= divergence):
                        require(step < len(history), "Controlled prefix exceeds historical trace")
                        assert_historical_step(row, history[step])
                    raw.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                    if done:
                        break
                result = episode_summary(rows, seed, reset, warmup_steps, forwards[0] - forward_start, time.monotonic() - start)
                require(instrumentation.steps - step_start == warmup_steps + len(rows), "Unaccounted SUMO steps")
                episodes.append(result)
                print("CONTROLLED_EPISODE %s %s %s episode=%d return=%.4f collision=%s" %
                      (job["job_id"], method, arm, episode, result["episode_return"], result["ego_collision_observed"]), flush=True)
    finally:
        if started:
            env.close()
        instrumentation.restore()
        hook.remove()
    victim.assert_frozen()
    require(forwards[0] == sum(e["resources"]["policy_forward_calls"] for e in episodes), "Unaccounted victim forward")
    write_json(directory / "episodes.json", episodes)
    write_json(directory / "resources.json", dict(sumo_simulation_step_calls=instrumentation.steps,
               policy_forward_calls=forwards[0], wall_seconds=time.monotonic() - total_start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output, config = args.manifest.parent, manifest["config"]
    for entry in manifest["jobs"]:
        job = json.loads(Path(entry["input_path"]).read_text(encoding="utf-8"))
        require(fingerprint(job) == entry["input_content_sha256"], "Prepared input changed")
        victim = VictimAdapter.from_reference(job["reference"], manifest["root"])
        run_arm(job, "none", "no_attack", victim, output, config)
        for method in config["attacks"]:
            for arm in ("single_then_clean", "sustained_reference"):
                run_arm(job, method, arm, victim, output, config)


if __name__ == "__main__":
    main()
