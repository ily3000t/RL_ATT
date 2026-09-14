"""Private SUMO subprocess. JSON IPC is used only by the owning evaluation."""

import base64
import copy
import json
import os
import pickle
import sys
import traceback
import numpy as np
from .replay_oracle import ReplayOracle


def main():
    # SUMO inherits OS descriptors, so redirect fd 1 as well as Python stdout.
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    from Environment.environment.envs.highway_env import HighwayEnv, traci
    env = HighwayEnv()
    snapshot = None

    def reset():
        attributes, rng = snapshot
        env.__dict__ = copy.deepcopy(attributes)
        np.random.set_state(rng)
        return dict(observation=np.asarray(env.reset()).tolist(), reward=0.0, done=False, collision=False)

    def step(action):
        obs, reward, done = env.step(action)[:3]
        return dict(observation=np.asarray(obs).tolist(), reward=float(reward), done=bool(done),
                    collision="Auto" in traci.simulation.getCollidingVehiclesIDList())

    oracle = ReplayOracle(reset, step)
    try:
        env.start(gui=False)
        for line in sys.stdin:
            message = json.loads(line)
            command = message["command"]
            if command == "reset":
                # Payload is created by this same local evaluator, never external input.
                snapshot = pickle.loads(base64.b64decode(message["snapshot"]))
                oracle.start_episode(message["initial"])
                result = oracle.counts
            elif command == "begin":
                result = oracle.begin()
            elif command == "step":
                result = oracle.step(message["action"])
            elif command == "observe":
                oracle.observe(message["action"], message["transition"])
                result = oracle.counts
            elif command == "counts":
                result = oracle.counts
            elif command == "close":
                channel.write(json.dumps(dict(ok=True, result=oracle.counts)) + "\n")
                channel.flush()
                break
            else:
                raise ValueError("Unknown oracle command")
            channel.write(json.dumps(dict(ok=True, result=result), allow_nan=False) + "\n")
            channel.flush()
    except Exception:
        channel.write(json.dumps(dict(ok=False, error=traceback.format_exc())) + "\n")
        channel.flush()
        raise
    finally:
        if traci.isLoaded():
            env.close()


if __name__ == "__main__":
    main()
