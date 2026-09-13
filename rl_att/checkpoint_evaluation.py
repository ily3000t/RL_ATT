"""Fresh-process verification and frozen, greedy-policy SUMO evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from Environment.environment.envs.highway_env import HighwayEnv, traci
from rl_att.utils.checkpoints import verify_checkpoint, weights_sha256
from rl_att.utils.seeding import seed_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = manifest["config"]
    run = Path(manifest["training_run"])
    training = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    run_seed = training["config"]["run_seed"]
    seeds = seed_manifest(run_seed, "controlled", config["episodes"], phase="evaluation")
    seeds.update(policy_rng="unused_greedy_argmax", attack_rng="unused_no_attack")
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    probes = np.load(str(run / "probe_observations.npy"))
    result = {"training_commit": training["git_commit"], "evaluation_commit": manifest["git_commit"],
              "victim": training["config"]["victim"], "run_seed": run_seed, "effective_seeds": seeds,
              "action_selection": "greedy_argmax", "attack": "none", "gate_enabled": False,
              "config": config, "checkpoints": []}
    for episode in config["checkpoints"]:
        proof = json.loads((run / "checkpoints" / ("policy%d.json" % episode)).read_text(encoding="utf-8"))
        checkpoint = Path(proof["checkpoint_path"])
        if hashlib.sha256((run / "probe_observations.npy").read_bytes()).hexdigest() != proof["probe_sha256"]:
            raise ValueError("Probe artifact hash mismatch")
        checked = verify_checkpoint(checkpoint, probes, proof["probabilities"], proof["weights_sha256"])
        if checked["checkpoint_sha256"] != proof["checkpoint_sha256"]:
            raise ValueError("Checkpoint file changed after training")
        actor = torch.load(str(checkpoint), map_location="cpu")
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        env = HighwayEnv(sumo_seed_schedule=seeds["episode_sumo_seeds"])
        rows = []
        try:
            env.start(gui=False)
            for index in range(config["episodes"]):
                obs = env.reset()
                score, done, collision = 0.0, False, False
                actions = [0, 0, 0]
                digest = hashlib.sha256()
                for step in range(config["max_steps"]):
                    with torch.no_grad():
                        prob = actor(torch.tensor(obs, dtype=torch.float32), softmax_dim=0)
                    if not bool(torch.isfinite(prob).all()):
                        raise FloatingPointError("Non-finite evaluation probabilities")
                    action = int(prob.argmax().item())
                    actions[action] += 1
                    obs, reward, done = env.step(action)[:3]
                    score += float(reward)
                    collision = collision or "Auto" in traci.simulation.getCollidingVehiclesIDList()
                    digest.update(np.asarray(obs, dtype=np.float64).tobytes())
                    digest.update(np.asarray([action, reward, done], dtype=np.float64).tobytes())
                    if done:
                        break
                rows.append({"episode": index + 1, "sumo_seed": seeds["episode_sumo_seeds"][index],
                             "episode_return": score, "steps": step + 1, "terminated": bool(done),
                             "ego_collision_observed": collision, "action_counts": actions,
                             "trajectory_sha256": digest.hexdigest()})
        finally:
            if traci.isLoaded():
                env.close()
        if weights_sha256(actor) != proof["weights_sha256"] or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != proof["checkpoint_sha256"]:
            raise ValueError("Frozen victim changed during evaluation")
        result["checkpoints"].append({"episode": episode, "verification": checked, "frozen_unchanged": True,
                                      "mean_return": float(np.mean([r["episode_return"] for r in rows])),
                                      "collision_episode_rate": float(np.mean([r["ego_collision_observed"] for r in rows])),
                                      "episodes": rows})
        (args.manifest.parent / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("CHECKPOINT_VALIDATED=%d" % episode, flush=True)


if __name__ == "__main__":
    main()
