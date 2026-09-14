"""Execute a committed attack evaluation snapshot; called by the launch script."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from Environment.environment.envs.highway_env import HighwayEnv, traci
from rl_att.agents.victim_adapter import VictimAdapter
from rl_att.attacks.registry import AttackRegistry
from rl_att.utils.seeding import seed_manifest
from .configuration import validate_config
from .evaluator import AttackEvaluator
from .sumo_metrics import SUMOMetrics
from .results import write_json


def compare_legacy(root, reference, rows):
    summary = json.loads((root / reference["verification_summary"]).read_text(encoding="utf-8"))
    run = next(r for r in summary["runs"] if r["run_seed"] == reference["run_seed"])
    path = root / Path(run["evaluation_manifest"]).parent / "evaluation.json"
    legacy = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = next(c for c in legacy["checkpoints"] if c["episode"] == reference["training_episode"])
    if checkpoint["verification"]["checkpoint_sha256"] != reference["checkpoint_sha256"]:
        raise ValueError("Legacy comparison checkpoint mismatch")
    expected = checkpoint["episodes"]
    if len(expected) != len(rows):
        raise ValueError("Legacy comparison episode count differs")
    # Compare every original field, including the exact post-step trajectory digest.
    for old, new in zip(expected, rows):
        for key, value in old.items():
            if new[key] != value:
                raise ValueError("NoAttack equivalence failed at episode %s: %s" % (old["episode"], key))
    return {"passed": True, "episodes": len(rows), "legacy_result": str(path),
            "legacy_result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fields": sorted(expected[0]), "checkpoint_sha256": reference["checkpoint_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config, root, output = validate_config(manifest["config"]), Path(manifest["root"]), args.manifest.parent
    result = {"git_commit": manifest["git_commit"], "config": config, "runs": [], "gate_enabled": False}
    registry = AttackRegistry.defaults()
    for reference in manifest["victim_references"]:
        victim = VictimAdapter.from_reference(reference, root)
        base_rows = None
        for specification in config["attacks"]:
            run_seed, name = reference["run_seed"], specification["name"]
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            seeds = seed_manifest(run_seed, "controlled", config["episodes"], phase="evaluation")
            seeds.update(policy_rng="unused_greedy_argmax", attack_rng=(
                "unused_no_attack" if name == "none" else "unused_deterministic_gradient" if name == "fgsm"
                else "seedsequence_attack_phase3_episode_step_local_numpy" if name in ("random", "pgd")
                else "optimizer_and_discarded_torch_draws_reinitialized_each_attack"))
            env = HighwayEnv(sumo_seed_schedule=seeds["episode_sumo_seeds"])
            directory = output / ("%s-seed%d-%s" % (reference["victim"], run_seed, name))
            try:
                env.start(gui=False)
                rows, summary = AttackEvaluator(
                    env, victim, registry.create(name, **specification["parameters"]),
                    SUMOMetrics(traci, lookahead_m=config["lookahead_m"]), config, seeds,
                    specification["budget"]).run(directory, base_rows)
            finally:
                if traci.isLoaded():
                    env.close()
            entry = {"victim": reference["victim"], "run_seed": run_seed, "attack": specification,
                     "effective_seeds": seeds, "checkpoint_sha256": reference["checkpoint_sha256"],
                     "weights_sha256": reference["weights_sha256"], "frozen_unchanged": True,
                     "results_directory": directory.name, "summary": summary}
            if name == "none":
                base_rows = rows
                if config["verify_legacy_no_attack"]:
                    entry["legacy_equivalence"] = compare_legacy(root, reference, rows)
            result["runs"].append(entry)
            write_json(output / "evaluation.json", result)
            print("ATTACK_EVALUATED=%s seed=%d attack=%s" % (reference["victim"], run_seed, name), flush=True)


if __name__ == "__main__":
    main()
