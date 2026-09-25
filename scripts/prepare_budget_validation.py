"""Freeze split-20 validation without changing the developed attacks or budgets."""

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from prepare_proposed_configs import protocol as seed_protocol, save

ROOT = Path(__file__).resolve().parents[1]
FROZEN_COMMIT = "57ae7e8"
REFERENCE_BATCH = "20260923T142718092754Z-attack-benchmark"


def validation_config(development, gradient_cap, attack_seed, traffic):
    config = copy.deepcopy(development)
    config["episodes"] = 20
    config["research_seeds"].update(split_id=20, attack_seed=attack_seed, episode_sumo_seeds=traffic)
    for attack in config["attacks"]:
        if attack["name"] != "none":
            attack["parameters"]["resource_limits"].update(gradient_evaluations=gradient_cap,
                                                         policy_forward_calls=2 * gradient_cap)
    return config


def prepare():
    seeds = json.loads((ROOT / "configs/research_seed_splits.json").read_text())
    if seed_protocol() != seeds:
        raise ValueError("Frozen traffic derivation or disjointness check changed")
    groups = []
    for cap in (400, 200, 100):
        for seed in range(3):
            files = []
            for checkpoint in range(5):
                base = json.loads((ROOT / ("configs/evaluation/budget_sweep_g200_attack0_seed%d.json" % checkpoint)).read_text())
                config = validation_config(base, cap, seed, seeds["splits"]["20"])
                name = "configs/evaluation/budget_validation_g%d_attack%d_seed%d.json" % (cap, seed, checkpoint)
                save(ROOT / name, config)
                files.append(name)
            groups.append(dict(gradient_cap=cap, forward_cap=2 * cap, attack_seed=seed, configs=files))
    references = []
    for checkpoint in range(5):
        path = ROOT / ".local/runs" / REFERENCE_BATCH / ("run-%d/manifest.json" % checkpoint)
        references.append(dict(checkpoint_seed=checkpoint, path=path.relative_to(ROOT).as_posix(),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    record = dict(kind="shared_compute_validation", frozen_method_commit=subprocess.check_output(
        ["git", "rev-parse", FROZEN_COMMIT], cwd=str(ROOT), universal_newlines=True).strip(),
        research_split_id=20, episodes=20, max_steps=200, checkpoint_seeds=list(range(5)), attack_seeds=[0, 1, 2],
        gradient_forward_caps=[[400, 800], [200, 400], [100, 200]], groups=groups,
        total_episodes=4500, attack_episodes=3600, repeated_clean_episodes=900, unique_clean_pairs=100,
        seed_protocol_sha256=hashlib.sha256((ROOT / "configs/research_seed_splits.json").read_bytes()).hexdigest(),
        victim_registry_sha256=hashlib.sha256((ROOT / "configs/frozen_victims.json").read_bytes()).hexdigest(),
        development_references=references,
        primary_endpoint="paired collision conversions among Clean noncollision episodes, separately for every budget/objective",
        limitations="Same scenario and frozen models; new traffic only. Attack seeds and objectives are correlated. No final-test outcomes.")
    save(ROOT / "configs/research/shared_compute_validation.json", record)
    print("FROZEN_VALIDATION_GRID configs=45 episodes=4500 split=20")


if __name__ == "__main__":
    prepare()
