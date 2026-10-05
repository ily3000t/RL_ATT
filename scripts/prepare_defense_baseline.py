"""Prepare controlled Clean/OARL defense controls without changing any policy."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_att.evaluation.configuration import validate_config
from rl_att.evaluation.seed_control import evaluation_seeds


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_config(root, smoke=False):
    basic = copy.deepcopy(read(root / "configs/evaluation/basic_validation_attack0_seed0.json"))
    search = read(root / "configs/evaluation/final_search_g400_attack0_seed0.json")
    selected = {a["name"]: a for a in search["attacks"]}
    if smoke:
        basic["attacks"] = [a for a in basic["attacks"] if a["name"] in ("none", "pgd")]
    else:
        basic["attacks"].append(copy.deepcopy(selected["zero_one_budgeted_return"]))
    basic["attacks"].append(copy.deepcopy(selected["ours_single_return"]))
    basic.update(victims=["clean", "oarl"], run_seeds=list(range(5)), episodes=1 if smoke else 5,
                 verify_legacy_no_attack=False)
    seed_spec = dict(root_seed=20260921, split_id=10, attack_seed=0, episode_sumo_seeds=[])
    # Derive from the registered development split, never from final split 30.
    registered = read(root / "configs/evaluation/zero_one_budgeted_development_attack0_seed0.json")
    seed_spec["episode_sumo_seeds"] = registered["research_seeds"]["episode_sumo_seeds"][:basic["episodes"]]
    basic["research_seeds"] = seed_spec
    validate_config(basic)
    evaluation_seeds(0, basic["episodes"], research_seeds=seed_spec)
    return basic


def build_protocol(root):
    configs = {"configs/evaluation/defense_oarl_smoke.json": build_config(root, True),
               "configs/evaluation/defense_oarl_development.json": build_config(root)}
    registry = read(root / "configs/frozen_victims.json")
    refs = [r for r in registry["victims"] if r["victim"] in ("clean", "oarl")]
    if len(refs) != 10 or {(r["victim"], r["run_seed"]) for r in refs} != {
            (v, s) for v in ("clean", "oarl") for s in range(5)}:
        raise ValueError("Exactly ten frozen matched training references required")
    protocol = dict(schema_version=1, kind="oarl_defense_baseline", gate_enabled=False,
        checkpoint_selection="episode_400_predeclared", victim_references=refs,
        groups=[dict(id="engineering_smoke" if "smoke" in p else "development_pilot", config=p,
                     config_sha256=canonical_hash(c), actual_episodes=10*c["episodes"]*len(c["attacks"]))
                for p, c in configs.items()],
        attack_mode="regenerate_against_each_actual_frozen_policy",
        adaptation_contract="Each victim is the attack's policy argument and the policy used for execution; do not replay Clean-targeted perturbations as adaptive evidence.",
        common_gradient_cap=400, common_perturbation_box=dict(relative_scale=.2, absolute_scale=.05, epsilon=1.),
        statistics=dict(unit="traffic_cluster", checkpoint_pairs=5, attack_repeats=1,
            primary_difference="OARL minus Clean for the same attack, checkpoint training seed and traffic",
            robustness_difference="OARL attack-induced return drop minus Clean attack-induced return drop",
            collision_denominators="Each policy has its own clean reference; compare raw collision rate as well as within-policy ASR.",
            efficacy_limit="Smoke is engineering only; five-traffic pilot is exploratory, no significance or general robustness claim."),
        source_commit_at_preparation=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root)).decode().strip(),
        reserved_traffic=dict(attack_final_split_30="never tune or use for defense development",
                             defense_final="Register a new, disjoint held-out split after defense freeze; not executable in this pilot."))
    return configs, protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Read-only reproducibility check")
    args = parser.parse_args()
    configs, protocol = build_protocol(ROOT)
    protocol_path = ROOT / "configs/research/defense_baseline.json"
    if args.check:
        existing = read(protocol_path)
        protocol["source_commit_at_preparation"] = existing["source_commit_at_preparation"]
        if existing != protocol or any(read(ROOT/p) != c for p, c in configs.items()):
            raise ValueError("Defense protocol or config changed")
    else:
        for p, value in list(configs.items()) + [("configs/research/defense_baseline.json", protocol)]:
            (ROOT/p).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print("DEFENSE_PROTOCOL_%s smoke=30 pilot=350 final_traffic_exposed=0" % ("VERIFIED" if args.check else "PREPARED"))


if __name__ == "__main__":
    main()
