"""Fail early on unsupported evaluation protocols and silent budget overrides."""

import math


def validate_config(config):
    required = {"victims", "run_seeds", "episodes", "max_steps", "action_selection", "gate_enabled",
                "sumo_schedule", "attacks", "lookahead_m", "metric_percentiles", "verify_legacy_no_attack"}
    if set(config) != required:
        raise ValueError("Unexpected or missing evaluation configuration fields")
    if config["gate_enabled"] is not False or config["action_selection"] != "greedy_argmax":
        raise ValueError("This stage requires greedy evaluation without Gate")
    if config["sumo_schedule"] != "phase_separated_derived":
        raise ValueError("Use the existing Protocol A evaluation seed derivation")
    for key in ("episodes", "max_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("victims", "run_seeds"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(key + " must be nonempty and unique")
    if any(v not in ("clean", "oarl") for v in config["victims"]):
        raise ValueError("Unknown frozen victim")
    if any(type(s) is not int or not 0 <= s < 5 for s in config["run_seeds"]):
        raise ValueError("The frozen registry contains run_seed 0 through 4")
    if not math.isfinite(config["lookahead_m"]) or config["lookahead_m"] <= 0:
        raise ValueError("lookahead_m must be positive")
    if set(config["metric_percentiles"]) != {"ttc_percentile", "drac_percentile"} or any(
            not math.isfinite(p) or not 0 <= p <= 100 for p in config["metric_percentiles"].values()):
        raise ValueError("Percentiles must be between 0 and 100")
    if type(config["verify_legacy_no_attack"]) is not bool:
        raise ValueError("verify_legacy_no_attack must be boolean")
    names = [a["name"] for a in config["attacks"]]
    if not names or names[0] != "none" or len(set(names)) != len(names):
        raise ValueError("A paired no-attack reference must run first; attacks must be unique")
    for attack in config["attacks"]:
        if set(attack) != {"name", "parameters", "budget"}:
            raise ValueError("Each attack requires name, parameters, and budget")
        if attack["name"] == "none":
            if attack["parameters"] or attack["budget"] != {"norm": "none"}:
                raise ValueError("NoAttack has no parameters or perturbation budget")
        elif attack["name"] == "oarl_bo":
            if set(attack["parameters"]) != {"evaluations", "every_n_steps"} or any(
                    type(v) is not int or v < 1 for v in attack["parameters"].values()):
                raise ValueError("BO requires explicit positive evaluations and every_n_steps")
            if attack["budget"] != {"norm": "affine_box", "multiplicative_bounds": [0.8, 1.2],
                                    "additive_bounds": [-0.05, 0.05]}:
                raise ValueError("Original BO uses its fixed affine box; epsilon projection would change it")
        else:
            raise ValueError("Attack is not implemented in this stage")
    if config["verify_legacy_no_attack"] and (config["episodes"], config["max_steps"]) != (20, 200):
        raise ValueError("Legacy comparison requires all 20 x 200 held-out episodes")
    return config
