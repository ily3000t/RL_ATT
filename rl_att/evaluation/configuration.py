"""Fail early on unsupported evaluation protocols and silent budget overrides."""

import math


def validate_envelope(budget):
    if set(budget) != {"norm", "epsilon", "relative_scale", "absolute_scale"} or budget["norm"] != "observation_scaled_linf":
        raise ValueError("Expected an explicit observation-scaled L-infinity envelope")
    if any(type(budget[k]) not in (int, float) or not math.isfinite(budget[k]) for k in
           ("epsilon", "relative_scale", "absolute_scale")):
        raise ValueError("Envelope parameters must be finite numbers")
    if budget["epsilon"] < 0 or budget["relative_scale"] < 0 or budget["absolute_scale"] <= 0:
        raise ValueError("Invalid envelope size")


def validate_config(config):
    required = {"victims", "run_seeds", "episodes", "max_steps", "action_selection", "gate_enabled",
                "sumo_schedule", "attacks", "lookahead_m", "metric_percentiles", "verify_legacy_no_attack"}
    if not required <= set(config) or set(config) - required - {"traffic_seed", "research_seeds"}:
        raise ValueError("Unexpected or missing evaluation configuration fields")
    if "traffic_seed" in config:
        if type(config["traffic_seed"]) is not int or not 0 <= config["traffic_seed"] < 5:
            raise ValueError("Diagnostic traffic_seed must be in 0 through 4")
        if config["verify_legacy_no_attack"]:
            raise ValueError("Crossed traffic uses explicit diagonal verification in diagnostic analysis")
        if [a["name"] for a in config["attacks"]] != ["none", "fgsm"]:
            raise ValueError("This diagnostic protocol uses deterministic Clean/FGSM controls")
    if config["gate_enabled"] is not False or config["action_selection"] != "greedy_argmax":
        raise ValueError("This stage requires greedy evaluation without Gate")
    if "research_seeds" in config:
        seeds = config["research_seeds"]
        if set(seeds) != {"root_seed", "split_id", "attack_seed", "episode_sumo_seeds"}:
            raise ValueError("Explicit research seed protocol required")
        if seeds["root_seed"] != 20260921 or type(seeds["split_id"]) is not int or seeds["split_id"] not in (10, 20, 30):
            raise ValueError("Use the frozen research split identifiers")
        if type(seeds["attack_seed"]) is not int or seeds["attack_seed"] not in (0, 1, 2):
            raise ValueError("Use independent attack replicate 0, 1 or 2")
        traffic = seeds["episode_sumo_seeds"]
        if not isinstance(traffic, list) or len(traffic) != config["episodes"] or len(set(traffic)) != len(traffic) or any(
                type(s) is not int or not 0 <= s < 2 ** 31 - 1 for s in traffic):
            raise ValueError("Explicit unique episode traffic seeds required")
        if config["sumo_schedule"] != "research_split_v1" or "traffic_seed" in config or config["verify_legacy_no_attack"]:
            raise ValueError("Research traffic cannot be mixed with the old held-out protocol")
    elif config["sumo_schedule"] != "phase_separated_derived":
        raise ValueError("Use the existing Protocol A evaluation seed derivation")
    for key in ("episodes", "max_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    for key in ("victims", "run_seeds"):
        if not config[key] or len(set(config[key])) != len(config[key]):
            raise ValueError(key + " must be nonempty and unique")
    if any(v not in ("clean", "oarl", "pgd_consistency") for v in config["victims"]):
        raise ValueError("Unknown frozen victim")
    if "pgd_consistency" in config["victims"] and config["verify_legacy_no_attack"]:
        raise ValueError("New defense policies have no upstream legacy evaluation reference")
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
            if attack["budget"] not in ({"norm": "affine_box", "multiplicative_bounds": [0.8, 1.2],
                                         "additive_bounds": [-0.05, 0.05]},
                                        {"norm": "observation_scaled_linf", "epsilon": 1.0,
                                         "relative_scale": 0.2, "absolute_scale": 0.05}):
                raise ValueError("Original BO uses its fixed affine box; epsilon projection would change it")
        elif attack["name"] in ("ours_return", "ours_safety", "zero_one_budgeted_return", "zero_one_budgeted_safety",
                              "ours_progress_return", "ours_progress_safety", "ours_single_return"):
            validate_envelope(attack["budget"])
            p = attack["parameters"]
            keys = {"epsilon", "relative_scale", "absolute_scale", "every_n_steps", "horizon", "evaluations",
                    "inner_steps", "step_size", "max_attempts", "objective", "resource_limits"}
            if attack["name"].startswith("ours_progress_"):
                keys.add("retry_rule")
                if p.get("retry_rule") != "strict_margin_progress":
                    raise ValueError("Explicit progress retry rule required")
            if set(p) != keys or p["objective"] != attack["name"].split("_")[-1]:
                raise ValueError("Search name must match its explicit objective and parameters")
            if any(type(p[k]) is not int or p[k] < 1 for k in ("horizon", "evaluations", "inner_steps", "max_attempts")):
                raise ValueError("Positive integer search parameters required")
            if attack["name"] == "ours_single_return" and p["max_attempts"] != 1:
                raise ValueError("Single-attempt control requires max_attempts=1")
            if p["every_n_steps"] != 1 or p["epsilon"] <= 0 or p["evaluations"] < 4:
                raise ValueError("Positive every-step budget and at least four candidates required")
            if type(p["step_size"]) not in (float, int) or not math.isfinite(p["step_size"]) or p["step_size"] <= 0:
                raise ValueError("Invalid inner PGD step size")
            resource_keys = {"gradient_evaluations", "policy_forward_calls", "new_shadow_transitions", "shadow_steps"}
            if set(p["resource_limits"]) != resource_keys or any(type(v) is not int or v < 0 for v in p["resource_limits"].values()):
                raise ValueError("Explicit nonnegative resource limits required")
            if p["resource_limits"]["policy_forward_calls"] < p["horizon"] + 2:
                raise ValueError("Reserve execution and emergency clean forward calls")
            if any(p[k] != attack["budget"][k] for k in ("epsilon", "relative_scale", "absolute_scale")):
                raise ValueError("Search envelope differs from declared perturbation budget")
            if "research_seeds" not in config:
                raise ValueError("New methods use separate development/validation/test traffic")
        elif attack["name"] in ("random", "fgsm", "pgd", "zero_one"):
            validate_envelope(attack["budget"])
            parameters = attack["parameters"]
            keys = {"epsilon", "relative_scale", "absolute_scale", "every_n_steps"}
            if attack["name"] == "zero_one":
                keys.update(("horizon", "evaluations", "inner_steps", "step_size", "objective"))
                if parameters.get("objective") != "targeted_logit_margin":
                    raise ValueError("Zero-One requires an explicit targeted logit objective")
                if any(type(parameters.get(k)) is not int or parameters[k] < 1
                       for k in ("horizon", "evaluations", "inner_steps")):
                    raise ValueError("Invalid Zero-One horizon or optimization budget")
                if parameters["evaluations"] < 4 or parameters.get("every_n_steps") != 1 or parameters.get("epsilon", 0) <= 0:
                    raise ValueError("Zero-One requires at least four candidates and a positive every-step budget")
                if type(parameters.get("step_size")) not in (int, float) or not math.isfinite(parameters["step_size"]) or parameters["step_size"] <= 0:
                    raise ValueError("Zero-One PGD step_size must be finite and positive")
            if attack["name"] in ("fgsm", "pgd"):
                keys.add("objective")
                if parameters.get("objective") != "untargeted_logit_margin":
                    raise ValueError("Gradient objective must be explicitly declared")
            if attack["name"] == "pgd":
                keys.update(("steps", "step_size"))
                if type(parameters.get("steps")) is not int or parameters["steps"] < 1:
                    raise ValueError("PGD steps must be positive")
                if type(parameters.get("step_size")) not in (int, float) or not math.isfinite(parameters["step_size"]) or parameters["step_size"] <= 0:
                    raise ValueError("PGD step_size must be finite and positive")
            if set(parameters) != keys or type(parameters["every_n_steps"]) is not int or parameters["every_n_steps"] < 1:
                raise ValueError("Invalid attack parameters or frequency")
            if any(parameters[k] != attack["budget"][k] for k in ("epsilon", "relative_scale", "absolute_scale")):
                raise ValueError("Attack implementation parameters differ from declared budget")
        else:
            raise ValueError("Attack is not implemented in this stage")
    if config["verify_legacy_no_attack"] and (config["episodes"], config["max_steps"]) != (20, 200):
        raise ValueError("Legacy comparison requires all 20 x 200 held-out episodes")
    return config
