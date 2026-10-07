"""Categorical clean-to-perturbed KL search in a clean-anchored feature box."""

import math
import numpy as np
import torch

from rl_att.training.state import require


DEFAULTS = dict(coefficient=0.1, multiplicative=0.2, additive=0.05, epsilon=1.0,
                steps=5, step_size=0.4, random_start=True, probability_floor=1e-8)


def validate_config(config):
    require(set(config) == set(DEFAULTS), "Declare every PGD consistency parameter")
    for key in ("coefficient", "multiplicative", "additive", "epsilon"):
        require(type(config[key]) in (int, float) and math.isfinite(config[key]) and config[key] >= 0,
                "Invalid nonnegative consistency parameter: " + key)
    require(type(config["steps"]) is int and config["steps"] > 0, "Positive integer PGD steps required")
    require(type(config["step_size"]) in (int, float) and math.isfinite(config["step_size"]) and config["step_size"] > 0,
            "Positive normalized PGD step size required")
    require(config["random_start"] is True, "This baseline declares one uniform random start")
    require(type(config["probability_floor"]) in (int, float) and
            0 < config["probability_floor"] < 1 / 3, "Invalid categorical probability floor")
    return dict(config)


def normalized_probability(probability, floor):
    require(probability.dim() == 2 and probability.shape[1] == 3 and
            bool(torch.isfinite(probability).all()) and bool((probability >= 0).all()) and
            bool((probability.sum(dim=1) > 0).all()), "Invalid three-action distribution")
    clipped = probability.clamp(min=floor)
    return clipped / clipped.sum(dim=1, keepdim=True)


def categorical_kl(reference, candidate, floor):
    # Caller explicitly controls stop-gradient on the reference distribution.
    p, q = normalized_probability(reference, floor), normalized_probability(candidate, floor)
    return (p * (p.log() - q.log())).sum(dim=1)


def search(actor, observation, clean_probability, rng, config):
    """Return detached per-row best observation and auditable actual search costs.

    Input gradients use autograd.grad, never accumulating parameter gradients.
    Zero perturbation is a candidate; ties preserve it. No categorical sampling.
    """
    config = validate_config(config)
    require(observation.dim() == 2 and observation.shape[1] == 16 and
            observation.device.type == "cpu" and bool(torch.isfinite(observation).all()), "Finite CPU 16D batch required")
    clean, reference = observation.detach(), clean_probability.detach()
    require(reference.shape[0] == clean.shape[0], "Clean probability batch mismatch")
    radius = config["epsilon"] * (config["multiplicative"] * clean.abs() + config["additive"])
    offset = torch.tensor(rng.uniform(-1., 1., tuple(clean.shape)), dtype=clean.dtype)
    best, best_score = clean.clone(), torch.zeros(clean.shape[0], dtype=clean.dtype)
    initial_score = None
    for index in range(config["steps"] + 1):
        with torch.enable_grad():
            offset = offset.detach().requires_grad_(index < config["steps"])
            candidate = clean + radius * offset
            values = categorical_kl(reference, actor(candidate, softmax_dim=-1), config["probability_floor"])
            require(bool(torch.isfinite(values).all()), "Non-finite PGD consistency objective")
            scores = values.detach()
            if initial_score is None:
                initial_score = scores.clone()
            better = scores > best_score
            best = torch.where(better.unsqueeze(1).expand_as(best), candidate.detach(), best)
            best_score = torch.max(best_score, scores)
            if index < config["steps"]:
                gradient = torch.autograd.grad(values.sum(), offset)[0]
                require(bool(torch.isfinite(gradient).all()), "Non-finite PGD consistency input gradient")
                offset = (offset + config["step_size"] * gradient.sign()).clamp(-1., 1.)
    violation = float(((best - clean).abs() - radius).max().item())
    require(violation <= 2e-6, "PGD perturbation exceeds its clean-anchored box")
    rows = int(clean.shape[0])
    return best.detach(), dict(gradient_evaluations=config["steps"],
        search_actor_forward_calls=config["steps"] + 1,
        search_observation_rows=(config["steps"] + 1) * rows,
        candidate_observations=(config["steps"] + 1) * rows,
        initial_kl_mean=float(initial_score.mean().item()), selected_kl_mean=float(best_score.mean().item()),
        max_box_violation=max(0., violation), selected_perturbed_rows=int(((best - clean).abs().sum(dim=1) > 0).sum().item()))
