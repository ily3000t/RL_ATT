"""Validation of locally produced inference checkpoints."""

import hashlib
from pathlib import Path
import numpy as np
import torch


def weights_sha256(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predictions(actor, observations):
    with torch.no_grad():
        values = actor(torch.as_tensor(observations, dtype=torch.float32), softmax_dim=1)
    if values.shape != (len(observations), 3) or not torch.isfinite(values).all():
        raise ValueError("Checkpoint produced invalid action probabilities")
    if not torch.allclose(values.sum(1), torch.ones(len(observations)), atol=1e-6, rtol=0):
        raise ValueError("Checkpoint probabilities do not sum to one")
    return values


def verify_checkpoint(path, observations, expected_probabilities=None, expected_weights=None):
    path = Path(path)
    actor = torch.load(str(path), map_location="cpu")
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
        if not torch.isfinite(parameter).all():
            raise ValueError("Checkpoint contains non-finite parameters")
    if actor.fc1.in_features != 16 or actor.pi.out_features != 3:
        raise ValueError("Unexpected victim architecture")
    probability = predictions(actor, observations)
    digest = weights_sha256(actor)
    if expected_weights is not None and digest != expected_weights:
        raise ValueError("Checkpoint weight hash mismatch")
    if expected_probabilities is not None:
        expected = torch.as_tensor(np.asarray(expected_probabilities), dtype=torch.float32)
        if not torch.equal(probability, expected):
            raise ValueError("Reloaded checkpoint predictions differ")
    return {
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "weights_sha256": digest,
        "observation_count": len(observations),
        "probabilities": probability.tolist(),
        "greedy_actions": probability.argmax(1).tolist(),
        "finite_parameters": True,
        "checkpoint_type": "inference_actor_only",
    }
