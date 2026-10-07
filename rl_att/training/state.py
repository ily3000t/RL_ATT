"""CPU episode-boundary training snapshots, separate from inference actors."""

import copy
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch


NETWORKS = ("actor", "qf1", "qf2", "qf1_target", "qf2_target")
OPTIMIZERS = ("actor_optimizer", "qf1_optimizer", "qf2_optimizer", "dual_cst_optimizer")
HYPERPARAMETERS = ("state_dim", "act_dim", "action_numb", "gamma", "hidden_sizes", "buffer_size",
                   "batch_size", "actor_lr", "qf_lr", "dual_cst_lr", "attack_optimizing_times",
                   "attack_seed", "target_robust_error")
BUFFER_ARRAYS = ("obs1_buf", "obs2_buf", "acts_buf", "rews_buf", "done_buf")


def require(value, message):
    if not value:
        raise ValueError(message)


def global_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state().clone())


def restore_global_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def network_state(network):
    require(all(p.device.type == "cpu" for p in network.parameters()), "Training support is frozen CPU-only")
    return dict(weights=copy.deepcopy(network.state_dict()), training=network.training,
                requires_grad={n: p.requires_grad for n, p in network.named_parameters()},
                gradients={n: None if p.grad is None else p.grad.detach().clone() for n, p in network.named_parameters()})


def restore_network(network, state):
    network.load_state_dict(state["weights"], strict=True)
    network.train(state["training"])
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(state["requires_grad"][name])
        gradient = state["gradients"][name]
        parameter.grad = None if gradient is None else gradient.clone()


def replay_state(buffer):
    require(0 <= buffer.size <= buffer.max_size and 0 <= buffer.ptr < buffer.max_size,
            "Invalid replay cursor")
    require(buffer.size == buffer.max_size or buffer.ptr == buffer.size, "Invalid partially filled replay")
    # Unused slots are zero in upstream FIFO and never sampled; preserve them as zeros.
    return dict(ptr=buffer.ptr, size=buffer.size, max_size=buffer.max_size,
                arrays={n: getattr(buffer, n)[:buffer.size].copy() for n in BUFFER_ARRAYS})


def restore_replay(buffer, state):
    require(buffer.max_size == state["max_size"], "Replay capacity mismatch")
    require(0 <= state["size"] <= state["max_size"] and 0 <= state["ptr"] < state["max_size"] and
            (state["size"] == state["max_size"] or state["ptr"] == state["size"]), "Invalid saved replay cursor")
    for name in BUFFER_ARRAYS:
        array, saved = getattr(buffer, name), state["arrays"][name]
        require(saved.shape == (state["size"],) + array.shape[1:] and saved.dtype == array.dtype and
                np.isfinite(saved).all(), "Invalid replay array: " + name)
        array.fill(0)
        array[:state["size"]] = saved
    buffer.ptr, buffer.size = state["ptr"], state["size"]


def optimizer_state(optimizer):
    """Torch 1.3 state_dict stores process-specific object IDs; normalize by order."""
    state = copy.deepcopy(optimizer.state_dict())
    order = [parameter for group in state["param_groups"] for parameter in group["params"]]
    require(len(set(order)) == len(order), "Duplicate optimizer parameters")
    mapping = {parameter: index for index, parameter in enumerate(order)}
    state["state"] = {mapping[key]: value for key, value in state["state"].items()}
    for group in state["param_groups"]:
        group["params"] = [mapping[key] for key in group["params"]]
    return state


def agent_state(agent):
    return dict(hyperparameters={k: getattr(agent, k) for k in HYPERPARAMETERS},
                networks={n: network_state(getattr(agent, n)) for n in NETWORKS},
                optimizers={n: optimizer_state(getattr(agent, n)) for n in OPTIMIZERS},
                dual=agent.dual_cst.detach().clone(), dual_gradient=None if agent.dual_cst.grad is None else agent.dual_cst.grad.detach().clone(),
                replay=replay_state(agent.replay_buffer), js_float64_evaluations=agent.js_float64_evaluations)


def restore_agent(agent, state):
    require(state["hyperparameters"] == {k: getattr(agent, k) for k in HYPERPARAMETERS}, "Agent hyperparameters mismatch")
    for name in NETWORKS:
        restore_network(getattr(agent, name), state["networks"][name])
    agent.dual_cst.data.copy_(state["dual"])
    agent.dual_cst.grad = None if state["dual_gradient"] is None else state["dual_gradient"].clone()
    for name in OPTIMIZERS:
        getattr(agent, name).load_state_dict(state["optimizers"][name])
    restore_replay(agent.replay_buffer, state["replay"])
    agent.js_float64_evaluations = state["js_float64_evaluations"]


def tree_digest(value):
    """Stable exact tensor/array/scalar digest, independent of pickle bytes."""
    digest = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            digest.update(b"torch")
            add(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode("ascii"))
            add(item.shape)
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            add(len(item))
            for child in item:
                add(child)
        elif isinstance(item, (str, int, float, bool, type(None), np.generic)):
            digest.update((type(item).__name__ + ":" + repr(item)).encode("utf-8"))
            digest.update(b"\x00")
        else:
            raise ValueError("Unsupported digest state: " + str(type(item)))
    add(value)
    return digest.hexdigest()


def save_snapshot(path, payload):
    """Atomic new artifact + SHA returned to a manifest; never overwrite old state."""
    path = Path(path)
    require(not path.exists() and not path.with_suffix(path.suffix + ".tmp").exists(), "Snapshot already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    os.replace(str(temporary), str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_snapshot(path, expected_sha256, expected_identity):
    path = Path(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256, "Training snapshot file hash mismatch")
    payload = torch.load(str(path), map_location="cpu")
    require(isinstance(payload, dict) and payload.get("schema_version") == 1 and
            payload.get("kind") == "episode_boundary_training_state", "Not a full training snapshot")
    require(payload["identity"] == expected_identity and payload["boundary"] == "before_next_episode_reset",
            "Training snapshot protocol/source identity mismatch")
    return payload
