"""Frozen CPU policy facade; input gradients remain available to attacks."""

import hashlib
from pathlib import Path
import numpy as np
import torch
from rl_att.utils.checkpoints import weights_sha256


class VictimAdapter:
    def __init__(self, actor, checkpoint=None, expected_file_hash=None, expected_weights=None):
        self.actor = actor.cpu().eval()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError("Non-finite victim parameter")
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.file_hash = expected_file_hash
        self.weight_hash = expected_weights or weights_sha256(self.actor)
        self.assert_frozen()

    @classmethod
    def from_reference(cls, reference, root):
        root = Path(root).resolve()
        path = (root / reference["checkpoint"]).resolve()
        if root not in path.parents:
            raise ValueError("Checkpoint must stay inside repository artifacts")
        # Hash before loading the locally generated pickle.
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference["checkpoint_sha256"]:
            raise ValueError("Checkpoint file hash mismatch")
        return cls(torch.load(str(path), map_location="cpu"), path,
                   reference["checkpoint_sha256"], reference["weights_sha256"])

    def probabilities(self, observation, input_grad=False):
        state = torch.as_tensor(observation, dtype=torch.float32, device="cpu")
        if state.dim() not in (1, 2) or state.shape[-1] != 16 or not bool(torch.isfinite(state).all()):
            raise ValueError("Expected finite 16D observation or batch")
        with torch.set_grad_enabled(input_grad):
            prob = self.actor(state, softmax_dim=state.dim() - 1)
        if prob.shape[-1] != 3 or not bool(torch.isfinite(prob).all()) or bool((prob < 0).any()):
            raise ValueError("Invalid victim probabilities")
        if not torch.allclose(prob.sum(-1), torch.ones_like(prob.sum(-1)), atol=1e-6, rtol=0):
            raise ValueError("Victim probabilities must sum to one")
        return prob

    def action(self, observation):
        if np.shape(observation) != (16,):
            raise ValueError("action expects one observation")
        return int(self.probabilities(observation).argmax().item())

    def assert_frozen(self):
        if self.actor.training or any(p.requires_grad for p in self.actor.parameters()):
            raise ValueError("Victim must remain in eval mode with frozen parameters")
        if weights_sha256(self.actor) != self.weight_hash:
            raise ValueError("Victim weights changed")
        if self.checkpoint and hashlib.sha256(self.checkpoint.read_bytes()).hexdigest() != self.file_hash:
            raise ValueError("Victim checkpoint file changed")
