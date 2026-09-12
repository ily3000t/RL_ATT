import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from oarl import ActorNet
from rl_att.utils.checkpoints import predictions, verify_checkpoint, weights_sha256


class CheckpointTests(unittest.TestCase):
    def test_reload_preserves_predictions_weights_and_rng(self):
        actor = ActorNet(16, 3, 128)
        obs = np.random.RandomState(3).normal(size=(8, 16)).astype(np.float32)
        expected = predictions(actor, obs).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pkl"
            torch.save(actor, str(path))
            rng = torch.get_rng_state().clone()
            result = verify_checkpoint(path, obs, expected, weights_sha256(actor))
            self.assertEqual(result["observation_count"], 8)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_checkpoint(path, obs, expected_weights="wrong")
            with self.assertRaisesRegex(ValueError, "predictions differ"):
                verify_checkpoint(path, obs, expected_probabilities=np.zeros((8, 3)))

    def test_nonfinite_weights_are_rejected(self):
        actor = ActorNet(16, 3, 128)
        actor.fc1.weight.data[0, 0] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pkl"
            torch.save(actor, str(path))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                verify_checkpoint(path, np.zeros((1, 16), dtype=np.float32))
