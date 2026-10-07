import copy
import unittest

import numpy as np
import torch

from oarl import ActorNet
from rl_att.defenses.pgd_consistency import DEFAULTS, categorical_kl, search
from rl_att.training.consistency_audit import audit_update, audit_training


def example():
    torch.manual_seed(67)
    actor = ActorNet(16, 3, 128)
    clean = torch.ones(4, 16)
    p = actor(clean, softmax_dim=-1)
    adversarial, measured = search(actor, clean, p, np.random.RandomState(5), DEFAULTS)
    q = actor(adversarial, softmax_dim=-1)
    regularizer = float(categorical_kl(p.detach(), q, 1e-8).clamp(min=0).mean().item())
    measured.update(base_actor_loss=-1., total_actor_loss=-1. + .1 * regularizer,
                    consistency_loss=regularizer, coefficient=.1, batch_size=4)
    return dict(training_update=1, metrics=measured, clean_observation=clean.tolist(),
                adversarial_observation=adversarial.tolist(), clean_probability=p.detach().tolist(),
                adversarial_probability=q.detach().tolist())


class ConsistencyAuditTests(unittest.TestCase):
    def test_numpy_recomputes_raw_torch_kl_and_bounds(self):
        self.assertTrue(audit_update(example(), DEFAULTS, 4)["kl"] > 0)

    def test_out_of_box_observation_cannot_hide_behind_summary(self):
        row = example()
        row["adversarial_observation"][0][0] = 9.
        with self.assertRaisesRegex(ValueError, "exceeds box"):
            audit_update(row, DEFAULTS, 4)

    def test_wrong_kl_direction_or_total_loss_is_rejected(self):
        row = example()
        p, q = np.array(row["clean_probability"]), np.array(row["adversarial_probability"])
        row["metrics"]["consistency_loss"] = float((q * (np.log(q) - np.log(p))).sum(axis=1).mean()) + .001
        with self.assertRaisesRegex(ValueError, "Raw KL"):
            audit_update(row, DEFAULTS, 4)
        row = example()
        row["metrics"]["total_actor_loss"] += .01
        with self.assertRaisesRegex(ValueError, "total actor loss"):
            audit_update(row, DEFAULTS, 4)

    def test_missing_gradient_work_and_invalid_distribution_are_rejected(self):
        row = example()
        row["metrics"]["gradient_evaluations"] -= 1
        with self.assertRaisesRegex(ValueError, "cost mismatch"):
            audit_update(row, DEFAULTS, 4)
        row = example()
        row["clean_probability"][0][0] = 0
        with self.assertRaisesRegex(ValueError, "probabilities"):
            audit_update(row, DEFAULTS, 4)

    def test_missing_update_rows_are_rejected_before_summary(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            audit_training([], DEFAULTS, dict(counts=dict(updates=2)),
                           dict(counts=dict(primary_updates=2)), 1)


if __name__ == "__main__":
    unittest.main()
