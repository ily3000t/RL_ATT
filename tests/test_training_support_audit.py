import copy
import unittest

from rl_att.training.support_audit import audit_equivalence


def modes():
    reference = dict(state_digest="final", episodes=[dict(episode=i, trajectory_sha256=str(i)) for i in range(1, 17)],
                     end_episode=16, begin_episode=0, counts=dict(real_interaction_steps=256, sumo_reset_warmup_steps=1312))
    continuous = copy.deepcopy(reference)
    continuous["boundary_digest"] = "boundary"
    continuous["counts"]["primary_updates"] = 40
    prefix = dict(state_digest="boundary", episodes=copy.deepcopy(reference["episodes"][:13]), end_episode=13, begin_episode=0,
                  counts=dict(real_interaction_steps=208, sumo_reset_warmup_steps=1066, primary_updates=16))
    resumed = dict(state_digest="final", episodes=copy.deepcopy(reference["episodes"][13:]), end_episode=16, begin_episode=13,
                   counts=dict(real_interaction_steps=48, sumo_reset_warmup_steps=246, primary_updates=24))
    return reference, continuous, prefix, resumed


class TrainingSupportAuditTests(unittest.TestCase):
    def test_full_state_and_split_trajectory_equivalence(self):
        audit_equivalence(*modes())

    def test_actor_only_agreement_cannot_hide_rng_or_optimizer_difference(self):
        values = modes()
        values[3]["state_digest"] = "different-full-state"
        with self.assertRaisesRegex(ValueError, "numeric training/RNG"):
            audit_equivalence(*values)

    def test_duplicate_episode_or_wrong_boundary_is_rejected(self):
        values = modes()
        values[3]["episodes"][0]["episode"] = 13
        with self.assertRaisesRegex(ValueError, "Episode trajectories"):
            audit_equivalence(*values)
        values = modes()
        values[3]["begin_episode"] = 12
        with self.assertRaisesRegex(ValueError, "boundary"):
            audit_equivalence(*values)

    def test_resume_prefix_cost_cannot_be_charged_twice(self):
        values = modes()
        values[3]["counts"]["real_interaction_steps"] = 256
        with self.assertRaisesRegex(ValueError, "Logical counts"):
            audit_equivalence(*values)

    def test_reference_reset_work_is_included(self):
        values = modes()
        values[0]["counts"]["sumo_reset_warmup_steps"] = 0
        with self.assertRaisesRegex(ValueError, "SUMO step"):
            audit_equivalence(*values)


if __name__ == "__main__":
    unittest.main()
