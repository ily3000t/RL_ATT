import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from rl_att.evaluation.defense_diagnostics import diagnose_pair, index_grid, summarize_cases, verify_trace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_defense_pilot import AuditedReader


def trace(length, changed=False, collision=None):
    return [dict(episode=1, step=i, observation=[0.] * 16, next_observation=[0.] * 16,
                 adversarial_observation=[0.] * 16, perturbation=[0.] * 16,
                 action=1 if changed else 2, clean_action_at_visited_state=2,
                 attacked=changed, reward=1., terminated=i == length - 1,
                 safety=dict(ego_present=i != collision, ego_collision_observed=i == collision, pairs=[]),
                 attack_cost={}, attack_metadata={}) for i in range(length)]


def episode(rows):
    digest = hashlib.sha256()
    for r in rows:
        digest.update(np.asarray(r["next_observation"], dtype=np.float64).tobytes())
        digest.update(np.asarray([r["action"], r["reward"], r["terminated"]], dtype=np.float64).tobytes())
    return dict(episode=1, sumo_seed=42, steps=len(rows), episode_return=sum(r["reward"] for r in rows),
                action_counts=[sum(r["action"] == a for r in rows) for a in range(3)],
                trajectory_sha256=digest.hexdigest(), terminated=rows[-1]["terminated"],
                truncated=not rows[-1]["terminated"],
                ego_collision_observed=any(r["safety"]["ego_collision_observed"] for r in rows),
                action_changed_steps=sum(r["action"] != r["clean_action_at_visited_state"] for r in rows),
                attacked_steps=sum(r["attacked"] for r in rows),
                changed_steps=sum(any(r["perturbation"]) for r in rows))


def case(length=3, collision=2, baseline_collision=None):
    clean, attack = trace(length, collision=baseline_collision), trace(length, changed=True, collision=collision)
    return diagnose_pair(clean, attack, episode(clean), episode(attack))


class DefenseDiagnosticTests(unittest.TestCase):
    def test_collision_conversion_uses_policy_own_baseline(self):
        good = case()
        already_colliding = case(baseline_collision=2)
        self.assertTrue(good["collision_conversion"])
        self.assertFalse(already_colliding["asr_eligible"])
        self.assertFalse(already_colliding["collision_conversion"])
        summary = summarize_cases([good, already_colliding])
        self.assertEqual((summary["collision_episodes"], summary["asr_eligible"], summary["collision_conversions"]), (2, 1, 1))
        self.assertEqual(summary["asr"], 1.)

    def test_early_and_late_timings_use_real_steps(self):
        third, fourth = case(4, 2), case(4, 3)
        twentieth, twenty_first = case(21, 19), case(21, 20)
        self.assertEqual(third["first_collision_real_step"], 3)
        self.assertEqual(third["collision_timing"], "first_3_real_steps")
        self.assertEqual(fourth["collision_timing"], "real_steps_4_to_20")
        self.assertEqual(twentieth["collision_timing"], "real_steps_4_to_20")
        self.assertEqual(twenty_first["collision_timing"], "real_step_21_or_later")

    def test_post_divergence_state_differences_are_not_cross_compared(self):
        clean, attack = trace(3), trace(3, changed=True)
        attack[0]["next_observation"][13] = .1
        attack[1]["observation"][13] = .1
        attack[1]["adversarial_observation"][13] = .1
        detail = diagnose_pair(clean, attack, episode(clean), episode(attack))
        self.assertEqual(detail["first_action_divergence"], 0)
        self.assertEqual(detail["lane_index_transition_steps"], [0, 1])

    def test_post_divergence_corruption_is_still_rejected(self):
        clean, attack = trace(3), trace(3, changed=True)
        attack[2]["step"] = 4
        with self.assertRaises(ValueError):
            diagnose_pair(clean, attack, episode(clean), episode(attack))
        attack = trace(3, changed=True)
        attack[2]["observation"][0] = .3
        attack[2]["adversarial_observation"][0] = .3
        with self.assertRaises(ValueError):
            diagnose_pair(clean, attack, episode(clean), episode(attack))

    def test_unchanged_prefix_transition_mismatch_is_rejected(self):
        clean, attack = trace(3), trace(3)
        attack[1]["reward"] = 2.
        with self.assertRaises(ValueError):
            diagnose_pair(clean, attack, episode(clean), episode(attack))

    def test_invalid_terminal_boundary_and_digest_are_rejected(self):
        rows = trace(3)
        stored = episode(rows)
        stored["trajectory_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            verify_trace(rows, stored, 200)
        rows[0]["terminated"] = True
        with self.assertRaises(ValueError):
            verify_trace(rows, episode(rows), 200)
        rows = trace(3)
        rows[-1]["terminated"] = False
        with self.assertRaises(ValueError):
            verify_trace(rows, episode(rows), 200)

    def test_state_local_action_count_denominator_is_pooled_steps(self):
        one, ten = case(1, 0), case(10, None)
        ten["local_action_changes"] = 0
        ten["local_action_counts"] = [[0, 0, 0], [0, 0, 0], [0, 0, 10]]
        summary = summarize_cases([one, ten])
        self.assertAlmostEqual(summary["action_change_rate"], 1 / 11)
        self.assertEqual(summary["local_action_counts"][2], [0, 1, 10])

    def test_no_eligible_samples_have_null_asr(self):
        summary = summarize_cases([case(baseline_collision=2)])
        self.assertIsNone(summary["asr"])
        self.assertEqual(summary["early_collision_conversions"], 0)

    def test_grid_rejects_missing_duplicate_and_wrong_cohort(self):
        entries = [dict(victim="clean", run_seed=0, attack=dict(name=name)) for name in ("none", "pgd")]
        self.assertEqual(len(index_grid(entries, ["clean"], [0], ["none", "pgd"])), 2)
        for wrong in (entries[:1], entries + [entries[0]], [dict(entries[0], run_seed=2), entries[1]]):
            with self.assertRaises(ValueError):
                index_grid(wrong, ["clean"], [0], ["none", "pgd"])

    def test_reader_rejects_modified_or_unregistered_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "trace.json"
            original = b'{"verified": true}'
            source.write_bytes(original)
            reader = AuditedReader(root, {"trace.json": hashlib.sha256(original).hexdigest()})
            self.assertEqual(reader.json(source), dict(verified=True))
            source.write_bytes(b'{"verified": false}')
            with self.assertRaises(ValueError):
                reader.json(source)
            with self.assertRaises(ValueError):
                reader.bytes(root / "unregistered.json")
            with self.assertRaises(ValueError):
                reader.bytes(root.parent / "outside.json")


if __name__ == "__main__":
    unittest.main()
