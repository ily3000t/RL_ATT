import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rl_att.evaluation.controlled_replay import (audit_case, audit_run, episode_summary,
                                               fingerprint, first_divergence, intervention, summarize)


def make_arm(changed_steps=(), collision_step=None, replay_steps=(), reset_collision=False):
    reset = dict(observation=[0.] * 16, observable_state=dict(colliding_ids=[]),
                 warmup=[dict(warmup_step=1, colliding_ids=["Auto"] if reset_collision else [])],
                 warmup_ego_collision_steps=[1] if reset_collision else [])
    if reset_collision:
        reset["observable_state"]["colliding_ids"] = ["Auto"]
    rows = []
    for step in range(4):
        adv = [0.] * 16
        if step in changed_steps:
            adv[0] = .05
        state = dict(time_s=step)
        rows.append(dict(episode=1, step=step, observation=[0.] * 16, next_observation=[0.] * 16,
                         adversarial_observation=adv, perturbation=adv.copy(),
                         action=1 if step in changed_steps else 2, clean_action_at_visited_state=2,
                         reward=0. if step in changed_steps else 1., terminated=step == 3,
                         attacked=step in changed_steps, safety=dict(ego_present=True,
                         ego_collision_observed=step == collision_step, pairs=[]),
                         attack_cost={}, attack_metadata={}, replayed_input=step in replay_steps,
                         lane_change_commands=[], pre_state=state, pre_state_sha256=fingerprint(state)))
    return dict(rows=rows, episode=episode_summary(rows, 7, reset, 1, 8, 1.))


def fixture():
    baseline = make_arm()
    single = make_arm((1,), replay_steps=(1,))
    sustained = make_arm((1, 2), collision_step=3, replay_steps=range(4))
    return baseline, single, sustained


class ControlledReplayTests(unittest.TestCase):
    def test_off_path_historical_input_is_rejected(self):
        history = fixture()[2]["rows"]
        with self.assertRaisesRegex(ValueError, "off-path"):
            intervention("sustained_reference", 1, np.ones(16), history, 1)
        supplied, replayed = intervention("single_then_clean", 2, np.ones(16), history, 1)
        self.assertFalse(replayed)
        np.testing.assert_array_equal(supplied, np.ones(16))

    def test_single_intervenes_at_first_action_change_then_cleans_new_state(self):
        history = fixture()[2]["rows"]
        self.assertEqual(first_divergence(history), 1)
        supplied, replayed = intervention("single_then_clean", 0, np.zeros(16), history, 1)
        self.assertFalse(replayed)
        supplied, replayed = intervention("single_then_clean", 1, np.zeros(16), history, 1)
        self.assertTrue(replayed)
        self.assertEqual(supplied[0], .05)
        supplied[0] = 9
        self.assertEqual(history[1]["adversarial_observation"][0], .05)

    def test_paired_damage_and_policy_own_collision_denominator(self):
        b, s, a = fixture()
        result = audit_case(b, s, a, a["rows"], a["episode"], 200)
        self.assertEqual((result["single_return_drop"], result["sustained_return_drop"]), (1., 2.))
        self.assertEqual((result["single_conversion"], result["sustained_conversion"]), (False, True))
        b = make_arm(collision_step=3)
        result = audit_case(b, s, a, a["rows"], a["episode"], 200)
        self.assertFalse(result["asr_eligible"])
        self.assertFalse(result["sustained_conversion"])

    def test_branch_state_mismatch_and_prefix_corruption_fail(self):
        b, s, a = fixture()
        s["rows"][1]["pre_state_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Branch-point"):
            audit_case(b, s, a, a["rows"], a["episode"], 200)
        b, s, a = fixture()
        s["rows"][0]["reward"] = 9.
        s["episode"] = episode_summary(s["rows"], 7, s["episode"]["reset"], 1, 8, 1.)
        with self.assertRaisesRegex(ValueError, "Historical transition"):
            audit_case(b, s, a, a["rows"], a["episode"], 200)

    def test_late_reused_input_rejected_by_single_arm_audit(self):
        b, s, a = fixture()
        s["rows"][2]["replayed_input"] = True
        with self.assertRaisesRegex(ValueError, "exactly once"):
            audit_case(b, s, a, a["rows"], a["episode"], 200)

    def test_no_divergence_is_not_a_recovery_sample(self):
        b, s, a = make_arm(), make_arm(), make_arm(replay_steps=range(4))
        result = audit_case(b, s, a, a["rows"], a["episode"], 200)
        self.assertTrue(result["no_intervention"])
        self.assertIsNone(result["first_action_divergence"])
        self.assertEqual(result["single_return_drop"], 0.)

    def test_reset_collision_reported_without_redefining_old_asr(self):
        b, s, a = fixture()
        reset = make_arm(reset_collision=True)["episode"]["reset"]
        for arm in (b, s, a):
            arm["episode"]["reset"] = copy.deepcopy(reset)
        result = audit_case(b, s, a, a["rows"], a["episode"], 200)
        self.assertTrue(result["asr_eligible"])
        self.assertTrue(result["reset_ego_collision"])
        self.assertFalse(result["reset_clear_eligible"])
        summary = summarize([result, result])
        self.assertEqual(summary["independent_traffic_clusters"], 1)
        self.assertEqual(summary["collision_pairs"]["sustained_only"], 2)

    def test_auditor_recomputes_raw_steps_and_costs(self):
        b, s, a = fixture()
        temporary_root = Path(__file__).resolve().parents[1] / ".local/tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(temporary_root)) as temporary:
            root = Path(temporary)
            job = dict(job_id="clean-seed0", reference=dict(victim="clean", run_seed=0),
                       traces={"none": {"1": b["rows"]}, "pgd": {"1": a["rows"]}},
                       episodes={"none": [b["episode"]], "pgd": [a["episode"]]})
            path = root / "input.json"
            path.write_text(json.dumps(job), encoding="utf-8")
            manifest = dict(config=dict(kind="test", attacks=["pgd"], victims=["clean"], traffic_seeds=[7],
                       episodes_per_arm=1, max_steps=200, expected_execution_commit="old", expected_audit_sha256="bound", interpretation="test"),
                       git_commit="new", group="test", expected_cases=1, expected_physical_episodes=3,
                       historical_inputs_sha256={}, jobs=[dict(job_id=job["job_id"], input_path=str(path), input_content_sha256=fingerprint(job))])
            mpath = root / "manifest.json"
            mpath.write_text(json.dumps(manifest), encoding="utf-8")
            for method, name, arm in (("none", "no_attack", b), ("pgd", "single_then_clean", s), ("pgd", "sustained_reference", a)):
                directory = root / job["job_id"] / method / name
                directory.mkdir(parents=True)
                (directory / "episodes.json").write_text(json.dumps([arm["episode"]]), encoding="utf-8")
                (directory / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in arm["rows"]) + "\n", encoding="utf-8")
                (directory / "resources.json").write_text(json.dumps(dict(sumo_simulation_step_calls=5, policy_forward_calls=8, wall_seconds=2.)), encoding="utf-8")
            report = audit_run(mpath)
            self.assertTrue(report["verified"])
            self.assertEqual((report["physical_episodes"], report["paired_cases"], report["total_policy_forward_calls"]), (3, 1, 24))
            resources = root / job["job_id"] / "pgd/single_then_clean/resources.json"
            resources.write_text(json.dumps(dict(sumo_simulation_step_calls=5, policy_forward_calls=0, wall_seconds=2.)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "resource totals"):
                audit_run(mpath)


if __name__ == "__main__":
    unittest.main()
