import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_attack_final_protocol import PROTOCOL
from summarize_attack_final import validate_reports, build_result


def synthetic_reports():
    protocol = json.loads(PROTOCOL.read_text())
    reports = []
    traffic = protocol["seed_protocol"]["splits"]["30"]
    for group in protocol["groups"]:
        names = [a["name"] for a in json.loads((ROOT / group["configs"][0]).read_text())["attacks"]]
        rows = []
        for model in range(5):
            victim = protocol["victims"][model]
            for name in names:
                value = -2. if name == "ours_single_return" else -1. if name == "fgsm" else 0.
                ep = [dict(episode=i+1, sumo_seed=s, episode_return=value, ego_collision_observed=False, steps=1,
                           attacked_steps=int(name != "none"), changed_steps=0, action_changed_steps=0,
                           linf_max=0., l2_max=0., scaled_linf_max=0., attack_wall_seconds=0.,
                           safety=dict(minimum_ttc_s=None, ttc_low_percentile_s=None, drac_high_percentile_mps2=None)) for i, s in enumerate(traffic)]
                rows.append(dict(checkpoint_seed=model, attack=name, episodes=50, checkpoint_sha256=victim["checkpoint_sha256"],
                                 episode_rows=ep, summary=dict(return_drop_mean=-value, attack_success_eligible_episodes=50, attack_successes=0),
                                 costs=dict(gradient_evaluations=50, policy_forward_calls=100, evaluator_policy_forward_calls=100,
                                            total_policy_forward_calls=200, new_shadow_transitions=0, shadow_steps=0)))
        reports.append(dict(kind="attack_final_group", verified=True, git_commit="mock", group_id=group["id"],
                            protocol_sha256=__import__("prepare_mechanism_controls").sha256(PROTOCOL), split_id=30,
                            gradient_cap=group["gradient_cap"], attack_seed=group["attack_seed"], episodes=group["episodes"],
                            clean_regression=dict(passed=True), rows=rows))
    return protocol, reports


class FinalMatrixTests(unittest.TestCase):
    def test_complete_table_and_deterministic_actual_counts(self):
        protocol, reports = synthetic_reports()
        result = build_result(reports, protocol, "mock")
        self.assertEqual(len(result["table"]), 17)
        fgsm = next(r for r in result["table"] if r["attack"] == "fgsm")
        self.assertEqual(fgsm["actual_episodes"], 250)
        c = next(c for c in result["contrasts"] if c["second"] == "fgsm")
        self.assertEqual(c["actual_episodes"], dict(first=750, second=250))
        self.assertEqual(c["mean_delta"], -1.)
        self.assertEqual(c["traffic_clusters"], 50)
        self.assertEqual(len(c["leave_one_traffic_out"]), 50)
        self.assertIsNone(fgsm["safety"]["minimum_ttc_s"]["mean_of_episode_values"])

    def test_missing_mislabeled_or_duplicate_grid_rejected(self):
        for change in ("group", "method", "duplicate", "traffic"):
            protocol, reports = synthetic_reports()
            if change == "group":
                reports.pop()
            elif change == "method":
                reports[0]["rows"].pop()
            elif change == "duplicate":
                reports[0]["rows"][1] = reports[0]["rows"][0]
            else:
                reports[0]["rows"][0]["episode_rows"][0]["sumo_seed"] = 123
            with self.assertRaises(ValueError):
                validate_reports(reports, protocol, "mock")


if __name__ == "__main__":
    unittest.main()
