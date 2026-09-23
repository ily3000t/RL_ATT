from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from test_basic_attacks import linear_victim
from test_proposed_attack import BudgetOracle
from rl_att.attacks.base import AttackContext
from rl_att.attacks.progress_retry import ProgressRetryAttack

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_progress_retry import audit_stops, verify_v0_reference
from summarize_proposed_smoke import validate_attack_seed
sys.path.pop(0)


class ProgressRetryAuditTests(unittest.TestCase):
    def test_expected_replicate_rejects_mixed_or_mislabeled_seeds(self):
        config = {"research_seeds": {"attack_seed": 1}}
        validate_attack_seed(config, {"attack_seed": 1}, 1)
        for effective, expected in ((0, 1), (1, 0), (1, 2), (1, 3)):
            with self.assertRaises(ValueError):
                validate_attack_seed(config, {"attack_seed": effective}, expected)

    def test_clean_reference_allows_attack_rng_only_and_rejects_changed_traffic_or_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for replica in (0, 1):
                batch = dict(status="passed", git_commit="fixture", runs=[])
                for checkpoint in range(5):
                    directory = root / ("replica%d-checkpoint%d" % (replica, checkpoint))
                    (directory / "none").mkdir(parents=True)
                    manifest = dict(status="passed", python_runtime="same", pip_freeze="same", sumo_version="same")
                    entry = dict(run_seed=checkpoint, attack=dict(name="none"), results_directory="none",
                                 effective_seeds=dict(attack_seed=replica, episode_sumo_seeds=[123]),
                                 checkpoint_sha256=str(checkpoint), weights_sha256=str(checkpoint))
                    (directory / "manifest.json").write_text(json.dumps(manifest))
                    (directory / "evaluation.json").write_text(json.dumps(dict(runs=[entry])))
                    (directory / "none/steps.jsonl").write_text(json.dumps(dict(action=1, attack_cost=dict(wall_seconds=replica))) + "\n")
                    batch["runs"].append(dict(run_dir=str(directory)))
                path = root / ("batch%d.json" % replica)
                path.write_text(json.dumps(batch))
                paths.append(path)
            with patch("analyze_progress_retry.ROOT", root):
                self.assertEqual(verify_v0_reference(paths[1], paths[0], clean_only=True)["verified_steps"], 5)
                directory = root / "replica1-checkpoint0"
                evaluation_path = directory / "evaluation.json"
                evaluation = json.loads(evaluation_path.read_text())
                changed = copy.deepcopy(evaluation)
                changed["runs"][0]["effective_seeds"]["episode_sumo_seeds"] = [456]
                evaluation_path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "provenance differs"):
                    verify_v0_reference(paths[1], paths[0], clean_only=True)
                evaluation_path.write_text(json.dumps(evaluation))
                (directory / "none/steps.jsonl").write_text(json.dumps(dict(action=2, attack_cost=dict(wall_seconds=0))) + "\n")
                with self.assertRaisesRegex(ValueError, "behavior changed"):
                    verify_v0_reference(paths[1], paths[0], clean_only=True)

    def test_actual_stops_match_trace_and_tampering_is_rejected(self):
        obs = np.full(16, .2)
        obs[0] = .01
        result = ProgressRetryAttack(horizon=2)(obs, linear_victim(),
                                                AttackContext(0, 0, 0, 0, BudgetOracle(obs), 2))
        rows = [dict(attack_metadata=result.metadata)]
        self.assertGreater(audit_stops(rows), 0)
        stop = result.metadata["retry_stop_trace"][0]
        stop["margin"] += .1
        with self.assertRaisesRegex(ValueError, "Stop margin"):
            audit_stops(rows)
