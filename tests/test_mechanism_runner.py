import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_mechanism_development as runner
from prepare_mechanism_controls import sha256
sys.path.pop(0)


class MechanismRunnerTests(unittest.TestCase):
    def exercise(self, fail, phase="development"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "configs/research").mkdir(parents=True)
            (root / ".local/runs").mkdir(parents=True)
            for name in ("research_seed_splits.json", "frozen_victims.json"):
                (root / "configs" / name).write_text("{}")
            reference_path = root / "runtime.json"
            reference_path.write_text(json.dumps(dict(environment_overrides={}, runtime_path_prepend=[], launch_command=["legacy"])))
            protocol = dict(frozen_method_commit="frozen", seed_splits_sha256=sha256(root / "configs/research_seed_splits.json"),
                            frozen_victims_sha256=sha256(root / "configs/frozen_victims.json"),
                            references=dict(original=dict(manifests=[dict(path="runtime.json", sha256=sha256(reference_path))])),
                            execution=dict(parallel_groups=2, jobs_per_group=5), total_episodes=750,
                            groups=[dict(gradient_cap=100, attack_seed=i, configs=[str(i)], episodes=250) for i in range(3)])
            candidate = phase == "candidate-validation"
            protocol_name = "single_candidate_validation.json" if candidate else "mechanism_development.json"
            (root / "configs/research" / protocol_name).write_text(json.dumps(protocol))
            output, activity_lock = root / ".local/runs/pipeline", threading.Lock()
            activity = dict(active=0, maximum=0)
            def git(*args):
                return "source" if args[0] == "rev-parse" else ""
            def launch(command, **kwargs):
                index = int(command[command.index("--configs") + 1])
                with activity_lock:
                    activity["active"] += 1
                    activity["maximum"] = max(activity["maximum"], activity["active"])
                time.sleep(.04 if index == 0 else .1)
                directory = root / ".local/runs" / ("batch%d" % index)
                directory.mkdir()
                (directory / "batch.json").write_text("{}")
                kwargs["stdout"].write(("BATCH_DIR=" + str(directory) + "\n").encode())
                with activity_lock:
                    activity["active"] -= 1
                return SimpleNamespace(returncode=1 if fail and index == 0 else 0)
            def check_call(command, **kwargs):
                if command[0] == "git":
                    return 0
                audit = Path(command[command.index("--output") + 1])
                if candidate:
                    self.assertIn(str(root / "scripts/analyze_single_validation.py"), command)
                    self.assertNotIn("--phase", command)
                    report = dict(verified=True, git_commit="source", kind="single_candidate_validation", new_episodes=250,
                                  rows=[dict(episodes=250), dict(episodes=250)], raw_episode_verified_steps=123)
                else:
                    report = dict(verified=True, git_commit="source", rows=[dict(episodes=250)], raw_episode_verified_steps=123)
                audit.write_text(json.dumps(report))
                return 0
            with patch.object(runner, "ROOT", root), patch.object(runner, "git", git), \
                    patch.object(runner.subprocess, "run", launch), patch.object(runner.subprocess, "check_call", check_call), \
                    patch.object(sys, "argv", ["runner", "--phase", phase, "--output", str(output)]):
                if fail:
                    with self.assertRaisesRegex(ValueError, "incomplete"):
                        runner.main()
                else:
                    runner.main()
                record = json.loads((output / ("candidate-validation.json" if candidate else "mechanism-development.json")).read_text())
                self.assertLessEqual(activity["maximum"], 2)
                if fail:
                    self.assertEqual(record["status"], "failed")
                    self.assertEqual(record["groups"][0]["status"], "failed")
                    self.assertEqual(record["groups"][2]["status"], "not_run_after_failure")
                    self.assertLess(record["verified_episodes"], 750)
                else:
                    self.assertEqual(record["status"], "passed")
                    self.assertEqual(record["verified_episodes"], 750)
                    self.assertTrue(all(g["status"] == "passed" for g in record["groups"]))
                    with self.assertRaises(SystemExit):
                        runner.main()

    def test_two_group_bound_and_verified_counts_without_overwriting_existing_pipeline(self):
        self.exercise(False)

    def test_failure_stops_pending_cells_and_keeps_partial_evidence(self):
        self.exercise(True)

    def test_candidate_phase_routes_auditor_and_counts_only_new_episodes(self):
        self.exercise(False, "candidate-validation")
