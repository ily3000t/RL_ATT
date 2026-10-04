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
from run_attack_final import dispatch
import run_attack_final as final_runner
from check_attack_final_execution import validate_certificate


class CertificateTests(unittest.TestCase):
    def test_certificate_rejects_changed_tooling(self):
        record = dict(kind="attack_final_execution_readiness", verified=True, final_execution_ready=True,
                      new_simulations=0, protocol_sha256="registered", tools_git_blob_sha256={}, inputs=[])
        with patch("check_attack_final_execution.sha256", return_value="registered"), \
             patch("check_attack_final_execution.tool_hashes", return_value={"tool": "changed"}):
            with self.assertRaises(ValueError):
                validate_certificate(record)


class DispatchTests(unittest.TestCase):
    def test_windows_status_sharing_retry_never_restarts_experiment(self):
        with patch.object(final_runner, "atomic_save", side_effect=[PermissionError(), None]) as save, \
             patch.object(final_runner.time, "sleep") as pause:
            final_runner.save(Path("status.json"), dict(status="running"))
            self.assertEqual(save.call_count, 2)
            pause.assert_called_once_with(.01)

    def test_anchor_passed_before_parallel_work(self):
        groups = [dict(status="pending") for _ in range(5)]
        called = []
        def worker(i):
            if i:
                self.assertEqual(groups[0]["status"], "passed")
            called.append(i)
            groups[i]["status"] = "passed"
        self.assertTrue(dispatch(groups, worker))
        self.assertEqual(called[0], 0)
        self.assertEqual(set(called), set(range(5)))

    def test_failed_anchor_blocks_all_remaining_groups(self):
        groups = [dict(status="pending") for _ in range(12)]
        called = []
        def worker(i):
            called.append(i)
            groups[i]["status"] = "failed"
        self.assertFalse(dispatch(groups, worker))
        self.assertEqual(called, [0])
        self.assertTrue(all(g["status"] == "not_run_after_failure" for g in groups[1:]))

    def test_failure_stops_new_dispatch_and_keeps_inflight_results(self):
        groups = [dict(status="pending") for _ in range(12)]
        called = []
        def worker(i):
            called.append(i)
            if i == 2:
                time.sleep(.05)
            groups[i]["status"] = "failed" if i == 1 else "passed"
        self.assertFalse(dispatch(groups, worker))
        self.assertEqual(set(called), {0, 1, 2})
        self.assertEqual(groups[2]["status"], "passed")


class FinalPipelineContractTests(unittest.TestCase):
    def exercise(self, fail_anchor=False):
        protocol = json.loads(PROTOCOL.read_text())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / ".local/runs/final"
            readiness = root / "readiness.json"
            readiness.write_text("{}")
            training_path = root / protocol["runtime_references"][0]["path"]
            training_path.parent.mkdir(parents=True)
            training_path.write_text(json.dumps(dict(environment_overrides={}, runtime_path_prepend=[], launch_command=["legacy-python"])))
            def process(command, **kwargs):
                configs = command[command.index("--configs")+1:command.index("--jobs")]
                registered = next(g for g in protocol["groups"] if g["configs"] == configs)
                batch = output / registered["id"]
                batch.mkdir()
                (batch / "batch.json").write_text("{}")
                kwargs["stdout"].write(("BATCH_DIR=" + str(batch) + "\n").encode())
                # A Windows observer can also race with replacement when opening
                # the status file; this is observer I/O, not a simulation failure.
                for _ in range(20):
                    try:
                        state = json.loads((output / "final-attack.json").read_text())
                        break
                    except PermissionError:
                        time.sleep(.01)
                else:
                    self.fail("Status observer could not reopen file")
                self.assertIsNotNone(state["final_exposure_declared_at_utc"])
                if registered["id"] != "basic_attack0":
                    self.assertEqual(state["groups"][0]["status"], "passed")
                return type("Result", (), dict(returncode=0))()
            def audit(command, **kwargs):
                target = Path(command[command.index("--output")+1])
                if "--group" not in command:
                    target.write_text("{}")
                    return
                group = next(g for g in protocol["groups"] if g["id"] == command[command.index("--group")+1])
                if fail_anchor and group["id"] == "basic_attack0":
                    raise RuntimeError("mock integrity audit failure")
                if group["id"] != "basic_attack0":
                    self.assertIn("--clean-reference", command)
                target.write_text(json.dumps(dict(verified=True, git_commit="mock", episodes=group["episodes"], raw_episode_verified_steps=10)))
            with patch.object(final_runner, "ROOT", root), \
                 patch.object(final_runner, "validate_certificate"), \
                 patch.object(final_runner, "preflight", return_value=dict(simulations_run=0)), \
                 patch.object(final_runner, "git", side_effect=lambda *args: "" if args[0] == "status" else "mock"), \
                 patch.object(final_runner.subprocess, "run", side_effect=process), \
                 patch.object(final_runner.subprocess, "check_call", side_effect=audit):
                if fail_anchor:
                    with self.assertRaises(ValueError):
                        final_runner.run(output, readiness, "mock")
                else:
                    final_runner.run(output, readiness, "mock")
            return json.loads((output / "final-attack.json").read_text())

    def test_whole_pipeline_records_anchor_audits_and_summary(self):
        result = self.exercise()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["verified_episodes"], 14500)
        self.assertTrue(all(g["status"] == "passed" for g in result["groups"]))
        self.assertIn("summary_sha256", result)

    def test_integrity_failure_cannot_publish_partial_matrix(self):
        result = self.exercise(True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["verified_episodes"], 0)
        self.assertNotIn("summary_sha256", result)
        self.assertTrue(all(g["status"] == "not_run_after_failure" for g in result["groups"][1:]))



if __name__ == "__main__":
    unittest.main()
