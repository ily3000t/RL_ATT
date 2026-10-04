import copy
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_attack_final_protocol import validate_runtime, assert_unexposed, preflight


class FinalPreflightTests(unittest.TestCase):
    def test_runtime_drift_and_failed_probe_rejected(self):
        probes = {k: dict(returncode=0, stdout=k) for k in ("python_runtime", "pip_freeze", "sumo_version")}
        validate_runtime(probes, [probes] * 5)
        for key in probes:
            changed = copy.deepcopy(probes)
            changed[key]["stdout"] = "different"
            with self.assertRaises(ValueError):
                validate_runtime(changed, [probes])
            changed = copy.deepcopy(probes)
            changed[key]["returncode"] = 1
            with self.assertRaises(ValueError):
                validate_runtime(changed, [probes])

    def test_possible_exposure_or_scan_error_fails_closed(self):
        for code in (0, 2):
            with patch("check_attack_final_protocol.subprocess.run", return_value=
                       subprocess.CompletedProcess([], code, "manifest.json", "scan error")):
                with self.assertRaises(ValueError):
                    assert_unexposed()
        with patch("check_attack_final_protocol.subprocess.run", return_value=
                   subprocess.CompletedProcess([], 1, "", "")):
            self.assertEqual(assert_unexposed()["recorded_run_metadata_matches"], 0)

    def test_dirty_or_wrong_execution_commit_rejected_before_any_probe(self):
        for answers in ([" M code.py"], ["", "different_sha"]):
            with patch("check_attack_final_protocol.git", side_effect=answers), \
                 patch("check_attack_final_protocol.capture") as probe:
                with self.assertRaises(ValueError):
                    preflight("expected_sha")
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
