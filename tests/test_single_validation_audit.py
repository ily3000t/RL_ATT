import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_single_validation import validate_reuse_sources, validate_runtime, ZOOPT_SHA256
from analyze_mechanism_controls import verify_shared_witnesses
sys.path.pop(0)


class SingleValidationAuditTests(unittest.TestCase):
    def test_source_reuse_rejects_policy_changes_and_unknown_runtime_files(self):
        old = {"rl_att/attacks/registry.py": "old", "rl_att/evaluation/configuration.py": "old", "oarl.py": "frozen"}
        new = dict(old, **{"rl_att/attacks/registry.py": "new", "rl_att/evaluation/configuration.py": "new",
                          "rl_att/attacks/single_attempt.py": "wrapper"})
        validate_reuse_sources(new, old)
        for path in ("oarl.py", "rl_att/attacks/unknown.py"):
            changed = dict(new)
            changed[path] = "unexpected"
            with self.assertRaises(ValueError):
                validate_reuse_sources(changed, old)

    def test_reused_controls_require_same_runtime_and_checkpoint(self):
        fields = ("python_runtime", "pip_freeze", "sumo_version", "victim_references")
        reference = {field: "frozen" for field in fields}
        reference["extra_dependencies"] = [dict(sha256=ZOOPT_SHA256)]
        current = dict(reference, extra_dependencies=[])
        validate_runtime(current, reference)
        for field in fields:
            wrong = copy.deepcopy(current)
            wrong[field] = "changed"
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_runtime(wrong, reference)

    def test_three_method_witness_proof_rejects_missing_method_and_changed_signature(self):
        names = ("zero_one_budgeted_return", "ours_single_return", "ours_progress_return")
        key = 1, (), 1, 0
        witnesses = {name: {key: (123, 1, .5)} for name in names}
        self.assertEqual(len(verify_shared_witnesses(witnesses, names)), 3)
        with self.assertRaises(ValueError):
            verify_shared_witnesses({names[0]: witnesses[names[0]]}, names)
        witnesses[names[1]][key] = (123, 2, .5)
        with self.assertRaisesRegex(ValueError, "primitive differs"):
            verify_shared_witnesses(witnesses, names)
