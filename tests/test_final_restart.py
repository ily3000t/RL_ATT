import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from final_restart import validate_restart
from prepare_mechanism_controls import sha256


class FinalRestartTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / '.local/runs/original'
        self.directory.mkdir(parents=True)
        self.protocol_path = self.root / 'configs/research/attack_final_test.json'
        self.protocol_path.parent.mkdir(parents=True)
        self.config_path = self.root / 'configs/evaluation/final.json'
        self.config_path.parent.mkdir(parents=True)
        self.config_path.write_text('{}')
        self.config_key = self.config_path.relative_to(self.root).as_posix()
        self.protocol = dict(groups=[dict(id='group')], configs={self.config_key: 'config'})
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.original_path = self.directory / 'final-attack.json'
        self.original = dict(kind='attack_final_pipeline', status='running', planned_episodes=14500,
                             verified_episodes=1250, final_exposure_declared_at_utc='original_time',
                             git_commit='original', protocol_sha256=sha256(self.protocol_path))
        self.original_path.write_text(json.dumps(self.original))
        raw = self.directory / 'existing-raw.jsonl'
        raw.write_text('preserve interrupted evidence')
        self.inventory_path = self.directory / 'inventory.json'
        inventory = dict(kind='attack_final_interruption_inventory', git_commit='original', final_matrix_complete=False,
                         original_files_modified=False, groups=[dict(id='group', stage='interrupted')],
                         preserved_control_evidence=[dict(path=raw.relative_to(self.root).as_posix(), sha256=sha256(raw))])
        self.inventory_path.write_text(json.dumps(inventory))
        self.receipt_path = self.directory / 'receipt.json'
        receipt = dict(kind='attack_final_interruption_receipt', git_commit='original', final_matrix_complete=False,
                       original_files_modified=False, evidence_inventory_sha256=sha256(self.inventory_path), search_raw_fingerprints=[])
        self.receipt_path.write_text(json.dumps(receipt))
        self.precheck_path = self.directory / 'preflight.json'
        self.precheck_path.write_text(json.dumps(dict(verified=True, git_commit='original',
            protocol_sha256=sha256(self.protocol_path), config_raw_sha256={self.config_key: sha256(self.config_path)})))
        self.process_path = self.directory / 'processes.json'
        self.process_path.write_text(json.dumps(dict(original_dispatcher_absent=True, matching_original_processes=[])))
        self.authorization_path = self.directory / 'authorization.json'
        self.authorization = dict(kind='attack_final_full_restart_authorization', allow_full_matrix_rerun=True,
                                  preserve_original_outputs=True, restart_attempt=1, user_authorization='Explicit full restart request',
                                  protocol_sha256=sha256(self.protocol_path), original_pipeline=self.proof(self.original_path),
                                  interruption_receipt=self.proof(self.receipt_path), interruption_inventory=self.proof(self.inventory_path),
                                  original_preflight=self.proof(self.precheck_path), process_absence_evidence=self.proof(self.process_path))
        self.write_authorization()

    def tearDown(self):
        self.temp.cleanup()

    def proof(self, path):
        return dict(path=path.relative_to(self.root).as_posix(), sha256=sha256(path))

    def write_authorization(self):
        self.authorization_path.write_text(json.dumps(self.authorization))

    def validate(self):
        with patch('final_restart.ROOT', self.root):
            return validate_restart(self.authorization_path, self.protocol)

    def test_authorized_restart_preserves_original_and_declares_prior_exposure(self):
        before = self.original_path.read_bytes()
        result = self.validate()
        self.assertEqual(self.original_path.read_bytes(), before)
        self.assertTrue(result['traffic_previously_exposed'])
        self.assertFalse(result['reuse_prior_results'])
        self.assertEqual(result['original_first_exposure_at_utc'], 'original_time')
        self.assertIn(self.original_path.relative_to(self.root).as_posix(), result['source_sha256'])

    def test_missing_or_ambiguous_authorization_rejected(self):
        original = copy.deepcopy(self.authorization)
        for key in ('allow_full_matrix_rerun', 'preserve_original_outputs', 'user_authorization', 'process_absence_evidence'):
            self.authorization = copy.deepcopy(original)
            self.authorization.pop(key)
            self.write_authorization()
            with self.assertRaises(ValueError):
                self.validate()

    def test_completed_matrix_nested_restart_or_output_rejected(self):
        for mutation in ('passed', 'count', 'nested', 'summary'):
            value = copy.deepcopy(self.original)
            if mutation == 'passed':
                value['status'] = 'passed'
            elif mutation == 'count':
                value['verified_episodes'] = 14500
            elif mutation == 'nested':
                value['restart'] = dict(attempt=1)
            else:
                (self.directory / 'final-comparison.json').write_text('{}')
            self.original_path.write_text(json.dumps(value))
            self.authorization['original_pipeline'] = self.proof(self.original_path)
            self.write_authorization()
            with self.assertRaises(ValueError):
                self.validate()

    def test_changed_raw_evidence_or_config_rejected(self):
        raw = self.directory / 'existing-raw.jsonl'
        raw.write_text('changed')
        with self.assertRaises(ValueError):
            self.validate()
        raw.write_text('preserve interrupted evidence')
        self.config_path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):
            self.validate()

    def test_running_original_process_evidence_rejected(self):
        self.process_path.write_text(json.dumps(dict(original_dispatcher_absent=False, matching_original_processes=[123])))
        self.authorization['process_absence_evidence'] = self.proof(self.process_path)
        self.write_authorization()
        with self.assertRaises(ValueError):
            self.validate()

    def test_consumed_authorization_cannot_restart_again(self):
        consumed = self.root / '.local/runs/new/final-attack.json'
        consumed.parent.mkdir(parents=True)
        consumed.write_text(json.dumps(dict(restart=dict(authorization_sha256=sha256(self.authorization_path)))))
        with self.assertRaises(ValueError):
            self.validate()


if __name__ == '__main__':
    unittest.main()
