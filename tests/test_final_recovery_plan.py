import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plan_attack_final_recovery import (validate_header, saved_progress,
                                      compare_clean, audit_complete, validate_coverage)
from analyze_attack_final import expected_seeds
from prepare_attack_final_protocol import final_config, PROTOCOL
from rl_att.attacks.no_attack import NoAttack
from rl_att.evaluation.evaluator import AttackEvaluator


class FakeEnv:
    def reset(self):
        return np.zeros(16)

    def step(self, action):
        return np.ones(16), 1., True


class FakeVictim:
    def assert_frozen(self):
        pass

    def action(self, obs):
        return 0


class FakeMetrics:
    def sample(self):
        return dict(ego_collision_observed=False, ego_present=True, pairs=[])


class FixtureReader:
    def path(self, path):
        return path

    def json(self, path):
        return json.loads(path.read_text())

    def steps(self, path):
        return [json.loads(line) for line in path.read_text().splitlines()]


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.directory = Path(self.temp.name)
        # Unit fixture for the raw helper only. Production plan validates all
        # sixty registered configs before it can invoke this helper.
        self.config = final_config(0, None, 0)
        self.config['attacks'] = [self.config['attacks'][0]]
        self.result_dir = self.directory / 'clean-seed0-none'
        seeds = expected_seeds(self.config, 0, 'none')
        self.episodes, summary = AttackEvaluator(FakeEnv(), FakeVictim(), NoAttack(), FakeMetrics(),
            self.config, seeds, dict(norm='none')).run(self.result_dir)
        self.steps = FixtureReader().steps(self.result_dir / 'steps.jsonl')
        self.victim = dict(run_seed=0, checkpoint='dummy', checkpoint_sha256='checkpoint', weights_sha256='weights')
        self.entry = dict(victim='clean', run_seed=0, attack=self.config['attacks'][0], effective_seeds=seeds,
                          checkpoint_sha256='checkpoint', weights_sha256='weights', frozen_unchanged=True,
                          results_directory=self.result_dir.name, summary=summary)
        self.evaluation = dict(git_commit='original', config=self.config, gate_enabled=False, runs=[self.entry])
        self.manifest = dict(kind='frozen_attack_evaluation', git_commit='original', config=self.config,
                             victim_references=[self.victim], status='passed', returncode=0, extra_dependencies=[])
        for key in ('python_runtime', 'pip_freeze', 'sumo_version'):
            self.manifest[key] = dict(returncode=0, stdout=key)
        self.protocol = dict(victims=[self.victim], frozen_source_sha256={},
                             runtime_references=[dict(path=str(self.directory / 'training.json'))])
        for name, value in (('manifest.json', self.manifest), ('evaluation.json', self.evaluation),
                            ('training.json', self.manifest)):
            (self.directory / name).write_text(json.dumps(value))

    def tearDown(self):
        self.temp.cleanup()

    def test_closed_condition_uses_raw_auditor_and_clean_regression(self):
        with patch('plan_attack_final_recovery.verify_snapshot'), \
             patch('plan_attack_final_recovery.sha256', return_value='checkpoint'):
            result = audit_complete(self.directory, self.config, dict(gradient_cap=None, attack_seed=0),
                                    self.protocol, 'original', FixtureReader(), {0: self.directory})
        self.assertTrue(result['verified'])
        self.assertEqual((result['episodes'], result['raw_verified_steps'], result['clean_regression_steps']), (50, 50, 50))

    def test_bad_closed_header_rejected(self):
        for mutation in ('running', 'returncode', 'commit', 'model', 'gate', 'method'):
            manifest, evaluation = copy.deepcopy(self.manifest), copy.deepcopy(self.evaluation)
            if mutation == 'running':
                manifest['status'] = 'running'
            elif mutation == 'returncode':
                manifest['returncode'] = 1
            elif mutation == 'commit':
                evaluation['git_commit'] = 'new'
            elif mutation == 'model':
                manifest['victim_references'][0]['checkpoint_sha256'] = 'other'
            elif mutation == 'gate':
                evaluation['gate_enabled'] = True
            else:
                evaluation['runs'] = []
            with self.assertRaises(ValueError):
                validate_header(manifest, evaluation, self.config, self.victim, 'original')

    def test_saved_complete_episode_rows_never_become_reusable(self):
        result = saved_progress(self.config, self.directory, FixtureReader())
        self.assertEqual(result[0]['saved_episode_rows'], 50)
        self.assertTrue(result[0]['summary_present'])
        self.assertFalse(result[0]['reusable'])

    def test_partial_seed_and_episode_id_drift_rejected(self):
        for key in ('episode', 'sumo_seed'):
            rows = copy.deepcopy(self.episodes[:33])
            rows[-1][key] += 1
            (self.result_dir / 'episodes.json').write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                saved_progress(self.config, self.directory, FixtureReader())

    def test_clean_comparison_ignores_only_wall_time_and_attack_seed(self):
        rows, entry = copy.deepcopy(self.steps), copy.deepcopy(self.entry)
        rows[0]['attack_cost']['wall_seconds'] = 10.
        entry['effective_seeds']['attack_seed'] = 2
        original = copy.deepcopy(self.steps)
        self.assertEqual(compare_clean(entry, rows, self.entry, self.steps), 50)
        self.assertEqual(self.steps, original)
        for key in ('reward', 'action'):
            changed = copy.deepcopy(rows)
            changed[0][key] += 1
            with self.assertRaises(ValueError):
                compare_clean(entry, changed, self.entry, self.steps)

    def test_grid_rejects_duplicates_missing_and_wrong_disposition(self):
        protocol = json.loads(PROTOCOL.read_text())
        grid = [dict(config=p, group=g['id'], planned_episodes=50 * len(final_config(i, g['gradient_cap'], g['attack_seed'])['attacks']),
                     disposition='first_run') for g in protocol['groups'] for i, p in enumerate(g['configs'])]
        validate_coverage(grid, protocol)
        for mutation in ('missing', 'duplicate', 'episodes', 'disposition'):
            changed = copy.deepcopy(grid)
            if mutation == 'missing':
                changed.pop()
            elif mutation == 'duplicate':
                changed[-1] = changed[0]
            elif mutation == 'episodes':
                changed[-1]['planned_episodes'] -= 1
            else:
                changed[-1]['disposition'] = 'automatically_retry'
            with self.assertRaises(ValueError):
                validate_coverage(changed, protocol)


if __name__ == '__main__':
    unittest.main()
