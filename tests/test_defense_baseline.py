import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import hashlib
import io
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_defense_baseline import build_config, canonical_hash
from analyze_defense_baseline import paired_tradeoffs, exported_source_hashes, analyze_batch


class DefenseProtocolTests(unittest.TestCase):
    def test_same_traffic_and_frozen_search_for_both_policies(self):
        config = build_config(ROOT)
        self.assertEqual(config["victims"], ["clean", "oarl"])
        self.assertEqual(config["run_seeds"], list(range(5)))
        self.assertEqual(config["research_seeds"]["split_id"], 10)
        self.assertFalse(config["gate_enabled"])
        final = json.loads((ROOT / "configs/evaluation/final_search_g400_attack0_seed0.json").read_text())
        for name in ("zero_one_budgeted_return", "ours_single_return"):
            self.assertEqual(next(a for a in config["attacks"] if a["name"] == name),
                             next(a for a in final["attacks"] if a["name"] == name))

    def test_smoke_is_full_cohort_but_not_an_efficacy_sample(self):
        config = build_config(ROOT, True)
        self.assertEqual(config["episodes"], 1)
        self.assertEqual([a["name"] for a in config["attacks"]], ["none", "pgd", "ours_single_return"])
        self.assertEqual(config["max_steps"], 200)

    def test_canonical_hash_catches_attacker_change(self):
        a = build_config(ROOT)
        b = copy.deepcopy(a)
        b["attacks"][-1]["parameters"]["resource_limits"]["gradient_evaluations"] = 200
        self.assertNotEqual(canonical_hash(a), canonical_hash(b))

    def fixture(self):
        config = build_config(ROOT, True)
        config["run_seeds"] = [0]
        rows = []
        for victim in config["victims"]:
            for spec in config["attacks"]:
                clean = spec["name"] == "none"
                rows.append(dict(victim=victim, checkpoint_seed=0, attack=spec["name"], costs={"gradient_evaluations":0},
                    episode_rows=[dict(sumo_seed=config["research_seeds"]["episode_sumo_seeds"][0], steps=1,
                        episode_return=(10 if clean else 2) if victim == "clean" else (8 if clean else 4),
                        ego_collision_observed=(victim == "clean" and not clean),
                        safety=dict(minimum_ttc_s=None,ttc_low_percentile_s=None,drac_high_percentile_mps2=None))]))
        return rows, config

    def test_clean_penalty_and_robustness_gain_are_separate(self):
        rows, config = self.fixture()
        results = paired_tradeoffs(rows,config)
        self.assertEqual(results[0]["return_robust_minus_clean_mean"], -2)
        self.assertEqual(results[1]["return_robust_minus_clean_mean"], 2)
        self.assertEqual(results[1]["drop_robust_minus_clean_mean"], -4)
        self.assertIsNone(results[1]["populations"]["clean"]["safety"]["minimum_ttc_s"]["mean_of_valid_episode_values"])

    def test_policy_specific_asr_denominators(self):
        rows, config = self.fixture()
        next(r for r in rows if r["victim"] == "oarl" and r["attack"] == "none")["episode_rows"][0]["ego_collision_observed"] = True
        results = paired_tradeoffs(rows,config)
        self.assertEqual(results[1]["populations"]["clean"]["asr_eligible"], 1)
        self.assertEqual(results[1]["populations"]["oarl"]["asr_eligible"], 0)
        self.assertIsNone(results[1]["populations"]["oarl"]["asr"])
        self.assertEqual(results[1]["common_clean_noncollision_pairs"], 0)

    def test_reject_missing_duplicate_or_unpaired_data(self):
        for failure in ("missing","duplicate","traffic"):
            rows, config = self.fixture()
            if failure == "missing": rows.pop()
            elif failure == "duplicate": rows.append(copy.deepcopy(rows[0]))
            else: rows[0]["episode_rows"][0]["sumo_seed"] += 1
            with self.assertRaises(ValueError): paired_tradeoffs(rows,config)

    def test_source_audit_hashes_export_bytes_and_file_set(self):
        buffer = io.BytesIO()
        content = b'first\r\nsecond\r\n'
        with tarfile.open(fileobj=buffer,mode='w') as archive:
            member = tarfile.TarInfo('main.py')
            member.size = len(content)
            archive.addfile(member,io.BytesIO(content))
        with patch('analyze_defense_baseline.subprocess.check_output',return_value=buffer.getvalue()):
            self.assertEqual(exported_source_hashes('mock',['main.py']),{'main.py':hashlib.sha256(content).hexdigest()})
            with self.assertRaises(ValueError): exported_source_hashes('mock',['main.py','oarl.py'])

    def batch_fixture(self):
        protocol = json.loads((ROOT/'configs/research/defense_baseline.json').read_text())
        config = build_config(ROOT)
        groups = [g for g in protocol['groups'] if g['id'].startswith('development_seed')]
        batch = dict(status='passed', git_commit='execution', configs=[g['config'] for g in groups],
                     runs=[dict(config=g['config'], run_dir='run-%d'%i, returncode=0) for i,g in enumerate(groups)])
        children = []
        for seed in config['run_seeds']:
            rows = []
            for victim in config['victims']:
                for spec in config['attacks']:
                    rows.append(dict(victim=victim,checkpoint_seed=seed,attack=spec['name'],costs={'gradient_evaluations':0},
                        episode_rows=[dict(sumo_seed=s,steps=1,episode_return=10,ego_collision_observed=False,
                            safety=dict(minimum_ttc_s=None,ttc_low_percentile_s=None,drac_high_percentile_mps2=None))
                            for s in config['research_seeds']['episode_sumo_seeds']]))
            children.append(dict(rows=rows,real_steps=len(rows)*config['episodes'],protocol_sha256='hash',source_sha256={}))
        return protocol, config, batch, children

    def test_batch_requires_explicit_historical_commit_and_records_both_commits(self):
        protocol, config, batch, children = self.batch_fixture()
        with patch('analyze_defense_baseline.Reader') as reader, \
             patch('analyze_defense_baseline.git',return_value='auditor'), \
             patch('analyze_defense_baseline.subprocess.check_output',return_value=json.dumps(protocol).encode()), \
             patch('analyze_defense_baseline.analyze',side_effect=children) as audit:
            reader.return_value.json.side_effect = [batch,protocol]
            with self.assertRaises(ValueError): analyze_batch(Path('batch.json'))
            self.assertEqual(audit.call_count,0)
            reader.return_value.json.side_effect = [batch,protocol,config]
            reader.return_value.sources = {}
            result = analyze_batch(Path('batch.json'),'execution')
            self.assertEqual(result['git_commit'],'execution')
            self.assertEqual(result['audit_git_commit'],'auditor')
            self.assertEqual(result['actual_episodes'],350)
            self.assertTrue(all(call[0][-1] == 'execution' for call in audit.call_args_list))

    def test_historical_batch_rejects_changed_protocol_or_aggregate_config(self):
        protocol, config, batch, children = self.batch_fixture()
        changed = copy.deepcopy(config)
        changed['attacks'][-1]['parameters']['resource_limits']['gradient_evaluations'] += 1
        with patch('analyze_defense_baseline.Reader') as reader, \
             patch('analyze_defense_baseline.git',return_value='auditor'), \
             patch('analyze_defense_baseline.subprocess.check_output',return_value=json.dumps(protocol).encode()), \
             patch('analyze_defense_baseline.analyze',side_effect=children) as audit:
            modified_protocol = copy.deepcopy(protocol)
            modified_protocol['statistics']['efficacy_limit'] = 'invalid claim'
            reader.return_value.json.side_effect = [batch,modified_protocol]
            with self.assertRaises(ValueError): analyze_batch(Path('batch.json'),'execution')
            self.assertEqual(audit.call_count,0)
            reader.return_value.json.side_effect = [batch,protocol,changed]
            with self.assertRaises(ValueError): analyze_batch(Path('batch.json'),'execution')


if __name__ == "__main__":
    unittest.main()
