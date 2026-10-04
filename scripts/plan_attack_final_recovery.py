"""Audit existing final conditions and plan recovery without launching episodes.

The planning commit is distinct from the immutable original experiment commit.
This tool neither marks incomplete batches passed nor implements a resume path.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from analyze_attack_final import (ROOT, PROTOCOL, Reader, SEARCH, SEARCH_COSTS,
                                 audit_raw, audit_witnesses, audit_oracle,
                                 expected_seeds, verify_prefix, verify_sources)
from analyze_mechanism_controls import verify_shared_witnesses
from check_attack_final_execution import TOOLS
from prepare_attack_final_protocol import canonical_sha256, validate_registration
from prepare_mechanism_controls import sha256
from rl_att.evaluation.sumo_metrics import summarize_safety
from run_baseline import git
from summarize_proposed_smoke import require


def validate_header(manifest, evaluation, config, victim, execution_commit):
    require(manifest['kind'] == 'frozen_attack_evaluation' and
            manifest['git_commit'] == evaluation['git_commit'] == execution_commit,
            'Mixed experiment commit')
    require(manifest['config'] == evaluation['config'] == config and
            not evaluation['gate_enabled'] and manifest['victim_references'] == [victim],
            'Run config/model/Gate differs')
    require(manifest['status'] == 'passed' and manifest['returncode'] == 0,
            'Only fully closed configurations are reusable')
    require([r['attack'] for r in evaluation['runs']] == config['attacks'],
            'Missing/reordered completed methods')


def saved_progress(config, directory, reader):
    """Inventory saved rows only; lack of closed oracle prevents acceptance."""
    progress = []
    for spec in config['attacks']:
        name = spec['name']
        result = directory / ('clean-seed%d-%s' % (config['run_seeds'][0], name))
        path = result / 'episodes.json'
        rows = reader.json(path) if path.exists() else []
        require(len(rows) <= config['episodes'] and
                [r['episode'] for r in rows] == list(range(1, len(rows) + 1)) and
                [r['sumo_seed'] for r in rows] == config['research_seeds']['episode_sumo_seeds'][:len(rows)],
                'Invalid saved episode identities')
        raw = result / 'steps.jsonl'
        if raw.exists():
            reader.path(raw)
        oracle = directory / (result.name + '-oracle') / 'manifest.json'
        progress.append(dict(attack=name, saved_episode_rows=len(rows),
                             summary_present=(result / 'summary.json').exists(),
                             closed_oracle_manifest_present=oracle.exists(), reusable=False))
    return progress


def verify_snapshot(manifest, frozen, directory, reader):
    verify_sources(manifest, frozen, directory)
    for name in manifest['source_sha256_after']:
        reader.path(directory / 'source' / name)


def compare_clean(new_entry, new_steps, old_entry, old_steps):
    for key in ('attack', 'checkpoint_sha256', 'weights_sha256', 'effective_seeds'):
        first, second = new_entry[key], old_entry[key]
        if key == 'effective_seeds':
            first, second = [{k: v for k, v in value.items() if k != 'attack_seed'}
                             for value in (first, second)]
        require(first == second, 'Canonical Clean provenance differs: ' + key)
    require(len(new_steps) == len(old_steps), 'Canonical Clean length differs')
    for new, old in zip(new_steps, old_steps):
        # Comparison copies only the cost mapping; do not mutate audit input.
        left, right = dict(new), dict(old)
        left['attack_cost'], right['attack_cost'] = [
            {k: v for k, v in row['attack_cost'].items() if k != 'wall_seconds'}
            for row in (new, old)]
        require(left == right, 'Canonical Clean trajectory differs')
    return len(new_steps)


def audit_complete(directory, config, group, protocol, execution_commit, reader, anchor):
    manifest, evaluation = reader.json(directory / 'manifest.json'), reader.json(directory / 'evaluation.json')
    checkpoint = config['run_seeds'][0]
    victim = next(v for v in protocol['victims'] if v['run_seed'] == checkpoint)
    validate_header(manifest, evaluation, config, victim, execution_commit)
    verify_snapshot(manifest, protocol['frozen_source_sha256'], directory, reader)
    require(sha256(reader.path(ROOT / victim['checkpoint'])) == victim['checkpoint_sha256'], 'Checkpoint changed')
    training = reader.json(ROOT / protocol['runtime_references'][checkpoint]['path'])
    for key in ('python_runtime', 'pip_freeze', 'sumo_version'):
        require(manifest[key]['returncode'] == 0 and manifest[key]['stdout'] == training[key]['stdout'],
                'Frozen runtime differs: ' + key)
    require([d['sha256'] for d in manifest['extra_dependencies']] ==
            ([protocol['zero_one_wheel']['sha256']] if group['gradient_cap'] is not None else []),
            'Dependency differs')
    clean, clean_steps, witnesses, rows = None, None, {}, []
    regression_steps = 0
    for entry, spec in zip(evaluation['runs'], config['attacks']):
        name = spec['name']
        require(entry['victim'] == 'clean' and entry['run_seed'] == checkpoint and entry['frozen_unchanged'] and
                all(entry[k] == victim[k] for k in ('checkpoint_sha256', 'weights_sha256')) and
                entry['effective_seeds'] == expected_seeds(config, checkpoint, name), 'Entry model/seeds differ')
        result = directory / entry['results_directory']
        require(result.parent == directory and result.name == 'clean-seed%d-%s' % (checkpoint, name),
                'Unexpected method output path')
        episodes, steps = reader.json(result / 'episodes.json'), reader.steps(result / 'steps.jsonl')
        count, summary = audit_raw(steps, episodes, config, spec, clean)
        summary['safety'] = summarize_safety([s['safety'] for s in steps], **config['metric_percentiles'])
        require(entry['summary'] == reader.json(result / 'summary.json') == summary, 'Raw summary differs')
        costs = {k: sum(s['attack_cost'].get(k, 0) for s in steps)
                 for k in ('objective_evaluations', 'gradient_evaluations', 'policy_forward_calls') + SEARCH_COSTS}
        if name == 'none':
            clean, clean_steps = episodes, steps
            anchor_dir = anchor[checkpoint]
            anchor_eval = reader.json(anchor_dir / 'evaluation.json')
            anchor_entry = next(r for r in anchor_eval['runs'] if r['attack']['name'] == 'none')
            anchor_steps = reader.steps(anchor_dir / anchor_entry['results_directory'] / 'steps.jsonl')
            regression_steps = compare_clean(entry, clean_steps, anchor_entry, anchor_steps)
        else:
            verify_prefix(steps, clean_steps)
        if name in SEARCH:
            witnesses[name] = audit_witnesses(steps, name, group['attack_seed'], spec['parameters'])
            costs.update(audit_oracle(reader, directory, entry, steps, episodes, protocol['frozen_source_sha256']))
        costs['evaluator_policy_forward_calls'] = 2 * count
        costs['total_policy_forward_calls'] = costs['policy_forward_calls'] + 2 * count
        rows.append(dict(attack=name, episodes=len(episodes), raw_verified_steps=count, costs=costs))
    shared = verify_shared_witnesses(witnesses, SEARCH) if witnesses else []
    return dict(verified=True, checkpoint_seed=checkpoint, rows=rows,
                episodes=sum(r['episodes'] for r in rows), raw_verified_steps=sum(r['raw_verified_steps'] for r in rows),
                clean_regression_steps=regression_steps, shared_first_attempt_comparisons=shared)


def validate_coverage(conditions, protocol):
    expected = {p: g['id'] for g in protocol['groups'] for p in g['configs']}
    require(len(conditions) == len(expected) == 60 and
            {c['config']: c['group'] for c in conditions} == expected, 'Incomplete/duplicate recovery grid')
    require(all(c['disposition'] in ('reuse', 'first_run', 'interrupted_retry_requires_authorization')
                for c in conditions), 'Unknown recovery disposition')
    require(sum(c['planned_episodes'] for c in conditions) == 14500, 'Incorrect episode grid')


def plan(inventory_path, receipt_path, pipeline_path, expected_tool_commit):
    require(git('rev-parse', 'HEAD') == expected_tool_commit and not git('status', '--porcelain'),
            'Plan on a fixed clean tool commit')
    reader = Reader()
    inventory, receipt, protocol = [reader.json(p) for p in (inventory_path, receipt_path, PROTOCOL)]
    execution_commit = inventory['git_commit']
    require(receipt['kind'] == 'attack_final_interruption_receipt' and
            receipt['git_commit'] == execution_commit and not receipt['simulation_launched'] and
            not receipt['experiment_retry'] and receipt['evidence_inventory_sha256'] == sha256(inventory_path),
            'Interruption receipt mismatch')
    require(inventory['kind'] == 'attack_final_interruption_inventory' and
            not inventory['final_matrix_complete'] and not inventory['simulation_launched'] and
            not inventory['original_files_modified'], 'Not an interrupted original run')
    # Compare the original certified tools and tests; newly added planning files
    # are not a replacement execution certificate and authorize no dispatch.
    original = reader.json(pipeline_path)
    certificate = reader.json(Path(original['readiness_path']))
    require(original['git_commit'] == execution_commit and original['final_exposure_declared_at_utc'] and
            original['protocol_sha256'] == sha256(PROTOCOL) and
            original['readiness_sha256'] == sha256(Path(original['readiness_path'])) and certificate['verified'],
            'Original pipeline/certificate differs')
    for path, digest in certificate['tools_git_blob_sha256'].items():
        for commit in (execution_commit, expected_tool_commit):
            blob = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=str(ROOT))
            require(hashlib.sha256(blob).hexdigest() == digest, 'Certified tool/test changed: ' + path)
    require(set(TOOLS).issubset(certificate['tools_git_blob_sha256']), 'Incomplete original tool certificate')
    for item in inventory['preserved_control_evidence'] + receipt['search_raw_fingerprints']:
        require(sha256(reader.path(ROOT / item['path'])) == item['sha256'], 'Preserved input changed: ' + item['path'])
    require(pipeline_path.relative_to(ROOT).as_posix() in
            {item['path'] for item in inventory['preserved_control_evidence']}, 'Pipeline outside preserved evidence')
    for proof in certificate['inputs']:
        require(sha256(reader.path(ROOT / proof['path'])) == proof['sha256'], 'Original readiness evidence changed')
    configs = {p: reader.json(ROOT / p) for p in protocol['configs']}
    validate_registration(protocol, configs)
    for victim, proof in zip(protocol['victims'], protocol['runtime_references']):
        require(sha256(reader.path(ROOT / victim['checkpoint'])) == victim['checkpoint_sha256'], 'Frozen checkpoint changed')
        require(sha256(reader.path(ROOT / proof['path'])) == proof['sha256'], 'Frozen training manifest changed')
    require(sha256(reader.path(ROOT / protocol['zero_one_wheel']['path'])) == protocol['zero_one_wheel']['sha256'],
            'ZOOpt wheel changed')
    require([g['id'] for g in inventory['groups']] == [g['id'] for g in protocol['groups']], 'Group inventory differs')
    anchor_item = inventory['groups'][0]
    require(anchor_item['stage'] == 'existing_audit_hash_verified', 'Canonical Clean missing')
    anchor_batch = reader.json(ROOT / anchor_item['batch'])
    anchor = {configs[r['config']]['run_seeds'][0]: Path(r['run_dir']) for r in anchor_batch['runs']}
    require(set(anchor) == set(range(5)), 'Incomplete canonical Clean')
    conditions, audits = [], []
    for prior, group in zip(inventory['groups'], protocol['groups']):
        batch = reader.json(ROOT / prior['batch']) if 'batch' in prior else None
        if batch:
            require(batch['git_commit'] == execution_commit and batch['configs'] == group['configs'] and
                    len({r['config'] for r in batch['runs']}) == len(batch['runs']), 'Mixed/duplicate batch')
        prior_runs = {r['config']: r for r in prior['runs']}
        if prior['stage'] == 'existing_audit_hash_verified':
            proof = reader.json(ROOT / prior['audit'])
            require(proof['verified'] and proof['group_id'] == group['id'] and
                    proof['git_commit'] == execution_commit and proof['episodes'] == group['episodes'] and
                    proof['protocol_sha256'] == sha256(PROTOCOL) and proof['clean_regression']['passed'] and
                    batch['status'] == 'passed', 'Original group proof differs')
            for path, digest in proof['source_sha256'].items():
                require(sha256(reader.path(ROOT / path)) == digest, 'Original group audit input changed')
        for index, path in enumerate(group['configs']):
            config = configs[path]
            require(canonical_sha256(config) == protocol['configs'][path], 'Unregistered config')
            condition = dict(config=path, group=group['id'], planned_episodes=50 * len(config['attacks']))
            if batch is None:
                require(prior['stage'] == 'not_dispatched' and not prior_runs, 'Missing original batch')
                condition.update(disposition='first_run', saved_episode_rows=0)
            else:
                directory = (ROOT / prior['batch']).parent / ('run-%d' % index)
                manifest = reader.json(directory / 'manifest.json')
                require(manifest['config_file'] == path and manifest['git_commit'] == execution_commit and
                        manifest['config'] == config, 'Original run identity differs')
                condition['original_directory'] = directory.relative_to(ROOT).as_posix()
                if manifest['status'] == 'passed':
                    require(any(r['config'] == path and r['returncode'] == 0 and Path(r['run_dir']) == directory
                                for r in batch['runs']), 'Missing normal configuration exit')
                    if prior['stage'] == 'existing_audit_hash_verified':
                        victim = next(v for v in protocol['victims'] if v['run_seed'] == config['run_seeds'][0])
                        validate_header(manifest, reader.json(directory / 'evaluation.json'), config, victim, execution_commit)
                        verify_snapshot(manifest, protocol['frozen_source_sha256'], directory, reader)
                        checked = dict(verified=True, episodes=condition['planned_episodes'],
                                       verification='Unchanged original full group audit inputs and executed snapshots')
                    else:
                        checked = audit_complete(directory, config, group, protocol, execution_commit, reader, anchor)
                        audits.append(dict(config=path, **checked))
                    require(checked['episodes'] == condition['planned_episodes'], 'Incomplete reusable configuration')
                    condition.update(disposition='reuse', saved_episode_rows=checked['episodes'], verified=True)
                else:
                    require(prior['stage'] == 'interrupted' and manifest['status'] == 'running' and
                            manifest.get('returncode') is None, 'Not a process-interrupted configuration')
                    require(manifest['source_sha256_before'] == protocol['frozen_source_sha256'], 'Partial source differs')
                    progress = saved_progress(config, directory, reader)
                    condition.update(disposition='interrupted_retry_requires_authorization', verified=False,
                                     saved_episode_rows=sum(p['saved_episode_rows'] for p in progress), methods=progress,
                                     reason='Incomplete launcher/oracle close evidence; no silent episode splice')
            conditions.append(condition)
    validate_coverage(conditions, protocol)
    for path, digest in reader.sources.items():
        require(sha256(ROOT / path) == digest, 'Input changed during recovery audit: ' + path)
    require(git('rev-parse', 'HEAD') == expected_tool_commit and not git('status', '--porcelain'), 'Planning source changed')
    reuse = [c for c in conditions if c['disposition'] == 'reuse']
    retries = [c for c in conditions if c['disposition'] == 'interrupted_retry_requires_authorization']
    fresh = [c for c in conditions if c['disposition'] == 'first_run']
    return dict(kind='attack_final_recovery_plan', verified=True, ready_to_execute=False,
                tool_commit=expected_tool_commit, experiment_commit=execution_commit,
                checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                protocol_sha256=sha256(PROTOCOL), command=[sys.executable] + sys.argv,
                simulations_run=0, conditions=conditions, completed_search_integrity_audits=audits,
                counts=dict(reuse_configs=len(reuse), reuse_episodes=sum(c['planned_episodes'] for c in reuse),
                    interrupted_configs=len(retries), saved_interrupted_episode_rows=sum(c['saved_episode_rows'] for c in retries),
                    proposed_retry_episodes=sum(c['planned_episodes'] for c in retries), first_run_configs=len(fresh),
                    first_run_episodes=sum(c['planned_episodes'] for c in fresh), final_episodes=14500),
                requirements=['Original experiment processes must be absent before dispatch',
                    'Prove infrastructure failure and obtain authorization for repeating saved work',
                    'At most one retry of each interrupted config in a new directory; preserve every old attempt',
                    'Use the original exact execution SHA/config/seeds/runtime/model, not the planning SHA',
                    'Commit and verify recovery dispatcher and full mixed-batch audit before any new episode',
                    'No outcome-dependent retry, tuning, seed substitution, or partial final table'],
                source_sha256=reader.sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--expected-tool-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    require((ROOT / '.local/runs').resolve() in output.parents and not output.exists(), 'Use a new ignored plan file')
    result = plan(args.inventory.resolve(), args.receipt.resolve(), args.pipeline.resolve(), args.expected_tool_commit)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('FINAL_RECOVERY_PLAN ' + json.dumps(result['counts'], sort_keys=True))


if __name__ == '__main__':
    main()
