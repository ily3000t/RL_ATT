"""Validate an explicit, one-use full restart of an interrupted final attempt."""

import json
from pathlib import Path

from prepare_attack_final_protocol import ROOT
from prepare_mechanism_controls import sha256


def validate_restart(authorization_path, protocol):
    sources = {}

    def checked(path, expected=None):
        path = Path(path).resolve()
        if (ROOT / '.local/runs').resolve() not in path.parents or not path.is_file():
            raise ValueError('Restart evidence must be an existing ignored run file')
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError('Restart evidence changed: ' + str(path))
        sources[path.relative_to(ROOT).as_posix()] = digest
        return path

    def read(path, expected=None):
        return json.loads(checked(path, expected).read_text(encoding='utf-8-sig'))

    authorization_path = checked(authorization_path)
    authorization = read(authorization_path)
    if (authorization.get('kind') != 'attack_final_full_restart_authorization' or
            authorization.get('allow_full_matrix_rerun') is not True or
            authorization.get('preserve_original_outputs') is not True or
            authorization.get('restart_attempt') != 1 or
            not authorization.get('user_authorization') or not authorization.get('process_absence_evidence')):
        raise ValueError('Explicit full-restart authorization and preserved evidence required')
    original_path = checked(ROOT / authorization['original_pipeline']['path'], authorization['original_pipeline']['sha256'])
    original = read(original_path)
    if (original.get('kind') != 'attack_final_pipeline' or original.get('status') not in ('running', 'failed') or
            original.get('planned_episodes') != 14500 or original.get('verified_episodes', 14500) >= 14500 or
            not original.get('final_exposure_declared_at_utc') or original.get('restart') or
            original.get('protocol_sha256') != authorization['protocol_sha256'] or
            (original_path.parent / 'final-comparison.json').exists()):
        raise ValueError('Only the incomplete original matrix may be restarted once')
    if authorization['protocol_sha256'] != sha256(ROOT / 'configs/research/attack_final_test.json'):
        raise ValueError('Restart protocol differs')
    receipt = read(ROOT / authorization['interruption_receipt']['path'], authorization['interruption_receipt']['sha256'])
    inventory_path = ROOT / authorization['interruption_inventory']['path']
    inventory = read(inventory_path, authorization['interruption_inventory']['sha256'])
    if (receipt.get('kind') != 'attack_final_interruption_receipt' or
            inventory.get('kind') != 'attack_final_interruption_inventory' or
            receipt['git_commit'] != inventory['git_commit'] or receipt['git_commit'] != original['git_commit'] or
            receipt['evidence_inventory_sha256'] != sha256(inventory_path) or
            receipt['final_matrix_complete'] or inventory['final_matrix_complete'] or
            receipt['original_files_modified'] or inventory['original_files_modified'] or
            [g['id'] for g in inventory['groups']] != [g['id'] for g in protocol['groups']] or
            not any(g['stage'] == 'interrupted' for g in inventory['groups'])):
        raise ValueError('Interruption proof does not match the original run')
    for item in inventory['preserved_control_evidence'] + receipt['search_raw_fingerprints']:
        checked(ROOT / item['path'], item['sha256'])
    precheck_proof = authorization['original_preflight']
    precheck = read(ROOT / precheck_proof['path'], precheck_proof['sha256'])
    if (not precheck['verified'] or precheck['git_commit'] != original['git_commit'] or
            precheck['protocol_sha256'] != authorization['protocol_sha256'] or
            set(precheck['config_raw_sha256']) != set(protocol['configs'])):
        raise ValueError('Original input preflight differs')
    for path, digest in precheck['config_raw_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Original configuration bytes changed: ' + path)
    process_proof = authorization['process_absence_evidence']
    processes = read(ROOT / process_proof['path'], process_proof['sha256'])
    if processes.get('original_dispatcher_absent') is not True or processes.get('matching_original_processes') != []:
        raise ValueError('Original experiment processes must be absent')
    authorization_digest = sha256(authorization_path)
    # A restarted pipeline declares consumption before its first launcher.
    # Evidence copies have the same digest too; either must prevent reuse.
    for path in (ROOT / '.local/runs').rglob('final-attack.json'):
        existing = json.loads(path.read_text(encoding='utf-8-sig'))
        if existing.get('restart', {}).get('authorization_sha256') == authorization_digest:
            raise ValueError('Restart authorization already consumed')
    for path, digest in sources.items():
        if sha256(ROOT / path) != digest:
            raise ValueError('Restart evidence changed during validation: ' + path)
    return dict(kind='authorized_full_matrix_restart', authorization_path=authorization_path.relative_to(ROOT).as_posix(),
                authorization_sha256=authorization_digest, original_pipeline=original_path.relative_to(ROOT).as_posix(),
                original_commit=original['git_commit'], original_first_exposure_at_utc=original['final_exposure_declared_at_utc'],
                restart_attempt=1, traffic_previously_exposed=True, reuse_prior_results=False,
                preserve_original_outputs=True, planned_episodes=14500, source_sha256=sources,
                policy_deviation='User explicitly requested a fresh complete matrix after process interruption, including previously completed configurations; not an additional independent traffic sample')
