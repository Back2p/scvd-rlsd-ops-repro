"""CPU-only split audit and aggregation of externally scored evaluation results.

This module does not implement a benchmark judge or execute an agent.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

DATASETS = {
    'cloudops': (604, 75, 75), 'openrca': (80, 13, 43),
    'cujbench': (57, 14, 14), 'itbench_aa': (19, 7, 5),
}
SPLITS = ('train', 'val', 'test')
MODES = ('nonthinking', 'thinking')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit_cases(rows, *, exact_counts=True, require_actor_hash=True):
    """Audit publisher IDs; never reconstruct a split from counts or a seed alone.

    incident_group must identify variants/questions sharing the same incident.
    prompt_sha256 is the hash of the actual actor-visible input, not of gold.
    These supplied metadata cannot prove semantic isolation or honest provenance.
    """
    seen, groups, fingerprints = set(), {}, {}
    counts = Counter()
    for row in rows:
        dataset, split, case = row.get('dataset'), row.get('split'), row.get('case_id')
        require(dataset in DATASETS and split in SPLITS, 'Unknown dataset/split')
        require(isinstance(case, str) and bool(case.strip()), 'case_id must be a nonempty string')
        key = (dataset, case)
        require(key not in seen, f'Duplicate case ID: {key}')
        seen.add(key)
        for field in ('incident_group', 'source_revision'):
            require(isinstance(row.get(field), str) and bool(row[field].strip()), f'Missing {field}: {key}')
        digest = row.get('prompt_sha256', '')
        if require_actor_hash:
            require(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), f'Invalid prompt hash: {key}')
        group = (dataset, row['incident_group'])
        require(groups.setdefault(group, split) == split, f'Incident family crosses splits: {group}')
        # Cross-dataset duplicate prompts also require explicit resolution.
        if isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest):
            require(fingerprints.setdefault(digest, split) == split, f'Prompt hash crosses splits: {key}')
        counts[dataset, split] += 1
    if exact_counts:
        for dataset, expected in DATASETS.items():
            actual = tuple(counts[dataset, split] for split in SPLITS)
            require(actual == expected, f'{dataset}: got {actual}; HTML requires {expected}')
    return {'cases': len(seen), 'counts': {d: {s: counts[d, s] for s in SPLITS} for d in DATASETS},
            'checks': ['unique_ids', 'declared_incident_group_isolation'] + (['exact_prompt_hash_isolation'] if require_actor_hash else []),
            'actor_hashes_required': require_actor_hash,
            'not_proven': ['same_case_ids_as_publisher', 'semantic_isolation', 'source_authenticity'] +
                          ([] if require_actor_hash else ['actor_input_hash_isolation'])}


def summarize(cases, scores, *, repeats=3, exact_counts=True, manifest_sha256=None):
    """Retain missing/infra errors in the expected denominator and flag incomplete.

    A result is publishable only if every cell was scored by the frozen judge.
    Repeats estimate mean single-attempt accuracy, never pass@3.
    """
    require(type(repeats) is int and repeats > 0, 'repeats must be positive integer')
    audit_cases(cases, exact_counts=exact_counts)
    index = {(r['dataset'], r['case_id']): r for r in cases}
    require(bool(scores), 'No scored results')
    cells, run_groups = {}, set()
    signature_fields = ('model_id', 'model_revision', 'protocol_sha256', 'scorer_revision',
                        'checkpoint_sha256', 'split_manifest_sha256', 'adapter_revision')
    for row in scores:
        key = (row.get('dataset'), row.get('case_id'))
        require(key in index and index[key]['split'] != 'train', f'Unknown/train evaluation case: {key}')
        require(row.get('mode') in MODES, 'Invalid thinking mode')
        require(isinstance(row.get('run_id'), str) and bool(row['run_id']), 'Missing run_id')
        trial = row.get('trial')
        require(type(trial) is int and trial in range(repeats), 'Invalid trial')
        signature = tuple(row.get(f) for f in signature_fields)
        require(all(isinstance(v, str) and bool(v.strip()) for v in signature), 'Missing immutable result provenance')
        for field in ('protocol_sha256', 'checkpoint_sha256', 'split_manifest_sha256'):
            require(bool(re.fullmatch('[0-9a-f]{64}', row[field])), f'Invalid {field}')
        if manifest_sha256 is not None:
            require(row['split_manifest_sha256'] == manifest_sha256, 'Result uses a different split manifest')
        status = row.get('status')
        require(status in ('scored', 'infra_error'), 'Invalid status')
        valid_score = (type(row.get('score')) is int and row['score'] in (0, 1)) if status == 'scored' else row.get('score') is None
        require(valid_score, 'Only official binary score or null infra_error accepted')
        require(isinstance(row.get('receipt'), str) and bool(row['receipt']), 'Missing judge/error receipt')
        group = (row['run_id'], row['mode'], signature)
        cell = (group, key, trial)
        require(cell not in cells, f'Duplicate evaluation cell: {key}, trial {trial}')
        cells[cell] = row
        run_groups.add(group)
    # A run ID names exactly one immutable model/configuration.
    require(len({(g[0], g[1]) for g in run_groups}) == len(run_groups), 'Mixed revisions under one run/mode')
    output = []
    for group in sorted(run_groups):
        for dataset in DATASETS:
            for splits in (('val',), ('test',), ('val', 'test')):
                selected = [k for k, r in index.items() if k[0] == dataset and r['split'] in splits]
                if not selected:
                    continue
                total, correct, scored, infra, missing = len(selected) * repeats, 0, 0, 0, 0
                for key in selected:
                    for trial in range(repeats):
                        row = cells.get((group, key, trial))
                        if row is None:
                            missing += 1
                        elif row['status'] == 'infra_error':
                            infra += 1
                        else:
                            scored += 1
                            correct += row['score']
                output.append({'run_id': group[0], 'mode': group[1], 'dataset': dataset,
                               'split': '+'.join(splits), 'cases': len(selected), 'expected_attempts': total,
                               'scored_attempts': scored, 'infra_errors': infra, 'missing': missing,
                               'correct': correct, 'accuracy': correct / total if scored == total else None,
                               'observed_success_fraction_of_planned_attempts': correct / total,
                               'publishable': scored == total,
                               **dict(zip(signature_fields, group[2]))})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--cases', required=True)
    audit.add_argument('--ids-only', action='store_true', help='Audit ID membership before the platform actor inputs exist; not training admission')
    summarize_cmd = sub.add_parser('summarize')
    summarize_cmd.add_argument('--cases', required=True)
    summarize_cmd.add_argument('--scores', required=True)
    summarize_cmd.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(audit_cases(read_jsonl(args.cases), require_actor_hash=not args.ids_only), indent=2))
    else:
        output = summarize(read_jsonl(args.cases), read_jsonl(args.scores), manifest_sha256=sha256(args.cases))
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
