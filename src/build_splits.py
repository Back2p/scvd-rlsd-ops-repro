"""Reconstruct HTML-compatible ID splits from verified official case metadata.

The published HTML gives aggregate constraints, not membership. Resulting ID
lists are a new deterministic reconstruction, never a claim of publisher IDs.
No model calls, benchmark execution, or gold-conditioned inference occur here.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re

from protocol import require, sha256

SEED = 20261106
ALGORITHM = 'html-marginals-stratum-swaps-v1'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows))


def marginals(rows, fields):
    return {field: dict(sorted(Counter(row[field] for row in rows).items())) for field in fields}


def select_train(rows, primary, primary_train, secondary_train, *, seed=SEED):
    """Exact HTML primary/marginal counts using fixed-seed within-stratum swaps.

    Start with proportional allocation inside the full joint strata, then swap
    selected/unselected rows within a primary category to meet the remaining
    declared marginals. Bounded search fails explicitly if no solution is found.
    Ordering and all tie breaks are fixed. This is our reconstruction algorithm.
    """
    rng = random.Random(seed)
    secondary = sorted(secondary_train)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['case_id']):
        groups[tuple(row[f] for f in [primary] + secondary)].append(row)
    keys = sorted(groups)
    for key in keys:
        rng.shuffle(groups[key])
    chosen = {}
    for category in sorted({r[primary] for r in rows}):
        strata = [key for key in keys if key[0] == category]
        total = sum(len(groups[key]) for key in strata)
        target = primary_train.get(category, 0)
        require(0 <= target <= total, f'Invalid category quota: {category}')
        fractions = {key: len(groups[key]) * target / total for key in strata}
        for key in strata:
            chosen[key] = int(fractions[key])
        left = target - sum(chosen[key] for key in strata)
        priority = sorted(strata, key=lambda key: (-(fractions[key] - chosen[key]), rng.random()))
        for key in priority[:left]:
            chosen[key] += 1
    dimensions = [(field, value) for field in secondary for value in sorted(secondary_train[field])]
    def residuals():
        return [sum(chosen[k] for k in keys if k[1 + secondary.index(field)] == value)
                - secondary_train[field][value] for field, value in dimensions]
    residual = residuals()
    pairs = [(a, b) for a in keys for b in keys if a != b and a[0] == b[0]]
    deltas = {(a, b): [int(b[1 + secondary.index(f)] == v) - int(a[1 + secondary.index(f)] == v)
                      for f, v in dimensions] for a, b in pairs}
    for step in range(20000):
        cost = sum(x*x for x in residual)
        if cost == 0:
            selected = [row for key in keys for row in groups[key][:chosen[key]]]
            require(marginals(selected, secondary) == secondary_train, 'Internal marginal mismatch')
            require(dict(Counter(r[primary] for r in selected)) == primary_train, 'Internal quota mismatch')
            return sorted(selected, key=lambda r: r['case_id']), step
        feasible = [(a, b) for a, b in pairs if chosen[a] and chosen[b] < len(groups[b])]
        require(bool(feasible), 'No feasible stratum swaps')
        costs = [sum((x+d)**2 for x, d in zip(residual, deltas[pair])) for pair in feasible]
        best_cost = min(costs)
        if best_cost <= cost:
            pair = rng.choice([p for p, c in zip(feasible, costs) if c == best_cost])
        else:
            # Explicit deterministic escape from a local minimum, never relaxation.
            pair = rng.choice(feasible)
        a, b = pair
        chosen[a] -= 1
        chosen[b] += 1
        residual = [x+d for x, d in zip(residual, deltas[pair])]
    raise ValueError('Could not satisfy exact marginal constraints within the fixed search budget')


def split_evaluation(rows, fields, val_count, *, seed=SEED):
    """Joint-stratum largest remainders; HTML specifies no val/test marginals."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r['case_id']):
        groups[tuple(row[f] for f in fields)].append(row)
    keys = sorted(groups)
    quota = {key: int(len(groups[key]) * val_count / len(rows)) for key in keys}
    priority = sorted(keys, key=lambda key: (-(len(groups[key]) * val_count / len(rows) - quota[key]), rng.random()))
    for key in priority[:val_count - sum(quota.values())]:
        quota[key] += 1
    val, test = [], []
    for key in keys:
        rng.shuffle(groups[key])
        val.extend(groups[key][:quota[key]])
        test.extend(groups[key][quota[key]:])
    return val, test


def cloud_catalog(root):
    tree = json.loads((root/'dataset_sources/cloudops-tree.json').read_text())
    files = tree['tree']
    evidence = defaultdict(list)
    for item in files:
        if item['type'] == 'blob' and '/raw_data/' in item['path']:
            parent, child = item['path'].split('/raw_data/', 1)
            evidence[parent].append((child, item['sha']))
    rows = []
    for item in files:
        path = item['path']
        if not (path.startswith('benchmark/') and path.endswith('/metadata.json')):
            continue
        source = root/'dataset_sources/cloudops'/path
        raw = source.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        require(blob == item['sha'], 'Cloud metadata identity mismatch')
        d = json.loads(raw)
        case = str(Path(path).parent)
        require(bool(evidence[case]), f'Missing raw snapshot: {case}')
        fingerprint = hashlib.sha256(json.dumps(sorted(evidence[case])).encode()).hexdigest()
        parts = path.split('/')
        rows.append(dict(dataset='cloudops', case_id='/'.join(parts[1:4]), system=parts[1],
            category=parts[2], difficulty=d['difficulty'].lower(),
            initial_query=d['query'], source_revision=tree['sha'], source_metadata_sha256=sha256(source),
            evidence_tree_sha256=fingerprint, incident_group='snapshot:'+fingerprint,
            payload_ready=False, prompt_sha256=None))
    require(len(rows) == 754, 'Cloud pool must contain 754 official metadata records')
    return rows


def cuj_catalog(root):
    source = root/'dataset_sources/cujbench/scenarios/registry.json'
    manifest = json.loads((root/'dataset_sources/cujbench-manifest.json').read_text())
    require(sha256(source) == manifest['registry_sha256'], 'CUJ registry identity mismatch')
    rows, excluded = [], []
    for d in json.loads(source.read_text()):
        if d['injection_type'] == 'none':
            excluded.append(dict(case_id=d['scenario_id'], reason='healthy_no_injected_failure',
                                 valid_public_case=True, exclusion_status='inferred_HTML_85_pool'))
            continue
        injection = d['injection_type']
        injection = d.get('params', {}).get('type', 'unlabelled') if injection == 'fault_proxy' else injection
        if injection == 'mutation':
            injection = 'unlabelled'
        rows.append(dict(dataset='cujbench', case_id=d['scenario_id'], app=d['app'],
            difficulty=d['taxonomy']['difficulty'].lower(), injection=injection,
            source_revision='bundle-sha256:'+manifest['bundle_sha256'], source_metadata_sha256=sha256(source),
            incident_group=d['scenario_id'], payload_ready=False, payload_downloaded=True,
            payload_status='locally_restored_source_corpus_runtime_not_integrated' if (root/'reports/cujbench-restore-receipt.json').exists()
                           else 'official_bundle_verified_not_runtime_restored', prompt_sha256=None))
    require(len(rows) == 85 and len(excluded) == 2, 'Unexpected CUJ pool')
    return rows, excluded


def openrca_catalog(root):
    source = root/'dataset_sources/openrca/rca/archive/agent-Bank.csv'
    rows = []
    for original in csv.DictReader(source.open()):
        instruction = original['instruction']
        match = re.search(r'March (\d{1,2}), 2021', instruction)
        require(match is not None, 'Unknown Bank query date')
        date = f'2021-03-{int(match.group(1)):02}'
        # Date grouping keeps all query variants of a failure on the same side.
        split = 'train' if date in ('2021-03-04','2021-03-06','2021-03-07','2021-03-09','2021-03-10') else 'val' if date == '2021-03-12' else 'test'
        require(date in ('2021-03-04','2021-03-06','2021-03-07','2021-03-09','2021-03-10',
                         '2021-03-12','2021-03-23','2021-03-24','2021-03-25'), 'Unexpected date')
        rows.append(dict(dataset='openrca', case_id='Bank/query-row-'+str(int(float(original['row_id']))),
            upstream_row_id=original['row_id'], upstream_task_index=original['task_index'],
            initial_query=instruction, date=date, split=split, incident_group='Bank/date/'+date,
            source_revision='official-archived-csv-sha256:'+sha256(source),
            query_provenance='instruction field of official rca/archive/agent-Bank.csv; original query.csv not acquired',
            source_metadata_sha256=sha256(source), payload_ready=False, prompt_sha256=None))
    require(len(rows) == 136 and Counter(r['split'] for r in rows) == {'train':80,'val':13,'test':43}, 'Bank date counts differ from HTML')
    return rows


def itbench_catalog(root, source):
    import yaml
    rows, excluded, normalized = [], [], []
    for original in map(json.loads, source.read_text().splitlines()):
        groundtruth = yaml.safe_load(original['ground_truth_yaml'])
        legacy = 'fault' in groundtruth
        normalized_spec = groundtruth if legacy else groundtruth.get('spec', {})
        fault = normalized_spec.get('fault', [])
        category = 'Compound' if len(fault) > 1 else fault[0].get('category', 'UNKNOWN') if fault else 'NoFault'
        row = dict(dataset='itbench_aa',case_id=original['scenario_id'],id_aa=original['id_aa'],
            category=category, source_revision='76df38a82288f75ba9e41dc8c515033332497473',
            source_metadata_sha256=sha256(source), incident_group=original['scenario_id'],
            label_schema='legacy_flat' if legacy else 'nested_spec', payload_ready=False, prompt_sha256=None)
        normalized.append(row)
        if legacy:
            rows.append(row)
        else:
            excluded.append(dict(case_id=original['scenario_id'],reason='nested_spec_schema_outside_inferred_HTML_31_pool',
                                 valid_public_case=True, exclusion_status='inferred_from_exact_HTML_category_totals'))
    require(len(rows) == 31 and len(normalized) == 40, 'ITBench pool no longer matches the investigated versions')
    return rows, excluded, normalized


def export(root, dataset, parts, fields):
    all_rows = []
    for split in ('train', 'val', 'test'):
        rows = sorted([dict(r, split=split, membership_status='RECONSTRUCTED_NOT_PUBLISHER_MEMBERSHIP',
                            split_seed=SEED, split_algorithm=ALGORITHM) for r in parts[split]], key=lambda r:r['case_id'])
        write_jsonl(root/'splits'/dataset/(split+'.jsonl'), rows)
        all_rows.extend(rows)
    ids = [row['case_id'] for row in all_rows]
    write_jsonl(root/'splits'/dataset/'eval.jsonl', [row for row in all_rows if row['split'] in ('val','test')])
    require(len(ids) == len(set(ids)), 'Split overlap/duplicate IDs')
    family_splits = defaultdict(set)
    for row in all_rows:
        family_splits[row['incident_group']].add(row['split'])
    overlaps = {group:sorted(splits) for group,splits in family_splits.items() if len(splits)>1}
    require(not overlaps, f'Observed incident/snapshot group crosses splits: {overlaps}')
    write_jsonl(root/'catalogs'/(dataset+'.jsonl'), sorted(all_rows,key=lambda r:r['case_id']))
    return dict(counts={s:len(parts[s]) for s in parts},
                train_marginals=marginals(parts['train'],fields),
                eval_marginals=marginals(parts['val']+parts['test'],fields),
                observed_group_overlap=overlaps,
                membership='reconstructed; publisher ID membership not confirmed',
                files={s:sha256(root/'splits'/dataset/(s+'.jsonl')) for s in (*parts,'eval')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--itbench-metadata',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    report={'seed':SEED,'algorithm':ALGORITHM,'status':'REAL_CASE_ID_SPLITS_RECONSTRUCTED_PAYLOADS_PARTIAL','datasets':{}}
    cloud=cloud_catalog(root)
    category_train=dict(admission=46,codedefect=78,infrastructure=32,performance=60,runtime=113,scheduling=132,service=73,startup=70)
    train,steps=select_train(cloud,'category',category_train,
        {'difficulty':{'easy':315,'medium':151,'hard':138},'system':{'boutique':442,'trainticket':162}})
    ids={r['case_id'] for r in train};held=[r for r in cloud if r['case_id'] not in ids]
    val,test=split_evaluation(held,['system','category','difficulty'],75)
    report['datasets']['cloudops']=export(root,'cloudops',dict(train=train,val=val,test=test),['system','category','difficulty'])
    report['datasets']['cloudops']['marginal_swap_steps']=steps
    cuj,excluded=cuj_catalog(root)
    train,steps=select_train(cuj,'injection',dict(unlabelled=3,compound=14,asset_block=2,delay=13,
        header_mutation=1,http_error=14,json_shape_mutation=1,response_body=6,url_rewrite=1,flagd=2),
        {'app':{'otel-demo':44,'tractor-store':13},'difficulty':{'easy':3,'medium':36,'hard':18}})
    ids={r['case_id'] for r in train};held=[r for r in cuj if r['case_id'] not in ids]
    val,test=split_evaluation(held,['app','difficulty','injection'],14)
    report['datasets']['cujbench']=export(root,'cujbench',dict(train=train,val=val,test=test),['app','difficulty','injection'])
    report['datasets']['cujbench'].update(excluded=excluded,marginal_swap_steps=steps)
    bank=openrca_catalog(root)
    report['datasets']['openrca']=export(root,'openrca',{s:[r for r in bank if r['split']==s] for s in ('train','val','test')},['date'])
    report['datasets']['openrca']['val_test_boundary_status']='Inferred chronological boundary: March 12 = 13 val queries; March 23/24/25 = 43 test queries. HTML gives only the aggregate eval dates.'
    aa,excluded,normalized=itbench_catalog(root,args.itbench_metadata)
    write_jsonl(root/'catalogs/itbench_aa_full40_normalized.jsonl',normalized)
    train,steps=select_train(aa,'category',{'Change':1,'Configuration':1,'Configuration Setting':1,
        'Latency':1,'Network':2,'Other':1,'Resource':1,'Resource Unavailable':1,'Compound':9,'custom':1},{})
    ids={r['case_id'] for r in train};held=[r for r in aa if r['case_id'] not in ids]
    val,test=split_evaluation(held,['category'],7)
    report['datasets']['itbench_aa']=export(root,'itbench_aa',dict(train=train,val=val,test=test),['category'])
    report['datasets']['itbench_aa'].update(excluded=excluded,pool_status='legacy-flat-31 pool inferred from exact HTML category totals; all 40 normalized separately')
    report['not_proven']=['same case membership as project party','full semantic/variant-family isolation',
        'CloudOps/OpenRCA/ITBench telemetry available on target platform','actor input hashes and tool/scorer integration']
    write_json(root/'reports/dataset_split_report.json',report)
    combined = [row for d in report['datasets'] for row in map(json.loads,(root/'catalogs'/(d+'.jsonl')).read_text().splitlines())]
    write_jsonl(root/'splits/all_cases.jsonl', combined)
    print(json.dumps({d:r['counts'] for d,r in report['datasets'].items()},indent=2))


if __name__=='__main__':
    main()
