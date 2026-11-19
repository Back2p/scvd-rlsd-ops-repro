"""Summarize completed or partial evaluations without dropping failed attempts."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

from .common import ROOT,append_jsonl,file_hash,load_json,read_jsonl,require,write_json


def summarize_run(run):
    run=Path(run);receipt=load_json(run/'run.json');cfg=receipt['config']
    require(file_hash(ROOT/cfg['cases'])==receipt['split_manifest_sha256'],'Case manifest differs from evaluated membership')
    view=cfg['_evaluation_view'];repeats=cfg['evaluation']['repeats']
    index={(r['dataset'],r['case_id']):r for r in read_jsonl(ROOT/cfg['cases'])
           if r['dataset'] in cfg['datasets'] and r['split'] in (('val','test') if view=='eval' else (view,))}
    data=read_jsonl(run/'scores.jsonl') if (run/'scores.jsonl').exists() else []
    cells={};badcases=[];inputs={};checkpoints=set();revisions=set();scorers={}
    for row in data:
        key=(row['dataset'],row['case_id']);cell=(key,row['mode'],row['trial'])
        require(key in index and row['mode'] in cfg['modes'] and 0<=row['trial']<repeats,'Unplanned evaluation cell')
        require(cell not in cells,'Duplicate evaluation cell')
        require(row['protocol_sha256']==receipt['protocol_sha256'],'Mixed evaluation protocol')
        require(row['split_manifest_sha256']==receipt['split_manifest_sha256'],'Mixed case membership')
        cells[cell]=row
        revisions.add(row['model_revision'])
        if row['status']=='scored':
            require(type(row['score']) is int and row['score'] in (0,1),'Invalid binary score')
            require(row.get('checkpoint_sha256'),'Missing checkpoint identity')
            require(row.get('actor_view_sha256'),'Missing actor input fingerprint')
            checkpoints.add(row['checkpoint_sha256'])
            require(row.get('scorer_revision'),'Missing scorer revision')
            require(scorers.setdefault(key[0],row['scorer_revision'])==row['scorer_revision'],'Scorer changed within an evaluation')
            old=inputs.setdefault(key,row['actor_view_sha256'])
            require(old==row['actor_view_sha256'],'Case evidence/prompt changed between attempts')
        else:require(row['status']=='infra_error' and row['score'] is None,'Invalid failure encoding')
        if row['status']!='scored' or row['score']==0:
            termination=row.get('termination','infra_error')
            layer='infrastructure' if row['status']=='infra_error' else 'output_contract' if termination.startswith('invalid') else 'budget' if termination.endswith('budget') else 'diagnosis_needs_trace_review'
            badcases.append(dict(dataset=key[0],case_id=key[1],mode=row['mode'],trial=row['trial'],
                layer=layer,termination=termination,receipt=row['receipt'],cause_status='triage only; not a verified root-cause attribution'))
    require(len(checkpoints)<=1,'Worker checkpoints differ within one frozen evaluation')
    require(len(revisions)<=1,'Model revision changed within one frozen evaluation')
    summaries=[]
    for dataset in cfg['datasets']:
        for mode in cfg['modes']:
            for splits in (('val',),('test',),('val','test')):
                keys=[k for k,r in index.items() if k[0]==dataset and r['split'] in splits]
                if not keys:continue
                planned=len(keys)*repeats;correct=0;scored=0;infra=0;missing=0
                for key in keys:
                    for trial in range(repeats):
                        row=cells.get((key,mode,trial))
                        if row is None:missing+=1
                        elif row['status']=='infra_error':infra+=1
                        else:scored+=1;correct+=row['score']
                summaries.append(dict(run_id=run.name,arm=cfg['_arm'],dataset=dataset,mode=mode,split='+'.join(splits),
                    model_revision=next(iter(revisions),None),checkpoint_sha256=next(iter(checkpoints),None),
                    expected_attempts=planned,scored_attempts=scored,infra_errors=infra,missing=missing,correct=correct,
                    accuracy=correct/planned if scored==planned else None,publishable=scored==planned,
                    protocol_sha256=receipt['protocol_sha256']))
    return summaries,badcases,inputs,receipt


def incomplete_suite(rows):
    expected = {'cloudops':(75,75), 'openrca':(13,43), 'cujbench':(14,14), 'itbench_aa':(7,5)}
    cells = defaultdict(list)
    identities = defaultdict(set)
    for row in rows:
        key = (row['arm'], row['dataset'], row['mode'], row['split'])
        if row['split'] in ('val','test'):
            cells[key].append(row)
            if row['publishable']:
                identities[row['arm']].add((row['model_revision'],row['checkpoint_sha256']))
    issues = []
    for arm in ('base_9b','scvd','rlsd','base_27b'):
        if len(identities[arm])>1:
            issues.append(dict(arm=arm,issue='different frozen policy across official evaluation cells'))
        for dataset,sizes in expected.items():
            for mode in ('nonthinking','thinking'):
                for split,size in zip(('val','test'),sizes):
                    matching = cells[(arm,dataset,mode,split)]
                    if len(matching)!=1 or not matching[0]['publishable'] or matching[0]['expected_attempts']!=size*3:
                        issues.append(dict(arm=arm,dataset=dataset,mode=mode,split=split,
                                           expected_attempts=size*3,run_count=len(matching),
                                           complete_runs=sum(row['publishable'] for row in matching)))
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--full-suite',action='store_true',help='Flag missing, incomplete or duplicate four-arm evaluation cells')
    args=parser.parse_args();output=Path(args.output)
    require(not output.exists(),'Report directory already exists; use a new version')
    output.mkdir(parents=True)
    all_rows=[];first_inputs={};protocol=None;comparison_issues=[]
    reference=None
    for run in args.runs:
        rows,bad,inputs,receipt=summarize_run(run);all_rows.extend(rows)
        identity=(receipt['split_manifest_sha256'],receipt['config'].get('seed'))
        if reference is None:reference=identity
        elif identity!=reference:comparison_issues.append(dict(run=str(run),issue='split manifest or evaluation seed differs'))
        if protocol is None:protocol=receipt['protocol_sha256']
        elif receipt['protocol_sha256']!=protocol:comparison_issues.append(dict(run=str(run),issue='protocol differs'))
        for key,value in inputs.items():
            if key in first_inputs and first_inputs[key]!=value:comparison_issues.append(dict(run=str(run),case=key,issue='actor input differs'))
            first_inputs.setdefault(key,value)
        for record in bad:append_jsonl(output/'badcases.jsonl',dict(run=str(run),**record))
    coverage_issues=incomplete_suite(all_rows) if args.full_suite else []
    write_json(output/'summary.json',dict(results=all_rows,comparison_issues=comparison_issues,
        full_suite_checked=args.full_suite,coverage_issues=coverage_issues,
        statistical_scope='Mean single-attempt strict accuracy; not pass@3; no significance inferred',
        original_html_comparability='Unknown original API; this is snapshot-reference-v2 only'))
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]) if all_rows else ['run_id'])
        writer.writeheader();writer.writerows(all_rows)
    lines=['# 训练后评测汇总','',
        '本报告来自 snapshot-reference-v2 的逐题记录。平均单次 strict accuracy 不等于 Pass@3。',
        '验证集用于选 checkpoint；测试集与验证+测试分别报告。缺失或基础设施失败时 accuracy 留空。','',
        '| 实验 | 数据集 | 模式 | 集合 | 正确/预定尝试 | 准确率 |','|---|---|---|---|---:|---:|']
    for row in all_rows:
        accuracy='待补齐' if row['accuracy'] is None else f"{row['accuracy']*100:.2f}%"
        lines.append(f"| {row['arm']} ({row['run_id']}) | {row['dataset']} | {row['mode']} | {row['split']} | {row['correct']}/{row['expected_attempts']} | {accuracy} |")
    lines.extend(['','协议/输入不一致项：'+str(len(comparison_issues))+'；存在不一致时不作方法增益归因。',
        '', 'badcase 自动分类只用于初筛；根因需回看 traces.jsonl 的工具证据、截断位置、提交内容和评分规则。',
        '', '待验证的改进方向：①依据错误候选的真实证据缺口补充 SCVD 核查/纠错覆盖；②先检查 RLSD 全对/全错组比例及 token 权重分布，再在训练集内研究采样分配。未运行改进实验前不声称收益。'])
    if args.full_suite:
        lines.extend(['', f'四数据集正式评测不完整或重复的单元数：{len(coverage_issues)}。详见 summary.json。'])
    (output/'报告.md').write_text('\n'.join(lines)+'\n')
    if args.full_suite and (coverage_issues or comparison_issues):
        raise SystemExit(2)


if __name__=='__main__':main()
