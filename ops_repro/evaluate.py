import argparse
import concurrent.futures
import json
from pathlib import Path

from .clients import TokenClient
from .common import ROOT,admit_data,append_jsonl,cases,config,digest,file_hash,new_run,seed_for,write_json
from .datasets import SnapshotCase
from .rollout import episode


def main():
    parser=argparse.ArgumentParser(description='Frozen repeated evaluation; validation and test are distinct views')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=['val','test','eval'],required=True)
    parser.add_argument('--policy-revision',required=True)
    parser.add_argument('--arm',choices=['base_9b','base_27b','scvd','rlsd','grpo'],required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    rows=[r for r in cases(cfg) if r['split'] in (('val','test') if args.split=='eval' else (args.split,))]
    tasks=[(r,mode,trial) for r in rows for mode in cfg['modes'] for trial in range(cfg['evaluation']['repeats'])]
    if args.plan:
        print(json.dumps(dict(stage='evaluation',cases=len(rows),attempts=len(tasks),split=args.split,model_calls_executed=0),indent=2));return
    admit_data(cfg)
    cfg['_evaluation_view']=args.split;cfg['_arm']=args.arm
    output=new_run(args.output,'evaluation',cfg)
    def work(index,task):
        row,mode,trial=task
        base=dict(dataset=row['dataset'],case_id=row['case_id'],split=row['split'],mode=mode,trial=trial,
            model_id=cfg['model']['id'],model_revision=args.policy_revision,protocol_sha256=digest(cfg['protocol']),
            split_manifest_sha256=file_hash(ROOT/cfg['cases']),adapter_revision='snapshot-reference-v2',run_id=output.name)
        base['arm']=args.arm
        try:
            case=SnapshotCase(row,cfg)
            trace=episode(case,TokenClient(cfg['rollout'],index),mode=mode,
                seed=seed_for(cfg['seed'],row['dataset'],row['case_id'],mode,trial),revision=args.policy_revision,
                budget=cfg['budget'],sampling=cfg['sampling'])
            identity=trace['checkpoint_identity']
            base.update(status='scored',score=trace['score'],receipt=digest(trace),
                        actor_view_sha256=trace['actor_view_sha256'],scorer_revision=case.scorer_revision(),
                        checkpoint_sha256=identity.get('checkpoint_sha256'),termination=trace['status'],
                        generated_tokens=trace['generated_tokens'],tool_calls=len(trace['session_state']['calls']))
            return base,trace
        except Exception as error:
            base.update(status='infra_error',score=None,receipt=digest({'case':row['case_id'],'error':type(error).__name__}),
                        error_type=type(error).__name__,error=str(error))
            return base,None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['rollout']['endpoints'])) as pool:
        futures=[pool.submit(work,i,task) for i,task in enumerate(tasks)]
        for future in concurrent.futures.as_completed(futures):
            result,trace=future.result();append_jsonl(output/'scores.jsonl',result)
            if trace is not None:append_jsonl(output/'traces.jsonl',dict(key={k:result[k] for k in ('dataset','case_id','mode','trial')},trace=trace))
    write_json(output/'completion.json',dict(planned_attempts=len(tasks),split=args.split,scores_sha256=file_hash(output/'scores.jsonl')))


if __name__=='__main__':main()
