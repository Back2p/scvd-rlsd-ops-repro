import argparse
import concurrent.futures
import json
from pathlib import Path

from .clients import TeacherClient,TokenClient
from .common import ROOT,admit_data,append_jsonl,cases,config,digest,file_hash,new_run,seed_for,write_json
from .datasets import SnapshotCase
from .rollout import collect_scvd


def main():
    parser=argparse.ArgumentParser(description='Collect genuine SCVD student-prefix/teacher-continuation training examples')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();cfg=config(args.config);rows=cases(cfg,'train')
    tasks=[(row,mode,rep) for row in rows for mode in cfg['modes'] for rep in range(cfg['scvd']['samples_per_case_mode'])]
    if args.plan:
        print(json.dumps(dict(stage='scvd_collection',training_cases=len(rows),attempts=len(tasks),model_calls_executed=0),indent=2));return
    TeacherClient(cfg['teacher'])  # validate provider settings before any student requests
    admit_data(cfg)
    output=new_run(args.output,'scvd_collection',cfg)
    def work(index,task):
        row,mode,rep=task
        try:
            case=SnapshotCase(row,cfg)
            result=collect_scvd(case,TokenClient(cfg['rollout'],index),TeacherClient(cfg['teacher']),
                mode=mode,seed=seed_for(cfg['seed'],row['dataset'],row['case_id'],mode,rep),
                student_revision=cfg['model']['revision'],teacher_revision=cfg['teacher']['revision'],
                budget=cfg['budget'],sampling=cfg['sampling'])
            if result['accepted']:
                result['record'].update(protocol_sha256=digest(cfg['protocol']),frozen_data_sha256=file_hash(cfg['frozen_data_manifest']))
            return dict(index=index,dataset=row['dataset'],case_id=row['case_id'],mode=mode,repeat=rep,**result)
        except Exception as error:
            return dict(index=index,dataset=row['dataset'],case_id=row['case_id'],mode=mode,repeat=rep,
                        accepted=False,reason='infra_error',error_type=type(error).__name__,error=str(error))
    accepted=0;infra_errors=0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['rollout']['endpoints'])) as pool:
        futures=[pool.submit(work,i,task) for i,task in enumerate(tasks)]
        for future in concurrent.futures.as_completed(futures):
            result=future.result();append_jsonl(output/'attempts.jsonl',result)
            infra_errors+=int(result['reason']=='infra_error')
            if result['accepted']:
                accepted+=1;append_jsonl(output/'accepted.jsonl',result['record'])
    if not (output/'accepted.jsonl').exists():(output/'accepted.jsonl').touch()
    write_json(output/'summary.json',dict(planned_attempts=len(tasks),accepted=accepted,rejected_or_failed=len(tasks)-accepted,infra_errors=infra_errors,
        acceptance_rate=accepted/len(tasks),source_manifest_sha256=file_hash(ROOT/cfg['cases']),
        accepted_sha256=file_hash(output/'accepted.jsonl'),protocol_sha256=digest(cfg['protocol'])))
    if infra_errors or accepted==0:raise SystemExit(2)


if __name__=='__main__':main()
