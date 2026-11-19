import hashlib
import json
import os
import platform
import importlib.metadata
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]


def require(value,message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.writing')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def append_jsonl(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def resolve(value):
    if isinstance(value,str):
        result=os.path.expandvars(value)
        require('${' not in result,'Set all environment placeholders before running: '+value)
        return result
    if isinstance(value,list):return [resolve(v) for v in value]
    if isinstance(value,dict):return {k:resolve(v) for k,v in value.items()}
    return value


def config(path):
    cfg=resolve(load_json(path))
    require(cfg['protocol']['name']=='snapshot-reference-v2','Unknown protocol; native project adapters must declare a different protocol version')
    require(cfg['model']['revision'],'Pin model revision or checkpoint identity')
    cfg['protocol']['implementation_sha256']=digest({p.name:file_hash(p) for p in sorted((ROOT/'ops_repro').glob('*.py'))})
    cfg['protocol']['inference_settings']=dict(budget=cfg['budget'],sampling=cfg['sampling'],
        actor_max_context=cfg['rollout'].get('max_context',30720),modes=cfg['modes'],
        evaluation_repeats=cfg['evaluation']['repeats'])
    require(cfg['budget']['max_turns']>0 and cfg['budget']['max_tool_calls']>0,'Budgets must be positive')
    require(bool(cfg['modes']) and set(cfg['modes'])<=set(('thinking','nonthinking')) and len(cfg['modes'])==len(set(cfg['modes'])),'Invalid thinking modes')
    require(cfg['rlsd']['rollouts_per_prompt']>=2 and cfg['rlsd']['prompt_batch_size']>=1,'Invalid GRPO group/batch size')
    require(cfg['training']['save_every']>=1 and cfg['training']['keep_resume_checkpoints']>=1
            and cfg['training']['keep_inference_publications']>=1,'Invalid checkpoint interval/retention')
    require(cfg['model']['attention']=='sdpa','This reference worker/trainer pair supports attention=sdpa only')
    require(cfg['rlsd']['ppo_epochs']==1,'Only one on-policy PPO epoch is implemented')
    require(cfg['scvd']['loss_reduction']=='global_teacher_token_mean','Unsupported SCVD loss reduction')
    require(cfg['rlsd']['loss_reduction']=='global_generated_token_mean','Unsupported RLSD loss reduction')
    for key in ('max_new_tokens','max_generated_tokens','max_context'):
        require(isinstance(cfg['budget'][key],int) and cfg['budget'][key]>0,'Invalid budget: '+key)
    require(0<cfg['rollout']['max_context']<=cfg['budget']['max_context'],'Actor context must fit training context')
    for section,keys in [('scvd',('epochs','global_batch_size','samples_per_case_mode')),
                         ('rlsd',('max_updates','max_rollout_batches','teacher_sync_steps','lambda_decay_steps')),
                         ('evaluation',('repeats',)),('training',('projection_chunk_size','control_timeout_seconds','collective_timeout_seconds'))]:
        for key in keys:require(isinstance(cfg[section][key],int) and cfg[section][key]>0,'Invalid positive integer: '+section+'.'+key)
    require(0<=cfg['scvd']['warmup_ratio']<=1,'Invalid warmup ratio')
    require(0<=cfg['rlsd']['lambda_start']<=1 and 0<=cfg['rlsd']['evidence_clip']<1,'Invalid teacher weight bounds')
    require(0<cfg['rlsd']['clip_low']<1 and cfg['rlsd']['clip_high']>0 and cfg['rlsd']['clip_dual']>1,'Invalid PPO clipping')
    require(bool(cfg['rollout']['endpoints']),'Missing rollout endpoints')
    require(set(cfg['datasets'])<=set(('cloudops','openrca','cujbench','itbench_aa')) and bool(cfg['datasets']),'Unknown/empty dataset selection')
    return cfg


def cases(cfg,split=None):
    rows=read_jsonl(ROOT/cfg['cases'])
    if cfg.get('datasets'):
        rows=[r for r in rows if r['dataset'] in cfg['datasets']]
    if split:rows=[r for r in rows if r['split']==split]
    require(bool(rows),'No cases selected')
    require(len({(r['dataset'],r['case_id']) for r in rows})==len(rows),'Duplicate case IDs')
    return rows


def selected_case(rows,dataset,case_id):
    matching=[r for r in rows if r['dataset']==dataset and r['case_id']==case_id]
    require(len(matching)==1,'Case missing or ambiguous')
    return matching[0]


def admit_data(cfg):
    path=Path(cfg['frozen_data_manifest'])
    require(path.is_file(),'Run prepare --freeze-data on the target-platform payloads first')
    freeze=load_json(path)
    require(freeze.get('complete') is True,'Frozen data manifest contains errors')
    require(freeze['protocol_sha256']==digest(cfg['protocol']),'Frozen data protocol differs')
    require(freeze['source_case_manifest_sha256']==file_hash(ROOT/cfg['cases']),'Frozen case membership differs')
    require(all(r['dataset']+'/'+r['case_id'] in freeze['entries'] for r in cases(cfg)),'Frozen manifest lacks selected cases')
    return freeze


def seed_for(base,*values):
    return (base+int(digest(values)[:8],16))%(2**31)


def new_run(path,kind,cfg):
    path=Path(path)
    require(not path.exists(),'Run directory already exists; use a new run_id or explicit checkpoint resume')
    path.mkdir(parents=True)
    versions={}
    for name in ('torch','transformers','tokenizers','safetensors','numpy','PyYAML','gdown'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    write_json(path/'run.json',dict(kind=kind,config=cfg,protocol_sha256=digest(cfg['protocol']),
               split_manifest_sha256=file_hash(ROOT/cfg['cases']),code_version='2.1.0',
               environment=dict(python=platform.python_version(),packages=versions)))
    return path
