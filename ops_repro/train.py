"""Full text-parameter FSDP trainers; importing this CLI never starts training."""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
from pathlib import Path
import random

from .clients import TokenClient
from .common import ROOT,admit_data,append_jsonl,cases,config,digest,file_hash,new_run,read_jsonl,require,seed_for,write_json


def plan(cfg,method,world_size):
    if method=='scvd':
        batch=cfg['scvd']['global_batch_size']
        require(batch%world_size==0,'SCVD global batch must be divisible by data-parallel world size')
        return dict(method=method,global_batch_size=batch,dp_world_size=world_size,
                    per_device_microbatch=1,accumulation_microbatches=batch//world_size,
                    epochs=cfg['scvd']['epochs'],parameter_scope='all text model and LM head parameters',training_started=False)
    r=cfg['rlsd']
    require(cfg['sampling']['temperature']==1 and cfg['sampling']['top_p']==1 and cfg['sampling']['top_k']==0,
            'RLSD requires unwarped temperature=1/top_p=1/top_k=0 in this implementation')
    return dict(method=method,dp_world_size=world_size,prompts_per_batch=r['prompt_batch_size'],
                rollouts_per_prompt=r['rollouts_per_prompt'],max_optimizer_updates=r['max_updates'],
                max_rollout_batches=r['max_rollout_batches'],training_started=False)


def collect_rlsd(cfg,rows,tok,publication,batch_index,output,*,privileged=True):
    """Coordinator-only. No labels are sent to the student inference service."""
    from .datasets import SnapshotCase,tools
    from .models import prompt_ids
    from .rollout import episode
    from .tensor_ops import grpo_advantages
    r=cfg['rlsd'];rng=random.Random(seed_for(cfg['seed'],'batch',batch_index))
    chosen=rng.choices(rows,k=r['prompt_batch_size'])
    tasks=[(draw,row,mode,sample) for draw,row in enumerate(chosen)
           for mode in [cfg['modes'][rng.randrange(len(cfg['modes']))]] for sample in range(r['rollouts_per_prompt'])]
    # One group is one drawn prompt and one thinking mode, never a mix of modes.
    def work(index,task):
        draw,row,mode,sample=task
        case=SnapshotCase(row,cfg)
        trace=episode(case,TokenClient(cfg['rollout'],index),mode=mode,
            seed=seed_for(cfg['seed'],batch_index,draw,sample),revision=publication['revision'],
            budget=cfg['budget'],sampling=cfg['sampling'])
        return dict(draw=draw,sample=sample,mode=mode,trace=trace,hint=case.teacher_hint() if privileged else None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cfg['rollout']['endpoints'])) as pool:
        futures=[pool.submit(work,i,t) for i,t in enumerate(tasks)]
        episodes=[];errors=[]
        for i,future in enumerate(futures):
            try:
                value=future.result();episodes.append(value)
                append_jsonl(output/'rollout_attempts'/f'batch-{batch_index:06}.jsonl',dict(index=i,status='collected',draw=value['draw'],sample=value['sample'],trace=value['trace']))
            except Exception as error:
                errors.append(dict(index=i,error_type=type(error).__name__,error=str(error)))
                append_jsonl(output/'rollout_attempts'/f'batch-{batch_index:06}.jsonl',dict(index=i,status='infra_error',error_type=type(error).__name__,error=str(error)))
        require(not errors,'Incomplete on-policy batch; no optimizer update. Inspect rollout_attempts receipts')
    flat=[];total_tokens=0;zero_groups=0
    template_hash=hashlib.sha256(tok.chat_template.encode()).hexdigest()
    for draw in range(len(chosen)):
        group=[e for e in episodes if e['draw']==draw]
        require(len(group)==r['rollouts_per_prompt'],'Incomplete GRPO group')
        adv=grpo_advantages([e['trace']['score'] for e in group])
        zero_groups+=int(all(a==0 for a in adv))
        for item,advantage in zip(group,adv):
            append_jsonl(output/'rollouts'/f'batch-{batch_index:06}.jsonl',dict(draw=draw,sample=item['sample'],trace=item['trace'],advantage=advantage))
            for turn in item['trace']['turns']:
                response=turn['response'];actions=response['generated_ids'];old=response['sampling_logprobs']
                require(response['template_sha256']==template_hash,'Sampler/trainer tokenizer template differs')
                require(response['checkpoint_sha256']==publication['checkpoint_sha256'],'Sampler checkpoint fingerprint differs from published weights')
                require(response['logprob_semantics']=='untempered_full_vocabulary','Unsupported sampling logprob convention')
                require(len(actions)==len(old)>0,'Missing actual rollout-time action logprobs')
                total_tokens+=len(actions)
                if advantage==0:continue
                prefix=response['prompt_ids']
                record=dict(input_ids=prefix+actions,labels=[-100]*len(prefix)+actions,
                            old_logp=old,advantage=advantage)
                if privileged:
                    teacher_messages=copy.deepcopy(turn['messages'])
                    teacher_messages[0]['content']+='\n\nTraining-only privileged final answer (not visible to the student):\n'+item['hint']
                    teacher_prefix=prompt_ids(tok,teacher_messages,tools(),item['mode'])
                    require(len(teacher_prefix)+len(actions)<=cfg['budget']['max_context'],'Teacher context overlength; abort instead of truncating sampled actions')
                    record.update(teacher_ids=teacher_prefix+actions,teacher_labels=[-100]*len(teacher_prefix)+actions)
                flat.append(record)
    return dict(records=flat,total_tokens=total_tokens,zero_advantage_groups=zero_groups,
                groups=len(chosen),episodes=len(episodes),mean_reward=sum(e['trace']['score'] for e in episodes)/len(episodes))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--method',choices=['scvd','rlsd','grpo'],required=True)
    parser.add_argument('--records',help='Encoded and receipted SCVD JSONL')
    parser.add_argument('--output',required=True)
    parser.add_argument('--resume',help='Trusted complete checkpoint made by this package; output must be a NEW run directory')
    parser.add_argument('--export-only',action='store_true',help='Restore a trusted checkpoint into a NEW run and publish it without sampling or optimizer updates; requires --resume')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--world-size',type=int,default=8,help='Planning only; actual size comes from torchrun')
    args=parser.parse_args();cfg=config(args.config)
    require(not args.export_only or args.resume,'--export-only requires a complete --resume checkpoint')
    if args.plan:
        print(json.dumps(plan(cfg,args.method,args.world_size),indent=2));return
    admit_data(cfg)
    cfg['_method']=args.method
    # Only this branch imports GPU training code. Do not execute it during package preparation.
    import torch
    import torch.distributed as dist
    from . import distributed as D
    from .models import tokenizer
    from .tensor_ops import mixture,ppo_token_losses,rlsd_weights
    rank,world,device=D.init(cfg['seed'],cfg['training']);plan(cfg,args.method,world)
    tok=tokenizer(cfg['model']['path']);rows=cases(cfg,'train')
    records=None
    if args.method=='scvd':
        require(args.records,'SCVD requires --records')
        records=read_jsonl(args.records);receipt=json.loads(Path(args.records+'.receipt.json').read_text())
        require(file_hash(args.records)==receipt['output_sha256'],'Encoded SCVD data hash differs from receipt')
        require(receipt['template_sha256']==hashlib.sha256(tok.chat_template.encode()).hexdigest(),'SCVD template differs')
        require(receipt.get('model_revision')==cfg['model']['revision'],'Encoded SCVD student revision differs')
        require(receipt.get('teacher_revision')==cfg['teacher']['revision'],'Encoded SCVD teacher revision differs')
        require(receipt.get('protocol_sha256')==digest(cfg['protocol']),'Encoded SCVD protocol differs')
        require(receipt.get('frozen_data_sha256')==file_hash(cfg['frozen_data_manifest']),'Encoded SCVD data freeze differs')
        train_ids={(r['dataset'],r['case_id']) for r in rows}
        require(all((r['dataset'],r['case_id']) in train_ids for r in records),'SCVD encoded data includes a non-training case')
        require(bool(records),'No SCVD records')
        cfg['_training_input_sha256']=file_hash(args.records)
        records.sort(key=lambda r:(r['dataset'],r['case_id'],r['mode'],r['record_sha256']))
    cfg['_frozen_data_sha256']=file_hash(cfg['frozen_data_manifest'])
    output=Path(args.output).resolve()
    D.primary(lambda:str(new_run(output,args.method,cfg)))
    student=D.wrap(cfg,device)
    teacher=D.wrap(cfg,device,frozen=True) if args.method=='rlsd' else None
    learning=cfg['scvd'] if args.method=='scvd' else cfg['rlsd']
    optimizer=torch.optim.AdamW(student.parameters(),lr=learning['learning_rate'],weight_decay=learning['weight_decay'])
    cursor={'updates':0,'batches':0}
    if args.resume:cursor=D.load_resume(student,teacher,optimizer,args.resume,cfg)
    elif teacher is not None:D.sync_teacher(student,teacher)
    publication=D.publish(student,tok,output,cursor['updates'],cfg)
    if args.export_only:
        D.primary(lambda:write_json(output/'result.json',dict(method=args.method,cursor=cursor,
            final_publication=publication,export_only=True,optimizer_updates_executed=0)))
        dist.barrier();dist.destroy_process_group();return

    def reload_all():
        return [TokenClient(cfg['rollout'],i).reload(publication['path'],publication['revision']) for i in range(len(cfg['rollout']['endpoints']))]
    if args.method!='scvd':D.primary(reload_all)

    def inputs(record,teacher_side=False):
        ids=record['teacher_ids' if teacher_side else 'input_ids']
        labels=record['teacher_labels' if teacher_side else 'labels']
        x=torch.tensor([ids],dtype=torch.long,device=device)
        return dict(input_ids=x,attention_mask=torch.ones_like(x),labels=torch.tensor([labels],dtype=torch.long,device=device))

    last_saved_cursor=None
    def checkpoint_if_due(final=False):
        nonlocal last_saved_cursor
        if last_saved_cursor!=cursor and (final or cursor['updates']%cfg['training']['save_every']==0):
            D.save_resume(student,teacher,optimizer,output,cursor,cfg)
            last_saved_cursor=dict(cursor)
            D.primary(lambda:D.prune_owned(output,'states',cfg['training']['keep_resume_checkpoints']))

    if args.method=='scvd':
        b=learning['global_batch_size'];all_batches=[]
        for epoch in range(learning['epochs']):
            order=list(range(len(records)));random.Random(cfg['seed']+epoch).shuffle(order)
            all_batches.extend([[records[i] for i in order[start:start+b]] for start in range(0,len(order),b)])
        warmup=max(1,math.ceil(len(all_batches)*learning['warmup_ratio']))
        for index,batch in enumerate(all_batches):
            if index<cursor['batches']:continue
            total_tokens=sum(sum(t!=-100 for t in row['labels'][1:]) for row in batch)
            require(total_tokens>0,'Empty SCVD optimizer batch')
            optimizer.zero_grad(set_to_none=True)
            slots=math.ceil(b/world)
            loss_total=torch.zeros((),device=device)
            for slot in range(slots):
                position=slot*world+rank
                record=batch[position] if position<len(batch) else {'input_ids':batch[0]['input_ids'][:2],'labels':[-100,-100]}
                value=student(**inputs(record))
                nll=-value['log_probs'].sum()+value['zero']
                loss=nll*world/total_tokens
                D.assert_finite(loss,'Nonfinite SCVD loss')
                loss.backward();loss_total+=nll.detach()
            step=cursor['updates']
            factor=min(1.,(step+1)/warmup) if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/max(1,len(all_batches)-warmup)))
            for group in optimizer.param_groups:group['lr']=learning['learning_rate']*factor
            norm=student.clip_grad_norm_(learning['max_grad_norm']);D.assert_finite(norm,'Nonfinite SCVD gradient norm')
            optimizer.step();cursor={'updates':step+1,'batches':index+1}
            dist.all_reduce(loss_total)
            D.primary(lambda:append_jsonl(output/'metrics.jsonl',dict(**cursor,teacher_tokens=total_tokens,teacher_token_nll=loss_total.item()/total_tokens,gradient_norm=norm.item())))
            checkpoint_if_due()
        if cursor['updates']!=int(Path(publication['path']).name.split('-')[1]):
            publication=D.publish(student,tok,output,cursor['updates'],cfg)
    else:
        while cursor['updates']<learning['max_updates'] and cursor['batches']<learning['max_rollout_batches']:
            batch,local_records=D.collect_sharded(lambda:collect_rlsd(cfg,rows,tok,publication,cursor['batches'],output,privileged=teacher is not None))
            cursor['batches']+=1
            D.primary(lambda:append_jsonl(output/'batch_metrics.jsonl',dict(**cursor,**batch,optimizer_step_skipped=not bool(local_records))))
            if not local_records:continue
            total_tokens=batch['total_tokens'];optimizer.zero_grad(set_to_none=True)
            loss_total=torch.zeros((),device=device);max_drift=torch.zeros((),device=device)
            for record in local_records:
                real=not record['padding']
                old=torch.tensor(record['old_logp'],device=device,dtype=torch.float32)
                if teacher is not None:
                    with torch.no_grad():teacher_lp=teacher(**inputs(record,True))['log_probs'].detach()
                    D.assert_finite(teacher_lp,'Nonfinite frozen-teacher logprob')
                    weights=rlsd_weights(record['advantage'],teacher_lp,old,
                        lam=mixture(cursor['updates']+1,learning['lambda_start'],learning['lambda_decay_steps']),evidence_clip=learning['evidence_clip'])
                else:weights=torch.ones_like(old)
                value=student(**inputs(record));new=value['log_probs']
                drift=(new.detach()-old).abs().max() if real else torch.zeros((),device=device)
                dist.all_reduce(drift,op=dist.ReduceOp.MAX);max_drift=torch.maximum(max_drift,drift)
                require(bool(torch.isfinite(drift)) and drift.item()<=learning['max_logprob_drift'],
                    'Sampler/trainer logprob mismatch: check exact checkpoint, tokenizer, dtype and attention implementation')
                losses=ppo_token_losses(new,old,record['advantage'],weights,
                    learning['clip_low'],learning['clip_high'],learning['clip_dual'])
                loss=(losses.sum()+value['zero'])*world/total_tokens
                D.assert_finite(loss,'Nonfinite policy loss')
                loss.backward();loss_total+=losses.detach().sum()
            norm=student.clip_grad_norm_(learning['max_grad_norm']);D.assert_finite(norm,'Nonfinite policy gradient norm')
            optimizer.step();cursor['updates']+=1
            if teacher is not None and cursor['updates']%learning['teacher_sync_steps']==0:D.sync_teacher(student,teacher)
            dist.all_reduce(loss_total)
            D.primary(lambda:append_jsonl(output/'metrics.jsonl',dict(**cursor,policy_token_loss=loss_total.item()/total_tokens,
                max_logprob_drift=max_drift.item(),gradient_norm=norm.item())))
            publication=D.publish(student,tok,output,cursor['updates'],cfg)
            D.primary(reload_all)
            checkpoint_if_due()
            D.primary(lambda:D.prune_owned(output,'publications',cfg['training']['keep_inference_publications'],active=[publication['path']]))
    checkpoint_if_due(final=True)
    D.primary(lambda:write_json(output/'result.json',dict(method=args.method,cursor=cursor,final_publication=publication,
        completed_schedule=(cursor['updates']>=learning['max_updates']) if args.method!='scvd' else cursor['batches']==len(all_batches))))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
