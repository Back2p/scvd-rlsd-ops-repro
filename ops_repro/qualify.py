"""Optional platform-only zero-update GPU qualification. Never run by packaging."""
import argparse
import json
from pathlib import Path

from .common import config,new_run,require,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sequence-length',type=int,default=128)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    require(8<=args.sequence_length<=cfg['budget']['max_context'],'Invalid qualification length')
    if args.plan:
        print(json.dumps(dict(stage='zero_update_gpu_qualification',sequence_length=args.sequence_length,
            optimizer_steps=0,training_started=False,model_loaded=False),indent=2));return
    import torch
    import torch.distributed as dist
    from . import distributed as D
    from .models import tokenizer
    rank,world,device=D.init(cfg['seed'],cfg['training']);output=Path(args.output).resolve()
    D.primary(lambda:str(new_run(output,'zero_update_synthetic_qualification',cfg)))
    model=D.wrap(cfg,device);tok=tokenizer(cfg['model']['path'])
    token=tok.encode(' check',add_special_tokens=False)[0]
    ids=torch.full((1,args.sequence_length),token,dtype=torch.long,device=device)
    labels=ids.clone();labels[:,:-4]=-100
    torch.cuda.reset_peak_memory_stats()
    result=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)
    loss=-result['log_probs'].mean()+result['zero'];D.assert_finite(loss,'Nonfinite synthetic loss')
    loss.backward();norm=model.clip_grad_norm_(1.);D.assert_finite(norm,'Nonfinite synthetic gradient')
    # No optimizer is constructed and no optimizer.step() is called.
    publication=D.publish(model,tok,output,0,cfg)
    stats=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device=device,dtype=torch.float64)
    dist.all_reduce(stats,op=dist.ReduceOp.MAX)
    D.primary(lambda:write_json(output/'qualification.json',dict(status='ZERO_UPDATE_SYNTHETIC_ONLY',world_size=world,
        sequence_length=args.sequence_length,loss=loss.item(),finite_gradient_norm=norm.item(),optimizer_steps=0,
        max_allocated_bytes=stats[0].item(),max_reserved_bytes=stats[1].item(),publication=publication,
        limitation='No Adam states, rollout integration, real benchmark execution, throughput or learning claim')))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
