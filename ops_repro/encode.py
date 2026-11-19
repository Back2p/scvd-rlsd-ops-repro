import argparse
import hashlib
import json
from pathlib import Path
import sys

from .common import ROOT,admit_data,append_jsonl,cases,config,digest,file_hash,read_jsonl,require,write_json


def main():
    parser=argparse.ArgumentParser(description='Encode verified SCVD continuations with the pinned native tokenizer')
    parser.add_argument('--config',required=True)
    parser.add_argument('--records',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();cfg=config(args.config)
    admit_data(cfg)
    freeze_sha=file_hash(cfg['frozen_data_manifest'])
    require(not Path(args.output).exists(),'Encoding output already exists')
    sys.path.insert(0,str(ROOT/'src'))
    from scvd_encode import encode
    from .models import tokenizer
    tok=tokenizer(cfg['model']['path'])
    template=hashlib.sha256(tok.chat_template.encode()).hexdigest()
    case_index={(r['dataset'],r['case_id']):r for r in cases(cfg,'train')}
    records=read_jsonl(args.records);count=0;held=0
    for record in records:
        try:
            require(record.get('student_revision')==cfg['model']['revision'],'SCVD student revision differs')
            require(record.get('teacher_revision')==cfg['teacher']['revision'],'SCVD teacher revision differs')
            require(record.get('protocol_sha256')==digest(cfg['protocol']),'SCVD collection protocol differs')
            require(record.get('frozen_data_sha256')==freeze_sha,'SCVD collection data freeze differs')
            value=encode(record,tok,case_index,template_sha256=template,max_length=cfg['budget']['max_context'])
            value.update(dataset=record['dataset'],case_id=record['case_id'],mode=record['mode'],
                         record_sha256=digest(record))
            append_jsonl(args.output,value);count+=1
        except ValueError as error:
            append_jsonl(str(args.output)+'.held.jsonl',dict(dataset=record.get('dataset'),case_id=record.get('case_id'),reason=str(error)));held+=1
    require(count>0,'No admitted SCVD records; inspect held records before training')
    write_json(str(args.output)+'.receipt.json',dict(source_sha256=file_hash(args.records),output_sha256=file_hash(args.output),
        admitted=count,held=held,template_sha256=template,source_records=len(records),
        model_revision=cfg['model']['revision'],teacher_revision=cfg['teacher']['revision'],
        protocol_sha256=digest(cfg['protocol']),frozen_data_sha256=freeze_sha))


if __name__=='__main__':main()
