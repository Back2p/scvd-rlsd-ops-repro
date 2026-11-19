"""Read-only CPU preflight; optionally inspect an installed GPU runtime without allocating a model."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

from .common import ROOT,cases,config,digest,file_hash,load_json,require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--runtime',action='store_true',help='Inspect installed Torch/Transformers/CUDA availability; no model construction')
    args=parser.parse_args();cfg=config(args.config)
    rows=cases(cfg)
    require(len(rows)==len({(r['dataset'],r['case_id']) for r in rows}),'Duplicate IDs')
    train={(r['dataset'],r['case_id']) for r in rows if r['split']=='train'}
    held={(r['dataset'],r['case_id']) for r in rows if r['split'] in ('val','test')}
    require(not train&held,'Train/evaluation overlap')
    result=dict(code_and_config_valid=True,case_ids=len(rows),train_cases=len(train),heldout_cases=len(held),
        protocol_sha256=digest(cfg['protocol']),parameter_scope='text-only Qwen3.5 + LM head',
        original_html_directly_comparable=False,training_started=False,
        frozen_data_manifest_present=Path(cfg['frozen_data_manifest']).is_file(),
        model_weights_present=bool(list(Path(cfg['model']['path']).glob('*.safetensors'))))
    if result['frozen_data_manifest_present']:
        freeze=load_json(cfg['frozen_data_manifest'])
        result['frozen_data_complete']=freeze.get('complete',False)
        result['frozen_protocol_matches']=freeze.get('protocol_sha256')==digest(cfg['protocol'])
        result['frozen_case_membership_matches']=freeze.get('source_case_manifest_sha256')==file_hash(ROOT/cfg['cases'])
    if args.runtime:
        import torch
        import transformers
        result['runtime']=dict(torch=torch.__version__,transformers=transformers.__version__,
            qwen_text_class_available=hasattr(transformers,'Qwen3_5ForCausalLM'),cuda_available=torch.cuda.is_available(),
            cuda_initialized=torch.cuda.is_initialized())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
