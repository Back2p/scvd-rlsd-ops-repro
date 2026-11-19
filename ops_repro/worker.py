"""Single-GPU HF rollout worker with an explicit checkpoint reload handshake."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import hashlib
import os
from pathlib import Path
import threading

from .common import digest,file_hash,load_json,require
from .parsing import assistant_message


def verify_publication(path,revision):
    manifest=load_json(Path(path)/'publication.json')
    require(manifest['policy_revision']==revision,'Checkpoint revision differs from requested policy')
    for name,expected in manifest['files'].items():
        candidate=(Path(path)/name).resolve()
        require(candidate.is_relative_to(Path(path).resolve()),'Publication file escapes checkpoint directory')
        require(file_hash(candidate)==expected,'Checkpoint file checksum mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--allowed-reload-root',required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8100)
    parser.add_argument('--api-key-env')
    parser.add_argument('--max-context',type=int,default=32768)
    args=parser.parse_args()
    if args.host not in ('127.0.0.1','localhost','::1'):
        require(args.api_key_env and os.environ.get(args.api_key_env),'Network-exposed worker requires an API key environment variable')
    # Lazy imports keep --help/plan commands independent of GPU initialization.
    import torch
    from transformers import GenerationConfig
    from .models import SelectedPolicy,load_policy,prompt_ids,tokenizer
    require(torch.cuda.is_available(),'Rollout worker requires CUDA; CPU is reserved for tests')
    device=torch.device('cuda:0');lock=threading.RLock()
    initial_files={p.name:file_hash(p) for p in sorted(Path(args.model).glob('*.safetensors'))}
    require(bool(initial_files),'Require local safetensors model weights')
    state={'base':load_policy(args.model,dtype=torch.bfloat16).to(device).eval(),
           'tokenizer':tokenizer(args.model),'revision':args.revision,'model_path':args.model}
    state['checkpoint_sha256']=digest(initial_files)

    def generate(body):
        require(body['expected_revision']==state['revision'],'Worker still has an older checkpoint')
        require(body['mode'] in ('thinking','nonthinking'),'Unknown thinking mode')
        tok=state['tokenizer'];base=state['base']
        ids=prompt_ids(tok,body['messages'],body['tools'],body['mode'])
        n=int(body['max_new_tokens'])
        limit=min(args.max_context,int(body.get('max_context',args.max_context)))
        n=min(n,limit-len(ids))
        if n<=0:
            return dict(policy_revision=state['revision'],checkpoint_sha256=state['checkpoint_sha256'],context_exhausted=True,
                        logprob_origin='no_generation_context_limit')
        params=body['sampling'];temperature=float(params['temperature'])
        require(temperature>0,'Sampling temperature must be positive')
        gen_cfg=GenerationConfig(do_sample=True,temperature=temperature,top_p=float(params.get('top_p',1.)),
            top_k=int(params.get('top_k',0)),repetition_penalty=1.,max_new_tokens=n,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True)
        torch.manual_seed(int(body['seed']));torch.cuda.manual_seed_all(int(body['seed']))
        inputs=torch.tensor([ids],device=device,dtype=torch.long)
        with torch.inference_mode():
            output=base.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),generation_config=gen_cfg)
            generated=output[0,len(ids):].tolist()
            require(bool(generated),'Model returned no generated tokens')
            labels=output.clone();labels[:,:len(ids)]=-100
            projection=SelectedPolicy(base)
            scores=projection(output,torch.ones_like(output),labels)['log_probs'].float().cpu().tolist()
        raw=tok.decode(generated,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        try:message=assistant_message(raw)
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):
            message=dict(role='assistant',content=raw,tool_calls=None)
        for i,call in enumerate(message.get('tool_calls') or []):
            call['id']='call_'+digest({'prompt':ids,'seed':body['seed']})[:12]+'_'+str(i)
        return dict(message=message,prompt_ids=ids,generated_ids=generated,sampling_logprobs=scores,
            logprob_origin='captured_during_rollout',logprob_semantics='untempered_full_vocabulary',
            policy_revision=state['revision'],checkpoint_sha256=state['checkpoint_sha256'],
            template_sha256=hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            sampling=params,stop_reason='eos' if generated[-1]==tok.eos_token_id else 'length',
            raw_completion=raw)

    def reload(body):
        path=Path(body['path']).resolve()
        require(path.is_relative_to(Path(args.allowed_reload_root).resolve()),'Reload path is outside the configured run directory')
        manifest=verify_publication(path,body['expected_revision'])
        # Load new weights before releasing old weights. This deliberately requires
        # enough HBM for two BF16 9B models during a reload (~36 GB plus buffers).
        new_base=load_policy(str(path),dtype=torch.bfloat16).to(device).eval()
        new_tok=tokenizer(str(path))
        old=state['base'];state.update(base=new_base,tokenizer=new_tok,revision=body['expected_revision'],model_path=str(path),checkpoint_sha256=digest(manifest['files']))
        del old;torch.cuda.empty_cache()
        return {'policy_revision':state['revision']}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*unused):
            # Request bodies and authorization headers must never enter access logs.
            return None
        def do_POST(self):
            try:
                if args.api_key_env:
                    require(self.headers.get('Authorization')=='Bearer '+os.environ[args.api_key_env],'Unauthorized')
                length=int(self.headers.get('Content-Length','0'))
                require(0<length<=32*1024*1024,'Invalid request size')
                body=json.loads(self.rfile.read(length))
                with lock:
                    if self.path=='/generate':result=generate(body)
                    elif self.path=='/reload':result=reload(body)
                    elif self.path=='/health':result={'policy_revision':state['revision']}
                    else:raise ValueError('Unknown endpoint')
                status=200
            except Exception as error:
                result={'error':type(error).__name__,'message':str(error) if isinstance(error,ValueError) else 'Worker operation failed; inspect coordinator logs'}
                print(json.dumps({'worker_request_error':result},ensure_ascii=False),flush=True)
                status=400
            data=json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    print(json.dumps({'worker_ready':True,'host':args.host,'port':args.port,'policy_revision':state['revision']}),flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=='__main__':main()
