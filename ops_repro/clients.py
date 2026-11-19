"""HTTP clients; requests are made only when their methods are called."""
import copy
import json
import os
import re
import urllib.error
import urllib.request

from .common import require


class ProviderError(RuntimeError):
    """Transport/provider failure; never convert it into a task reward."""


class ContextLimit(RuntimeError):
    def __init__(self,message,identity=None):
        super().__init__(message)
        self.identity=identity or {}



def post(url,body,key_env=None,timeout=600):
    headers={'Content-Type':'application/json'}
    if key_env:
        key=os.environ.get(key_env)
        require(bool(key),'Missing API key environment variable: '+key_env)
        headers['Authorization']='Bearer '+key
    data=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
    try:
        request=urllib.request.Request(url,data=data,headers=headers,method='POST')
        with urllib.request.urlopen(request,timeout=timeout) as response:
            return json.load(response)
    except (urllib.error.URLError,TimeoutError,OSError) as error:
        # Do not echo request headers or full provider bodies into logs.
        raise ProviderError(f'Provider request failed: {type(error).__name__}') from error


class TokenClient:
    def __init__(self,cfg,index=0):
        self.cfg=cfg
        endpoints=cfg['endpoints']
        require(bool(endpoints),'Configure at least one token-worker endpoint')
        self.endpoint=endpoints[index%len(endpoints)].rstrip('/')

    def generate(self,messages,tools,*,mode,seed,revision,max_new_tokens,sampling):
        result=post(self.endpoint+'/generate',dict(messages=messages,tools=tools,mode=mode,seed=seed,
            expected_revision=revision,max_new_tokens=max_new_tokens,max_context=self.cfg.get('max_context',30720),
            sampling=sampling),self.cfg.get('api_key_env'))
        require(result.get('policy_revision')==revision,'Rollout server did not use the requested policy revision')
        require(isinstance(result.get('checkpoint_sha256'),str) and len(result['checkpoint_sha256'])==64,'Missing sampler checkpoint fingerprint')
        if result.get('context_exhausted'):raise ContextLimit('Actor context budget exhausted',{k:result[k] for k in ('policy_revision','checkpoint_sha256')})
        require(result.get('logprob_origin')=='captured_during_rollout','Raw rollout-time logprobs are required')
        ids=result.get('generated_ids',[])
        require(bool(ids) and len(ids)==len(result.get('sampling_logprobs',[])),'Misaligned sampled tokens/logprobs')
        return result

    def reload(self,path,revision):
        result=post(self.endpoint+'/reload',dict(path=str(path),expected_revision=revision),self.cfg.get('api_key_env'),timeout=1800)
        require(result.get('policy_revision')==revision,'Weight publication/reload was not acknowledged')
        return result


class TeacherClient:
    """External OpenAI-compatible teacher for SCVD hard-label continuations."""
    def __init__(self,cfg):
        require(all(cfg.get(k) and 'SET_' not in cfg[k] for k in ('url','model','revision')),
                'Configure the actual SCVD teacher URL, model and provider version before collection')
        if cfg.get('api_key_env'):require(bool(os.environ.get(cfg['api_key_env'])),'Missing teacher API key environment variable')
        reserved={'model','messages','tools','temperature','top_p','max_tokens','max_completion_tokens','stream'}
        for extra in [cfg.get('extra_body',{})]+list(cfg.get('mode_extra_body',{}).values()):
            require(isinstance(extra,dict) and not reserved.intersection(extra),'Teacher extra_body cannot override identity, inputs, budget or sampling')
        self.cfg=cfg

    def generate(self,messages,tools,*,mode,seed,revision,max_new_tokens,sampling):
        wire=copy.deepcopy(messages)
        for message in wire:
            if self.cfg.get('message_reasoning_format','inline')=='inline':
                reasoning=message.pop('reasoning_content',None)
                if reasoning:message['content']='<think>\n'+reasoning+'\n</think>\n\n'+(message.get('content') or '')
            for call in message.get('tool_calls') or []:
                if isinstance(call['function']['arguments'],dict):
                    call['function']['arguments']=json.dumps(call['function']['arguments'],ensure_ascii=False)
        body=dict(model=self.cfg['model'],messages=wire,tools=tools,temperature=sampling['temperature'],
                  top_p=sampling.get('top_p',1.),max_tokens=max_new_tokens)
        body.update(self.cfg.get('extra_body',{}))
        body.update(self.cfg.get('mode_extra_body',{}).get(mode,{}))
        result=post(self.cfg['url'],body,self.cfg.get('api_key_env'))
        message=result['choices'][0]['message']
        message['content']=message.get('content') or ''
        if mode=='nonthinking' and self.cfg.get('nonthinking_reasoning')=='omit_from_supervision':
            message.pop('reasoning_content',None)
            message['content']=re.sub(r'<think>.*?</think>','',message['content'],flags=re.S).strip()
        for call in message.get('tool_calls') or []:
            if isinstance(call['function']['arguments'],str):
                call['function']['arguments']=json.loads(call['function']['arguments'])
        return dict(message=message,policy_revision=revision,usage=result.get('usage',{}),
                    stop_reason=result['choices'][0].get('finish_reason'),provider_response_id=result.get('id'),
                    provider_response_model=result.get('model'),requested_model=self.cfg['model'])
