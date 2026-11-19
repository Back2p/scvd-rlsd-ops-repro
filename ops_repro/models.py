"""Selected-token projection avoids a [sequence, full-vocabulary] loss tensor."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def load_policy(path,*,dtype=torch.float32,attention='sdpa'):
    from transformers import Qwen3_5ForCausalLM
    # Official Transformers text-only model class; vision parameters are omitted.
    base,loading=Qwen3_5ForCausalLM.from_pretrained(path,torch_dtype=dtype,local_files_only=True,
        trust_remote_code=False,attn_implementation=attention,output_loading_info=True)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError('Text checkpoint loading is incomplete; do not train newly initialized missing weights')
    base._loading_evidence=loading
    return base


def tokenizer(path):
    from transformers import AutoTokenizer
    result=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    if result.pad_token_id is None:result.pad_token_id=result.eos_token_id
    return result


def prompt_ids(tok,messages,tools,mode):
    return tok.apply_chat_template(messages,tools=tools,tokenize=True,add_generation_prompt=True,
                                   enable_thinking=mode=='thinking')


class SelectedPolicy(torch.nn.Module):
    def __init__(self,base,chunk_size=64):
        super().__init__();self.base=base;self.chunk_size=chunk_size

    def forward(self,input_ids,attention_mask,labels):
        # labels are unshifted. Only generated/teacher tokens have labels != -100.
        if input_ids.shape!=labels.shape or labels[:,0].ne(-100).any():raise ValueError('Invalid unshifted labels')
        if ((attention_mask==0)&(labels!=-100)).any():raise ValueError('Padding is supervised')
        states=self.base.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True).last_hidden_state
        batch,index=torch.nonzero(labels[:,1:]!=-100,as_tuple=True)
        count=index.numel()
        if count:
            hidden=states[batch,index];targets=labels[batch,index+1]
        else:
            hidden=states[:1,:1,:].reshape(1,-1);targets=input_ids[:1,0]
        def project(h,t):
            logits=self.base.lm_head(h).float()
            return -F.cross_entropy(logits,t,reduction='none')
        pieces=[]
        for start in range(0,hidden.shape[0],self.chunk_size):
            h,t=hidden[start:start+self.chunk_size],targets[start:start+self.chunk_size]
            lp=checkpoint(project,h,t,use_reentrant=False) if torch.is_grad_enabled() else project(h,t)
            pieces.append(lp)
        values=torch.cat(pieces)
        return {'log_probs':values if count else values[:0],'zero':values.sum()*0,'valid_tokens':count}
