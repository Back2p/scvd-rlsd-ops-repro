"""Parse native Qwen3.5 tool XML without executing any model-generated text."""
import json
import re

from .common import require


def assistant_message(raw):
    text=raw.replace('<|im_end|>','').replace('<|endoftext|>','').strip()
    reasoning=None
    if '</think>' in text:
        reasoning,text=text.split('</think>',1)
        reasoning=reasoning.removeprefix('<think>').strip()
        text=text.strip()
    calls=[]
    for block in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S):
        function=re.fullmatch(r'<function=([A-Za-z0-9_]+)>\s*(.*?)\s*</function>',block,re.S)
        if function:
            name,body=function.groups();arguments={}
            for key,value in re.findall(r'<parameter=([A-Za-z0-9_]+)>\s*(.*?)\s*</parameter>',body,re.S):
                require(key not in arguments,'Duplicate function parameter')
                try:arguments[key]=json.loads(value)
                except json.JSONDecodeError:arguments[key]=value
        else:
            item=json.loads(block);name=item['name'];arguments=item.get('arguments',{})
        require(isinstance(arguments,dict),'Tool arguments must be an object')
        calls.append(dict(id='call_'+str(len(calls)),type='function',function=dict(name=name,arguments=arguments)))
    content=re.sub(r'<tool_call>.*?</tool_call>','',text,flags=re.S).strip()
    return dict(role='assistant',content=content,reasoning_content=reasoning,tool_calls=calls or None)
