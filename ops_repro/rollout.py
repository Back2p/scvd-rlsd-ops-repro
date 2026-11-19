import copy
import json

from .common import digest,require,seed_for
from .datasets import Session,tools
from .clients import ContextLimit


def clean(messages):
    return [{k:copy.deepcopy(v) for k,v in message.items() if k in
             ('role','content','tool_calls','reasoning_content','tool_call_id','name')} for message in messages]


def episode(case,client,*,mode,seed,revision,budget,sampling,messages=None,session_state=None,
            phase='student_prefix',stop_at_candidate=False,start_turn=0,start_generated_tokens=0):
    session=Session(case)
    if session_state is not None:session.restore(session_state)
    messages=copy.deepcopy(messages) if messages is not None else [dict(m,source='task') for m in case.initial_messages()]
    turns=[];used_tokens=start_generated_tokens;status='turn_budget';identity={}
    for turn in range(start_turn,budget['max_turns']):
        if len(session.calls)>=budget['max_tool_calls']:status='tool_budget';break
        remaining=budget['max_generated_tokens']-used_tokens
        if remaining<=0:status='token_budget';break
        try:
            response=client.generate(clean(messages),tools(),mode=mode,
                seed=seed_for(seed,turn),revision=revision,max_new_tokens=min(budget['max_new_tokens'],remaining),sampling=sampling)
        except ContextLimit as exc:
            identity=exc.identity;status='context_budget';break
        identity={k:response[k] for k in ('policy_revision','checkpoint_sha256') if k in response}
        message=copy.deepcopy(response['message']);message['role']='assistant';message['content']=message.get('content') or ''
        ids=response.get('generated_ids',[])
        tokens=len(ids) if ids else response.get('usage',{}).get('completion_tokens')
        require(isinstance(tokens,int) and tokens>0,'Provider must supply actual generated IDs or completion_tokens for budget accounting')
        require(tokens<=min(budget['max_new_tokens'],remaining),'Provider exceeded requested completion budget')
        used_tokens+=tokens
        turns.append(dict(response=response,messages=clean(messages),turn=turn))
        message['source']=phase;messages.append(message)
        calls=message.get('tool_calls') or []
        if not calls:status='invalid_no_tool_call';break
        if len(calls)>4:status='invalid_tool_batch';break
        if len(calls)>1 and any(isinstance(c,dict) and isinstance(c.get('function'),dict) and c['function'].get('name') in ('propose_diagnosis','submit_diagnosis') for c in calls):
            status='invalid_batched_diagnosis';break
        stop=False
        if any(not isinstance(call,dict) or not isinstance(call.get('function'),dict) for call in calls):
            status='invalid_tool_schema';break
        for index,call in enumerate(calls):
            function=call['function'];name=function.get('name');arguments=function.get('arguments',{})
            try:
                if isinstance(arguments,str):arguments=json.loads(arguments)
                require(isinstance(arguments,dict) and isinstance(name,str),'Tool name/arguments must be a string/JSON object')
            except (ValueError,TypeError):
                status='invalid_tool_arguments';stop=True;break
            if len(session.calls)>=budget['max_tool_calls']:
                result={'error':'tool budget exhausted'};status='tool_budget';stop=True
            else:
                try:result=session.execute(name,arguments)
                except (ValueError,TypeError,KeyError) as exc:result={'error':str(exc)}
            messages.append(dict(role='tool',tool_call_id=call.get('id','call_'+str(index)),
                                 content=json.dumps(result,ensure_ascii=False),source='tool_observation'))
            if session.closed:status='submitted';stop=True
            if stop_at_candidate and name=='propose_diagnosis' and session.draft is not None:
                status='candidate';stop=True
            if stop:break
        if stop:break
    return dict(dataset=case.dataset,case_id=case.row['case_id'],split=case.row['split'],mode=mode,
        seed=seed,policy_revision=revision,messages=messages,turns=turns,session_state=session.snapshot(),
        candidate=copy.deepcopy(session.draft),submission=copy.deepcopy(session.submission),
        status=status,generated_tokens=used_tokens,phase_generated_tokens=used_tokens-start_generated_tokens,
        checkpoint_identity=identity,context_revision=case.context_revision,
        score=case.score(session.submission) if session.submission is not None else 0,
        scorer_revision=case.scorer_revision(),actor_view_sha256=digest({'messages':case.initial_messages(),'tools':tools(),'context':case.context_revision}))


def collect_scvd(case,student,teacher,*,mode,seed,student_revision,teacher_revision,budget,sampling):
    require(case.row['split']=='train','SCVD collection only accepts training cases')
    prefix=episode(case,student,mode=mode,seed=seed,revision=student_revision,budget=budget,
                   sampling=sampling,stop_at_candidate=True)
    if prefix['status']!='candidate':return dict(accepted=False,reason=prefix['status'],prefix=prefix)
    handoff=digest({'messages':prefix['messages'],'session':prefix['session_state'],'context':case.context_revision})
    messages=copy.deepcopy(prefix['messages'])
    messages.append(dict(role='user',source='collection_instruction',content=
        'The existing diagnosis is a student candidate. Independently check it using the available evidence, '
        'correct any mistakes, and submit a final diagnosis. Do not assume that the candidate is wrong.'))
    continuation=episode(case,teacher,mode=mode,seed=seed_for(seed,'teacher'),revision=teacher_revision,
        budget=budget,sampling=sampling,messages=messages,session_state=prefix['session_state'],
        phase='teacher_continuation',start_turn=len(prefix['turns']),start_generated_tokens=prefix['generated_tokens'])
    accepted=continuation['status']=='submitted' and continuation['score']==1
    record=dict(dataset=case.dataset,case_id=case.row['case_id'],split='train',mode=mode,
        student_revision=student_revision,teacher_revision=teacher_revision,
        candidate_state_receipt=handoff,verifier_receipt=digest({'submission':continuation['submission'],
        'score':continuation['score'],'scorer':case.scorer_revision(),'case':case.row['case_id']}),
        verifier_passed=accepted,messages=continuation['messages'],tools=tools(),
        protocol='snapshot-reference-v2',teacher_generated_tokens=continuation['phase_generated_tokens'])
    return dict(accepted=accepted,reason=continuation['status'],record=record,prefix=prefix,continuation=continuation)
