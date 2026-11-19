"""Native Qwen3.5 text/tool encoding. No model inference or environment replay.

Receipts are references to platform evidence; this encoder cannot authenticate
them. Platform collection and independent scoring must establish their validity.
"""
import copy
import hashlib
import json
import re

from protocol import require


def encode(record, tokenizer, cases, *, template_sha256, max_length=32768):
    require(type(max_length) is int and max_length > 0, 'Invalid max_length')
    template = tokenizer.chat_template
    require(isinstance(template, str), 'One explicit native chat template required')
    require(hashlib.sha256(template.encode()).hexdigest() == template_sha256, 'Chat template hash mismatch')
    case = cases.get((record.get('dataset'), record.get('case_id')))
    require(case is not None and case['split'] == 'train', 'Only publisher training cases accepted')
    require(record.get('mode') in ('thinking', 'nonthinking'), 'Explicit thinking mode required')
    for field in ('student_revision', 'teacher_revision', 'candidate_state_receipt', 'verifier_receipt'):
        require(isinstance(record.get(field), str) and bool(record[field]), f'Missing {field}')
    require(record.get('verifier_passed') is True, 'Require independent final verifier success')
    messages, sources, teacher_started = [], [], False
    for raw in record['messages']:
        source, role = raw.get('source'), raw.get('role')
        if source == 'collection_instruction':
            continue
        allowed = {'task': ('system', 'user'), 'student_prefix': ('assistant',),
                   'teacher_continuation': ('assistant',), 'tool_observation': ('tool',)}
        require(source in allowed and role in allowed[source], 'Invalid source/role')
        require(isinstance(raw.get('content'), str), 'Text-only schema; images need a separate processor')
        require(not teacher_started or source not in ('student_prefix', 'task'), 'Prefix/task after teacher takeover')
        teacher_started |= source == 'teacher_continuation'
        message = {k: copy.deepcopy(raw[k]) for k in
                   ('role', 'content', 'tool_calls', 'reasoning_content', 'tool_call_id', 'name') if k in raw}
        message.setdefault('tool_calls', None)
        message.setdefault('reasoning_content', None)
        for call in message.get('tool_calls') or []:
            require(isinstance(call.get('function', {}).get('arguments'), dict),
                    'Normalize tool arguments to objects before native Qwen rendering')
        require(not any(tag in json.dumps(message, ensure_ascii=False) for tag in
                        ('<|im_start|>', '<|im_end|>', '<|endoftext|>')), 'Reserved chat delimiter in message')
        if record['mode'] == 'nonthinking' and role == 'assistant':
            require(not message['reasoning_content'] and '<think>' not in message['content']
                    and '</think>' not in message['content'], 'Nonthinking record contains reasoning')
        messages.append(message)
        if role == 'assistant':
            sources.append(source)
    require('student_prefix' in sources and 'teacher_continuation' in sources, 'Missing student/teacher phase')
    require(isinstance(record.get('tools'), list) and bool(record['tools']), 'Frozen tool schemas required')
    rendered = tokenizer.apply_chat_template(messages, tools=record['tools'], tokenize=False,
                add_generation_prompt=False, enable_thinking=record['mode'] == 'thinking')
    segments = list(re.finditer(r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>', rendered, re.DOTALL))
    require(len(segments) == len(sources), 'Assistant boundary mismatch')
    spans = [(m.start(1), m.end()) for m, source in zip(segments, sources) if source == 'teacher_continuation']
    tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = tokens['input_ids']
    require(len(ids) <= max_length, 'Overlength; quarantine instead of silently truncating')
    labels = [-100] * len(ids)
    for i, (lo, hi) in enumerate(tokens['offset_mapping']):
        if hi <= lo:
            continue
        for start, end in spans:
            if lo >= start and hi <= end:
                labels[i] = ids[i]
            elif lo < end and hi > start:
                raise ValueError('Teacher boundary crosses a token')
    require(labels[0] == -100 and any(x != -100 for x in labels), 'Invalid/empty causal supervision')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels,
            'template_sha256': template_sha256,
            'rendered_sha256': hashlib.sha256(rendered.encode()).hexdigest(),
            'teacher_tokens': sum(x != -100 for x in labels)}
