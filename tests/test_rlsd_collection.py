import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.train import collect_rlsd


class Tokenizer:
    chat_template='SYNTHETIC_TEMPLATE_ONLY'
    def apply_chat_template(self,messages,**kwargs):return [ord(c) for c in json.dumps(messages,ensure_ascii=False)]


class FakeCase:
    def __init__(self,row,cfg):self.row=row;self.dataset=row['dataset'];self.context_revision='SYNTHETIC';self.evidence={}
    def initial_messages(self):return [dict(role='system',content='Synthetic task'),dict(role='user',content='Find synthetic fault')]
    def score(self,submission):return int(submission=={'answer':'correct'})
    def scorer_revision(self):return 'SYNTHETIC_SCORER'
    def teacher_hint(self):return 'GOLD_CANARY_TEACHER_ONLY'


class FakeClient:
    student_views=[]
    def __init__(self,cfg,index=0):self.index=index;self.turn=0
    def generate(self,messages,tools,**kwargs):
        FakeClient.student_views.append(json.dumps(messages))
        action='propose_diagnosis' if self.turn==0 else 'submit_diagnosis'
        answer='correct' if self.index%2==0 else 'wrong'
        self.turn+=1
        return dict(message=dict(role='assistant',content='',tool_calls=[dict(id='synthetic_'+str(self.turn),type='function',
            function=dict(name=action,arguments=dict(diagnosis={'answer':answer})))]),
            prompt_ids=Tokenizer().apply_chat_template(messages),generated_ids=[33],sampling_logprobs=[-.1],
            policy_revision=kwargs['revision'],checkpoint_sha256='a'*64,
            template_sha256=hashlib.sha256(Tokenizer.chat_template.encode()).hexdigest(),
            logprob_origin='captured_during_rollout',logprob_semantics='untempered_full_vocabulary')


class RLSDCollectionTests(unittest.TestCase):
    def config(self):
        return dict(seed=7,modes=['nonthinking'],rlsd=dict(prompt_batch_size=1,rollouts_per_prompt=2),
            rollout=dict(endpoints=['synthetic-worker-1','synthetic-worker-2']),
            budget=dict(max_turns=3,max_tool_calls=4,max_new_tokens=10,max_generated_tokens=30,max_context=32768),
            sampling=dict(temperature=1.,top_p=1.,top_k=0))

    def test_online_tokens_and_teacher_hint_alignment(self):
        FakeClient.student_views=[]
        row=dict(dataset='cloudops',case_id='SYNTHETIC_CASE',split='train')
        with tempfile.TemporaryDirectory() as folder,patch('ops_repro.datasets.SnapshotCase',FakeCase),patch('ops_repro.train.TokenClient',FakeClient):
            result=collect_rlsd(self.config(),[row],Tokenizer(),dict(revision='SYNTHETIC_REVISION',checkpoint_sha256='a'*64),0,Path(folder))
        self.assertEqual(result['episodes'],2)
        self.assertEqual(result['zero_advantage_groups'],0)
        self.assertEqual(result['total_tokens'],4)
        self.assertEqual(len(result['records']),4)
        self.assertTrue(all('GOLD_CANARY' not in text for text in FakeClient.student_views))
        for record in result['records']:
            self.assertIn('GOLD_CANARY_TEACHER_ONLY',''.join(map(chr,record['teacher_ids'][:-1])))
            self.assertNotIn('GOLD_CANARY',''.join(map(chr,record['input_ids'][:-1])))
            self.assertEqual(record['labels'][-1],record['teacher_labels'][-1])
            self.assertEqual(record['old_logp'],[-.1])

    def test_policy_hash_mismatch_aborts_before_any_update(self):
        row=dict(dataset='cloudops',case_id='SYNTHETIC_CASE',split='train')
        with tempfile.TemporaryDirectory() as folder,patch('ops_repro.datasets.SnapshotCase',FakeCase),patch('ops_repro.train.TokenClient',FakeClient):
            with self.assertRaisesRegex(ValueError,'fingerprint differs'):
                collect_rlsd(self.config(),[row],Tokenizer(),dict(revision='SYNTHETIC',checkpoint_sha256='b'*64),0,Path(folder))


if __name__=='__main__':unittest.main()
