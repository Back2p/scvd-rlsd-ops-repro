import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.clients import TeacherClient


class TeacherClientTests(unittest.TestCase):
    def test_generic_api_format_and_nonthinking_supervision(self):
        captured={}
        def fake_post(url,body,key_env):
            captured.update(body)
            return {'model':'SYNTHETIC_MODEL','id':'synthetic-response','choices':[{'finish_reason':'tool_calls',
                'message':{'role':'assistant','content':'<think>PRIVATE_TEACHER_REASON</think>visible response',
                           'reasoning_content':'PRIVATE_TEACHER_REASON','tool_calls':None}}]}
        cfg=dict(url='https://synthetic.invalid/v1/chat/completions',model='SYNTHETIC_MODEL',revision='synthetic-version',
                 nonthinking_reasoning='omit_from_supervision',message_reasoning_format='inline')
        messages=[dict(role='assistant',content='candidate',reasoning_content='STUDENT_REASON')]
        original=copy.deepcopy(messages)
        with patch('ops_repro.clients.post',fake_post):
            result=TeacherClient(cfg).generate(messages,[],mode='nonthinking',seed=1,revision='synthetic-version',max_new_tokens=10,sampling={'temperature':1.,'top_p':1.})
        self.assertEqual(messages,original)
        self.assertNotIn('reasoning_content',captured['messages'][0])
        self.assertIn('STUDENT_REASON',captured['messages'][0]['content'])
        self.assertNotIn('PRIVATE_TEACHER_REASON',str(result['message']))
        self.assertEqual(result['provider_response_model'],'SYNTHETIC_MODEL')


if __name__=='__main__':unittest.main()
