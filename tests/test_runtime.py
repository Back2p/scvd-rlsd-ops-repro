"""Synthetic sessions exercise real control flow without inference or training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.common import digest
from ops_repro.datasets import Session,SnapshotCase,safe_files
from ops_repro.parsing import assistant_message
from ops_repro.rollout import collect_scvd,episode
from ops_repro.clients import ContextLimit


class Case:
    def __init__(self):
        self.dataset='cloudops';self.row={'dataset':'cloudops','case_id':'SYNTHETIC','split':'train'}
        self.context_revision='synthetic-context';self.evidence={}
    def initial_messages(self):return [dict(role='system',content='Synthetic task only'),dict(role='user',content='Find synthetic fault')]
    def score(self,submission):return int(submission=={'answer':'correct'})
    def scorer_revision(self):return 'synthetic-scorer'


class Client:
    def __init__(self,actions):self.actions=copy.deepcopy(actions);self.inputs=[]
    def generate(self,messages,tools,**kwargs):
        self.inputs.append(copy.deepcopy(messages))
        name,args=self.actions.pop(0)
        return dict(message=dict(role='assistant',content='',tool_calls=[dict(id='synthetic-'+str(len(self.inputs)),type='function',function=dict(name=name,arguments=args))]),
            policy_revision=kwargs['revision'],generated_ids=[1],sampling_logprobs=[-.1],prompt_ids=[2],
            checkpoint_sha256='a'*64,logprob_origin='captured_during_rollout')


class RuntimeTests(unittest.TestCase):
    budget=dict(max_turns=5,max_tool_calls=8,max_new_tokens=100,max_generated_tokens=500)
    sampling=dict(temperature=1.,top_p=1.,top_k=0)

    def test_qwen_xml_reasoning_and_tool_arguments(self):
        raw='<think>check evidence</think><tool_call>\n<function=read_evidence>\n<parameter=evidence_id>e00000</parameter>\n<parameter=offset>12</parameter>\n</function>\n</tool_call><|im_end|>'
        result=assistant_message(raw)
        self.assertEqual(result['reasoning_content'],'check evidence')
        self.assertEqual(result['tool_calls'][0]['function']['arguments'],dict(evidence_id='e00000',offset=12))

    def test_scvd_restores_same_state_and_only_accepts_verified_continuation(self):
        student=Client([('propose_diagnosis',{'diagnosis':{'answer':'wrong'}})])
        teacher=Client([('list_evidence',{}),('submit_diagnosis',{'diagnosis':{'answer':'correct'}})])
        result=collect_scvd(Case(),student,teacher,mode='nonthinking',seed=7,student_revision='student-test',teacher_revision='teacher-test',budget=self.budget,sampling=self.sampling)
        self.assertTrue(result['accepted'])
        record=result['record']
        self.assertTrue(record['verifier_passed'])
        self.assertIn('student_prefix',{m['source'] for m in record['messages']})
        self.assertIn('teacher_continuation',{m['source'] for m in record['messages']})
        self.assertEqual(result['prefix']['session_state']['draft'],{'answer':'wrong'})
        self.assertEqual(result['continuation']['session_state']['calls'],['propose_diagnosis','list_evidence','submit_diagnosis'])

    def test_direct_submit_does_not_create_a_fake_candidate_boundary(self):
        student=Client([('submit_diagnosis',{'diagnosis':{'answer':'correct'}})])
        result=collect_scvd(Case(),student,Client([]),mode='nonthinking',seed=7,student_revision='s',teacher_revision='t',budget=self.budget,sampling=self.sampling)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'],'submitted')

    def test_privileged_hint_is_denied_for_heldout(self):
        case=object.__new__(SnapshotCase);case.row={'split':'test'}
        with self.assertRaisesRegex(ValueError,'forbidden'):case.teacher_hint()

    def test_context_limit_is_agent_budget_failure_not_infrastructure_reward(self):
        class Limited:
            def generate(self,*args,**kwargs):raise ContextLimit('synthetic limit')
        result=episode(Case(),Limited(),mode='nonthinking',seed=1,revision='synthetic',budget=self.budget,sampling=self.sampling)
        self.assertEqual(result['status'],'context_budget')
        self.assertEqual(result['score'],0)
        self.assertEqual(result['turns'],[])

    def test_tools_do_not_accept_paths_or_read_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'evidence.txt').write_text('readable evidence')
            (root/'ground_truth').mkdir();(root/'ground_truth/secret.json').write_text('GOLD_CANARY')
            (root/'secret-link.txt').symlink_to(root/'ground_truth/secret.json')
            paths=safe_files(root)
            self.assertEqual([p.name for p in paths],['evidence.txt'])
            case=Case();case.evidence={'e00000':root/'evidence.txt'}
            session=Session(case)
            with self.assertRaisesRegex(ValueError,'Unknown evidence'):
                session.execute('read_evidence',dict(evidence_id='../ground_truth/secret.json'))
            result=session.execute('read_evidence',dict(evidence_id='e00000',offset=0,limit=8))
            self.assertEqual(result['content'],'readable')
            self.assertEqual(result['next_byte'],8)

    def test_heldout_openrca_multiple_causes_and_time_tolerance(self):
        case=object.__new__(SnapshotCase);case.dataset='openrca'
        case.gold=('The 1-th predicted root cause component is A\nThe 2-th predicted root cause component is B\n'
                   'The 1-th root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-04 01:00:00\n'
                   'The 2-th root cause occurrence time is within 1 minutes (i.e., <=1min) of 2021-03-04 02:00:00')
        prediction={'2':{'root cause component':'B','root cause occurrence datetime':'2021-03-04 02:01:00'},
                    '1':{'root cause component':'A','root cause occurrence datetime':'2021-03-04 01:00:01'}}
        self.assertEqual(case.score(prediction),1)
        prediction['2']['root cause occurrence datetime']='2021-03-04 02:01:01'
        self.assertEqual(case.score(prediction),0)

    def test_extra_itbench_entity_is_not_ignored(self):
        case=object.__new__(SnapshotCase);case.dataset='itbench_aa'
        case.gold={'groups':[dict(id='bad-pod',kind='Pod',root_cause=True,filter=['bad-pod-.*'])]}
        correct={'entities':[dict(kind='Pod',name='bad-pod-123')]}
        self.assertEqual(case.score(correct),1)
        correct['entities'].append(dict(kind='Pod',name='innocent-pod'))
        self.assertEqual(case.score(correct),0)


if __name__=='__main__':unittest.main()
