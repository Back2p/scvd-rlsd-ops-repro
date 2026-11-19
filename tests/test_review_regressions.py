"""Audit regressions: CPU/synthetic fixtures only; no optimizer or model service."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.clients import ContextLimit,TeacherClient,TokenClient
from ops_repro.common import config,load_json
from ops_repro.datasets import SnapshotCase
from ops_repro.prepare import extract_zip
from ops_repro.rollout import collect_scvd,episode
from ops_repro.training_state import checkpoint_name,shard_records
import test_runtime as runtime_fixtures
import test_rlsd_collection as collection_fixtures
Case=runtime_fixtures.Case
Client=runtime_fixtures.Client
FakeCase=collection_fixtures.FakeCase
FakeClient=collection_fixtures.FakeClient
Tokenizer=collection_fixtures.Tokenizer
from ops_repro.train import collect_rlsd


class ReviewRegressionTests(unittest.TestCase):
    budget=runtime_fixtures.RuntimeTests.budget
    sampling=runtime_fixtures.RuntimeTests.sampling

    def test_checkpoint_cursor_includes_skipped_rollout_batches(self):
        a=checkpoint_name(dict(updates=10,batches=10))
        b=checkpoint_name(dict(updates=10,batches=13))
        self.assertNotEqual(a,b)
        self.assertEqual(b,'step-000010-batch-000013')

    def test_record_shards_preserve_every_real_turn_once_and_collective_count(self):
        records=[dict(input_ids=[1,2,3],labels=[-100,-100,3],old_logp=[-.1],advantage=1.,id=i) for i in range(5)]
        shards=shard_records(records,4)
        self.assertEqual([len(s) for s in shards],[2]*4)
        self.assertEqual(sorted(r['id'] for s in shards for r in s if not r['padding']),list(range(5)))
        self.assertTrue(all(r['labels']==[-100,-100] for s in shards for r in s if r['padding']))
        self.assertEqual(shard_records([],4),[[],[],[],[]])

    def test_grpo_does_not_read_or_encode_privileged_answers(self):
        cfg=collection_fixtures.RLSDCollectionTests().config()
        row=dict(dataset='cloudops',case_id='SYNTHETIC_CASE',split='train')
        with tempfile.TemporaryDirectory() as folder,patch('ops_repro.datasets.SnapshotCase',FakeCase),patch('ops_repro.train.TokenClient',FakeClient),patch.object(FakeCase,'teacher_hint',side_effect=AssertionError('GRPO cannot ask for gold')):
            result=collect_rlsd(cfg,[row],Tokenizer(),dict(revision='SYNTHETIC',checkpoint_sha256='a'*64),0,Path(folder),privileged=False)
        self.assertTrue(result['records'])
        self.assertTrue(all('teacher_ids' not in r for r in result['records']))

    def test_context_limit_retains_actor_checkpoint_without_generated_turn(self):
        identity=dict(policy_revision='synthetic',checkpoint_sha256='a'*64)
        with patch('ops_repro.clients.post',return_value=dict(**identity,context_exhausted=True)):
            result=episode(Case(),TokenClient(dict(endpoints=['https://synthetic.invalid'])),mode='nonthinking',seed=1,revision='synthetic',budget=self.budget,sampling=self.sampling)
        self.assertEqual(result['checkpoint_identity'],identity)
        self.assertEqual(result['status'],'context_budget')
        self.assertEqual(result['score'],0)

    def test_scvd_prefix_and_teacher_share_one_generated_token_budget(self):
        budget=dict(self.budget,max_generated_tokens=1)
        result=collect_scvd(Case(),Client([('propose_diagnosis',{'diagnosis':{'answer':'wrong'}})]),Client([]),mode='nonthinking',seed=1,student_revision='s',teacher_revision='t',budget=budget,sampling=self.sampling)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['continuation']['status'],'token_budget')
        self.assertEqual(result['continuation']['phase_generated_tokens'],0)

    def test_malformed_model_tool_arguments_are_scored_failure(self):
        for args in ('{broken',None,[1]):
            result=episode(Case(),Client([('submit_diagnosis',args)]),mode='nonthinking',seed=1,revision='s',budget=self.budget,sampling=self.sampling)
            self.assertEqual(result['status'],'invalid_tool_arguments')
            self.assertEqual(result['score'],0)

    def test_malformed_diagnosis_is_zero_instead_of_scorer_crash(self):
        case=object.__new__(SnapshotCase);case.dataset='openrca';case.gold='The only predicted root cause component is A'
        for diagnosis in (None,{'1':None},{'1':'A'}):self.assertEqual(case.score(diagnosis),0)
        case.dataset='itbench_aa';case.gold={'groups':[dict(id='bad',kind='Pod',root_cause=True)]}
        for diagnosis in ({'entities':None},{'entities':[None]},{'entities':[{'name':'bad','kind':5}]}):self.assertEqual(case.score(diagnosis),0)

    def test_cuj_taxonomy_is_part_of_scorer_fingerprint(self):
        case=object.__new__(SnapshotCase);case.dataset='cujbench';case.row={'app':'synthetic'};case.gold={'component':'a','type':'b'}
        with tempfile.TemporaryDirectory() as folder:
            case.taxonomy=Path(folder)
            for name in ('components_synthetic.json','root_cause_types.json'):(case.taxonomy/name).write_text('{}')
            before=case.scoring_fingerprint()
            (case.taxonomy/'root_cause_types.json').write_text('{"changed":true}')
            self.assertNotEqual(before,case.scoring_fingerprint())

    def test_teacher_extra_body_cannot_replace_model_inputs_or_budget(self):
        cfg=dict(url='https://synthetic.invalid',model='synthetic',revision='synthetic')
        for key in ('model','messages','max_tokens','temperature'):
            with self.assertRaisesRegex(ValueError,'cannot override'):TeacherClient(dict(cfg,extra_body={key:'bad'}))

    def test_unknown_budget_usage_is_rejected_before_accepting_teacher(self):
        class NoUsage:
            def generate(self,*args,**kwargs):return dict(message={'content':'x'})
        with self.assertRaisesRegex(ValueError,'completion_tokens'):
            episode(Case(),NoUsage(),mode='nonthinking',seed=1,revision='s',budget=self.budget,sampling=self.sampling)

    def test_zip_rerun_checks_existing_bytes_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'synthetic.zip';out=root/'out'
            with zipfile.ZipFile(archive,'w') as target:target.writestr('Bank/query.csv','synthetic bytes')
            extract_zip(archive,out);extract_zip(archive,out)
            (out/'Bank/query.csv').write_text('changed bytes')
            with self.assertRaisesRegex(ValueError,'differs'):extract_zip(archive,out)
            self.assertEqual((out/'Bank/query.csv').read_text(),'changed bytes')

    def test_unsupported_training_knobs_fail_during_config_load(self):
        base=load_json(Path(__file__).resolve().parents[1]/'configs/runtime.example.json')
        for section,key,value in [('rlsd','ppo_epochs',2),('scvd','loss_reduction','unknown'),('budget','max_context',0),('training','control_timeout_seconds',0)]:
            cfg=copy.deepcopy(base);cfg[section][key]=value
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'cfg.json';path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):config(path)


if __name__=='__main__':unittest.main()
