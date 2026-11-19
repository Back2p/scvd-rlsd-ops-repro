import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.common import append_jsonl,file_hash,write_json
from ops_repro.report import summarize_run


class ReportingTests(unittest.TestCase):
    def fixture(self,root):
        casefile=root/'synthetic_cases.jsonl'
        for case,split in [('SYNTHETIC_VAL','val'),('SYNTHETIC_TEST','test')]:
            append_jsonl(casefile,dict(dataset='cloudops',case_id=case,split=split))
        run=root/'synthetic-run';run.mkdir()
        cfg=dict(_evaluation_view='eval',_arm='SYNTHETIC_ONLY',evaluation=dict(repeats=3),
                 cases='synthetic_cases.jsonl',datasets=['cloudops'],modes=['nonthinking'])
        write_json(run/'run.json',dict(config=cfg,protocol_sha256='synthetic-protocol',split_manifest_sha256=file_hash(casefile)))
        rows=[]
        for case in ('SYNTHETIC_VAL','SYNTHETIC_TEST'):
            for trial in range(3):
                rows.append(dict(dataset='cloudops',case_id=case,mode='nonthinking',trial=trial,
                    model_revision='synthetic-policy',protocol_sha256='synthetic-protocol',split_manifest_sha256=file_hash(casefile),
                    status='scored',score=int(case=='SYNTHETIC_TEST'),checkpoint_sha256='synthetic-checkpoint',
                    actor_view_sha256=case,scorer_revision='synthetic-scorer',receipt='synthetic-receipt',termination='submitted'))
        return run,rows

    def test_failed_and_missing_attempts_do_not_disappear(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,rows=self.fixture(root)
            rows[0].update(status='infra_error',score=None)
            for row in rows[:-1]:append_jsonl(run/'scores.jsonl',row)
            with patch('ops_repro.report.ROOT',root):summary,bad,inputs,receipt=summarize_run(run)
            combined=next(row for row in summary if row['split']=='val+test')
            self.assertEqual(combined['expected_attempts'],6)
            self.assertEqual(combined['missing'],1)
            self.assertEqual(combined['infra_errors'],1)
            self.assertIsNone(combined['accuracy'])
            self.assertFalse(combined['publishable'])

    def test_mixed_checkpoint_cannot_be_reported_as_one_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run,rows=self.fixture(root)
            rows[-1]['checkpoint_sha256']='different-synthetic-checkpoint'
            for row in rows:append_jsonl(run/'scores.jsonl',row)
            with patch('ops_repro.report.ROOT',root),self.assertRaisesRegex(ValueError,'checkpoints differ'):
                summarize_run(run)


class FourDatasetReportTests(unittest.TestCase):
    def test_completeness_rejects_missing_duplicate_and_mixed_policy(self):
        from ops_repro.report import incomplete_suite
        sizes={'cloudops':(75,75),'openrca':(13,43),'cujbench':(14,14),'itbench_aa':(7,5)}
        rows=[dict(arm=arm,dataset=name,mode=mode,split=split,expected_attempts=3*n,
                   publishable=True,model_revision=arm,checkpoint_sha256=arm)
              for arm in ('base_9b','scvd','rlsd','base_27b') for name,counts in sizes.items()
              for mode in ('thinking','nonthinking') for split,n in zip(('val','test'),counts)]
        self.assertEqual(incomplete_suite(rows),[])
        self.assertEqual(len(incomplete_suite(rows[:-1])),1)
        self.assertEqual(len(incomplete_suite(rows+[dict(rows[0],publishable=False)])),1)
        rows[-1]['checkpoint_sha256']='different'
        self.assertIn('different frozen policy',incomplete_suite(rows)[0]['issue'])

    def test_diagnostics_keep_zero_rewards_and_skip_undefined_fraction(self):
        from ops_repro.diagnostics import chart, series
        stats=dict(successes=0,episodes=8,zero_advantage_groups=1,groups=1,all_wrong_groups=1,all_correct_groups=0)
        curves=dict(series([], [dict(batches=1,mean_reward=0,zero_advantage_groups=1,groups=1,
                                      active_tokens=0,total_tokens=0,by_dataset={'cloudops':stats})], ['cloudops']))
        self.assertEqual(curves['cloudops / rollout reward'],[(1,0.0)])
        self.assertNotIn('active token fraction',curves)
        self.assertIn('<circle',chart('cloudops / rollout reward',curves['cloudops / rollout reward']))


if __name__=='__main__':unittest.main()
