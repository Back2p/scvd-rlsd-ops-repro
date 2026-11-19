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


if __name__=='__main__':unittest.main()
