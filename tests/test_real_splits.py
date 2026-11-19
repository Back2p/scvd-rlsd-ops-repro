"""Checks actual source metadata and reconstructed ID lists, not model quality."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from protocol import audit_cases, read_jsonl


@unittest.skipUnless((ROOT/'splits/all_cases.jsonl').exists(), 'Reconstructed source metadata not prepared')
class RealSplitTests(unittest.TestCase):
    def test_1006_real_ids_match_html_and_have_no_declared_group_overlap(self):
        result=audit_cases(read_jsonl(ROOT/'splits/all_cases.jsonl'),require_actor_hash=False)
        self.assertEqual(result['cases'],1006)
        self.assertEqual(result['counts']['cloudops'],dict(train=604,val=75,test=75))
        self.assertEqual(result['counts']['openrca'],dict(train=80,val=13,test=43))
        self.assertEqual(result['counts']['cujbench'],dict(train=57,val=14,test=14))
        self.assertEqual(result['counts']['itbench_aa'],dict(train=19,val=7,test=5))

    def test_metadata_lists_do_not_pass_actor_input_readiness(self):
        with self.assertRaisesRegex(ValueError,'prompt hash'):
            audit_cases(read_jsonl(ROOT/'splits/all_cases.jsonl'))

    def test_exclusions_are_explicit_and_full_itbench_is_retained(self):
        report=json.loads((ROOT/'reports/dataset_split_report.json').read_text())
        self.assertEqual(len(report['datasets']['itbench_aa']['excluded']),9)
        self.assertEqual(len(report['datasets']['cujbench']['excluded']),2)
        self.assertEqual(len(read_jsonl(ROOT/'catalogs/itbench_aa_full40_normalized.jsonl')),40)
        self.assertTrue(all(r['valid_public_case'] for r in report['datasets']['itbench_aa']['excluded']))

    def test_openrca_dates_are_disjoint(self):
        rows=read_jsonl(ROOT/'catalogs/openrca.jsonl')
        dates={split:{r['date'] for r in rows if r['split']==split} for split in ('train','val','test')}
        self.assertEqual(dates['val'],{'2021-03-12'})
        self.assertFalse(dates['train'] & dates['test'])
        self.assertFalse(dates['val'] & dates['test'])

    def test_all_html_marginals_are_exactly_matched(self):
        report=json.loads((ROOT/'reports/dataset_split_report.json').read_text())['datasets']
        self.assertEqual(report['cloudops']['eval_marginals']['difficulty'],dict(easy=76,medium=38,hard=36))
        self.assertEqual(report['cloudops']['eval_marginals']['system'],dict(boutique=108,trainticket=42))
        self.assertEqual(report['cujbench']['eval_marginals']['app'],{'otel-demo':14,'tractor-store':14})
        self.assertEqual(report['cujbench']['eval_marginals']['difficulty'],dict(easy=2,medium=19,hard=7))
        self.assertEqual(report['itbench_aa']['train_marginals']['category']['Compound'],9)
        self.assertEqual(report['itbench_aa']['eval_marginals']['category']['Compound'],2)


if __name__=='__main__':
    unittest.main()
