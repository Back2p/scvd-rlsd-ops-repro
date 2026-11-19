import copy
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from protocol import audit_cases, summarize


def cases():
    return [dict(dataset='itbench_aa', case_id=name, split=split,
                 incident_group=name, source_revision='synthetic-test-only',
                 prompt_sha256=hashlib.sha256(name.encode()).hexdigest())
            for name, split in [('train-1', 'train'), ('val-1', 'val'), ('test-1', 'test')]]


def scores():
    return [dict(dataset='itbench_aa', case_id=case, mode='thinking', trial=trial,
                 run_id='synthetic-test-run', model_id='synthetic', model_revision='fake-test-only',
                 protocol_sha256='a'*64, checkpoint_sha256='b'*64, split_manifest_sha256='c'*64,
                 scorer_revision='synthetic-scorer', adapter_revision='synthetic-adapter',
                 status='scored', score=int(case == 'test-1'), receipt='synthetic-receipt')
            for case in ('val-1', 'test-1') for trial in range(3)]


class SplitAuditTests(unittest.TestCase):
    def test_html_counts_are_not_fabricated_from_fixture(self):
        with self.assertRaisesRegex(ValueError, 'HTML requires'):
            audit_cases(cases())

    def test_group_leakage_is_rejected(self):
        rows = cases()
        rows[1]['incident_group'] = rows[0]['incident_group']
        with self.assertRaisesRegex(ValueError, 'family crosses'):
            audit_cases(rows, exact_counts=False)

    def test_duplicate_prompt_across_ids_is_rejected(self):
        rows = cases()
        rows[2]['prompt_sha256'] = rows[0]['prompt_sha256']
        with self.assertRaisesRegex(ValueError, 'hash crosses'):
            audit_cases(rows, exact_counts=False)

    def test_mean_accuracy_and_test_separation(self):
        result = summarize(cases(), scores(), exact_counts=False)
        by_split = {r['split']: r for r in result}
        self.assertEqual(by_split['test']['accuracy'], 1)
        self.assertEqual(by_split['val']['accuracy'], 0)
        self.assertEqual(by_split['val+test']['accuracy'], .5)
        self.assertEqual(by_split['val+test']['expected_attempts'], 6)

    def test_missing_attempt_does_not_raise_accuracy(self):
        result = summarize(cases(), scores()[1:], exact_counts=False)
        combined = next(r for r in result if r['split'] == 'val+test')
        self.assertFalse(combined['publishable'])
        self.assertIsNone(combined['accuracy'])
        self.assertEqual(combined['missing'], 1)
        self.assertEqual(combined['observed_success_fraction_of_planned_attempts'], .5)

    def test_infra_error_is_disclosed(self):
        rows = scores()
        rows[0].update(status='infra_error', score=None)
        combined = next(r for r in summarize(cases(), rows, exact_counts=False) if r['split'] == 'val+test')
        self.assertEqual(combined['infra_errors'], 1)
        self.assertFalse(combined['publishable'])

    def test_mixed_checkpoint_cannot_be_averaged(self):
        rows = scores()
        rows[0]['checkpoint_sha256'] = 'd'*64
        with self.assertRaisesRegex(ValueError, 'Mixed revisions'):
            summarize(cases(), rows, exact_counts=False)

    def test_train_scoring_rejected(self):
        rows = scores()
        rows[0]['case_id'] = 'train-1'
        with self.assertRaisesRegex(ValueError, 'Unknown/train'):
            summarize(cases(), rows, exact_counts=False)

    def test_duplicate_attempt_rejected(self):
        rows = scores()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate evaluation'):
            summarize(cases(), rows, exact_counts=False)

    def test_different_manifest_cannot_be_aggregated(self):
        with self.assertRaisesRegex(ValueError, 'different split manifest'):
            summarize(cases(), scores(), exact_counts=False, manifest_sha256='d'*64)


if __name__ == '__main__':
    unittest.main()
