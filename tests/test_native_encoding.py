"""Synthetic mask checks with a real local native tokenizer. No weight loading."""
import copy
import hashlib
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(os.environ.get('QWEN_TOKENIZER_DIR'), 'Set local QWEN_TOKENIZER_DIR for native template checks')
class NativeEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        from scvd_encode import encode
        cls.encode = staticmethod(encode)
        cls.tokenizer = AutoTokenizer.from_pretrained(os.environ['QWEN_TOKENIZER_DIR'],
            tokenizer_type='qwen2', local_files_only=True, trust_remote_code=False)
        cls.template_hash = hashlib.sha256(cls.tokenizer.chat_template.encode()).hexdigest()

    def record(self):
        return dict(dataset='itbench_aa', case_id='synthetic-1', mode='nonthinking',
            student_revision='synthetic-student', teacher_revision='synthetic-teacher',
            candidate_state_receipt='synthetic-state', verifier_receipt='synthetic-judge', verifier_passed=True,
            tools=[{'type':'function','function':{'name':'query_logs','description':'Read logs',
                   'parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}],
            messages=[
                dict(role='user',source='task',content='Diagnose a synthetic fault.'),
                dict(role='assistant',source='student_prefix',content='STUDENT_CANDIDATE_UNIQUE'),
                dict(role='user',source='collection_instruction',content='COLLECTION_ONLY_UNIQUE'),
                dict(role='assistant',source='teacher_continuation',content='VERIFY_UNIQUE',
                     tool_calls=[{'type':'function','function':{'name':'query_logs','arguments':{'query':'errors'}}}]),
                dict(role='tool',source='tool_observation',content='TOOL_RESULT_UNIQUE'),
                dict(role='assistant',source='teacher_continuation',content='TEACHER_ANSWER_UNIQUE')])

    def run_encode(self, record=None, split='train', **kwargs):
        return self.encode(record or self.record(), self.tokenizer,
            {('itbench_aa','synthetic-1'):{'split':split}}, template_sha256=self.template_hash, **kwargs)

    def test_only_teacher_continuation_is_supervised(self):
        result = self.run_encode()
        all_text = self.tokenizer.decode(result['input_ids'])
        target = self.tokenizer.decode([x for x in result['labels'] if x != -100])
        self.assertIn('STUDENT_CANDIDATE_UNIQUE', all_text)
        self.assertNotIn('COLLECTION_ONLY_UNIQUE', all_text)
        self.assertNotIn('STUDENT_CANDIDATE_UNIQUE', target)
        self.assertNotIn('TOOL_RESULT_UNIQUE', target)
        self.assertIn('TEACHER_ANSWER_UNIQUE', target)
        self.assertIn('<function=query_logs>', target)
        self.assertEqual(target.count('<|im_end|>'), 2)

    def test_heldout_record_rejected(self):
        with self.assertRaisesRegex(ValueError, 'training cases'):
            self.run_encode(split='test')

    def test_failed_verifier_rejected(self):
        record = self.record()
        record['verifier_passed'] = False
        with self.assertRaisesRegex(ValueError, 'verifier success'):
            self.run_encode(record)

    def test_nonthinking_cannot_include_reasoning(self):
        record = self.record()
        record['messages'][-1]['reasoning_content'] = 'hidden reasoning'
        with self.assertRaisesRegex(ValueError, 'contains reasoning'):
            self.run_encode(record)

    def test_thinking_target_is_kept(self):
        record = self.record()
        record['mode'] = 'thinking'
        record['messages'][-1]['reasoning_content'] = 'THINKING_TARGET_UNIQUE'
        result = self.run_encode(record)
        self.assertIn('THINKING_TARGET_UNIQUE', self.tokenizer.decode([x for x in result['labels'] if x != -100]))

    def test_overlength_is_not_truncated(self):
        with self.assertRaisesRegex(ValueError, 'Overlength'):
            self.run_encode(max_length=10)

    def test_student_after_handoff_rejected(self):
        record = self.record()
        record['messages'][-1]['source'] = 'student_prefix'
        with self.assertRaisesRegex(ValueError, 'after teacher'):
            self.run_encode(record)


if __name__ == '__main__':
    unittest.main()
