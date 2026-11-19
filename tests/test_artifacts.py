import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ops_repro.common import digest,file_hash,write_json
from ops_repro.prepare import extract_zip
from ops_repro.worker import verify_publication


class ArtifactTests(unittest.TestCase):
    def test_publication_is_rejected_if_weight_bytes_change(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);weight=root/'SYNTHETIC_NOT_A_MODEL.safetensors';weight.write_bytes(b'original synthetic bytes')
            write_json(root/'publication.json',dict(policy_revision='synthetic',files={weight.name:file_hash(weight)}))
            verify_publication(root,'synthetic')
            weight.write_bytes(b'changed synthetic bytes')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):verify_publication(root,'synthetic')

    def test_archive_cannot_escape_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'synthetic.zip'
            with zipfile.ZipFile(archive,'w') as output:output.writestr('../escape.txt','synthetic')
            with self.assertRaisesRegex(ValueError,'escapes destination'):extract_zip(archive,root/'destination')
            self.assertFalse((root/'escape.txt').exists())

    def test_retention_refuses_an_unowned_directory(self):
        from ops_repro.distributed import prune_owned
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for n in (1,2):(root/'publications'/f'policy-{n:06}').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'unowned'):prune_owned(root,'publications',1)
            self.assertTrue((root/'publications/policy-000001').exists())


if __name__=='__main__':unittest.main()
