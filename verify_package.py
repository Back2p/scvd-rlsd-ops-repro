"""Standard-library integrity check for the pristine delivered package."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
errors=[]
for name,expected in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        errors.append({'path':name,'error':'missing/unsafe'});continue
    value=hashlib.sha256(path.read_bytes()).hexdigest()
    if value!=expected:errors.append({'path':name,'error':'sha256 mismatch'})
actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not {'.git', '__pycache__'}.intersection(p.relative_to(root).parts) and p.suffix!='.pyc' and p.name!='PACKAGE_MANIFEST.json'}
for name in sorted(actual-set(manifest['files'])):errors.append({'path':name,'error':'unexpected file'})
print(json.dumps(dict(verified_files=len(manifest['files'])-sum(e['error']!='unexpected file' for e in errors),errors=errors,
    scope='Pristine package contents only; run before editing configs. No model or training starts.'),ensure_ascii=False,indent=2))
raise SystemExit(2 if errors else 0)
