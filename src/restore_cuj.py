"""Restore a checksum-verified official CUJ corpus without executing its code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from protocol import require, sha256


def restore(bundle, manifest_path, destination):
    manifest=json.loads(manifest_path.read_text())
    require(bundle.stat().st_size==manifest['bundle_size_bytes'], 'Bundle length differs from publisher manifest')
    require(sha256(bundle)==manifest['bundle_sha256'], 'Bundle hash differs from publisher manifest')
    destination=destination.resolve()
    receipts={}
    with tarfile.open(bundle,'r:gz') as archive:
        members=archive.getmembers()
        for member in members:
            target=(destination/member.name).resolve()
            require(target.is_relative_to(destination), 'Archive path escapes destination')
            require(not member.issym() and not member.islnk(), 'Links are not allowed')
            require(member.isfile() or member.isdir(), 'Special files are not allowed')
        for member in members:
            if member.isdir():
                continue
            target=destination/member.name
            require(member.name not in receipts, 'Duplicate archive member')
            raw=archive.extractfile(member).read()
            digest=hashlib.sha256(raw).hexdigest()
            if target.exists():
                require(target.is_file() and sha256(target)==digest, 'Refusing to overwrite a different existing file')
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw)
            receipts[member.name]=digest
    registry=destination/'scenarios/registry.json'
    require(sha256(registry)==manifest['registry_sha256'], 'Restored registry checksum mismatch')
    require(len(json.loads(registry.read_text()))==87, 'Unexpected scenario registry count')
    return dict(bundle_sha256=manifest['bundle_sha256'],restored_files=len(receipts),
                registry_scenarios=87,files=receipts,
                scope='Source corpus restored; no model, tools, scorer or runtime execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    result=restore(args.bundle,args.manifest,args.destination)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
