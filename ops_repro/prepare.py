"""Target-platform download and freeze commands. All actions require an explicit CLI invocation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile
import zlib

from .common import ROOT,cases,config,digest,file_hash,load_json,require,write_json


def download(url,target,expected_hash=None,expected_bytes=None):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        temporary=target.with_name(target.name+'.partial')
        require(not temporary.exists(),'A partial download exists; inspect it or choose a fresh destination')
        with urllib.request.urlopen(url,timeout=120) as response,temporary.open('xb') as output:shutil.copyfileobj(response,output)
        if expected_bytes:require(temporary.stat().st_size==expected_bytes,'Downloaded archive length mismatch')
        if expected_hash:require(file_hash(temporary)==expected_hash,'Downloaded archive SHA256 mismatch')
        temporary.rename(target)
    if expected_hash:require(file_hash(target)==expected_hash,'Existing archive SHA256 mismatch')
    if expected_bytes:require(target.stat().st_size==expected_bytes,'Existing archive length mismatch')
    return target


def extract_zip(archive,destination):
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        needed=sum(member.file_size for member in source.infolist())
        require(shutil.disk_usage(destination).free>needed+2*1024**3,'Insufficient free data-volume space')
        for member in source.infolist():
            target=(destination/member.filename).resolve()
            require(target.is_relative_to(destination),'ZIP member escapes destination')
            require((member.external_attr>>16)&0o170000!=0o120000,'ZIP symlink not allowed')
            if member.is_dir():continue
            if target.exists():
                require(target.is_file() and not target.is_symlink(),'ZIP destination contains a non-file')
                crc=0
                with target.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):crc=zlib.crc32(chunk,crc)
                require(target.stat().st_size==member.file_size and crc==member.CRC,'Existing ZIP member differs; choose a fresh destination')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.extracting')
            require(not temporary.exists(),'Partial extraction exists; inspect it or choose a fresh destination')
            with source.open(member) as stream,temporary.open('xb') as output:shutil.copyfileobj(stream,output)
            temporary.rename(target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--download',nargs='+',choices=['model','cloudops','cujbench','openrca','itbench_aa'])
    parser.add_argument('--freeze-data',action='store_true')
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();cfg=config(args.config)
    requested=args.download or []
    if args.plan:
        print(json.dumps(dict(downloads=requested,freeze_data=args.freeze_data,
            targets=cfg['data_roots'],model_target=cfg['model']['path'],actions_executed=0),indent=2));return
    for dataset in requested:
        if dataset=='model':
            subprocess.run(['hf','download',cfg['model']['id'],'--revision',cfg['model']['revision'],'--local-dir',cfg['model']['path']],check=True)
        elif dataset=='itbench_aa':
            subprocess.run(['hf','download','ArtificialAnalysis/ITBench-AA','--type','dataset','--revision',
                            '76df38a82288f75ba9e41dc8c515033332497473','--local-dir',cfg['data_roots'][dataset]],check=True)
        elif dataset=='cloudops':
            subprocess.run([sys.executable,str(ROOT/'src/fetch_cloud_payload.py'),'--tree',str(ROOT/'dataset_sources/cloudops-tree.json'),
                '--destination',cfg['data_roots'][dataset],'--receipt',str(Path(cfg['data_roots'][dataset])/'download-receipt.json')],check=True)
        elif dataset=='cujbench':
            manifest=load_json(ROOT/'dataset_sources/cujbench-manifest.json')
            folder=Path(cfg['data_roots'][dataset]);archive=folder/'archives'/manifest['bundle_file']
            download('https://media.githubusercontent.com/media/haoming29/CUJBench/main/artifacts/'+manifest['bundle_file'],
                     archive,manifest['bundle_sha256'],manifest['bundle_size_bytes'])
            subprocess.run([sys.executable,str(ROOT/'src/restore_cuj.py'),'--bundle',str(archive),
                '--manifest',str(ROOT/'dataset_sources/cujbench-manifest.json'),'--destination',str(folder),
                '--receipt',str(folder/'restore-receipt.json')],check=True)
        elif dataset=='openrca':
            folder=Path(cfg['data_roots'][dataset]);folder.mkdir(parents=True,exist_ok=True)
            require(shutil.disk_usage(folder).free>60*1024**3,'Reserve at least 60 GiB on the OpenRCA data volume before downloading/unpacking Bank')
            import gdown
            archive=folder/'archives/Bank.zip';archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():
                result=gdown.download(id='1enBrdPT3wLG94ITGbSOwUFg9fkLR-16R',output=str(archive),quiet=False,use_cookies=False)
                require(bool(result),'Official public Drive download failed; download Bank.zip manually from the documented source')
            require(archive.stat().st_size==2829139841,'Bank.zip differs from the observed official Drive file size')
            extract_zip(archive,folder)
            require((folder/'Bank/query.csv').is_file(),'Expected Bank/query.csv missing; inspect archive layout and set data root explicitly')
            write_json(folder/'download-receipt.json',dict(source_drive_id='1enBrdPT3wLG94ITGbSOwUFg9fkLR-16R',
                local_archive_sha256=file_hash(archive),checksum_authenticity='local fingerprint; publisher SHA256 not supplied'))
    if args.freeze_data:
        from .datasets import SnapshotCase,tools
        raw_cfg=dict(cfg);raw_cfg.pop('frozen_data_manifest',None)
        entries={};errors=[]
        for row in cases(cfg):
            key=row['dataset']+'/'+row['case_id']
            try:
                case=SnapshotCase(row,raw_cfg)
                entries[key]=dict(context_revision=case.context_revision,source_revision=row['source_revision'],
                    scoring_source_sha256=case.scoring_fingerprint(),
                    actor_view_sha256=digest({'messages':case.initial_messages(),'tools':tools(),'context':case.context_revision}))
            except Exception as error:errors.append(dict(dataset=row['dataset'],case_id=row['case_id'],error=str(error)))
        result=dict(complete=not errors,entries=entries,errors=errors,
                    source_case_manifest_sha256=file_hash(ROOT/cfg['cases']),protocol_sha256=digest(cfg['protocol']))
        path=Path(cfg['frozen_data_manifest']);require(not path.exists(),'Frozen manifest already exists; choose a new versioned path')
        write_json(path,result)
        print(json.dumps(dict(complete=result['complete'],cases=len(entries),errors=len(errors)),indent=2))
        if errors:raise SystemExit(2)


if __name__=='__main__':main()
