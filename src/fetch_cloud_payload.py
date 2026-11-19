"""Download pinned Cloud-OpsBench observation files on the target data volume.

Does not call models or execute downloaded code. Gold metadata is not fetched by
this observation-only downloader. The private coordinator metadata is separate.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from protocol import require


def identity(path, entry):
    if path.stat().st_size != entry['size']:
        return False
    digest=hashlib.sha1(b'blob '+str(entry['size']).encode()+b'\0')
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()==entry['sha']


def fetch(entry, destination, source_revision):
    require('/raw_data/' in entry['path'] and entry['path'].startswith('benchmark/'), 'Only observation files allowed')
    path=(destination/entry['path']).resolve()
    require(path.is_relative_to(destination.resolve()), 'Path escapes destination')
    if path.exists():
        require(path.is_file() and identity(path,entry), 'Existing file differs from official Git blob; preserved')
        return 'verified_existing'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    require(not temporary.exists(), 'Existing partial download preserved; choose a fresh destination or inspect it')
    url=f"https://raw.githubusercontent.com/LLM4Ops/Cloud-OpsBench/{source_revision}/{entry['path']}"
    with urllib.request.urlopen(url,timeout=120) as response, temporary.open('xb') as output:
        shutil.copyfileobj(response,output)
    require(identity(temporary,entry), 'Downloaded observation differs from official Git blob')
    temporary.rename(path)
    return 'downloaded_and_verified'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--receipt',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    require(1<=args.workers<=8,'Use 1 to 8 workers')
    tree=json.loads(args.tree.read_text())
    entries=[x for x in tree['tree'] if x['type']=='blob' and x['path'].startswith('benchmark/') and '/raw_data/' in x['path']]
    args.destination.mkdir(parents=True,exist_ok=True)
    needed=sum(e['size'] for e in entries if not (args.destination/e['path']).exists())
    require(shutil.disk_usage(args.destination).free>=needed+2*1024**3,'Insufficient space including 2 GiB reserve')
    counts,errors={},[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(fetch,e,args.destination,tree['sha']):e for e in entries}
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            try:
                status=future.result()
                counts[status]=counts.get(status,0)+1
            except Exception as error:
                errors.append(dict(path=futures[future]['path'],error=str(error)))
            if i%100==0:
                print(f'{i}/{len(entries)} observations processed',flush=True)
    result=dict(source_revision=tree['sha'],expected_files=len(entries),counts=counts,errors=errors,
                complete=not errors,scope='observation files only; adapter/scorer not executed')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2))
    raise SystemExit(0 if result['complete'] else 2)
