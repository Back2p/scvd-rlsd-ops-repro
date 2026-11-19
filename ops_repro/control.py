"""Explicit worker health/reload commands for platform operators."""
import argparse
import json
from pathlib import Path

from .clients import TokenClient,post
from .common import config,load_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--publication',help='A publication directory, or a trainer result.json containing final_publication')
    args=parser.parse_args();cfg=config(args.config)
    publication=None
    if args.publication:
        path=Path(args.publication)
        publication=load_json(path)['final_publication'] if path.is_file() else dict(path=str(path.resolve()),revision=load_json(path/'publication.json')['policy_revision'])
    results=[]
    for i in range(len(cfg['rollout']['endpoints'])):
        client=TokenClient(cfg['rollout'],i)
        result=client.reload(publication['path'],publication['revision']) if publication else post(client.endpoint+'/health',{},cfg['rollout'].get('api_key_env'))
        results.append(dict(worker=i,**result))
    print(json.dumps(dict(publication=publication,workers=results),indent=2))


if __name__=='__main__':main()
