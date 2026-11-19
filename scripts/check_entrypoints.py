"""Read-only CLI plans/help and shell syntax checks. Never invokes live actions."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    checks=[]
    def run(label,args):
        result=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        entry=dict(check=label,returncode=result.returncode)
        if result.returncode:entry['error']=result.stderr[-2500:]
        checks.append(entry)
    modules=('prepare','doctor','worker','collect','encode','train','evaluate','report','control','qualify')
    for name in modules:run(name+' --help',[sys.executable,'-m','ops_repro.'+name,'--help'])
    cfg=['--config','configs/runtime.example.json']
    for method,world in [('scvd',8),('scvd',16),('rlsd',8),('grpo',8)]:
        run(f'{method} world={world} plan',[sys.executable,'-m','ops_repro.train',*cfg,'--method',method,'--world-size',str(world),'--output','/platform/output/PLAN_ONLY','--plan'])
    for method in ('scvd','rlsd'):
        run(method+' export plan',[sys.executable,'-m','ops_repro.train',*cfg,'--method',method,'--output','/platform/output/PLAN_ONLY','--resume','/platform/output/TRUSTED_STATE','--export-only','--plan'])
    run('collection plan',[sys.executable,'-m','ops_repro.collect',*cfg,'--output','/platform/output/PLAN_ONLY','--plan'])
    run('evaluation plan',[sys.executable,'-m','ops_repro.evaluate',*cfg,'--output','/platform/output/PLAN_ONLY','--split','eval','--policy-revision','c202236235762e1c871ad0ccb60c8ee5ba337b9a','--arm','base_9b','--plan'])
    run('download freeze plan',[sys.executable,'-m','ops_repro.prepare',*cfg,'--download','model','cloudops','openrca','cujbench','itbench_aa','--freeze-data','--plan'])
    run('zero update qualification plan',[sys.executable,'-m','ops_repro.qualify',*cfg,'--sequence-length','32768','--output','/platform/output/PLAN_ONLY','--plan'])
    for script in sorted((ROOT/'scripts').glob('*.sh')):run(script.name+' syntax',['bash','-n',str(script)])
    result=dict(success=all(c['returncode']==0 for c in checks),checks=checks,actions='help, plans and bash syntax only; no worker, model inference, download, training, GPU qualification or optimizer update')
    print(json.dumps(result,indent=2));return int(not result['success'])

if __name__=='__main__':sys.exit(main())
