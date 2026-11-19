"""Read-only snapshot adapters and coordinator-only deterministic scorers.

This is the declared snapshot-reference-v2 protocol, not the unknown project
party's original tool API. All models must be evaluated through this same API.
"""
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import re
from functools import lru_cache

import yaml

from .common import ROOT,digest,file_hash,load_json,require

OUTPUT_FORMATS={
 'cloudops':{'fault_object':'resource name from the evidence','root_cause':'fault mechanism identifier'},
 'openrca':{'1':{'root cause occurrence datetime':'YYYY-MM-DD HH:MM:SS','root cause component':'component','root cause reason':'reason'}},
 'cujbench':{'component':'owning subsystem','type':'fault family','layer':'browser/frontend/backend/infrastructure','explanation':'evidence-based reasoning','evidence_ids':[]},
 'itbench_aa':{'entities':[{'kind':'Kubernetes kind','name':'responsible resource name'}]},
}
FORBIDDEN={'ground_truth','groundtruth','gold','answers','reference_trajectory.json','public-metadata-with-gold.jsonl'}


@lru_cache(maxsize=20000)
def observation_hash(path,size,mtime_ns):
    return file_hash(path)


@lru_cache(maxsize=8)
def frozen_manifest(path,mtime_ns):
    return load_json(path)


def safe_files(directory,allowed=None):
    directory=Path(directory).resolve()
    require(directory.is_dir(),'Missing observation directory; download/mount data first')
    files=[]
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        real=path.resolve()
        if not real.is_relative_to(directory):continue
        parts=set(real.relative_to(directory).parts)
        if parts & FORBIDDEN or any('ground_truth' in s.lower() for s in parts):continue
        if allowed and not allowed(path.relative_to(directory)):continue
        files.append(path)
    require(bool(files),'No eligible observation files; data is incomplete')
    return files


class SnapshotCase:
    """Coordinator object. The model sees schema+tool results, never this object."""
    def __init__(self,row,cfg):
        self.row=row;self.cfg=cfg;self.dataset=row['dataset'];self.gold=None
        patterns={'cloudops':r'(boutique|trainticket)/[a-z]+/[0-9]+','cujbench':r'SCEN-[0-9]+-[A-Za-z0-9_-]+',
                  'openrca':r'Bank/query-row-[0-9]+','itbench_aa':r'Scenario-[0-9]+'}
        require(self.dataset in patterns and re.fullmatch(patterns[self.dataset],row['case_id']),
                'Invalid original case ID; paths are not accepted')
        roots=cfg['data_roots']
        dataset=self.dataset
        if dataset=='cloudops':
            base=Path(roots[dataset])/'benchmark'/row['case_id']
            self.paths=safe_files(base/'raw_data')
            self.query=row['initial_query']
            metadata=ROOT/'dataset_sources/cloudops/benchmark'/row['case_id']/'metadata.json'
            require(file_hash(metadata)==row['source_metadata_sha256'],'Cloud label/source metadata changed')
            self.gold=load_json(metadata)['result']
        elif dataset=='cujbench':
            base=Path(roots[dataset])/'scenarios'/row['case_id']
            self.paths=safe_files(base/'tool_cache',lambda p:p.name!='view_screenshot.json')
            self.query='Diagnose the failed critical user journey using the available browser and backend observations. Screenshots are not provided.'
            self.gold=load_json(base/'ground_truth/root_cause.json')['root_cause']
            self.taxonomy=Path(roots[dataset])/'scenarios/taxonomy'
        elif dataset=='openrca':
            base=Path(roots[dataset])/'Bank'
            self.paths=safe_files(base/'telemetry'/row['date'].replace('-','_'))
            self.query=row['initial_query']
            with (base/'query.csv').open() as stream:
                queries=list(csv.DictReader(stream))
            native=queries[int(float(row['upstream_row_id']))]
            require(native['instruction'].strip()==self.query.strip(),'OpenRCA query.csv differs from archived query/row ID; stop before evaluating')
            require(native.get('scoring_points'),'Native OpenRCA scoring_points are required; archived predictions are not labels')
            self.gold=native['scoring_points']
            require('predicted root cause' in self.gold or 'root cause occurrence time' in self.gold,'Unrecognized official OpenRCA scoring point schema')
        elif dataset=='itbench_aa':
            base=Path(roots[dataset])/'sre'/row['case_id']
            visible={'k8s_events_raw.tsv','k8s_objects_raw.tsv','otel_logs_raw.tsv','otel_traces_raw.tsv'}
            self.paths=safe_files(base,lambda p:p.parts[0] in ('alerts','metrics') or p.name in visible or p.name.startswith('alerts_in_alerting_state_'))
            source=ROOT/'dataset_sources/itbench-aa/public-metadata-with-gold.jsonl'
            require(file_hash(source)==row['source_metadata_sha256'],'ITBench metadata differs from the split source')
            originals=[json.loads(line) for line in source.read_text().splitlines()]
            original=next(r for r in originals if r['scenario_id']==row['case_id'])
            d=yaml.safe_load(original['ground_truth_yaml']);self.gold=d.get('spec',d)
            self.query='Diagnose the Kubernetes incident from the supplied offline alerts, objects, events, logs and telemetry. Identify all responsible entities.'
        else:raise ValueError('Unknown dataset')
        self.evidence={f'e{i:05}':p for i,p in enumerate(self.paths)}
        self.context_revision=digest({'case':row['case_id'],'source':row['source_revision'],
            'files':[(key,p.name,observation_hash(str(p),p.stat().st_size,p.stat().st_mtime_ns)) for key,p in self.evidence.items()]})
        if self.dataset=='cujbench':require(self._score_cuj(self.gold)==1,'CUJ gold labels are outside the supplied canonical taxonomy')
        if self.dataset=='openrca':self._openrca_points()
        if self.dataset=='itbench_aa':require(any(g.get('root_cause') is True for g in self.gold.get('groups',[])),'ITBench case has no normalized root-cause group')
        if cfg.get('frozen_data_manifest'):
            path=Path(cfg['frozen_data_manifest'])
            freeze=frozen_manifest(str(path),path.stat().st_mtime_ns)
            require(freeze.get('complete') is True,'Frozen data manifest is incomplete')
            require(freeze['protocol_sha256']==digest(cfg['protocol']),'Protocol changed after data freezing')
            require(freeze['source_case_manifest_sha256']==file_hash(ROOT/cfg['cases']),'Case membership changed after data freezing')
            entry=freeze['entries'][self.dataset+'/'+row['case_id']]
            require(entry['context_revision']==self.context_revision,'Observation data changed after freezing')
            require(entry['scoring_source_sha256']==self.scoring_fingerprint(),'Scoring labels changed after freezing')

    def initial_messages(self):
        system=('You diagnose an incident using read-only evidence tools. Never guess unseen evidence. '
                'Form an initial complete candidate with propose_diagnosis, then check it against evidence and submit_diagnosis. '
                'These are diagnosis tasks; no production state changes are requested. '
                'The diagnosis object must follow this dataset schema: '+json.dumps(OUTPUT_FORMATS[self.dataset],ensure_ascii=False))
        return [dict(role='system',content=system),dict(role='user',content=self.query)]

    def teacher_hint(self):
        require(self.row['split']=='train','Teacher privilege is forbidden for val/test')
        if self.dataset=='itbench_aa':
            answer={'entities':[{'kind':g['kind'],'name':g.get('name',g['id'])} for g in self.gold['groups'] if g.get('root_cause') is True]}
        else:answer=self.gold
        return json.dumps(answer,ensure_ascii=False)

    def scoring_fingerprint(self):
        sources={'gold':self.gold}
        if self.dataset=='cujbench':
            sources['taxonomy']={name:file_hash(self.taxonomy/name) for name in
                ('components_'+self.row['app']+'.json','root_cause_types.json')}
        return digest(sources)

    def scorer_revision(self):
        return 'snapshot-reference-v2/'+self.dataset+'/strict-v1'

    def score(self,diagnosis):
        if not isinstance(diagnosis,dict):return 0
        try:
            if self.dataset=='cloudops':
                fields=('fault_object','root_cause')
                return int(all(str(diagnosis.get(k,'')).strip().casefold()==str(self.gold[k]).strip().casefold() for k in fields))
            if self.dataset=='openrca':return self._score_openrca(diagnosis)
            if self.dataset=='cujbench':return self._score_cuj(diagnosis)
            return self._score_itbench(diagnosis)
        except (KeyError,TypeError,ValueError):return 0

    def _openrca_points(self):
        points=self.gold
        components=re.findall(r'The (?:\d+-th|only) predicted root cause component is ([^\n]+)',points)
        reasons=re.findall(r'The (?:\d+-th|only) predicted root cause reason is ([^\n]+)',points)
        times=re.findall(r'The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)',points)
        n=max(map(len,(components,reasons,times)))
        require(n>0 and all(len(values) in (0,n) for values in (components,reasons,times)),'Unrecognized/inconsistent official OpenRCA scoring point schema')
        for value in times:datetime.strptime(value,'%Y-%m-%d %H:%M:%S')
        return components,reasons,times,n

    def _score_openrca(self,diagnosis):
        components,reasons,times,n=self._openrca_points()
        predicted=list(diagnosis.values()) if isinstance(diagnosis,dict) else diagnosis
        if len(predicted)!=n or not all(isinstance(p,dict) for p in predicted):return 0
        def match(p,i):
            if len(components)==n and p.get('root cause component')!=components[i]:return False
            if len(reasons)==n and p.get('root cause reason')!=reasons[i]:return False
            if len(times)==n:
                a=datetime.strptime(p.get('root cause occurrence datetime',''),'%Y-%m-%d %H:%M:%S')
                b=datetime.strptime(times[i],'%Y-%m-%d %H:%M:%S')
                if abs((a-b).total_seconds())>60:return False
            return True
        # Bipartite matching reproduces all-correct permutation semantics without factorial work.
        assigned={}
        def augment(i,seen):
            for j,p in enumerate(predicted):
                if j in seen or not match(p,i):continue
                seen.add(j)
                if j not in assigned or augment(assigned[j],seen):assigned[j]=i;return True
            return False
        return int(all(augment(i,set()) for i in range(n)))

    def _score_cuj(self,diagnosis):
        def norm(s):return re.sub('[^a-z0-9]+','_',str(s).lower()).strip('_')
        def canonical(filename,value):
            aliases={}
            for entry in load_json(self.taxonomy/filename)['values']:
                for name in [entry['name']]+entry.get('maps_from',[]):aliases[norm(name)]=entry['name']
            return aliases.get(norm(value))
        app=self.row['app']
        expected_component=canonical('components_'+app+'.json',self.gold['component'])
        expected_type=canonical('root_cause_types.json',self.gold['type'])
        return int(expected_component is not None and expected_type is not None
            and canonical('components_'+app+'.json',diagnosis.get('component'))==expected_component
            and canonical('root_cause_types.json',diagnosis.get('type'))==expected_type)

    def _score_itbench(self,diagnosis):
        groups={g['id']:g for g in self.gold.get('groups',[])}
        roots={gid for gid,g in groups.items() if g.get('root_cause') is True}
        if not roots:return 0
        found=set()
        if not isinstance(diagnosis.get('entities'),list):return 0
        for entity in diagnosis['entities']:
            if not isinstance(entity,dict):return 0
            name,kind=entity['name'],entity['kind']
            if not isinstance(name,str) or not isinstance(kind,str) or len(name)>256:return 0
            matched={gid for gid,g in groups.items() if str(g.get('kind','')).casefold()==kind.casefold()
                     and (name==gid or name==g.get('name') or any(re.search(pattern,name) for pattern in g.get('filter',[])))}
            expanded=set(matched)
            for alias in self.gold.get('aliases',[]):
                if set(alias)&matched:expanded.update(alias)
            candidates=roots & expanded
            if not candidates:return 0
            found.update(candidates)
        return int(found==roots)


def tools():
    def function(name,description,properties,required):
        return dict(type='function',function=dict(name=name,description=description,
            parameters=dict(type='object',properties=properties,required=required,additionalProperties=False)))
    integer=lambda description:dict(type='integer',description=description,minimum=0)
    diagnosis={'diagnosis':dict(type='object',description='Complete diagnosis following the dataset schema in the system message')}
    return [function('list_evidence','List paged evidence IDs, file names and sizes.',{'offset':integer('Start entry')},[]),
        function('read_evidence','Read UTF-8 bytes from an allowlisted evidence ID; offsets are bytes.',
                 {'evidence_id':dict(type='string'),'offset':integer('Byte offset'),'limit':integer('Maximum returned bytes')},['evidence_id']),
        function('search_evidence','Search one evidence file for a literal string within a bounded byte window.',
                 {'evidence_id':dict(type='string'),'query':dict(type='string'),'offset':integer('Start byte')},['evidence_id','query']),
        function('propose_diagnosis','Store a complete candidate without ending the session. Verify before final submission.',diagnosis,['diagnosis']),
        function('submit_diagnosis','Submit the final diagnosis and end the session.',diagnosis,['diagnosis'])]


class Session:
    def __init__(self,case):self.case=case;self.draft=None;self.submission=None;self.calls=[];self.closed=False

    def snapshot(self):return dict(draft=copy.deepcopy(self.draft),submission=copy.deepcopy(self.submission),calls=copy.deepcopy(self.calls),closed=self.closed)

    def restore(self,state):
        self.draft=copy.deepcopy(state['draft']);self.submission=copy.deepcopy(state['submission'])
        self.calls=copy.deepcopy(state['calls']);self.closed=state['closed']

    def execute(self,name,args):
        require(not self.closed,'Session has ended')
        self.calls.append(name)
        if name in ('propose_diagnosis','submit_diagnosis'):
            require(isinstance(args.get('diagnosis'),dict),'diagnosis must be an object')
            self.draft=copy.deepcopy(args['diagnosis'])
            if name=='submit_diagnosis':self.submission=copy.deepcopy(self.draft);self.closed=True
            return {'status':'submitted' if self.closed else 'candidate_stored','candidate_sha256':digest(self.draft)}
        if name=='list_evidence':
            offset=max(0,int(args.get('offset',0)));items=list(self.case.evidence.items())
            return dict(items=[dict(evidence_id=k,name=p.name,bytes=p.stat().st_size) for k,p in items[offset:offset+50]],
                        total=len(items),next_offset=offset+50 if offset+50<len(items) else None)
        require(name in ('read_evidence','search_evidence'),'Unknown tool')
        require(args.get('evidence_id') in self.case.evidence,'Unknown evidence ID; paths are not accepted')
        path=self.case.evidence[args['evidence_id']]
        offset=max(0,int(args.get('offset',0)))
        limit=min(max(1,int(args.get('limit',6000))),12000) if name=='read_evidence' else 256*1024
        with path.open('rb') as stream:stream.seek(offset);raw=stream.read(limit)
        text=raw.decode('utf-8',errors='replace')
        if name=='search_evidence':
            query=str(args['query']);require(0<len(query)<=256,'Search text must have 1..256 characters')
            positions=[];start=0
            while len(positions)<10:
                found=text.casefold().find(query.casefold(),start)
                if found<0:break
                positions.append(dict(window_character_offset=found,text=text[max(0,found-200):found+400]));start=found+max(1,len(query))
            content=positions
        else:content=text
        return dict(evidence_id=args['evidence_id'],start_byte=offset,bytes_read=len(raw),content=content,
                    next_byte=offset+len(raw) if offset+len(raw)<path.stat().st_size else None,
                    receipt=digest({'case':self.case.row['case_id'],'id':args['evidence_id'],'offset':offset,'raw':raw.hex()}))
