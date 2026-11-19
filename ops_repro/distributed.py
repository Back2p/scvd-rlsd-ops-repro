"""FSDP reference backend. Full-model GPU execution is deferred to the platform."""
from contextlib import contextmanager
from functools import partial
from datetime import timedelta
import os
from pathlib import Path
import random
import shutil

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,MixedPrecision,
    StateDictType,FullStateDictConfig,ShardedStateDictConfig)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .common import digest,file_hash,load_json,require,write_json
from .models import SelectedPolicy,load_policy
from .training_state import checkpoint_name,shard_records


CONTROL_GROUP=None


def init(seed,training=None):
    require(torch.cuda.is_available(),'CUDA training is required; CPU tests never enter the trainer')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    global CONTROL_GROUP
    training=training or {}
    if not dist.is_initialized():
        dist.init_process_group('nccl',timeout=timedelta(seconds=training.get('collective_timeout_seconds',1800)))
    # Rollout/download-free CPU coordination can take hours. Never leave an NCCL
    # collective pending while rank zero waits for external inference requests.
    CONTROL_GROUP=dist.new_group(backend='gloo',timeout=timedelta(seconds=training.get('control_timeout_seconds',86400)))
    random.seed(seed);torch.manual_seed(seed)
    return dist.get_rank(),dist.get_world_size(),torch.device('cuda',local)


def wrap(cfg,device,*,frozen=False):
    base=load_policy(cfg['model']['path'],dtype=torch.float32,attention=cfg['model'].get('attention','sdpa'))
    if not frozen:base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    base.config.use_cache=False
    for module in base.modules():
        if isinstance(module,torch.nn.Dropout):require(module.p==0,'Nonzero dropout is not supported by this exact-policy reference trainer')
    if frozen:base.requires_grad_(False)
    names=set(getattr(base,'_no_split_modules',[]))
    layer_types={type(m) for m in base.modules() if type(m).__name__ in names and 'Decoder' in type(m).__name__}
    require(bool(layer_types),'Qwen decoder wrap policy could not be identified')
    policy=SelectedPolicy(base,chunk_size=cfg['training']['projection_chunk_size'])
    wrapped=FSDP(policy,auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls=layer_types),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32),
        use_orig_params=True,device_id=device,sync_module_states=True,limit_all_gathers=True)
    wrapped.eval() if frozen else wrapped.train()
    return wrapped


def primary(function):
    packet=[None]
    if dist.get_rank()==0:
        try:packet[0]={'ok':True,'value':function()}
        except Exception as error:packet[0]={'ok':False,'error_type':type(error).__name__,'error':str(error)}
    dist.broadcast_object_list(packet,src=0,group=CONTROL_GROUP)
    require(packet[0]['ok'],'Coordinator failed: '+str(packet[0].get('error')))
    return packet[0]['value']


def collect_sharded(function):
    """Rank zero collects once; each rank receives only its own padded turn shard."""
    rank=dist.get_rank();world=dist.get_world_size();result=None
    def collect():
        nonlocal result
        result=function()
        return {k:v for k,v in result.items() if k!='records'}
    metadata=primary(collect)
    packets=shard_records(result['records'],world) if rank==0 else None
    local=[None]
    dist.scatter_object_list(local,packets,src=0,group=CONTROL_GROUP)
    return metadata,local[0]


def assert_finite(value,message):
    flag=torch.isfinite(value).all().to(dtype=torch.int32)
    dist.all_reduce(flag,op=dist.ReduceOp.MIN)
    require(flag.item()==1,message)


def publish(model,tok,run,step,cfg):
    """Immutable BF16 inference checkpoint; all ranks participate in gathering."""
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        state=model.state_dict()
    def save():
        path=Path(run)/'publications'/f'policy-{step:06}'
        require(not path.exists(),'Publication already exists; use a fresh output or a resume checkpoint')
        path.mkdir(parents=True)
        weights={k.removeprefix('base.'):v.to(torch.bfloat16) if v.is_floating_point() else v for k,v in state.items()}
        model.module.base.save_pretrained(path,state_dict=weights,safe_serialization=True,max_shard_size='4GB')
        tok.save_pretrained(path)
        files={p.name:file_hash(p) for p in sorted(path.iterdir()) if p.is_file()}
        revision=digest({'step':step,'files':files,'protocol':cfg['protocol']})
        write_json(path/'publication.json',dict(step=step,policy_revision=revision,files=files,
            owner_run=digest(str(Path(run).resolve())),parameter_scope='Qwen3.5 text model and LM head; no vision tower'))
        return dict(path=str(path.resolve()),revision=revision,checkpoint_sha256=digest(files))
    result=primary(save)
    del state
    dist.barrier()
    return result


def sync_teacher(student,teacher):
    with FSDP.state_dict_type(student,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=False)):
        state=student.state_dict()
    with FSDP.state_dict_type(teacher,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=False)):
        teacher.load_state_dict(state,strict=True)
    teacher.eval();dist.barrier()


def save_resume(student,teacher,optimizer,run,cursor,cfg):
    folder=Path(run)/'states'/checkpoint_name(cursor)
    primary(lambda:folder.mkdir(parents=True,exist_ok=False))
    with FSDP.state_dict_type(student,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
        student_state=student.state_dict()
    teacher_state=None
    if teacher is not None:
        with FSDP.state_dict_type(teacher,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
            teacher_state=teacher.state_dict()
    local_file=folder/f'rank-{dist.get_rank():04}.pt'
    torch.save(dict(student=student_state,teacher=teacher_state,optimizer=optimizer.state_dict(),
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),python_rng=random.getstate()),local_file)
    hashes=[None]*dist.get_world_size()
    dist.all_gather_object(hashes,{local_file.name:file_hash(local_file)},group=CONTROL_GROUP)
    files={key:value for entry in hashes for key,value in entry.items()}
    dist.barrier()
    primary(lambda:write_json(folder/'complete.json',dict(cursor=cursor,world_size=dist.get_world_size(),config_sha256=digest(cfg),files=files,
        owner_run=digest(str(Path(run).resolve())),split_manifest_sha256=file_hash(Path(__file__).resolve().parents[1]/cfg['cases']))))
    return str(folder)


def load_resume(student,teacher,optimizer,folder,cfg):
    folder=Path(folder);receipt=load_json(folder/'complete.json')
    require(receipt['world_size']==dist.get_world_size(),'Resume requires the original data-parallel world size')
    require(receipt['config_sha256']==digest(cfg),'Resume configuration differs from checkpoint')
    require(receipt['split_manifest_sha256']==file_hash(Path(__file__).resolve().parents[1]/cfg['cases']),'Resume case manifest differs')
    def verify_shards():
        for name,expected in receipt['files'].items():
            candidate=(folder/name).resolve()
            require(candidate.is_relative_to(folder.resolve()),'Resume shard escapes checkpoint directory')
            require(file_hash(candidate)==expected,'Resume shard checksum mismatch')
    primary(verify_shards)
    # These pickle files are created by this package inside the user's run, not downloaded data.
    state=torch.load(folder/f'rank-{dist.get_rank():04}.pt',map_location='cpu',weights_only=False)
    with FSDP.state_dict_type(student,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
        student.load_state_dict(state['student'],strict=True)
    if teacher is not None:
        require(state['teacher'] is not None,'Missing frozen teacher state')
        with FSDP.state_dict_type(teacher,StateDictType.SHARDED_STATE_DICT,ShardedStateDictConfig(offload_to_cpu=True)):
            teacher.load_state_dict(state['teacher'],strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['rng']);torch.cuda.set_rng_state(state['cuda_rng']);random.setstate(state['python_rng'])
    dist.barrier();return receipt['cursor']


def prune_owned(run,section,keep,active=()):
    """Retention affects only complete artifacts created inside this NEW run.

    The user reviews this runtime policy in config. No cleanup is executed while
    building the package. Existing research folders cannot pass ownership checks.
    """
    root=Path(run).resolve();owner=digest(str(root))
    parent=root/section
    if not parent.exists():return
    folders=sorted(p for p in parent.iterdir() if p.is_dir() and not p.is_symlink())
    survivors=set(p.resolve() for p in folders[-keep:])|{Path(p).resolve() for p in active}
    for path in folders:
        if path.resolve() in survivors:continue
        require(path.resolve().is_relative_to(parent.resolve()),'Retention path escapes managed directory')
        marker=path/('publication.json' if section=='publications' else 'complete.json')
        require(marker.is_file() and load_json(marker).get('owner_run')==owner,'Refusing to remove an unowned artifact')
        shutil.rmtree(path)
