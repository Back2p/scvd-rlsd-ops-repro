"""Pure control helpers shared by the trainer and CPU regression checks."""
import copy
from .common import require


def checkpoint_name(cursor):
    return f"step-{cursor['updates']:06}-batch-{cursor['batches']:06}"


def shard_records(records,world):
    require(world>0,'World size must be positive')
    if not records:return [[] for _ in range(world)]
    slots=(len(records)+world-1)//world
    dummy=dict(input_ids=records[0]['input_ids'][:2],labels=[-100,-100],
               teacher_ids=records[0].get('teacher_ids',records[0]['input_ids'])[:2],
               teacher_labels=[-100,-100],old_logp=[],advantage=0.,padding=True)
    return [[dict(records[i],padding=False) if i<len(records) else copy.deepcopy(dummy)
             for i in range(rank,slots*world,world)] for rank in range(world)]
