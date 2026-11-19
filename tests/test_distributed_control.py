"""Two local CPU Gloo processes; no model, optimizer, sampler or CUDA init."""
from datetime import timedelta
import multiprocessing
from pathlib import Path
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def control_probe(rank,uri):
    import os
    import socket
    # Local-only probe: avoid macOS sandbox hostname/interface discovery failures.
    interfaces={name for _,name in socket.if_nameindex()}
    loopback=next((name for name in ('lo0','lo') if name in interfaces),None)
    if loopback:os.environ['GLOO_SOCKET_IFNAME']=loopback
    import torch
    import torch.distributed as dist
    from ops_repro import distributed as D
    dist.init_process_group('gloo',init_method=uri,rank=rank,world_size=2,timeout=timedelta(seconds=25))
    D.CONTROL_GROUP=dist.new_group(backend='gloo',timeout=timedelta(seconds=25))
    def collect():
        time.sleep(.15)
        return dict(records=[dict(id=i,input_ids=[1,2],labels=[-100,2]) for i in range(3)],groups=7)
    metadata,records=D.collect_sharded(collect)
    assert metadata=={'groups':7}
    assert len(records)==2
    assert [r['id'] for r in records if not r['padding']]==([0,2] if rank==0 else [1])
    def fail():raise ValueError('synthetic coordinator failure')
    try:D.primary(fail)
    except ValueError as error:assert 'synthetic coordinator failure' in str(error)
    else:raise AssertionError('Coordinator error was not propagated')
    assert not torch.cuda.is_initialized()
    dist.destroy_process_group()


class CPUControlTests(unittest.TestCase):
    def test_coordinator_scatter_and_failure_reach_both_cpu_ranks(self):
        context=multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as folder:
            uri=(Path(folder)/'gloo-init').as_uri()
            children=[context.Process(target=control_probe,args=(rank,uri)) for rank in range(2)]
            for child in children:child.start()
            try:
                for child in children:child.join(35)
                self.assertTrue(all(not child.is_alive() for child in children),'CPU control check timed out')
                self.assertEqual([child.exitcode for child in children],[0,0])
            finally:
                for child in children:
                    if child.is_alive():child.terminate();child.join(5)


if __name__=='__main__':unittest.main()
