import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
  
  
def get_local_rank() -> int:
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
  

def is_main_process() -> bool:
    return get_local_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()
    
    
def init_env(rank,cfg,model):
    world_size = get_world_size()
    dist.init_process_group(backend="NCCL", init_method="tcp://127.0.0.1:25052", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True) 
    return model