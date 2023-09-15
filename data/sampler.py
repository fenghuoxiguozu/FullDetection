import torch,itertools
import utils.comm as comm
from torch.utils.data.sampler import Sampler


class TrainSampler(Sampler):
    """
    随机采样,根据world_size间隔分组
    """
    def __init__(self, size, shuffle=True,seed=9999):
        self._size = size
        self._world_size = comm.get_world_size()
        self._rank = comm.get_local_rank()
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self):
        yield from itertools.islice(self._local_indices(), self._rank, None, self._world_size)

    def _local_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size,generator=g)
            else:
                yield from torch.arange(self._size)

class InferSampler(Sampler):
    """
    顺序采样
    size:100 world_size:3 rank: 0,1,2   => per:34  
    _local_indices: (0,34)(34,68)(68,100s)
    """

    def __init__(self, size):
        self._world_size = comm.get_world_size()
        self._rank = comm.get_local_rank()
        per = (size-1) // self._world_size + 1
        begin = self._rank * per
        end = min((self._rank+1) * per, size)
        self._local_indices = range(begin, end)

    def __len__(self):
        return len(self._local_indices)

    def __iter__(self):
        yield from self._local_indices
