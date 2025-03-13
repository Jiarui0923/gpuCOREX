from itertools import product
import torch

from .sampler import Sampler

class ExhaustiveSampler(Sampler):
    
    def __init__(self, peptide, partition_id, batch_size=1024, samples=10000):
        super().__init__(peptide, partition_id, batch_size, samples)
        self._sample_num = 2**self._part_num - 2
        if partition_id == (len(peptide.partitions) - 1): self._sample_num += 1
        self._visited_states_num = 0
        
    def __iter__(self):
        self._iter = product([0, 1], repeat=self._part_num)
        next(self._iter)
        return self
    
    def accept(self, states, delta_g, exp_k, state_weight):
        self._visited_states_num += len(states)
        return states, exp_k, state_weight
        
    @property
    def accessed(self): return self._visited_states_num

    def __next__(self):
        if self._iter is None: raise StopIteration
        _states = []
        for _ in range(self._batch_size):
            try: _states.append(next(self._iter))
            except:
                _states = _states[:-1]
                if self._partition_id == (len(self._peptide.partitions) - 1):
                    _states.append([0]*self._part_num)
                self._iter = None
                break
        if len(_states) <= 0: raise StopIteration
        return torch.tensor(_states, dtype=torch.bool)
    
    @property
    def status(self): return None
    
def exhaustive(peptide, partition_id, batch_size=1024, samples=10000):
    return ExhaustiveSampler(peptide=peptide, partition_id=partition_id,
                            batch_size=batch_size, samples=samples)