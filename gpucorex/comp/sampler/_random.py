import numpy as np
import torch

from .sampler import Sampler

class RandomSampler(Sampler):
    def __init__(self, peptide, partition_id, batch_size=1024, samples=10000, duplicate=True):
        super().__init__(peptide, partition_id, batch_size, samples)
        if np.log2(samples) > self._part_num:
            raise ValueError(f'No enough states to be sampled. ({2**self._part_num} samples in total)')
        self._duplicate = duplicate
        if self._duplicate:
            self._rand_samples = torch.randint(0, 2, [samples, self._part_num], dtype=torch.bool)
        else:
            self._rand_samples = torch.randint(0, 2, [samples, self._part_num], dtype=torch.bool)
            self._rand_samples = self._rand_samples.unique(dim=0)
            while len(self._rand_samples) < self._sample_num:
                _append_samples = torch.randint(0, 2, [self._sample_num-len(self._rand_samples), self._part_num], dtype=torch.bool)
                self._rand_samples = torch.concat([self._rand_samples, _append_samples], dim=0)
                self._rand_samples = self._rand_samples.unique(dim=0)
            
    def __iter__(self):
        self._iter_index = 0
        return self
    def __next__(self):
        if self._iter_index >= self._sample_num: raise StopIteration
        _states = self._rand_samples[self._iter_index:self._iter_index+self._batch_size]
        self._iter_index += len(_states)
        return _states

def random(peptide, partition_id, batch_size=1024, samples=10000, duplicate=False):
    return RandomSampler(peptide=peptide, partition_id=partition_id,
                            batch_size=batch_size, samples=samples, duplicate=duplicate)
def randupicate(peptide, partition_id, batch_size=1024, samples=10000):
    return RandomSampler(peptide=peptide, partition_id=partition_id,
                            batch_size=batch_size, samples=samples, duplicate=True)
def randunique(peptide, partition_id, batch_size=1024, samples=10000):
    return RandomSampler(peptide=peptide, partition_id=partition_id,
                            batch_size=batch_size, samples=samples, duplicate=False)