import torch
class Sampler:
    
    def __init__(self, peptide, partition_id, batch_size=1024, samples=10000):
        self._sample_num = samples
        self._peptide = peptide
        self._partition_id = partition_id
        self._part_num = len(peptide.partitions[partition_id].partition)
        self._batch_size = batch_size
        self._visited_states = torch.tensor([], dtype=torch.bool)
    
    def accept(self, states, delta_g, exp_k, state_weight):
        self._visited_states = torch.concat([self._visited_states, states.cpu()])
        return states, exp_k, state_weight
        
    @property
    def accessed(self): return len(self._visited_states)
    @property
    def status(self): return None

    def __len__(self): return self._sample_num
    def __iter__(self): raise NotImplemented
    def __next__(self): raise NotImplemented
    def __call__(self, states, delta_g, exp_k, state_weight):
        self.accept(states=states, delta_g=delta_g, exp_k=exp_k, state_weight=state_weight)