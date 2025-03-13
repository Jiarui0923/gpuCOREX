import numpy as np
import torch

from .sampler import Sampler

class AdaptiveMonteCarloSampler(Sampler):
    def __init__(self, peptide, partition_id, batch_size=1024, samples=10000,
                 probability=0.75, temperature=298.15, adaptive_rate=0.5, min_rate=0.01):
        super().__init__(peptide, partition_id, batch_size, samples)
        if np.log2(samples) > self._part_num:
            raise ValueError(f'No enough states to be sampled. ({2**self._part_num} samples in total)')
        
        if probability < 1: _ex_sample_num = int(samples / probability * 1.2)
        else: _ex_sample_num = int(samples * 1.5)
            
        _ex_sample_num = _ex_sample_num if np.log2(_ex_sample_num) <= self._part_num else 2**self._part_num
        
        self._rand_samples = torch.randint(0, 2, [_ex_sample_num, self._part_num], dtype=torch.bool)
        self._rand_samples = self._rand_samples.unique(dim=0)
        self._rand_samples = self._rand_samples[torch.any(self._rand_samples, dim=1)]
        self._rand_samples = self._rand_samples[~torch.all(self._rand_samples, dim=1)]
        while len(self._rand_samples) < _ex_sample_num:
            _append_samples = torch.randint(0, 2, [_ex_sample_num-len(self._rand_samples), self._part_num], dtype=torch.bool)
            self._rand_samples = torch.concat([self._rand_samples, _append_samples], dim=0)
            self._rand_samples = self._rand_samples.unique(dim=0)
            self._rand_samples = self._rand_samples[torch.any(self._rand_samples, dim=1)]
            self._rand_samples = self._rand_samples[~torch.all(self._rand_samples, dim=1)]
        # self._rand_samples = torch.concat([self._rand_samples, torch.ones([1, self._part_num])], dim=0)
        self._rand_samples = torch.concat([torch.zeros([1, self._part_num]), self._rand_samples], dim=0)
        
        # self._base_factor = None
        self._base_delta_g = None
        self._prob = probability
        self._adaptive_rate = adaptive_rate
        self._target_prob = probability
        self._temperature = temperature
        self._min_rate = min_rate
        self._visited_states = torch.concat([self._visited_states,
                                             torch.zeros([1, self._part_num]),
                                             torch.ones([1, self._part_num])])
        self._accessed_samples_num = 0
        self._total_comp_samples = 0
        self._accept_rate = None
    
    
    def _comp_delta_g(self, delta_g):
        delta_g = torch.relu(delta_g)
        _thres = torch.rand([len(delta_g)], device=delta_g.device)
        
        if self._accept_rate is not None:
            # if self._accept_rate == 0: self._prob = 2 * self._prob
            # else: self._prob = (self._target_prob / self._accept_rate) * self._prob
            self._prob = (self._target_prob - self._accept_rate) * self._adaptive_rate + self._prob
            if self._prob <= self._min_rate: self._prob = self._min_rate
        _base_factor = self._get_base_factor_value(self._prob)
        delta_g = delta_g * _base_factor
        selection = torch.exp((-delta_g) * (1.0/ (1.9872 * self._temperature)))
        
        index = ~(_thres > selection)
        # for _ in range(0, 9):
        #     _thres = torch.rand([len(selection)], device=delta_g.device)
        #     index = torch.logical_or(~(_thres > selection), index)
        
        return index
        
    def _get_base_factor(self, states, delta_g, exp_k, state_weight):
        _base_index = 0
        self._base_delta_g = delta_g[_base_index]
        
        return states[1:], delta_g[1:], exp_k[1:], state_weight[1:]
    
    def _get_base_factor_value(self, prob):
        if self._base_delta_g < 5000:
            _base_factor = -np.log(prob) / (5000 * (1 / (1.9872 * self._temperature)))
        else:
            _base_factor = -np.log(prob) / (self._base_delta_g * (1 / (1.9872 * self._temperature)))
        return _base_factor
        
    def accept(self, states, delta_g, exp_k, state_weight):
        if self._base_delta_g is None:
            states, delta_g, exp_k, state_weight = self._get_base_factor(states, delta_g, exp_k, state_weight)
        _accept_index = self._comp_delta_g(delta_g)
        
        self._total_comp_samples += len(states)
        _total_samples = len(states)
        
        states = states[_accept_index]
        self._visited_states = torch.concat([self._visited_states, states.cpu()])
        
        exp_k = exp_k[_accept_index]
        state_weight = state_weight[_accept_index]
        
        if len(states) + self._accessed_samples_num > self._sample_num:
            _end = self._sample_num - self._accessed_samples_num
            states, exp_k, state_weight = states[:_end], exp_k[:_end], state_weight[:_end]
        self._accessed_samples_num += len(states)
        
        # self._accept_rate = self._accessed_samples_num / self._total_comp_samples
        self._accept_rate = len(states)/_total_samples
        
        return states, exp_k, state_weight
            
    def __iter__(self):
        self._iter_index = 0
        return self
    def __next__(self):
        if self._accessed_samples_num >= self._sample_num: raise StopIteration
        _end = self._iter_index+self._batch_size
        # if _end > self._sample_num: _end = self._sample_num
        _states = self._rand_samples[self._iter_index:_end]
        self._iter_index += len(_states)
        if len(_states) >= self._batch_size: return _states.to(dtype=torch.bool)
        elif len(_states) + self._accessed_samples_num >= self._sample_num: return _states.to(dtype=torch.bool)
        else:
            # if self._sample_num - (len(_states) + self._accessed_samples_num) > self._batch_size:
            #     _num = self._batch_size
            # else: _num = self._sample_num - (len(_states) + self._accessed_samples_num)
            _num = self._batch_size
            
            _match_tensor = lambda _e, _elems : not (~torch.logical_xor(_e, _elems)).all(dim=1).any()
            _append_states = torch.randint(0, 2, [_num, self._part_num], dtype=torch.bool)
            _append_states = [i for i in _append_states if _match_tensor(i, self._visited_states) and _match_tensor(i, _states)]
            while len(_append_states) < _num:
                if len(_append_states) <= 0:
                    _append_states = torch.randint(0, 2, [_num, self._part_num], dtype=torch.bool)
                    _append_states = [i for i in _append_states if _match_tensor(i, self._visited_states) and _match_tensor(i, _states)]
                else:
                    _append_states = torch.concat([torch.stack(_append_states),
                                                torch.randint(0, 2, [_num-len(_append_states), self._part_num],
                                                                dtype=torch.bool)])
                    _append_states = [i for i in _append_states if _match_tensor(i, self._visited_states) and _match_tensor(i, _states)]
            _append_states = torch.stack(_append_states)
            _states = torch.concat([_states, _append_states])
            return _states.to(dtype=torch.bool)
        
    @property
    def status(self): return f'AcceptRate:{self._accept_rate*100:.4f}%, DynamicProbablity:{self._prob*100:.4f}%'
    @property
    def accessed(self): return self._accessed_samples_num
        
def adaptive_montecarlo(peptide, partition_id, batch_size=1024, samples=10000, probability=0.75, temperature=298.15, adaptive_rate=0.5, min_rate=0.01):
    return AdaptiveMonteCarloSampler(peptide=peptide, partition_id=partition_id,
                            batch_size=batch_size, samples=samples,
                            probability=probability, temperature=temperature,
                            adaptive_rate=adaptive_rate, min_rate=min_rate)
