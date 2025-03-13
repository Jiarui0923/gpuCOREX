import numpy as np
import warnings
import torch
import time
from torch.multiprocessing import Manager
from torch.multiprocessing import Lock
from torch.multiprocessing import get_context

from tqdm import tqdm

from .compute import Compute
from .sasa import SolventAccessableSurfaceArea
from . import sampler
from .criterion import Criterion, MaxMinCriterion

class CalSConf(Compute):
    
    def __init__(self, peptide, partition, dtype=torch.float32, device='cpu'):
        self.is_carbon = torch.tensor(peptide.atoms.is_carbon, dtype=torch.bool, device=device)
        self.is_side_chain = torch.tensor(peptide.atoms.is_side_chain, dtype=torch.bool, device=device)
        self.polar_unfolded_a = torch.tensor(partition.polar_unfolded_a, dtype=dtype, device=device)
        self.polar_unfolded = torch.tensor(partition.polar_unfolded, dtype=dtype, device=device)
        self.sconf_unfolded = torch.tensor(partition.sconf_unfolded, dtype=dtype, device=device)
        self.residue_asa_sc = torch.tensor(peptide.residues.ASAsc, dtype=dtype, device=device)
        self.residue_dSbuex = torch.tensor(peptide.residues.dSbuex, dtype=dtype, device=device)
        self.native_exposed = torch.tensor(peptide.residues.NativeExposed, dtype=dtype, device=device)
        self.peptide = peptide
        self.partition = partition
        self.device = device
        
    def convert_residue_shape(self, x, partition):
        partition = np.array(partition).T
        full_size = np.max(partition[1] - partition[0])
        result = []
        for p in partition.T:
            fragment = x[:, p[0] - 1:p[1]]
            fragment = torch.concat([fragment,
                                    torch.zeros([x.shape[0], full_size-(p[1]-p[0])],
                                                dtype=x.dtype,
                                                device=x.device)], dim=1)
            result.append(fragment)
        result = torch.permute(torch.stack(result), [1, 0, 2])
        return result
        
    def forward(self, states, asa_folded):
        res_fold_mask = torch.concat([torch.ones([1, shape[1] - shape[0] + 1], dtype=torch.bool, device=self.device) & status[...,None]
                                for shape, status
                                in zip(self.partition.partition, states.permute([1, 0]))], dim=1)
        atom_fold_mask = torch.concat([torch.ones([1, shape[1] - shape[0] + 1], dtype=torch.bool, device=self.device) & status[...,None]
                                    for shape, status
                                    in zip(self.peptide.residues.shape, res_fold_mask.permute([1, 0]))], dim=1)
        
        state_polar_folded_a = torch.nansum(asa_folded * (self.is_carbon & atom_fold_mask), dim=1)
        state_polar_folded = torch.nansum(asa_folded * (~self.is_carbon & atom_fold_mask), dim=1)

        reversed_states = ~states
        state_polar_unfoled_a = torch.sum(self.polar_unfolded_a * reversed_states, dim=1)
        state_polar_unfoled = torch.sum(self.polar_unfolded * reversed_states, dim=1)
        state_sconf_unfolded = torch.sum(self.sconf_unfolded * reversed_states, dim=1)

        side_chain_folded = self.convert_residue_shape(self.is_side_chain * asa_folded,
                                                       self.peptide.residues.shape)
        state_exposed_fraction = torch.sum(side_chain_folded, dim=2) / self.residue_asa_sc
        state_sconf_folded = torch.nansum((state_exposed_fraction - self.native_exposed) * self.residue_dSbuex, dim=1)

        sconf = state_sconf_unfolded + state_sconf_folded
        delta_polar_a = state_polar_folded_a + state_polar_unfoled_a - self.peptide.polar_natural_a
        delta_polar = state_polar_folded + state_polar_unfoled - self.peptide.polar_natural
        
        return sconf, delta_polar_a, delta_polar
    
class Aggregate(Compute):
    
    def __init__(self, peptide, sconf_weight = 0.5, temperature = 298.15,
                 temp_zero = 273.15+60, aCp = 0.44, bCp = -0.26,
                 adeltaH = -8.44, bdeltaH = 31.4, TsPolar = 335.15, TsApolar = 385.15,
                 dtype = torch.float32, device = 'cpu'):
        
        self.dtype = dtype
        self.device = device
        self.peptide = peptide
        self.sconf_weight = sconf_weight
        self.temperature = temperature
        self.temp_zero = temp_zero
        self.aCp = aCp
        self.bCp = bCp
        self.adeltaH = adeltaH
        self.bdeltaH = bdeltaH
        self.TsPolar = TsPolar
        self.TsApolar = TsApolar
        
    def forward(self, states, partition, sconf, delta_polar_a, delta_polar):

        delta_H_a_polar = delta_polar_a * (self.adeltaH + self.aCp * (self.temperature - self.temp_zero))
        delta_H_polar = delta_polar * (self.bdeltaH + self.bCp * (self.temperature - self.temp_zero))
        dSsolv_ap = delta_polar_a * self.aCp * np.log(self.temperature / self.TsApolar)
        dSsolv_pol = delta_polar * self.bCp * np.log(self.temperature / self.TsPolar)
        dG_solv = delta_H_a_polar + delta_H_polar - self.temperature * (dSsolv_ap + dSsolv_pol)
        dG25 = dG_solv - self.temperature * sconf * self.sconf_weight
        
        dCp = delta_polar_a * self.aCp + delta_polar * self.bCp
        dH60 = delta_polar_a * self.adeltaH + delta_polar * self.bdeltaH
        dH25 = dH60 + dCp * (self.temperature - self.temp_zero)
        dG25_origin = dH25 - self.temperature * (sconf + dSsolv_ap + dSsolv_pol)
        
        state_weight = torch.exp(-dG25 / (1.9872 * self.temperature))

        states_residue = state_weight[...,None] * (~states)
        prob_unfolded = torch.concat([torch.ones([shape[1] - shape[0] + 1], dtype=self.dtype, device=self.device) * state[...,None]
                                      for state, shape in zip(states_residue.T, partition.partition)], dim=1)

        return prob_unfolded, state_weight, dG25, dG25_origin


class WSconfSearch(Compute):
    
    def __init__(self, peptide, criterion=MaxMinCriterion(2.0, 0.1), temperature = 298.15,
                 temp_zero = 273.15+60, aCp = 0.44, bCp = -0.26,
                 adeltaH = -8.44, bdeltaH = 31.4, TsPolar = 335.15, TsApolar = 385.15,
                 dtype = torch.float32, device = 'cpu'):
        
        self.dtype = dtype
        self.device = device
        self.peptide = peptide
        self.temperature = temperature
        self.temp_zero = temp_zero
        self.aCp = aCp
        self.bCp = bCp
        self.adeltaH = adeltaH
        self.bdeltaH = bdeltaH
        self.TsPolar = TsPolar
        self.TsApolar = TsApolar
        self.criterion = criterion
        
    def forward(self, sconf, delta_polar_a, delta_polar, sconf_weights):

        delta_H_a_polar = delta_polar_a * (self.adeltaH + self.aCp * (self.temperature - self.temp_zero))
        delta_H_polar = delta_polar * (self.bdeltaH + self.bCp * (self.temperature - self.temp_zero))
        dSsolv_ap = delta_polar_a * self.aCp * np.log(self.temperature / self.TsApolar)
        dSsolv_pol = delta_polar * self.bCp * np.log(self.temperature / self.TsPolar)
        dG_solv = delta_H_a_polar + delta_H_polar - self.temperature * (dSsolv_ap + dSsolv_pol)

        dG25 = dG_solv - self.temperature * sconf * sconf_weights

        return self.criterion(dG25)

def _corex_process_wrap(kwargs):
    return _corex_process(**kwargs)
    
def _corex_process(peptide, partition_id, batch_size=1000, samples=10000,
                   sampler=sampler.exhaustive, sampler_args={}, device_mutex=Lock(),
                   silence=False, dtype=torch.float32, device='cpu',
                   probe_radius=1.4, point_number=1000, sconf_weight = 0.5,
                   temperature = 298.15, temp_zero = 273.15+60, aCp = 0.44,
                   bCp = -0.26, adeltaH = -8.44, bdeltaH = 31.4,
                   TsPolar = 335.15, TsApolar = 385.15):
    
    _exec_start = time.perf_counter()
    _states_sampler = sampler(peptide, partition_id, batch_size=batch_size, samples=samples, **sampler_args)
    samples = len(_states_sampler)
    partition = peptide.partitions[partition_id]
    if not silence:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bar = tqdm(total=samples,
                                desc=f'[PIPE:{partition_id} <CPU>] Initializing (Pre-Computing)',
                                position=partition_id, leave=True)
    
    _sasa = SolventAccessableSurfaceArea(partition, probe_radius=probe_radius, point_number=point_number,
                                         dtype=dtype, device='cpu')
    _exec_precomp = time.perf_counter()
    if not silence:
        bar.set_description(f'[PIPE:{partition_id} <CPU>] Initialized, Waiting {device}')
    
    with device_mutex:
        _exec_wait = time.perf_counter()
        _sasa.to(device)
        _cal_sconf = CalSConf(peptide, partition, dtype = dtype, device = device)
        _agg = Aggregate(peptide, dtype = dtype, device = device,
                         sconf_weight = sconf_weight, temperature = temperature,
                         temp_zero = temp_zero, aCp = aCp, bCp = bCp,
                         adeltaH = adeltaH, bdeltaH = bdeltaH, TsPolar = TsPolar,
                         TsApolar = TsApolar)
        
        
        prob_unfoldeds, weight_sums = None, None
        for _states in _states_sampler:
            
            _batch_states = _states.to(device=device)

            asa_folded = _sasa(_batch_states)
            sconf, delta_polar_a, delta_polar = _cal_sconf(_batch_states, asa_folded)
            prob_unfolded, state_weight, delta_g, delta_g_origin = _agg(_batch_states, partition, sconf, delta_polar_a, delta_polar)
            
            _batch_states, prob_unfolded, state_weight = _states_sampler.accept(_batch_states, delta_g_origin, prob_unfolded, state_weight)
            
            prob_unfolded = torch.sum(prob_unfolded, dim=0).cpu()
            weight_sum = torch.sum(state_weight).cpu()
            if len(_batch_states) > 0:
                if prob_unfoldeds is None:
                    prob_unfoldeds = prob_unfolded
                    weight_sums = weight_sum
                else:
                    weight_sums += weight_sum
                    prob_unfoldeds += prob_unfolded
                    
            if not silence:
                bar.update(len(_batch_states))
                _sampler_status = _states_sampler.status
                if _sampler_status is not None:
                    bar.set_description(f'[PIPE:{partition_id} <{device}>] {_sampler_status}')
                else:
                    bar.set_description(f'[PIPE:{partition_id} <{device}>] Computing')
                
        _sasa.to('cpu')   
        _cal_sconf.to('cpu') 
        _agg.to('cpu')
        del _sasa
        del _cal_sconf
        del _agg
        del _states_sampler
        torch.cuda.empty_cache()
        
            
    if not silence:
        bar.set_description(f'[PIPE:{partition_id} <{device}>] Finished')
        bar.close()
    _exec_end = time.perf_counter()
    
    _time_pack = (_exec_start, _exec_end, _exec_wait, _exec_precomp)
    return prob_unfoldeds, weight_sums, _time_pack
        
class COREX(Compute):
    """
    COREX is a class that extends the Compute class, providing methods and properties to compute the COREX values
    for a given peptide using specific configurations, such as sampling methods and temperature parameters.

    Attributes
    ----------
    workers : int
        The max number of processes (default 10).
    batch_size : int
        The number of samples in each batch (default 1000).
    samples : int
        The total number of samples to be processed.
    device : str or list[str]
        Specifies the device or devices to be used (default 'cpu').
    dtype : torch.dtype
        Data type for PyTorch computations (default torch.float32).
    sampler : Sampler
        The sampler object used to generate micro-states (default sampler.exhaustive).
    sampler_args : dict
        Additional arguments for the sampler configuration (default {}).
    base_fraction : float
        Base fraction used in COREX computations (default 1.0).
    silence : bool
        If True, suppresses the progress of computation (default False).
    probe_radius : float
        Radius of the probe used in the computation (default 1.4).
    point_number : int
        Number of points used in the computation (default 1000).
    sconf_weight : float or Criterion
        Entropy factor used in the COREX computation, either a float or optimized using the provided criterion (default 1.0).
    temperature : float
        Temperature in Kelvin (default 298.15).
    temp_zero : float
        Temperature offset in Kelvin (default 273.15 + 60).
    aCp : float
        Heat capacity coefficient a (default 0.44).
    bCp : float
        Heat capacity coefficient b (default -0.26).
    adeltaH : float
        Enthalpy change coefficient a (default -8.44).
    bdeltaH : float
        Enthalpy change coefficient b (default 31.4).
    TsPolar : float
        Transition temperature for polar interactions (default 335.15).
    TsApolar : float
        Transition temperature for apolar interactions (default 385.15).
    context_method : str
        Method for creating a multiprocessing context (default 'spawn').

    Methods
    -------
    forward(peptide)
        Computes the COREX values for the given peptide.
    optimize(peptide, criterion=MaxMinCriterion(5.0, 0.1, (0.01, 100)))
        Optimizes the entropy factor (sconf_weight) for the given peptide.
    _build_device_lock(_manager)
        Constructs locks for the devices based on the device configuration.
    time_cost_total()
        Calculates the total execution time for the last run.
    time_start_total()
        Retrieves the start time of the last execution.
    time_end_total()
        Retrieves the end time of the last execution.
    time_cost_process()
        Calculates the execution time for each process.
    time_wait_cost_process()
        Calculates the wait time cost for each process.
    time_precomp_cost_process()
        Calculates the precomputation time cost for each process.
    time_start_process()
        Retrieves the start time for each process.
    time_end_process()
        Retrieves the end time for each process.
    time_wait_process()
        Retrieves the wait time for each process.
    time_precomp_process()
        Retrieves the precomputation time for each process.
    """
    
    def __init__(self, workers=10, batch_size=1000, samples=10000, device='cpu', dtype=torch.float32,
                 sampler=sampler.exhaustive, sampler_args={}, base_fraction=1.0,
                 silence=False, probe_radius=1.4, point_number=1000, sconf_weight = 1.0,
                 temperature = 298.15, temp_zero = 273.15+60, aCp = 0.44, bCp = -0.26,
                 adeltaH = -8.44, bdeltaH = 31.4, TsPolar = 335.15, TsApolar = 385.15,
                 context_method='spawn'):
        """
        Initializes a COREX object with the given parameters.

        Parameters
        ----------
        workers : int, optional
            The max number of processes to be used (default is 10).
        batch_size : int, optional
            The number of samples in each batch (default is 1000).
        samples : int, optional
            The total number of samples to be processed (default is 10000).
        device : str or list[str], optional
            Specifies the device or devices to be used (default is 'cpu').
            - 'cuda': Use all available CUDA devices.
            - 'cpu': Use CPU for computations.
            - ['cuda:0', 'cuda:1']: Use specified CUDA devices.
        dtype : torch.dtype, optional
            Data type for PyTorch computations (default is torch.float32).
        sampler : Sampler, optional
            The sampler object used to generate micro-states (default is sampler.exhaustive).
        sampler_args : dict, optional
            Additional arguments for the sampler configuration (default is {}).
        base_fraction : float, optional
            Base fraction used in COREX computations (default is 1.0).
        silence : bool, optional
            If True, suppresses the progress of computation (default is False).
        probe_radius : float, optional
            Radius of the probe used in the computation (default is 1.4).
        point_number : int, optional
            Number of points used in the computation (default is 1000).
        sconf_weight : float or Criterion, optional
            Entropy factor used in the COREX computation, either a float or optimized using the provided criterion (default is 1.0).
        temperature : float, optional
            Temperature in Kelvin (default is 298.15).
        temp_zero : float, optional
            Temperature offset in Kelvin (default is 273.15 + 60).
        aCp : float, optional
            Heat capacity coefficient a (default is 0.44).
        bCp : float, optional
            Heat capacity coefficient b (default is -0.26).
        adeltaH : float, optional
            Enthalpy change coefficient a (default is -8.44).
        bdeltaH : float, optional
            Enthalpy change coefficient b (default is 31.4).
        TsPolar : float, optional
            Transition temperature for polar interactions (default is 335.15).
        TsApolar : float, optional
            Transition temperature for apolar interactions (default is 385.15).
        context_method : str, optional
            Method for creating a multiprocessing context (default is 'spawn').
        """
        self.sampler = sampler
        self.sampler_args = sampler_args
        self.workers = workers
        self.dtype = dtype
        self.batch_size = batch_size
        self.samples = samples
        self.base_fraction = base_fraction
        self.silence = silence
        self.device = device
        self.context_method = context_method
        
        self.probe_radius = probe_radius
        self.point_number = point_number
        self.sconf_weight = sconf_weight
        self.temperature = temperature
        self.temp_zero = temp_zero
        self.aCp = aCp
        self.bCp = bCp
        self.adeltaH = adeltaH
        self.bdeltaH = bdeltaH
        self.TsPolar = TsPolar
        self.TsApolar = TsApolar
        
        self._last_exec = None
        self._last_exec_end = None
        self._process_exec_times = None
        
    @property
    def time_cost_total(self):
        """
        Returns the total execution time for the last run.

        Returns
        -------
        float
            The total execution time.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._last_exec is not None and self._last_exec_end is not None:
            return self._last_exec_end - self._last_exec
        else: raise RuntimeError('Execution not ready')
    @property
    def time_start_total(self):
        """
        Returns the start time of the last execution.

        Returns
        -------
        float
            The start time of the last execution.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._last_exec: return self._last_exec
        else: raise RuntimeError('Execution not ready')
    @property
    def time_end_total(self):
        """
        Returns the end time of the last execution.

        Returns
        -------
        float
            The end time of the last execution.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._last_exec_end: return self._last_exec_end
        else: raise RuntimeError('Execution not ready')
    @property
    def time_cost_process(self):
        """
        Returns the execution time for each process.

        Returns
        -------
        numpy.array
            Array of execution times for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times:
            _t = np.array(self._process_exec_times).T
            return _t[1] - _t[0]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_wait_cost_process(self):
        """
        Returns the wait time cost for each process.

        Returns
        -------
        numpy.array
            Array of wait time costs for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times:
            _t = np.array(self._process_exec_times).T
            return _t[2] - _t[0]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_precomp_cost_process(self):
        """
        Returns the precomputation time cost for each process.

        Returns
        -------
        numpy.array
            Array of precomputation time costs for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times:
            _t = np.array(self._process_exec_times).T
            return _t[3] - _t[0]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_start_process(self):
        """
        Returns the start time for each process.

        Returns
        -------
        numpy.array
            Array of start times for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times: return np.array(self._process_exec_times).T[0]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_end_process(self):
        """
        Returns the end time for each process.

        Returns
        -------
        numpy.array
            Array of end times for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times: return np.array(self._process_exec_times).T[1]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_wait_process(self):
        """
        Returns the wait time for each process.

        Returns
        -------
        numpy.array
            Array of wait times for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times: return np.array(self._process_exec_times).T[2]
        else: raise RuntimeError('Execution not ready')
    @property
    def time_precomp_process(self):
        """
        Returns the precomputation time for each process.

        Returns
        -------
        numpy.array
            Array of precomputation times for each process.
        
        Raises
        ------
        RuntimeError
            If the execution is not ready.
        """
        if self._process_exec_times: return np.array(self._process_exec_times).T[3]
        else: raise RuntimeError('Execution not ready')
    
    def _build_device_lock(self, _manager):
        """
        Constructs device locks based on the device configuration.

        Parameters
        ----------
        _manager : multiprocessing.Manager
            Manager object to create locks for devices.

        Returns
        -------
        tuple
            A tuple containing the device count and a dictionary of device locks.

        Raises
        ------
        ValueError
            If the device configuration is not supported.
        """
        if isinstance(self.device, list):
            device_count = len(self.device)
            device_mutexs = {device_id: (_device, _manager.Lock())
                                for device_id, _device
                                in enumerate(self.device)}
        elif self.device.lower() == 'cuda' or self.device.lower() == 'gpu':
            device_count = torch.cuda.device_count()
            device_mutexs = {device_id: (f'cuda:{device_id}', _manager.Lock()) for device_id in range(device_count)}
        elif self.device.lower() == 'cpu':
            device_count = self.workers
            device_mutexs = {device_id: ('cpu', _manager.Lock()) for device_id in range(device_count)}
        else: raise ValueError(f'Device {self.device} Not Supported')
        return device_count, device_mutexs
        
    
    def forward(self, peptide):
        """
        Computes the COREX values for the given peptide using the configured parameters.

        Parameters
        ----------
        peptide : Peptide
            The peptide object used to compute COREX values.

        Returns
        -------
        torch.Tensor
            A tensor containing the COREX values.

        Raises
        ------
        RuntimeError
            If the execution is not ready or if the sampler is not configured correctly.
        """
        
        self.peptide = peptide
        if isinstance(self.sconf_weight, Criterion):
            self.sconf_weight = self.optimize(peptide, criterion=self.sconf_weight)
        with Manager() as manager:
            device_count, device_mutexs = self._build_device_lock(manager)

            _pool_context = get_context(self.context_method)
            _tqdm_lock = _pool_context.RLock()
            with _pool_context.Pool(self.workers, initializer=tqdm.set_lock, initargs=(_tqdm_lock, )) as pool:
                _args = []
                for partition_id in range(len(self.peptide.partitions)):
                    device_id = partition_id % device_count
                    device, device_mutex = device_mutexs[device_id]
                    _arg = {'peptide': self.peptide,
                            'partition_id': partition_id,
                            'batch_size': self.batch_size,
                            'samples': self.samples,
                            'sampler': self.sampler,
                            'sampler_args': self.sampler_args,
                            'device_mutex': device_mutex,
                            'silence': self.silence,
                            'dtype': self.dtype,
                            'device': device,
                            'probe_radius': self.probe_radius,
                            'point_number': self.point_number,
                            'sconf_weight': self.sconf_weight,
                            'temperature': self.temperature,
                            'temp_zero': self.temp_zero,
                            'aCp': self.aCp,
                            'bCp': self.bCp,
                            'adeltaH': self.adeltaH,
                            'bdeltaH': self.bdeltaH,
                            'TsPolar': self.TsPolar,
                            'TsApolar': self.TsApolar}
                    _args.append(_arg)
                
                self._last_exec = time.perf_counter()
                _tasks = pool.map(_corex_process_wrap, _args)
                prob_unfoldeds, weight_sums = [], []
                self._process_exec_times = []
                for prob_unfolded, weight_sum, _exec_time  in _tasks:
                    prob_unfoldeds.append(prob_unfolded)
                    weight_sums.append(weight_sum)
                    self._process_exec_times.append(_exec_time)
                prob_unfoldeds, weight_sums = torch.stack(prob_unfoldeds), torch.stack(weight_sums)
                prob_unfolded = torch.sum(prob_unfoldeds, dim=0) / (torch.sum(weight_sums) + self.base_fraction)
                ln_kf = torch.log((1 - prob_unfolded) / prob_unfolded)
                self._last_exec_end = time.perf_counter()
        return ln_kf

    
    def optimize(self, peptide, criterion=MaxMinCriterion(5.0, 0.1, (0.01, 100))):
        """
        Optimizes the entropy factor (sconf_weight) for the given peptide based on the provided criterion.

        Parameters
        ----------
        peptide : Peptide
            The peptide object used to compute COREX values.
        criterion : Criterion, optional
            The criterion used for optimizing the entropy factor (default is MaxMinCriterion(5.0, 0.1, (0.01, 100))).

        Returns
        -------
        float
            The optimized entropy factor (sconf_weight).
        """
        _asa_fold = torch.tensor(peptide.atoms.natural_area, device=self.device, dtype=self.dtype)
        _asa_fold = torch.stack([_asa_fold])
        _cal_sconf = CalSConf(peptide, peptide.partitions[0], dtype = self.dtype, device = self.device)
        _wsonf_search = WSconfSearch(peptide, criterion=criterion, dtype = self.dtype, device = self.device,
                         temperature = self.temperature, temp_zero = self.temp_zero, aCp = self.aCp, bCp = self.bCp,
                         adeltaH = self.adeltaH, bdeltaH = self.bdeltaH, TsPolar = self.TsPolar, TsApolar = self.TsApolar)    
        sconf, delta_polar_a, delta_polar = _cal_sconf.forward(torch.ones([1, len(peptide.partitions[0].partition)], device=self.device, dtype=torch.bool), _asa_fold)
        
        if not self.silence:
            bar = tqdm(desc=f'Optimize Entropy Factor, Search Field: {criterion._search_field}')
        _weights = criterion.parameters(self.batch_size)
        _weights = _weights.to(device=self.device)
        _diff, _weight = _wsonf_search.forward(sconf, delta_polar_a, delta_polar, _weights)      
        if not self.silence:
            bar.update(1)
            bar.set_description(desc=f'Optimize Entropy Factor, Search Field: {criterion._search_field}') 
        while not criterion.done(_diff):
            _weights = criterion.parameters(self.batch_size)
            _weights = _weights.to(device=self.device)
            _diff, _weight = _wsonf_search.forward(sconf, delta_polar_a, delta_polar, _weights)  
            if not self.silence:
                bar.update(1)
                bar.set_description(desc=f'Optimize Entropy Factor, Search Field: {criterion._search_field}')
        if not self.silence:
            bar.set_description(desc=f'Optimize Entropy Factor, Done, Sconf_Weight={_weight.cpu().numpy()}')    
        return float(_weight.cpu().numpy())