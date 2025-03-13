import numpy as np
import torch
from scipy.spatial import KDTree

from .compute import Compute

class SolventAccessableSurfaceArea(Compute):
    
    def __init__(self, partition, probe_radius=1.4, point_number=1000, dtype=torch.float32, device='cpu'):
        self._probe_radius    = probe_radius
        self._point_number    = point_number
        self._area_per_point  = 4.0 * np.pi / point_number
        
        self.sphere_points    = self._create_fibonacci_points(point_number)
        self.sphere_kd_tree   = self._create_kd_tree(self.sphere_points)
        self.device = device
        self.dtype = dtype
        self.setup(partition.coords, partition.radius)
        
        
    def _create_fibonacci_points(self, n):
        phi         = (3 - np.sqrt(5)) * np.pi * np.arange(n)
        z           = np.linspace(1 - 1.0/n, 1.0/n - 1, n)
        radius      = np.sqrt(1 - z*z)
        coords      = np.zeros((n, 3))
        coords[:,0] = radius * np.cos(phi)
        coords[:,1] = radius * np.sin(phi)
        coords[:,2] = z
        return coords
    
    def _create_kd_tree(self, coords, leafsize=10):
        return KDTree(coords,
                      leafsize=leafsize,
                      copy_data=True,
                      balanced_tree=True)
    
    def _build_microstate_ball_tree(self, X):
        _atom_ball_trees = []
        for x in X:
            _atom_ball_trees.append(self._create_kd_tree(x))
        return _atom_ball_trees
    
    def _build_sphere_mask(self, coord, r, _rel_atoms_coords, _rel_atoms_radius):
        
        _rel_atoms_coords = (_rel_atoms_coords - coord) / r
        _rel_atoms_radius = _rel_atoms_radius / r
        
        points = self.sphere_kd_tree.query_ball_point(_rel_atoms_coords, _rel_atoms_radius)
        _mask  = np.zeros(self._point_number)
        if len(points) > 0:
            points = np.asarray(np.unique(np.concatenate(points)), np.int16)
            _mask[points] = 1
        return _mask
    
    def _build_base_mask(self, X, radius, ball_trees):
        
        _base_mask = []
        for coords, rs, tree in zip(X, radius, ball_trees):
            _microstate_base_mask = []
            for i, (coord, r) in enumerate(zip(coords, rs)):
                _atoms_cover = tree.query_ball_point(coord, r + self._max_radius)
                del _atoms_cover[_atoms_cover.index(i)]
                _mask = self._build_sphere_mask(coord, r, coords[_atoms_cover], rs[_atoms_cover])
                _microstate_base_mask.append(_mask)
            _microstate_base_mask = np.asarray(_microstate_base_mask, np.int8)
            _base_mask.append(_microstate_base_mask)
        return _base_mask
    
    def _build_microstate_neighbor_mask(self, coords, radius, neighbors):
        _neighbor_masks = []
        for i, neighbor in enumerate(neighbors):
            _ms_coord, _ms_radius = coords[i], radius[i]
            _neighbor_mask = []
            for _n in neighbor:
                _mask = []
                for _c, _r in zip(_ms_coord, _ms_radius):
                    _atoms = self._atom_ball_trees[_n].query_ball_point(_c, _r + self._max_radius)
                    _mask.append(self._build_sphere_mask(_c, _r, coords[_n][_atoms], radius[_n][_atoms]))
                _neighbor_mask.append(_mask)
            _neighbor_masks.append(np.asarray(_neighbor_mask, np.int8))
        return _neighbor_masks
                
    def _cluster_partition(self, _coord, _radius):
        _cluster_centers = []
        _cluster_radius  = []
        for coord, radius in zip(_coord, _radius):
            _c = np.array([np.average(coord[:,0]), np.average(coord[:,1]), np.average(coord[:,2])])
            _dist = [np.sqrt(np.sum((_c - i)**2)) + r for i, r in zip(coord, radius)]
            _r = max(_dist)
            _cluster_centers.append(_c)
            _cluster_radius.append(_r)
        _cluster_centers = np.array(_cluster_centers)
        _cluster_radius = np.array(_cluster_radius)
        return _cluster_centers, _cluster_radius
    
    def _get_microstate_neighbor(self, _centers, _radius):
        _microstate_kd_tree = self._create_kd_tree(_centers)
        _neighbors = []
        for i, (_c, _r) in enumerate(zip(_centers, _radius)):
            _ns = []
            for j, (_cn, _rn) in enumerate(zip(_centers, _radius)):
                if i != j:
                    if np.sqrt(np.sum((_cn - _c) ** 2)) <= (_r + _rn):
                        _ns.append(j)
            _neighbors.append(_ns)
        return _neighbors, _microstate_kd_tree
    
    def _get_microstate_occupy_nums(self, _micro_state):
        X_ = _micro_state
        X_T = X_.T
        Y = []
        for i in range(X_T.shape[0]):
            X_n = X_T[self._microstate_neighbors[i]].T
            X_m = X_n[:,None,None,:] * self._neighbor_masks[i].permute([1, 2, 0])
            X_m = X_m.permute([0, 3, 1, 2])
            X_o = torch.any(X_m, 1)
            X_o = X_o | self._base_mask[i][None,...]
            X_c = torch.sum(X_o, 2)
            nan_mask = torch.zeros_like(X_T[i], dtype=self.dtype)
            nan_mask[torch.logical_not(X_T[i])] = torch.nan
            y = nan_mask[..., None] + X_c
            Y.append(y)
        Y = torch.concatenate(Y, dim=1)
        return Y
           
    
    def setup(self, X, radius):
        '''
        X: (microstate, atoms, 3)
        radius: (microstate, atoms, 1)
        '''
        self._X                   = X
        self._radius              = [r + self._probe_radius for r in radius]
        self._flatten_radius      = np.concatenate(self._radius)
        self._max_radius          = max(self._flatten_radius)
        self._flatten_radius_sqrt = self._flatten_radius ** 2
        
        self._atom_ball_trees     = self._build_microstate_ball_tree(X)
        self._base_mask           = self._build_base_mask(X, self._radius,
                                                          self._atom_ball_trees)
        self._cluster_centers, self._cluster_radius          = self._cluster_partition(X, self._radius)
        self._microstate_neighbors, self._microstate_kd_tree = self._get_microstate_neighbor(self._cluster_centers,
                                                                                             self._cluster_radius)
        self._neighbor_masks      = self._build_microstate_neighbor_mask(X, self._radius,
                                                                         self._microstate_neighbors)
        self._neighbor_masks = [torch.tensor(i, dtype=torch.bool, device=self.device) for i in self._neighbor_masks]
        self._base_mask = [torch.tensor(i, dtype=torch.bool, device=self.device) for i in self._base_mask]
        self._flatten_radius_sqrt = torch.tensor(self._flatten_radius_sqrt, device=self.device, dtype=self.dtype)
        
    def forward(self, micro_states):
        _points = self._point_number - self._get_microstate_occupy_nums(micro_states)
        _output = _points * self._area_per_point * self._flatten_radius_sqrt
        return _output