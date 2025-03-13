import pandas as pd
import numpy as np

from .propobj import PropertyObject
from ._convert import convert_batch
from ._manifest import _get_manifest

class Partition(PropertyObject):
    """The Peptide Partition Data Structure

    Attributes
    ----------
    polar_unfolded_a : np.array(float32)
        The polar of the unfolded state of the A chain.
    polar_unfolded : np.array(float32)
        The polar of the unfolded state.
    sconf_unfolded : float32
        The delta conformation stability of the unfolded state.    
    """
    
    def __init__(self, partition, residue, atom_stack, atoms, atom_radius_path=None):
        if atom_radius_path is None: self.atom_radius_path = _get_manifest('atom_radius')
        else: self.atom_radius_path = atom_radius_path
        self.partition = partition
        self.atoms = atoms
        self._init_native_state(partition, residue)
        self._slice(atom_stack)
        
    def __len__(self): len(self.partition)
    def __repr__(self): f'< Partition UnfoldedSconf:{self.sconf_unfolded} >'
     
    def _init_native_state(self, partition, residue):

        asa_expose_a_polar = convert_batch(residue.ASAexapol, partition)
        asa_unfoled_a_polar = np.sum(asa_expose_a_polar, axis=1)
        asa_unfoled_a_polar[-1] += 30.0

        asa_expose_polar = convert_batch(residue.ASAexpol, partition)
        asa_unfoled_polar = np.sum(asa_expose_polar, axis=1)
        asa_unfoled_polar[0] += 45.0
        ot_num = np.sum(self.atoms.is_ot)
        asa_unfoled_polar[-1] += (ot_num * 30.0)

        global_sconf_unfolded = np.sum(convert_batch(residue.GlobalSconfUnfold, partition), axis=1)

        self.polar_unfolded_a = asa_unfoled_a_polar
        self.polar_unfolded = asa_unfoled_polar
        self.sconf_unfolded = global_sconf_unfolded
        
    def _slice(self, atom_stack):
        radius_table = pd.read_csv(self.atom_radius_path, index_col=0).to_dict()['RADIUS']
        coords, radius = [], []
        for i in range(len(self.partition)):
            _atoms, _radius = [], []
            for j in range(self.partition[i][0], self.partition[i][1] + 1):
                k = atom_stack[atom_stack.res_id == j]
                _atoms.append(k.coord)
                _radius.append(np.array([radius_table[e] for e in k.atom_name]))
            coords.append(np.concatenate(_atoms))
            radius.append(np.concatenate(_radius))
        self.coords, self.radius = coords, radius