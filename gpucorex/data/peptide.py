import os
import numpy as np
from biotite.structure.io.pdb import PDBFile

from .propobj import PropertyObject
from .atoms import Atoms
from .residues import Residues
from .partition import Partition
from ._manifest import _get_manifest

class SlicePeptide(PropertyObject):
    """The Peptide Data Structure

    Attributes
    ----------
    atoms : Atoms
        The data structures store atoms information.
    residues: Residues
        The data structures store residues information.
    partitions: list[Partition]
        The list of partitions of the folding units.
    polar_natural: float32
        The natural state polar value for the peptide.
    polar_natural_a: float32
        The natural state polar value for the peptide chain A.
    """
    def __init__(self, path, window_size=10, min_size=4,
                 dSbb_len_correlation = -0.12,
                 residue_constant=None,
                 radius_table=None):
        """Build a peptide structure for COREX computation

        Parameters
        ----------
        path : str
            The path for the PDB file.
        window_size : int
            The partition window size. (Defaults to 10.)
        min_size : int
            The minimal window size of the partition. (Defaults to 4.)
        dSbb_len_correlation : float
            The delta entropy Gibbs free energy factor for the native state. (Defaults to -0.12.)
        residue_constant : str
            The path for the residue constants table. (Defaults to 'amino_acid_constants.csv')
        radius_table : str
            The path for the atom radius table. (Defaults to 'atom_radius.csv'.)
        """
        if residue_constant is None: residue_constant = _get_manifest('amino_acid_constants')
        if radius_table is None: radius_table = _get_manifest('atom_radius')
        self._path = path
        self._name = os.path.basename(path)
        self._atom_stack = self._read_pdb_file(path)
        self._radius_table = radius_table
        self._win_size = window_size
        self.atoms = Atoms(self._atom_stack, radius_table=radius_table)
        self.residues = Residues(self._atom_stack, self.atoms,
                                 dSbb_len_correlation=dSbb_len_correlation,
                                 residue_constant=residue_constant)
        self.partitions = self._partition(len(self.residues), window_size, min_size)
        
        self._init_polar()
        
    def __repr__(self): return f'< Peptide {self._name} PartitionNumber:{len(self.partitions)}, NaturalPolar:{self.polar_natural}, NaturalPolarA:{self.polar_natural_a} >'
        
    def _read_pdb_file(self, file_path):
        file = PDBFile.read(file_path)
        atom_stack = file.get_structure(altloc='first').get_array(0)
        return atom_stack
    
    def _init_polar(self):
        asa_natural_polar = np.sum(self.atoms.natural_area * (1 - self.atoms.is_carbon))
        asa_natural_a_polar = np.sum(self.atoms.natural_area * self.atoms.is_carbon)
        
        self.polar_natural = asa_natural_polar
        self.polar_natural_a = asa_natural_a_polar
        
    
    def _partition(self, length, window_size, min_size):
        full_windows = length // window_size
        partition_schemes = []
        for start in range(window_size):
            windows = []
            for i in range(full_windows):
                if (start + window_size - 1 > length - 1):
                    windows.append([start, length - 1])
                else:
                    windows.append([start, start + window_size - 1])
                start = start + window_size
            if windows[-1][1] < length - 1:
                windows.append([windows[-1][1] + 1, length - 1])
            if (windows[0][0] - 0 <= window_size and windows[0][0] != 0):
                windows.insert(0, [0, windows[0][0] - 1])

            if (windows[-1][1] - windows[-1][0] + 1 < min_size):
                windows.remove(windows[-1])
                windows[-1][1] = length - 1

            if (windows[0][1] - windows[0][0] + 1 < min_size):
                windows.remove(windows[0])
                windows[0][0] = 0

            partition_schemes.append(np.array(windows) + 1)
        partition_schemes = [Partition(windows, self.residues, self._atom_stack, self.atoms, self._radius_table)
                             for windows in partition_schemes]
        return partition_schemes