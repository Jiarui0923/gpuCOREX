import pandas as pd
from biotite.structure import sasa

from .propobj import PropertyObject
from ._manifest import _get_manifest

class Atoms(PropertyObject):
    """The Atom Data Structure

    Attributes
    ----------
    is_carbon : np.array(bool)
        The mask for carbon atoms.
    is_amide : np.array(bool)
        The mask for amide atoms.
    is_side_chain : np.array(bool)
        The mask for side chain atoms.
    is_ot : np.array(bool)
        The mask for OT atoms.
    coord : np.array([float32, float32, float32])
        The coordinates of the atoms.
    radius : np.array(float32)
        The radii of the atoms.
    natural_area : np.array(float32)
        The natural area values for the atoms.
    """
    
    def __init__(self, atom_stack, radius_table=None):
        if radius_table is None: self.radius_table = _get_manifest('atom_radius')
        else: self.radius_table = radius_table
        self._atom_stack = atom_stack
        self._radius_table = radius_table
        self._init_property(atom_stack)
    
    def __len__(self): return len(self._atom_stack)
    def __repr__(self): return f'< ATOMS Number:{len(self)}, OT:{sum(self.is_ot)}, Carbon:{sum(self.is_carbon)}, SideChain:{sum(self.is_side_chain)}  >'
    def _repr_html_(self): return self._atom_df._repr_html_()
    
    def _init_property(self, atom_stack):
        radius_table = pd.read_csv(self._radius_table, index_col=0).to_dict()['RADIUS']
        atom_matrix = []
        for atom, element, coord in zip(atom_stack.atom_name, atom_stack.element, atom_stack.coord):
            atom_matrix.append({'is_carbon': (element == 'C'),
                                'is_amide': (element == 'N'),
                                'is_side_chain': (atom not in ['N', 'CA', 'C', 'O']),
                                'is_ot': (atom == 'OT' or atom == 'OXT'),
                                'radius': radius_table[atom],
                                'coord': coord})
        atom_df = pd.DataFrame(atom_matrix)
        atom_df['natural_area'] = sasa(atom_stack, point_number=1000, vdw_radii=atom_df['radius'].values)
        self._atom_df = atom_df
        self._load_dataframe(atom_df)
        self._atom_df['atom_name'] = self._atom_stack.atom_name
        self._atom_df['res_name'] = self._atom_stack.res_name
        self._atom_df['res_id'] = self._atom_stack.res_id