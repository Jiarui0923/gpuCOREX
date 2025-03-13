import pandas as pd
import numpy as np

from .propobj import PropertyObject
from ._convert import convert_batch
from ._manifest import _get_manifest

class Residues(PropertyObject):
    """The Residue Data Structure

    Attributes
    ----------
    NativeExposed : np.array(float32)
        The native state exposed values.
    GlobalSconfUnfold : np.array(float32)
        The delta conformation stability of the native unfolded state.
    mw : np.array(float32)
        molecule weight
    ASAexapol : np.array(float32)
        Accessible surface area exposed polar(A Chain).
    ASAexpol : np.array(float32)
        Accessible surface area exposed polar.
    ASAsc : np.array(float32)
        Accessible surface area of side chain.
    dSbuex : np.array(float32)
        Delta entropy burial unfolded exposed.
    dSexu : np.array(float32)
        Delta entropy unfolded exposed.
    dSbb : np.array(float32)
        Delta entropy Gibbs free energy.
    ASAexOH : np.array(float32)
        Accessible surface area of exposed alcohol residue.
    ASA_U_Apolar : np.array(float32)
        Accessible surface area polar of unfolded atoms (Chain A).
    ASA_U_Polar : np.array(float32)
        Accessible surface area polar of unfolded atoms.
    Sconf : np.array(float32)
        conformational entropy
    """
    def __init__(self, atom_stack, atom, dSbb_len_correlation = -0.12, residue_constant='aaConstants.csv'):
        if residue_constant is None: self.residue_constant = _get_manifest('amino_acid_constants')
        else: self.residue_constant = residue_constant
        self._init_property(atom_stack, atom, dSbb_len_correlation, residue_constant)
       
    def __len__(self): return len(self.shape)
    def __repr__(self): return f'< Residues Number:{len(self)}, NativeExposeMean:{np.mean(self.NativeExposed):.4f}, NativeUnfoldMean:{np.mean(self.GlobalSconfUnfold):.4f} >'
    def _repr_html_(self): return self._res_df._repr_html_()
    
    def _load_constant(self, path='aaConstants.csv'):
        amio_acid_table = pd.read_csv(path)
        amio_acid_table = amio_acid_table.iloc[:, 1:]
        amio_acid_table = amio_acid_table.set_index('aa')
        return amio_acid_table
    
    def _init_property(self, atom_stack, atom, dSbb_len_correlation = -0.12, residue_constant='aaConstants.csv'):
        amio_acid_table = self._load_constant(residue_constant)
        res_matrix = np.array([amio_acid_table.loc[atom_stack[atom_stack.res_id == id].res_name[0]].values for id in np.unique(atom_stack.res_id)])
        res_sizes = np.array([np.sum(atom_stack.res_id == id) for id in np.unique(atom_stack.res_id)], dtype=np.int32)
        res_shape = np.array([[np.sum(res_sizes[:i]) + 1, np.sum(res_sizes[:i + 1])]
                            for i in range(len(res_sizes))], dtype=np.int32)
        res_df = pd.DataFrame(res_matrix, columns=amio_acid_table.columns)
        natural_area_side_chain = convert_batch(atom.natural_area * atom.is_side_chain, res_shape)
        native_exposed_fraction = np.sum(natural_area_side_chain, axis=1) / res_df['ASAsc'].values
        global_sconf_unfolded = res_df['dSexu'].values + res_df['dSbb'].values + dSbb_len_correlation + (1 - native_exposed_fraction) * res_df['dSbuex'].values
        res_df['NativeExposed'] = native_exposed_fraction
        res_df['GlobalSconfUnfold'] = global_sconf_unfolded
        self.shape = res_shape
        self._load_dataframe(res_df)
        self._res_df = res_df