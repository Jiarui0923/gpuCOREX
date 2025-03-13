import os

_manifest_folder = 'manifest/'
_manifest_dict = {
    'amino_acid_constants': 'amino_acid_constants.csv',
    'atom_radius': 'atom_radius.csv'
}
def _get_manifest(_resource_name):
    
    if _resource_name not in _manifest_dict:
        raise FileExistsError(f'Resource {_resource_name} not found!')
    _pack_path = os.path.dirname(__file__)
    _resource_file = _manifest_dict[_resource_name]
    return os.path.join(_pack_path, _manifest_folder, _resource_file)