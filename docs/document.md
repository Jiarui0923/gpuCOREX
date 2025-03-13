<img src="/images/tulane_long.png" width="128px">
<img src="/images/icon_bar.png" width="164px">

# gpuCOREX Documentation

**`UPDATED: 2024/06/26`**  

## Protein Data Structure

The gpuCOREX accepts a protein data structure designed for the algorithm, which can minimize the number of variables.

#### Peptide
The Peptide Data Structure
```python
Peptide(path, window_size=10, min_size=4, dSbb_len_correlation = -0.12,
        residue_constant='amino_acid_constants.csv', radius_table='atom_radius.csv')
```
##### Parameters  
- **path**(`str`)   
    The path for the PDB file.
- **window_size**(`int`)  
    The partition window size. (Defaults to 10.)
- **min_size**(`int`)  
    The minimal window size of the partition. (Defaults to 4.)
- **dSbb_len_correlation**(`float`)  
    The delta entropy Gibbs free energy factor for the native state. (Defaults to -0.12.)
- **residue_constant**(`str`)  
    The path for the residue constants table. (Defaults to 'amino_acid_constants.csv')
- **radius_table**(`str`)  
    The path for the atom radius table. (Defaults to 'atom_radius.csv'.)

##### Attributes
- **atoms**(`Atoms`)  
    The data structures store atoms information.
- **residues**(`Residues`)  
    The data structures store residues information.
- **partitions**(`list[Partition]`)  
    The list of partitions of the folding units.
- **polar_natural**(`float32`)  
    The natural state polar value for the peptide.
- **polar_natural_a**(`float32`)  
    The natural state polar value for the peptide chain A.

##### Example
```python
from gpucorex import Peptide
peptide = Peptide('./3JWO.pdb',
                  window_size=10, min_size=4, dSbb_len_correlation = -0.12,
                  residue_constant='amino_acid_constants.csv', radius_table='atom_radius.csv')
```

## gpuCOREX Code

### COREX
This is the main entrance used to compute COREX
```python
COREX(workers=10, batch_size=1000, samples=10000, device='cpu', dtype=torch.float32,
      sampler=sampler.exhaustive, sampler_args={}, base_fraction=1.0,
      silence=False, probe_radius=1.4, point_number=1000, sconf_weight = 0.5,
      temperature = 298.15, temp_zero = 273.15+60, aCp = 0.44, bCp = -0.26,
      adeltaH = -8.44, bdeltaH = 31.4, TsPolar = 335.15, TsApolar = 385.15,
      context_method='spawn')
```

##### Parameters
- **worker**(`int`)  
    The max number of processes. (Defaults to 10.)
- **batch_size**(`int`)  
    The batch size. (Number of samples in each batch.) (Defaults to 1000.)
- **samples**(`int`)  
    The number of samples.
- **device** (`str|list[str]`)  
    The list of devices or an automatic deivce. (Defaults to 'cpu'.)  
    - 'cuda': Use all cuda devices (Auto assign tasks to all cuda devices).
    - 'cpu': Use CPU devices.
    - ['cuda:0', 'cuda:1']: Use device cuda:0 and cuda:1.
- **dtype**(`torch.type`)  
    The data type for pytorch. (Defaults to `torch.float32`.)
- **sampler**(`Sampler`):  
    The sampler object for micro-states. (Defaults to `sampler.exhaustive`.)
- **sampler_args**(`dict`)  
    The sampler parameters for sampler configuration. (Defaults to `{}`.)
- **silence**(`bool`)  
    Whether show the progress of the computation. (Defualts to `False`.)

##### Example
```python
from gpucorex import COREX
corex = COREX(workers=10, batch_size=1000, samples=10000, device='cpu',
              dtype=torch.float32, sampler=sampler.adaptive_montecarlo, 
              sampler_args=dict(probability=0.75, adaptive_rate=0.05, min_rate=0.01), base_fraction=1.0,
              silence=False, probe_radius=1.4, point_number=1000, sconf_weight = 0.5,
              temperature = 298.15, temp_zero = 273.15+60, aCp = 0.44, bCp = -0.26,
              adeltaH = -8.44, bdeltaH = 31.4, TsPolar = 335.15, TsApolar = 385.15,
              context_method='spawn')
corex_values = corex(peptide)
```
## Constants Table
### Amino Acid Property Table
|Location|Abbrevation|Full Note|
|:-------|:---------:|:--------|
|`file::aaConstants.csv`|`mw`       |molecule weight|
|`file::aaConstants.csv`|`ASAexapol`|accessible surface area exposed polar(A Chain)|
|`file::aaConstants.csv`|`ASAexpol`|accessible surface area exposed polar|
|`file::aaConstants.csv`|`ASAsc`|accessible surface area of side chain|
|`file::aaConstants.csv`|`dSbuex`|$\Delta$entropy burial unfolded exposed|
|`file::aaConstants.csv`|`dSexu`|$\Delta$entropy unfolded exposed|
|`file::aaConstants.csv`|`dSbb`|$\Delta$entropy Gibbs free energy|
|`file::aaConstants.csv`|`ASAexOH`|accessible surface area of exposed `-OH` residue|
|`code::Native_State`|`ASA_U_Apolar`|accessible surface area polar of unfolded atoms (Chain A)|
|`code::Native_State`|`ASA_U_Polar`|accessible surface area polar of unfolded atoms|
|`code::task`|`Sconf`|conformational entropy|