# MCGAT
We developed a Metapath-based Cross-type Synchronized Graph Attention Network, MCGAT for herb-disease association prediction.

## Requirements
Python version
* `python` == 3.11.5


## Require packages
* `pandas` == 2.0.3
* `numpy` == 1.24.3
* `torch` == 2.1.2+cu121
* `dgl` == 2.0.0+cu121


## Require input files
Input files need to run the codes. These files should be in the `data` folder.

* `coconut_he_cp.csv` - The relationships between herb and compound from COCONUT

* `coconut_he_ph.csv` - The relationships between herb and phenotype from COCONUT

* `cp_cp_id.csv` - The relationships between compound and compound from CODA

* `cp_ph_id.csv` - The relationships between compound and phenotype from CODA

* `ph_ph_id.csv` - The relationships between phenotype and phenotype from CODA

## Run analysis
Run `main.py` for model training and testing.
If you want to select metapaths with recursive metapath selection, run `recursive_metapath_selection.py` several times and select metapaths with the highest performance.
