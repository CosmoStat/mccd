

[![Build Status](https://travis-ci.org/CosmoStat/mccd.svg?branch=master)](https://travis-ci.org/CosmoStat/mccd)
[![PyPI version](https://badge.fury.io/py/mccd.svg)](https://badge.fury.io/py/mccd)
[![PyPI pyversions](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)
[![arXiv](https://img.shields.io/badge/arXiv-2011.09835-B31B1B)](https://arxiv.org/abs/2011.09835)

# MCCD PSF Modelling

Multi-CCD Point Spread Function Modelling.

---
> Main contributor: <a href="https://tobias-liaudat.github.io" target="_blank" style="text-decoration:none; color: #F08080">Tobias Liaudat</a>  
> Email: <a href="mailto:tobias.liaudat@cea.fr" style="text-decoration:none; color: #F08080">tobias.liaudat@cea.fr</a>  
> Documentation: <a href="https://cosmostat.github.io/mccd/" target="_blank" style="text-decoration:none; color: #F08080">https://cosmostat.github.io/mccd/</a>  
> Article: <a href="https://doi.org/10.1051/0004-6361/202039584" style="text-decoration:none; color: #F08080">DOI - A&A</a>  
> Current release: 16/06/2021
---

The non-parametric MCCD PSF modelling, or MCCD for short, is a Point Spread Function modelling
pure python package.  
It is used to generate a PSF model based on stars observations in the field of view.
Once trained, the MCCD PSF model can then recover the PSF at any position in the field of view.

## Contents

1. [Dependencies](#Dependencies)
1. [Installation](#Installation)
1. [Quick usage](#quick-usage)
1. [Recommendations](#Recommendations)



## Dependencies

The following python packages should be installed with their specific dependencies:

- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [astropy](https://github.com/astropy/astropy)
- [GalSim](https://github.com/GalSim-developers/GalSim)
- [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
- [PySAP](https://github.com/CEA-COSMIC/pysap)

It is of utmost importance that the PySAP package is correctly installed as we will be using the wavelet transforms provided by it.

_Note: The GalSim package was removed from ``requirements.txt``, it is expected to be installed (preferably with conda) before installing the MCCD package._

## Installation

After installing all the dependencies one can perform the MCCD package installation:

#### Locally
```bash
git clone https://github.com/CosmoStat/mccd.git
cd mccd
python setup.py install
```

To verify that the PySAP package is correctly installed and that the MCCD package is
accessing the needed wavelet transforms one can run: ``python setup.py test`` and
check that all the tests are passed.

#### From Pypi
```bash
pip install mccd
```


## Quick usage

The easiest usage of the method is to go through the configuration file ``config_MCCD.ini`` using the helper classes found
in [auxiliary_fun.py](https://github.com/CosmoStat/mccd/blob/master/mccd/auxiliary_fun.py)
([documentation](https://cosmostat.github.io/mccd/mccd.auxiliary_fun.html#)).
Description of the parameters can be found directly in the configuration file [config_MCCD.ini](https://github.com/CosmoStat/mccd/blob/master/config_MCCD.ini).
The MCCD method can handle SExtractor dataset as input catalogs given that they follow an appropriate naming convention.

The main MCCD model parameters are:

- ``LOC_MODEL``:  Indicating the type of local model to be used (MCCD-HYB, MCCD-RCA, or MCCD-POL),
- ``N_COMP_LOC``: Indicating the number of eigenPSFs to use in the local model.
- ``D_COMP_GLOB``: Indicating the maximum polynomial degree for the global model.

After setting up all the parameters from the configuration file there are three main functions, one to fit the model,
one to validate the model and the last one to fit and then validate the model. The usage is as follows:

```python
import mccd

config_file_path = 'path_to_config_file.ini'

run_mccd_instance = mccd.auxiliary_fun.RunMCCD(config_file_path,
                                               fits_table_pos=1)

run_mccd_instance.fit_MCCD_models()
```

For the validation one should replace the last line with:

```python
run_mccd_instance.validate_MCCD_models()
```

Finally for the fit and validation one should change the last line to:

```python
run_mccd_instance.run_MCCD()
```

All the output file will be saved on the directories specified on the configuration files.


#### PSF recovery

To recover PSFs from the model at specific positions ```test_pos``` from
the CCD ```ccd_id``` one could use the following example:

```python
import numpy as np
import mccd

config_file_path = 'path_to_config_file.ini'
mccd_model_path = 'path_to_fitted_mccd_model.npy'
test_pos = np.load(..)
ccd_id = np.load(..)
local_pos = True

mccd_instance = mccd.auxiliary_fun.RunMCCD(config_file_path,
                                           fits_table_pos=1)

rec_PSFs = mccd_instance.recover_MCCD_PSFs(mccd_model_path,
                                           positions=test_pos,
                                           ccd_id=ccd_id,
                                           local_pos=local_pos)
```

See the [documentation](https://cosmostat.github.io/mccd/mccd.auxiliary_fun.html)
of the ```recover_MCCD_PSFs()``` function for more information.

#### Recommendations

Some notebook examples can be found
[here](https://github.com/CosmoStat/mccd/tree/master/notebooks).
