# MCCD PSF Modelling

Multi-CCD Point Spread Function Modelling.

---
> Main contributor: <a href="https://tobias-liaudat.github.io" target="_blank" style="text-decoration:none; color: #F08080">Tobias Liaudat</a>  
> Email: <a href="mailto:tobias.liaudat@cea.fr" style="text-decoration:none; color: #F08080">tobias.liaudat@cea.fr</a>  
> Year: 2020
---

The non-parametric MCCD PSF modelling, or MCCD for short, is a Point Spread Function modelling
pure python package. 

## Contents
---

1. [Dependencies](#Dependencies)
1. [Installation](#Installation)
1. [Recomendations](#Recomendations)


## Dependencies
---

The following python packages should be installed with their specific dependencies:

- [numpy](https://github.com/numpy/numpy)
- [scipy](https://github.com/scipy/scipy)
- [astropy](https://github.com/astropy/astropy)
- [GalSim](https://github.com/GalSim-developers/GalSim)
- [ModOpt](https://github.com/CEA-COSMIC/ModOpt)
- [PySAP](https://github.com/CEA-COSMIC/pysap)

It is of utmost importance that the PySAP package is correctly installed as we will be using
the wavelet transforms provided by it.

## Installation
---

After installing all the dependencies one can perform the MCCD package installation:

#### Locally
```bash
git clone https://github.com/CosmoStat/mccd.git
cd mccd
python setup.py install
```

To verify that the PySAP package is correctly installed and that the MCCD package is
accesing the needed wavelet transforms one can run: ``python setup.py test`` and 
check that all the tests are passed.

#### From Pypi
```bash
pip install mccd
```

## Recomendations
---

A useful example notebook ``testing-simulated-data.ipynb`` can be found
[here](https://github.com/CosmoStat/mccd/tree/master/notebooks).

Quick tutorial will be written soon.
