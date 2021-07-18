# -*- coding: utf-8 -*-

"""MCCD PACKAGE.

MCCD is a non-parametric PSF modelling method.

:Author:   Tobias Liaudat <tobias.liaudat@cea.fr>

"""

from .mccd import MCCD, mccd_quickload
from . import mccd_utils, utils, grads, proxs, dataset_generation
from . import auxiliary_fun
from .info import __version__, __about__

__all__ = []  # List of submodules
__all__ += ['mccd']
__all__ += ['proxs']
__all__ += ['grads']
__all__ += ['utils']
__all__ += ['mccd_utils']
__all__ += ['auxiliary_fun']
__all__ += ['dataset_generation']
