# -*- coding: utf-8 -*-

"""PACKAGE INFO.

This module provides some basic information about the package.

"""

# Set the package release version
version_info = (1, 2, 2)
__version__ = '.'.join(str(c) for c in version_info)

# Set the package details
__author__ = 'Tobias Liaudat'
__email__ = 'tobiasliaudat@gmail.com'
__year__ = '2020'
__url__ = 'https://github.com/CosmoStat/mccd'
__description__ = 'A non-parametric Multi-CCD Point Spread Function modelling.'

# Default package properties
__license__ = 'MIT'
__about__ = ('{}\nAuthor: {} \nEmail: {} \nYear: {} \nInfo: {}'
             ''.format(__name__, __author__, __email__, __year__,
                       __description__))
__setup_requires__ = ['pytest-runner', ]
__tests_require__ = ['pytest', 'pytest-cov', 'pytest-pycodestyle', 'galsim']
