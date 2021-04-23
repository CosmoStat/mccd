.. test documentation master file, created by
   sphinx-quickstart on Mon Oct 24 16:46:22 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MCCD Package Contents
=====================

Multi-CCD Point Spread Function Modelling.

:Main contributor: `Tobias Liaudat <https://tobias-liaudat.github.io>`_ `(tobias.liaudat@cea.fr) <tobias.liaudat@cea.fr>`_

:Release date: 08/10/2020

:Documentation: |link-to-docs|

:Repository: |link-to-repo|

.. |link-to-docs| raw:: html

  <a href="https://cosmostat.github.io/mccd/"
  target="_blank">https://cosmostat.github.io/mccd/</a>

.. |link-to-repo| raw:: html

  <a href="https://github.com/CosmoStat/mccd"
  target="_blank">https://github.com/CosmoStat/mccd</a>

----

This package is used to generate a PSF model based on stars observations in the field of view.
Once trained, the MCCD PSF model can then recover the PSF at any position in the field of view.

Installation
============

To install using `pip` run the following command:

.. code-block:: bash

  $ pip install mccd


To clone the MCCD package from GitHub run the following command:

.. code-block:: bash

  $ git clone https://github.com/CosmoStat/mccd.git
  $ cd mccd
  $ python setupy.py install

Check that the installation of PySAP went correctly by launching:

.. code-block:: bash

  $ python setupy.py test

----

Package Contents
================

.. toctree::
   :numbered:
   :maxdepth: 3

   mccd
