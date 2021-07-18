Installation
============

Dependencies
------------

The following python packages should be installed with their specific dependencies:

- |link-to-numpy|
- |link-to-scipy|
- |link-to-astropy|
- |link-to-GalSim|
- |link-to-ModOpt|
- |link-to-pysap|

It is of utmost importance that the PySAP package is correctly installed as we will be using the wavelet transforms provided by it.

.. note::

  The GalSim package was removed from ``requirements.txt``, it is expected to be installed (preferably with conda) before installing the MCCD package.


Users
-----

You can install the latest release from `PyPi <https://pypi.org/project/mccd/>`_
as follows:

.. code-block:: bash

  pip install mccd


Alternatively clone the repository and build the package locally as follows:

.. code-block:: bash

  pip install .


Developers
----------

Developers are recommend to clone the repository and build the package locally
in development mode with testing and documentation packages as follows:

.. code-block:: bash

  pip install -e .





.. |link-to-numpy| raw:: html

  <a href="https://github.com/numpy/numpy/" target="_blank">Numpy</a>

.. |link-to-scipy| raw:: html

  <a href="https://github.com/scipy/scipy" target="_blank">Scipy</a>

.. |link-to-astropy| raw:: html

  <a href="https://github.com/astropy/astropy" target="_blank">astropy</a>

.. |link-to-GalSim| raw:: html

  <a href="https://github.com/GalSim-developers/GalSim" target="_blank">GalSim</a>

.. |link-to-ModOpt| raw:: html

  <a href="https://github.com/CEA-COSMIC/ModOpt" target="_blank">ModOpt</a>

.. |link-to-pysap| raw:: html

  <a href="https://github.com/CEA-COSMIC/pysap" target="_blank">PySap</a>

