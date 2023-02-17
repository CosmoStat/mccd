Quickstart Tutorial
===================


You can import the package as follows:

.. code-block:: python

  import mccd


The easiest usage of the method is to go through the configuration file ``config_MCCD.ini`` using the helper classes found
in |link-to-auxiliary_fun_py|
(|link-to-documentation-aux|).
Description of the parameters can be found directly in the configuration file |link-to-config_MCCD_ini|.
The MCCD method can handle SExtractor dataset as input catalogs given that they follow an appropriate naming convention.

The main MCCD model parameters are:

- ``LOC_MODEL``:  Indicating the type of local model to be used (MCCD-HYB, MCCD-RCA, or MCCD-POL),
- ``N_COMP_LOC``: Indicating the number of eigenPSFs to use in the local model.
- ``D_COMP_GLOB``: Indicating the maximum polynomial degree for the global model.

After setting up all the parameters from the configuration file there are three main functions, one to fit the model,
one to validate the model and the last one to fit and then validate the model. The usage is as follows:

.. code-block:: python

  import mccd

  config_file_path = 'path_to_config_file.ini'

  run_mccd_instance = mccd.auxiliary_fun.RunMCCD(config_file_path,
                                               fits_table_pos=1)

  run_mccd_instance.fit_MCCD_models()


For the validation one should replace the last line with:

.. code-block:: python

  run_mccd_instance.validate_MCCD_models()


Finally for the fit and validation one should change the last line to:

.. code-block:: python

  run_mccd_instance.run_MCCD()


All the output file will be saved on the directories specified on the configuration files.


PSF recovery
------------

To recover PSFs from the model at specific positions ```test_pos``` from
the CCD ```ccd_id``` one could use the following example:

.. code-block:: python

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


See the |link-to-documentation|
of the ```recover_MCCD_PSFs()``` function for more information.


Extra information
-----------------

.. tip::

  There are more interpolation functions. For usage inside shape measurement pipelines, the
  new PSF interpolation function included in the MCCD model ```interpolate_psf_pipeline()```.
  This function allows to output interpolated PSFs with a specific centroid.



.. |link-to-documentation| raw:: html

  <a href="https://cosmostat.github.io/mccd/mccd.auxiliary_fun.html" target="_blank">documentation</a>


.. |link-to-auxiliary_fun_py| raw:: html

  <a href="https://github.com/CosmoStat/mccd/blob/master/mccd/auxiliary_fun.py" target="_blank">auxiliary_fun.py</a>


.. |link-to-documentation-aux| raw:: html

  <a href="https://cosmostat.github.io/mccd/mccd.auxiliary_fun.html#" target="_blank">documentation</a>

.. |link-to-config_MCCD_ini| raw:: html

  <a href="https://github.com/CosmoStat/mccd/blob/master/config_MCCD.ini" target="_blank">config_MCCD.ini</a>




