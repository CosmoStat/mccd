Changelog
=========

- Added new module for realisitic simulations ```dataset_generation.py```.
    It is capable of simulating realistic simulations from the UNIONS/CFIS survey,
    including realistic atmospherical simulations following a realisation of a Von Kármán model.
    See the above-mentioned module documentation for more information.
    See also the ```testing-realistic-data.ipynb``` in the notebook folder for an example. 

- Added outlier rejection based on a pixel residual criterion.
    The main parameters, ```RMSE_THRESH``` and ```CCD_STAR_THRESH``` can be found in the MCCD config file.
    See then parameter documentation for more information.  

- New interpolation function useful for shape measurement pipelines.
    For usage inside shape measurement pipelines: new PSF interpolation
    function included in the MCCD model ```interpolate_psf_pipeline()```.
    This function allows to output interpolated PSFs with a specific centroid.

- New handling of position polynomials, local as well as global polynomials.
    Increased model performance.

- New functionalities added.
    Handling of the max polynomial degree in the local hybrid model by
    the ```D_HYB_LOC```. Also adding a parameter ```MIN_D_COMP_GLOB``` to remove
    lower polynomial degrees in the global polynomial model.

- Increased number of iterations.
    Increased default number of iterations to have a better convergence in the PSF wings.

- Algorithm modifications. 
    Augmented algorithm updates to increase performance.
    Dropping the normalisation proximal operator.
    Harder sparsity constraint for spatial variations.
    Forcing RBF interpolation for the global part.
    Skipping the last weight optimization so that we always finish with a features/components optimization.

- Changed default denoising.
    Set default denoising to zero as wavelet denoising (using starlets) introduce an
    important bias in the ellipticity estimates of the model.

