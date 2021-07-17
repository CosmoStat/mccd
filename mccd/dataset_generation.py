# -*- coding: utf-8 -*-

r"""DATASET GENERATION UTILITY FUNCTIONS.

Useful function to generate dataset including that include:

- Analytic ellipticity and size variations.

- Variations based on an interpolation of a binned image.

- Random atmospheric contributions with a Von Karman power spectrum.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import mccd
import scipy as sp
import galsim as gs
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt


class GenerateRealisticDataset(object):
    r"""Generate realistic dataset for training and validating PSF models.

    **General considerations**:

    The PSF will have a Moffat profile with a specific beta parameter that
    by default is fixed for the CFIS observations.

    *Optical contributions*:
    This realistic simulation is based on the CFIS survey of CFHT. The idea is
    to use the mean star ellipticity measurements to characterise the optical
    aberrations in terms of ellipticity contribution. We use the mean
    measurements and we interpolate from those images to get the e1, e2
    contribution at the target positions. For the size, we draw a random
    sample from the size distribution of the observed CFIS mean exposure FWHM.

    *Atmospheric contribution*:
    In this case we will be using the Von Karman power function as it has been
    proposed in the Heymans et al. 2012 paper. We use Galsim's PowerSpectrum
    class to generate an atmospheric realisation. We use the same Von Karman
    power law for the E-mode and the B-mode. The outer scale parameter, also
    known as ``theta_zero`` is set with the value measured in Heymans paper.
    Then we simulate a grid of points that correspond to our focal plane, and
    then we interpolate that grid to the target positions. This gives us the
    ellipticity contribution of the atmosphere to our PSF. We adjust the
    variance of the variations to a range that makes sense with our optical
    contributions. We use the magnification of the lensing field to multiply
    the constant exposure size.

    *Total contribution*:
    We need to add the ellipticity contributions of the optical and the
    atmospheric part to get the overall ellipticity distribution of the
    exposure. Concerning the size, as many star selection methods for PSF
    modelling consist in making cuts on the star sizes, we know that the
    observed stars in our exposure have limited size variations. That is
    why we add the ``max_fwhm_var`` parameter that scales the size variations
    so that the maximal variations are within the required range.

    *Star quantity and positions*:
    For the number of stars we draw a random number of stars per CCD that will
    be our mean number of stars per CCD for this particular exposure.
    The distribution is uniform within the range ``range_mean_star_qt``.
    Then for each CCD we draw a random uniform sample from the range
    ``range_dev_star_nb`` that will deviate the star number from the exposure
    mean. However, this deviation shuold have a zero mean,ex [-10, 10].
    Once we have set the number of stars a CCD will have, we draw random
    positions in the corresponding CCD.

    The test dataset can be generated at random positions or at a regular grid.
    Once the train dataset is generated, the ``exposure_sim`` attribute is
    saved for reproductibility issues. The plotting functions
    ``plot_realisation`` and ``plot_correlation`` of the
    ``AtmosphereGenerator`` allows to see the atmospheric realisations.

    **Usage example**::

    sim_dataset_gen = mccd.dataset_generation.GenerateRealisticDataset(
        e1_path=e1_path,
        e2_path=e2_path,
        size_path=fwhm_path,
        output_path=output_path,
        catalog_id=cat_id)
    sim_dataset_gen.generate_train_data()
    sim_dataset_gen.generate_test_data()

    Parameters
    ----------
    e1_path: str
        Path to the binned e1 data.
    e2_path: str
        Path to the binned e2 data.
    size_path: str
        Path to the size distribution. FWHM in arcsec.
    output_path: str
        Path to the folder to save the simulated datasets.
    image_size: int
        Dimension of the squared image stamp. (image_size x image_size)
        Default is ``51``.
    psf_flux: float
        Total PSF photometric flux.
        Default is ``1``.
    beta_psf: float
        Moffat beta parameter.
        Default is ``4.765``.
    pix_scale: float
        Pixel scale.
        Default is ``0.187``.
    catalog_id: int
        Catalog identifier number.
        Default is ``2086592``.
    n_ccd: int
        Total number of CCDs.
        Default is ``40``.
    range_mean_star_qt: [float, float]
        Range of the uniform distribution from where to
        sample the mean star number per exposure.
        Default is ``[40, 90]``.
    range_dev_star_nb: [float, float]
        Range of the uniform distribution from where to
        sample the deviation of star number for each with
        respect to the mean star number.
        Default is ``[-10, 10]``.
    max_fwhm_var: float
        Maximum FWHM variation with respect to the mean allowed
        in one exposure. Units are in arcsec.
        As many times the star selection is done in size cuts,
        the maximum FWHM variations are known.
        Default is ``0.04``.
    save_realisation: bool
        If we need to save the exposure realisation in order to be able to
        reproduce the simulation.

    """
    def __init__(self, e1_path, e2_path, size_path, output_path,
                 image_size=51, psf_flux=1., beta_psf=4.765, pix_scale=0.187,
                 catalog_id=2086592, n_ccd=40, range_mean_star_qt=[40, 100],
                 range_dev_star_nb=[-10, 10], max_fwhm_var=0.04,
                 save_realisation=False,
                 atmos_kwargs={'ngrid': 8192}, e1_kwargs={},
                 e2_kwargs={}):
        # Load the paths
        self.output_path = output_path
        self.e1_path = e1_path
        self.e2_path = e2_path
        self.size_path = size_path

        # Train/Test data params
        self.image_size = image_size
        self.psf_flux = psf_flux
        self.beta_psf = beta_psf
        self.pix_scale = pix_scale
        self.catalog_id = catalog_id
        self.n_ccd = n_ccd
        self.range_mean_star_qt = range_mean_star_qt
        self.range_dev_star_nb = range_dev_star_nb
        self.max_fwhm_var = max_fwhm_var
        self.save_realisation = save_realisation

        # To initialise
        self.test_grid_xy = None
        self.mean_star_qt = None
        self.positions = None
        self.ccd_list = None

        # Define camera geometry
        self.loc2glob = mccd.mccd_utils.Loc2Glob()
        self.max_x = self.loc2glob.x_npix * 6 + self.loc2glob.x_gap * 5
        self.min_x = self.loc2glob.x_npix * (-5) + self.loc2glob.x_gap * (-5)
        self.max_y = self.loc2glob.y_npix * 2 + self.loc2glob.y_gap * 1
        self.min_y = self.loc2glob.y_npix * (-2) + self.loc2glob.y_gap * (-2)

        # Generate exposure instance
        self.exposure_sim = mccd.dataset_generation.ExposureSimulation(
                              e1_bin_path=self.e1_path,
                              e2_bin_path=self.e2_path,
                              fwhm_dist_path=self.size_path,
                              atmos_kwargs=atmos_kwargs,
                              e1_kwargs=e1_kwargs,
                              e2_kwargs=e2_kwargs)

    def init_random_positions(self):
        r""" Initialise random positions."""

        # Draw a random mean star quantity per ccd
        self.mean_star_qt = int(np.ceil(
            np.random.uniform(self.range_mean_star_qt[0],
                              self.range_mean_star_qt[1])))

        # Re-initialise values
        self.positions = None
        self.ccd_list = None

        for ccd_it in range(self.n_ccd):
            # Draw random deviation of stars from the previous mean
            star_dev_nb = np.ceil(
                np.random.uniform(self.range_dev_star_nb[0],
                                  self.range_dev_star_nb[1]))
            # Calculate current CCD star number
            current_star_nb = int(self.mean_star_qt + star_dev_nb)

            # Simulate random positions
            x = np.random.uniform(0,
                                  self.loc2glob.x_npix,
                                  size=(current_star_nb))
            y = np.random.uniform(0,
                                  self.loc2glob.y_npix,
                                  size=(current_star_nb))
            # Shift them to the corresponding CCD
            current_pos = np.array([
                self.loc2glob.loc2glob_img_coord(ccd_it, _x, _y)
                for _x, _y in zip(x, y)])

            # Generate current CCD ids
            current_ccd = np.ones(current_star_nb) * ccd_it

            # Concatenate
            if self.positions is not None:
                self.positions = np.concatenate(
                    (self.positions, current_pos), axis=0)
                self.ccd_list = np.concatenate(
                    (self.ccd_list, current_ccd), axis=0)
            else:
                self.positions = current_pos
                self.ccd_list = current_ccd

    def init_grid_positions(self, x_grid=5, y_grid=10):
        r""" Initialise positions in a regular grid."""
        # Re-initialise values
        self.positions = None
        self.ccd_list = None

        # Parameters
        self.test_grid_xy = [x_grid,
                             y_grid]  # Grid size for the PSF generation

        # Generation of the test positions
        ccd_unique_list = np.arange(self.n_ccd)

        # Generate local generic grid
        x_lin = np.linspace(start=self.image_size,
                            stop=self.loc2glob.x_npix - self.image_size,
                            num=self.test_grid_xy[0])
        y_lin = np.linspace(start=self.image_size,
                            stop=self.loc2glob.y_npix - self.image_size,
                            num=self.test_grid_xy[1])

        xv, yv = np.meshgrid(x_lin, y_lin)
        x_coor = xv.flatten()
        y_coor = yv.flatten()

        position_list = []
        ccd_list = []

        for it in range(len(ccd_unique_list)):
            x_glob, y_glob = self.loc2glob.loc2glob_img_coord(
                ccd_n=ccd_unique_list[it],
                x_coor=np.copy(x_coor), y_coor=np.copy(y_coor))
            position_list.append(np.array([x_glob, y_glob]).T)
            ccd_list.append(
                (np.ones(len(x_glob), dtype=int) * ccd_unique_list[it]).astype(
                    int))

        # Obtain final positions and ccd_id list
        self.positions = np.concatenate(position_list, axis=0)
        self.ccd_list = np.concatenate(ccd_list, axis=0)

    def scale_fwhms(self, fwhms):
        r""" Scale the FWHM values so that they are in the desired range."""
        # Substract the mean
        scaled_fwhms = fwhms - self.exposure_sim.mean_fwhm
        # Divide the total range
        scaled_fwhms = scaled_fwhms / (
            np.max(scaled_fwhms) - np.min(scaled_fwhms))
        # Multiply by the desired total range and restore the mean
        scaled_fwhms = scaled_fwhms * 2 * self.max_fwhm_var + \
            self.exposure_sim.mean_fwhm
        return scaled_fwhms

    def generate_train_data(self):
        r"""Generate the training dataset and saves it in fits format.

        The positions are drawn randomly.
        """
        # Initialise positions
        self.init_random_positions()

        # Define the ellipticities for each stars
        e1s, e2s, fwhms = self.exposure_sim.interpolate_values(
            self.positions[:, 0], self.positions[:, 1])

        # Verify the max fwhm variations
        # We need to scale the variation range
        fwhms = self.scale_fwhms(fwhms)

        # Generate the vignets
        new_vignets = np.zeros((e1s.shape[0],
                                self.image_size,
                                self.image_size))
        new_e1_HSM = np.zeros(e1s.shape)
        new_e2_HSM = np.zeros(e1s.shape)
        new_sig_HSM = np.zeros(e1s.shape)

        for it in range(e1s.shape[0]):
            # PSF generation. Define size
            psf = gs.Moffat(fwhm=fwhms[it], beta=self.beta_psf)

            # Define the Flux
            psf = psf.withFlux(self.psf_flux)
            # Define the shear
            psf = psf.shear(g1=e1s[it], g2=e2s[it])
            # Draw the PSF on a vignet
            image_epsf = gs.ImageF(self.image_size, self.image_size)
            # Define intrapixel shift (uniform distribution in [-0.5,0.5])
            rand_shift = np.random.rand(2) - 0.5
            psf.drawImage(image=image_epsf, offset=rand_shift,
                          scale=self.pix_scale)

            # Generate Gaussian noise for the PSF
            # sigma_noise = 0
            # gaussian_noise = gs.GaussianNoise(sigma=sigma_noise)

            # Before adding the noise, we measure the ellipticity components
            my_moments = gs.hsm.FindAdaptiveMom(image_epsf)
            new_e1_HSM[it] = my_moments.observed_shape.g1
            new_e2_HSM[it] = my_moments.observed_shape.g2
            new_sig_HSM[it] = my_moments.moments_sigma

            # Add Gaussian noise to the PSF
            # image_epsf.addNoise(gaussian_noise)

            new_vignets[it, :, :] = image_epsf.array

        new_masks = self.handle_SExtractor_mask(new_vignets, thresh=-1e5)

        # Build the dictionary
        train_dic = {'VIGNET_LIST': new_vignets,
                     'GLOB_POSITION_IMG_LIST': self.positions,
                     'MASK_LIST': new_masks, 'CCD_ID_LIST': self.ccd_list,
                     'TRUE_E1_HSM': new_e1_HSM, 'TRUE_E2_HSM': new_e2_HSM,
                     'TRUE_SIG_HSM': new_sig_HSM}

        # Save the fits file
        mccd.mccd_utils.save_fits(train_dic,
                                  train_bool=True,
                                  cat_id=self.catalog_id,
                                  output_path=self.output_path)

        if self.save_realisation:
            # Save the exposure object realisation
            cat_id_str = "%07d" % self.catalog_id
            save_str = self.output_path + 'exposure_sim' + '-' + \
                cat_id_str + '.npy'
            np.save(save_str, self.exposure_sim)

    def generate_test_data(self, grid_pos_bool=False, x_grid=5, y_grid=10):
        r"""Generate the test dataset and save it into a fits file.

        Parameters
        ----------
        x_grid: int
            Horizontal number of elements of the testing grid in one CCD.
        y_grid: int
            Vertical number of elements of the testing grid in one CCD.

        """
        # Generate positions (on the grid or at random places)
        if grid_pos_bool:
            self.init_grid_positions(x_grid, y_grid)
        else:
            self.init_random_positions()

        # Calculate the ellipticities on the testing positions
        test_e1s, test_e2s, test_fwhms = self.exposure_sim.interpolate_values(
            self.positions[:, 0], self.positions[:, 1])

        # Verify the max fwhm variations
        # We need to scale the variation range
        test_fwhms = self.scale_fwhms(test_fwhms)

        # Define the constant shape of the stars (before the shearing)
        # sigma_vect = np.sqrt(test_r2s/2)
        # test_fwhms = (2 * np.sqrt(2 * np.log(2))) * sigma_vect

        # Generate the vignets
        test_vignets = np.zeros(
            (test_e1s.shape[0], self.image_size, self.image_size))
        test_e1_HSM = np.zeros(test_e1s.shape)
        test_e2_HSM = np.zeros(test_e1s.shape)
        test_sig_HSM = np.zeros(test_e1s.shape)
        for it in range(test_e1s.shape[0]):
            # PSF generation. Define size
            psf = gs.Moffat(fwhm=test_fwhms[it],
                            beta=self.beta_psf)
            # Define the Flux
            psf = psf.withFlux(self.psf_flux)
            # Define the shear
            psf = psf.shear(g1=test_e1s[it], g2=test_e2s[it])
            # Draw the PSF on a vignet
            image_epsf = gs.ImageF(self.image_size, self.image_size)
            psf.drawImage(image=image_epsf, scale=self.pix_scale)

            # Before adding the noise, we measure the ellipticity components
            my_moments = gs.hsm.FindAdaptiveMom(image_epsf)
            test_e1_HSM[it] = my_moments.observed_shape.g1
            test_e2_HSM[it] = my_moments.observed_shape.g2
            test_sig_HSM[it] = my_moments.moments_sigma

            test_vignets[it, :, :] = image_epsf.array

        # Build the masks
        test_masks = self.handle_SExtractor_mask(test_vignets, thresh=-1e5)

        # Build the dictionary
        test_dic = {'VIGNET_LIST': test_vignets,
                    'GLOB_POSITION_IMG_LIST': self.positions,
                    'MASK_LIST': test_masks, 'CCD_ID_LIST': self.ccd_list,
                    'TRUE_E1_HSM': test_e1_HSM, 'TRUE_E2_HSM': test_e2_HSM,
                    'TRUE_SIG_HSM': test_sig_HSM}

        # Save the fits file
        mccd.mccd_utils.save_fits(test_dic,
                                  train_bool=False,
                                  cat_id=self.catalog_id,
                                  output_path=self.output_path)

    @staticmethod
    def handle_SExtractor_mask(stars, thresh):
        r"""Handle SExtractor masks.
        Reads SExtracted star stamps, generates MCCD-compatible masks
        (that is, binary weights), and replaces bad pixels with 0s - they will
        not be used by MCCD, but the ridiculous numerical values can
        otherwise still lead to problems because of convolutions.
        """
        mask = np.ones(stars.shape)
        mask[stars < thresh] = 0
        stars[stars < thresh] = 0
        return mask


class MomentInterpolator(object):
    r"""Allow to interpolate moments from a bin image.

    Bin image like the one from the MeanShapes function.

    Notes
    -----
    Not used for the moment.
    """
    def __init__(self, moment_map, n_neighbors=1000,
                 rbf_function='thin_plate'):
        r"""Initialize class attributes."""
        # Save variables
        self.loc2glob = mccd.mccd_utils.Loc2Glob()
        self.n_neighbors = n_neighbors
        self.rbf_function = rbf_function
        self.moment_map = np.zeros(moment_map.shape)
        self.x_pix = self.loc2glob.x_npix
        self.y_pix = self.loc2glob.y_npix

        # Define parameters
        self.n_ccd = moment_map.shape[0]  # 40
        self.x_grid = moment_map.shape[1]  # 20
        self.y_grid = moment_map.shape[2]  # 40
        self.bin_x = self.x_pix / self.x_grid
        self.bin_y = self.y_pix / self.y_grid

        # Correct MegaCam origin conventions
        for ccd_it in range(self.n_ccd):
            for it_x in range(self.x_grid):
                for it_y in range(self.y_grid):

                    if ccd_it < 18 or ccd_it in [36, 37]:
                        # swap x axis so origin is on top-right
                        x = it_x
                        y = it_y

                    else:
                        # swap y axis so origin is on bottom-left
                        x = self.x_grid - it_x - 1
                        y = self.y_grid - it_y - 1

                    self.moment_map[ccd_it, x, y] = moment_map[ccd_it, it_x,
                                                               it_y]

        # Generate local generic grid
        x_lin = np.linspace(start=self.bin_x / 2,
                            stop=self.x_pix - self.bin_x / 2, num=self.x_grid)
        y_lin = np.linspace(start=self.bin_y / 2,
                            stop=self.y_pix - self.bin_y / 2, num=self.y_grid)
        xv, yv = np.meshgrid(x_lin, y_lin, indexing='ij')
        self.xv = xv
        self.yv = yv

        # Generate global positions for the bins
        self.x_pos = np.zeros(moment_map.shape)
        self.y_pos = np.zeros(moment_map.shape)
        for ccd_it in range(self.n_ccd):
            x_glob, y_glob = self.loc2glob.loc2glob_img_coord(
                ccd_n=ccd_it,
                x_coor=np.copy(self.xv.flatten()),
                y_coor=np.copy(self.yv.flatten()))

            self.x_pos[ccd_it, :, :] = x_glob.reshape(self.x_grid,
                                                      self.y_grid)
            self.y_pos[ccd_it, :, :] = y_glob.reshape(self.x_grid,
                                                      self.y_grid)

    def interpolate_position(self, target_x, target_y):
        r"""Interpolate positions."""
        # Calculate distances
        res_x = self.x_pos.flatten() - target_x
        res_y = self.y_pos.flatten() - target_y
        dist = np.sqrt(res_x ** 2 + res_y ** 2)

        # Select bins to use. The n_neighbors closest positions
        sort_idxs = np.argsort(dist)[:self.n_neighbors]

        # Extract values
        x_pos_interp = self.x_pos.flatten()[sort_idxs]
        y_pos_interp = self.y_pos.flatten()[sort_idxs]
        val_interp = self.moment_map.flatten()[sort_idxs]

        # Generate the interpolation function
        rbf = Rbf(x_pos_interp, y_pos_interp,
                  val_interp, function=self.rbf_function)
        output_val = rbf(target_x, target_y)

        return output_val


class AtmosphereGenerator(object):
    r""" Generate atmospheric variations.

    This class generates a random atmospheric contributition in terms of
    elloipticity and size.
    The simulation is done using the Von Karman model for isotropic
    atmospheric turbulence. We use the model's 2D power spectrum to generate
    a realisation of the atmosphere of the dimensions of our focal plane.

    The parameter `theta_zero` of the model, also known as the outer scale,
    is by default fixed for th CFHT telescope based on the results of
    Heymans et al. 2012 (DOI: 10.1111/j.1365-2966.2011.20312.x).


    Parameters
    ----------
    theta_zero: float
        Outer scale parameter of the Von Karman model. In arcsec.
    r_trunc: float
        Gaussian truncation parameter of the power spectrum. In arcsec.
    ngrid: int
        Number of grid points to use for our power spectrum realisation.
        Should be a power of 2.
    map_std: float
        Standard deviation of our realisation.
    pix_scale: float
        Pixel scale of our instrument. In arcsec/pixel.

    """
    def __init__(self, theta_zero=3. * 60, r_trunc=1., ngrid=8192,
                 map_std=0.008, pix_scale=0.187):
        # Variables initialised
        self.theta_zero = theta_zero
        self.r_trunc = r_trunc
        self.pix_scale = pix_scale  # arcsec/pixel
        self.ngrid = ngrid  # 2048 # 4096 # 8192
        self.map_std = map_std
        self.loc2glob = mccd.mccd_utils.Loc2Glob()

        # Other variables to initialise
        self.my_ps = None
        self.total_arcsec = None
        self.grid_spacing = None
        self.g1 = None
        self.g2 = None
        self.kappa = None

        # Initialise powerspectrum (might be slow)
        self.init_powerspectrum()

    def power_fun(self, freq):
        r""" Von Karman power function.

        Parameters should be in arcsec.
        Heymans' parameter for the CFHT telescope is in the range
        [2.62, 3.22] arcmin.
        """
        # theta = self.theta_zero
        # r = self.r_trunc
        return (freq**2 + 1 / (self.theta_zero**2))**(- 11 / 6) * \
            np.exp(-freq**2 * (self.r_trunc**2))

    def init_powerspectrum(self):
        r""" Initialise the powerspectrum. """
        # We need to have the hole area of the focal plane expressed in arcsec.
        # Get the maximum values for the global positions (in pixels])
        max_x = self.loc2glob.x_npix * 6 + self.loc2glob.x_gap * 5
        min_x = self.loc2glob.x_npix * (-5) + self.loc2glob.x_gap * (-5)
        max_y = self.loc2glob.y_npix * 2 + self.loc2glob.y_gap * 1
        min_y = self.loc2glob.y_npix * (-2) + self.loc2glob.y_gap * (-2)

        # Max absolute value in pixels
        # This gives us the maximum value of a square [-max_val, max_val]^2
        max_val = np.max(abs(np.array([max_x, min_x, max_y, min_y])))
        # Convert to total arcsec. As it is the total we need to multiply by 2.
        self.total_arcsec = 2 * max_val * self.pix_scale
        # For CFIS this given ~ 4676.06
        # We want to use a power of 2 for the FFTs so we fix the `ngrid`
        # variable (recommended 8192) and we adjust the `grid_spacing`
        # `grid_spacing` is in arcsec/grid_point
        self.grid_spacing = self.total_arcsec / self.ngrid

        # Create the powerspectrum instance
        self.my_ps = gs.PowerSpectrum(
            e_power_function=self.power_fun,
            b_power_function=self.power_fun)

        # Generate grid points of the powerspectrum
        self.g1, self.g2, self.kappa = self.my_ps.buildGrid(
                                            grid_spacing=self.grid_spacing,
                                            ngrid=self.ngrid,
                                            get_convergence=True,
                                            bandlimit='soft',
                                            variance=self.map_std**2)

    def regenerate_atmosphere(self):
        r""" Generate a new random atmosphere."""
        self.init_powerspectrum()

    def interpolate_position(self, target_x, target_y):
        r""" Get the ellipticity and size factor for a target position.

        It is recommended to calculate with 1D arrays as it is much faster.

        Parameters
        ----------
        target_x: 1D np.ndarray or float
            Position x in global MCCD coordinates.
        target_y: 1D np.ndarray or float
            Position y in global MCCD coordinates.

        Returns
        -------
        e1: 1D np.ndarray or float
            At1D np.ndarray or floatmospheric contribution to the first
            ellipticity component.
        e2: 1D np.ndarray or float
            Atmospheric contribution to the second ellipticity component.
        size_factor: 1D np.ndarray or float
            Atmospheric factor afecting the PSF size.

        """
        # Calculate the position in arcsec
        x_asec = target_x * self.pix_scale
        y_asec = target_y * self.pix_scale

        # Interpolate positions
        interp_g1, interp_g2, interp_mu = self.my_ps.getLensing(
                                            [x_asec, y_asec], periodic=True)

        return interp_g1, interp_g2, interp_mu

    def plot_realisation(self, ccd_corner=None, save_path=None,
                         save_fig=False):
        r""" Plot atmospheric realisation.

        Plot the entire focal plane and the dimensions of a CCD.
        """
        if save_path is None:
            save_path = './'
        if ccd_corner is None:
            ccd_corner = int(np.floor(self.ngrid / 2.))

        # Plot the entire focal plane
        plt.figure(figsize=(18, 5))

        plt.subplot(131)
        plt.imshow(self.g1)
        plt.colorbar()
        plt.gca().set_title('g1', fontsize='18')

        plt.subplot(132)
        plt.imshow(self.g2)
        plt.colorbar()
        plt.gca().set_title('g2', fontsize='18')

        plt.subplot(133)
        plt.imshow(self.kappa)
        plt.colorbar()
        plt.gca().set_title('kappa', fontsize='18')

        if save_fig:
            plt.savefig(save_path + 'focal_plane_atmosphere.pdf',
                bbox_inches='tight')
        plt.show()

        # Plot only one CCD
        # We want to calculate how many grid points represent the CCD dimension
        # Only on one CCD. Number of pixels
        pix_x = self.loc2glob.x_npix  # 2048
        pix_y = self.loc2glob.y_npix  # 4612
        # Now CCD dimensions in arcsec
        arcsec_ccd_x = pix_x * self.pix_scale
        arcsec_ccd_y = pix_y * self.pix_scale
        # Now in grid points
        p_ccd_x = int(np.ceil(arcsec_ccd_x / self.grid_spacing))
        p_ccd_y = int(np.ceil(arcsec_ccd_y / self.grid_spacing))

        plt.figure(figsize=(18, 5))
        plt.subplot(131)
        plt.imshow(self.g1[ccd_corner:ccd_corner + p_ccd_x,
                           ccd_corner:ccd_corner + p_ccd_y].T)
        plt.colorbar()
        plt.gca().set_title('g1', fontsize='18')

        plt.subplot(132)
        plt.imshow(self.g2[ccd_corner:ccd_corner + p_ccd_x,
                           ccd_corner:ccd_corner + p_ccd_y].T)
        plt.colorbar()
        plt.gca().set_title('g2', fontsize='18')

        plt.subplot(133)
        plt.imshow(self.kappa[ccd_corner:ccd_corner + p_ccd_x,
                              ccd_corner:ccd_corner + p_ccd_y].T)
        plt.colorbar()
        plt.gca().set_title('kappa', fontsize='18')

        if save_fig:
            plt.savefig(save_path + 'one_ccd_atmosphere.pdf',
                        bbox_inches='tight')
        plt.show()

    def plot_correlation(self, save_path=None, n_points=100, kmin_factor=10.,
                         kmax_factor=10., save_fig=False):
        r""" Plot correlation functions. """
        if save_path is None:
            save_path = './'

        kmin = 2. * np.pi / (self.ngrid * self.grid_spacing) / kmin_factor
        kmax = np.pi / self.grid_spacing * kmax_factor
        print('kmin: ', kmin)
        print('kmax: ', kmax)

        theta, xi_p, xi_m = self.my_ps.calculateXi(
            grid_spacing=self.grid_spacing,
            ngrid=self.ngrid,
            kmax_factor=kmax_factor,
            kmin_factor=kmin_factor,
            n_theta=n_points,
            bandlimit='hard')

        # Convert theta to arcmin
        theta_amin = theta / 60
        theta_zero_amin = self.theta_zero / 60

        plt.figure(figsize=(15, 7))
        plt.semilogx(theta_amin, xi_p, '-o', label=r'$\xi_{+}$', markersize=3)
        plt.semilogx(theta_amin, xi_m, '--o', label=r'$\xi_{-}$', markersize=3)
        plt.vlines(theta_zero_amin, np.min(xi_p), np.max(xi_p), colors='r',
            linestyles='dashed', label=r'$\theta_{0}$')
        plt.grid('minor')
        plt.xlabel(r'$\theta$ [arcmin]', fontsize=18)
        plt.ylabel(r'$\xi_{\pm}$', fontsize=18)
        plt.legend(fontsize=12)
        if save_fig:
            plt.savefig(save_path + 'correlation_atmosphere.pdf',
                bbox_inches='tight')
        plt.show()


class ExposureSimulation(object):
    r""" Simulate one exposure.

    Generate a random exposure and the give the ellipticities and size
    of the PSF for any position in the focal plane.

    Parameters
    ----------
    e1_bin_path: str
        e1 data path.
    e2_bin_path: str
        e2 data path.
    fwhm_dist_path: str
        fwhm distribution path.
    fwhm_range: [float, float]
        The range for the possible fwhm. Units in arcsec.
        Default for CFIS data.
    atmos_kwargs: dict
        Atmosphere arguments.
    e1_kwargs: dict
        e1 interpolator arguments.
    e2_kwargs: dict
        e2 interpolator arguments.

    """
    def __init__(self, e1_bin_path=None, e2_bin_path=None, fwhm_dist_path=None,
                 fwhm_range=[0.45, 1.], atmos_kwargs={}, e1_kwargs={},
                 e2_kwargs={}):
        # Variables
        self.atmos_kwargs = atmos_kwargs
        self.e1_kwargs = e1_kwargs
        self.e2_kwargs = e2_kwargs
        self.e1_bin_path = e1_bin_path
        self.e2_bin_path = e2_bin_path
        self.fwhm_dist_path = fwhm_dist_path
        self.fwhm_range = fwhm_range

        # Variables to init
        self.atmosphere = None
        self.e1_bins = None
        self.e1_interp = None
        self.e2_bins = None
        self.e2_interp = None
        self.mean_fwhm = None

        self.current_e1 = None
        self.current_e2 = None
        self.current_fwhm = None
        self.current_pos = None

        # Initialize exposure
        self.init_exposure()

    def init_exposure(self):
        r""" Initialise exposure variables. """
        # Generate atmosphere
        self.atmosphere = mccd.dataset_generation.AtmosphereGenerator(
            **self.atmos_kwargs)

        # Generate e1 interpolator
        self.e1_bins = np.load(self.e1_bin_path, allow_pickle=True)
        self.e1_interp = mccd.dataset_generation.MomentInterpolator(
            moment_map=self.e1_bins, **self.e1_kwargs)

        # Generate e2 interpolator
        self.e2_bins = np.load(self.e2_bin_path, allow_pickle=True)
        self.e2_interp = mccd.dataset_generation.MomentInterpolator(
            moment_map=self.e2_bins, **self.e2_kwargs)

        # Sample the mean size from the size distribution
        self.fwhm_dist = np.load(self.fwhm_dist_path, allow_pickle=True)[()]
        # Draw a sample from the distribution
        self.mean_fwhm = self.fwhm_dist.ppf(np.random.rand(1))
        # Check that it is inside the limits and draw again if is not
        while self.mean_fwhm < self.fwhm_range[0] or \
                self.mean_fwhm > self.fwhm_range[1]:
            self.mean_fwhm = self.fwhm_dist.ppf(np.random.rand(1))

    def regenerate_exposure(self):
        r""" Regenerate a random exposure. """
        # Regenerate atmosphere
        self.atmosphere.regenerate_atmosphere()
        # Regenerate mean size
        # Draw a sample from the distribution
        self.mean_fwhm = self.fwhm_dist.ppf(np.random.rand(1))
        # Check that it is inside the limits and draw again if is not
        while self.mean_fwhm < self.fwhm_range[0] or \
                self.mean_fwhm > self.fwhm_range[1]:
            self.mean_fwhm = self.fwhm_dist.ppf(np.random.rand(1))

    def interpolate_values(self, target_x, target_y):
        r""" Interpolate exposure values.

        For some target positions interpolate the values (e1, e2, fwhm).
        The input positions are in global MCCD coordinates.
        Faster if the several positions are passed as a np.ndarray.

        Parameters
        ----------
        target_x: float or np.ndarray
            Target positions x coordinate from the global MCCD coordinate
            system.
        target_y: float or np.ndarray
            Target positions y coordinate from the global MCCD coordinate
            system.

        Returns
        -------
        current_e1: float or np.ndarray
            Interpolated e1 values at target positions.
        current_e2: float or np.ndarray
            Interpolated e2 values at target positions.
        current_fwhm: float or np.ndarray
            Interpolated fwhm values at target positions.
            Units in arcsec.
        """
        # Save current positions
        self.current_pos = [target_x, target_y]
        # First calculate the mean variations
        if np.isscalar(target_x):
            self.current_e1 = self.e1_interp.interpolate_position(target_x,
                                                                  target_y)
            self.current_e2 = self.e2_interp.interpolate_position(target_x,
                                                                  target_y)
        else:
            self.current_e1 = np.array([
                self.e1_interp.interpolate_position(_x, _y)
                for _x, _y in zip(target_x, target_y)])
            self.current_e2 = np.array([
                self.e2_interp.interpolate_position(_x, _y)
                for _x, _y in zip(target_x, target_y)])

        self.current_fwhm = np.ones_like(self.current_e1) * self.mean_fwhm

        # Calculate and add the atmospheric part
        atm_contribution = self.atmosphere.interpolate_position(target_x,
                                                                target_y)
        self.current_e1 += atm_contribution[0]
        self.current_e2 += atm_contribution[1]
        self.current_fwhm = self.current_fwhm * atm_contribution[2]

        return self.current_e1, self.current_e2, self.current_fwhm


class GenerateSimDataset(object):
    r"""Generate simulated dataset for training and validating PSF models.

    Parameters
    ----------
    input_pos_path: str
        Path to the global positions of the PSF that will be used for the
        training.
    input_ccd_path: str
        Path to the corresponding CCDs of the global positions.
    output_path: str
        Path to the folder to save the simulated datasets.
    e1_analytic_fun: function
        The analytic e1 ellipticity function that will define an ellipticity
        e1 for each position in the focal plane.
    e2_analytic_fun: function
        The analytic e2 ellipticity function that will define an ellipticity
        e2 for each position in the focal plane.

    Notes
    -----
    The simulated PSFs are based on the Moffat profile and we are using Galsim
    to generate them.
    We base ourselves on two analytic functions that have to output an
    ellipticity for each position in the focal plane.
    """

    def __init__(self, input_pos_path, input_ccd_path, output_path,
                 e1_analytic_fun=None, e2_analytic_fun=None):
        r"""Initialize the class."""
        self.input_pos_path = input_pos_path
        self.input_ccd_path = input_ccd_path
        self.output_path = output_path
        self.e1_analytic_fun = e1_analytic_fun
        self.e2_analytic_fun = e2_analytic_fun

        self.positions = None
        self.ccd_id = None

        # Train/Test data params
        self.sigma = None
        self.image_size = None
        self.psf_flux = None
        self.beta_psf = None
        self.pix_scale = None
        self.desired_SNR = None
        self.catalog_id = None
        self.test_grid_xy = None
        self.n_ccd = None

        if e1_analytic_fun is None:
            self.e1_analytic_fun = self.e1_catalog_fun
        if e2_analytic_fun is None:
            self.e2_analytic_fun = self.e2_catalog_fun

    def load_data(self):
        r"""Load data from input paths."""
        try:
            self.positions = np.load(self.input_pos_path, allow_pickle=True)
            self.ccd_id = np.load(self.input_ccd_path, allow_pickle=True)
        except FileNotFoundError:
            print('The positions or ccd path was not found. Check the paths.')
            raise FileNotFoundError

    def generate_train_data(self, sigma=1.6, image_size=51, psf_flux=1.,
                            beta_psf=4.8, pix_scale=0.187,
                            desired_SNR=30, catalog_id=2086592):
        r"""Generate the training dataset and saves it in fits format.

        Parameters
        ----------
        sigma: float
            Size of the PSF in sigma's. (Sigma from Galsim's HSM adaptive
            moments).
            Default is ``1.6``.
        image_size: int
            Dimension of the squared image stamp. (image_size x image_size)
            Default is ``51``.
        psf_flux: float
            Total PSF photometric flux.
            Default is ``1``.
        beta_psf: float
            Moffat beta parameter.
            Default is ``4.8``.
        pix_scale: float
            Pixel scale.
            Default is ``0.187``.
        desired_SNR: float
            Desired SNR
            Default is ``30``.
        catalog_id: int
            Catalog identifier number.
            Default is ``2086592``.

        """
        # Parameters
        self.sigma = sigma
        self.image_size = image_size
        self.psf_flux = psf_flux
        self.beta_psf = beta_psf
        self.pix_scale = pix_scale
        self.desired_SNR = desired_SNR
        self.catalog_id = catalog_id

        # Define the ellipticities for each stars
        e1s = np.array(
            [self.e1_catalog_fun(position[0], position[1]) for position in
             self.positions])
        e2s = np.array(
            [self.e2_catalog_fun(position[0], position[1]) for position in
             self.positions])

        # Define the constant shape of the stars (before the shearing)
        fwhm_psf = (2 * np.sqrt(2 * np.log(2))) * sigma
        fwhms = np.ones(e1s.shape) * fwhm_psf  # Arround 5. and 6. in sigma

        # Generate the vignets
        new_vignets = np.zeros((e1s.shape[0], image_size, image_size))
        new_e1_HSM = np.zeros(e1s.shape)
        new_e2_HSM = np.zeros(e1s.shape)
        new_sig_HSM = np.zeros(e1s.shape)
        for it in range(e1s.shape[0]):
            # PSF generation. Define size
            psf = gs.Moffat(fwhm=fwhms[it] * pix_scale, beta=beta_psf)

            # Define the Flux
            psf = psf.withFlux(psf_flux)
            # Define the shear
            psf = psf.shear(g1=e1s[it], g2=e2s[it])
            # Draw the PSF on a vignet
            noisy_image_epsf = gs.ImageF(image_size, image_size)
            # Define intrapixel shift (uniform distribution in [-0.5,0.5])
            rand_shift = np.random.rand(2) - 0.5
            psf.drawImage(image=noisy_image_epsf, offset=rand_shift,
                          scale=pix_scale)

            sigma_noise = np.sqrt((np.sum(noisy_image_epsf.array ** 2)) / (
                    desired_SNR * image_size ** 2))
            # Generate Gaussian noise for the PSF
            gaussian_noise = gs.GaussianNoise(sigma=sigma_noise)

            # Before adding the noise, we measure the ellipticity components
            my_moments = gs.hsm.FindAdaptiveMom(noisy_image_epsf)
            new_e1_HSM[it] = my_moments.observed_shape.g1
            new_e2_HSM[it] = my_moments.observed_shape.g2
            new_sig_HSM[it] = my_moments.moments_sigma

            # Add Gaussian noise to the PSF
            noisy_image_epsf.addNoise(gaussian_noise)

            new_vignets[it, :, :] = noisy_image_epsf.array

        new_masks = self.handle_SExtractor_mask(new_vignets, thresh=-1e5)

        # Build the dictionary
        train_dic = {'VIGNET_LIST': new_vignets,
                     'GLOB_POSITION_IMG_LIST': self.positions,
                     'MASK_LIST': new_masks, 'CCD_ID_LIST': self.ccd_id,
                     'TRUE_E1_HSM': new_e1_HSM, 'TRUE_E2_HSM': new_e2_HSM,
                     'TRUE_SIG_HSM': new_sig_HSM}

        # Save the fits file
        mccd.mccd_utils.save_fits(train_dic,
                                  train_bool=True,
                                  cat_id=catalog_id,
                                  output_path=self.output_path)

    def generate_test_data(self, x_grid=5, y_grid=10, n_ccd=40):
        r"""Generate the test dataset and save it into a fits file.

        Parameters
        ----------
        x_grid: int
            Horizontal number of elements of the testing grid in one CCD.
        y_grid: int
            Vertical number of elements of the testing grid in one CCD.
        n_ccd: int
            Number of CCDs in the instrument.

        Notes
        -----
        ``n_ccd`` should be coherent with the corresponding functions on
        ``mccd.mccd_utils`` that do the change of coordiante system.

        """
        # Parameters
        self.test_grid_xy = [x_grid,
                             y_grid]  # Grid size for the PSF generation
        self.n_ccd = n_ccd

        # Generation of the test positions
        ccd_unique_list = np.arange(self.n_ccd)

        # Saving file dictionary
        loc2glob = mccd.mccd_utils.Loc2Glob()

        # Generate local generic grid
        x_lin = np.linspace(start=self.image_size,
                            stop=loc2glob.x_npix - self.image_size,
                            num=self.test_grid_xy[0])
        y_lin = np.linspace(start=self.image_size,
                            stop=loc2glob.y_npix - self.image_size,
                            num=self.test_grid_xy[1])
        xv, yv = np.meshgrid(x_lin, y_lin)
        x_coor = xv.flatten()
        y_coor = yv.flatten()

        position_list = []
        ccd_list = []

        for it in range(len(ccd_unique_list)):
            x_glob, y_glob = loc2glob.loc2glob_img_coord(
                ccd_n=ccd_unique_list[it],
                x_coor=np.copy(x_coor), y_coor=np.copy(y_coor))
            position_list.append(np.array([x_glob, y_glob]).T)
            ccd_list.append(
                (np.ones(len(x_glob), dtype=int) * ccd_unique_list[it]).astype(
                    int))

        # Obtain final positions and ccd_id list
        test_positions = np.concatenate(position_list, axis=0)
        test_ccd_id = np.concatenate(ccd_list, axis=0)

        # Calculate the ellipticities on the testing positions
        test_e1s = np.array(
            [self.e1_catalog_fun(position[0], position[1]) for position in
             test_positions])
        test_e2s = np.array(
            [self.e2_catalog_fun(position[0], position[1]) for position in
             test_positions])

        fwhm_psf = (2 * np.sqrt(2 * np.log(2))) * self.sigma
        test_fwhms = np.ones(
            test_e1s.shape) * fwhm_psf  # Arround 5. and 6. in sigma

        # Generate the vignets
        test_vignets = np.zeros(
            (test_e1s.shape[0], self.image_size, self.image_size))
        test_e1_HSM = np.zeros(test_e1s.shape)
        test_e2_HSM = np.zeros(test_e1s.shape)
        test_sig_HSM = np.zeros(test_e1s.shape)
        for it in range(test_e1s.shape[0]):
            # PSF generation. Define size
            psf = gs.Moffat(fwhm=test_fwhms[it] * self.pix_scale,
                            beta=self.beta_psf)
            # Define the Flux
            psf = psf.withFlux(self.psf_flux)
            # Define the shear
            psf = psf.shear(g1=test_e1s[it], g2=test_e2s[it])
            # Draw the PSF on a vignet
            image_epsf = gs.ImageF(self.image_size, self.image_size)
            psf.drawImage(image=image_epsf, scale=self.pix_scale)

            # Before adding the noise, we measure the ellipticity components
            my_moments = gs.hsm.FindAdaptiveMom(image_epsf)
            test_e1_HSM[it] = my_moments.observed_shape.g1
            test_e2_HSM[it] = my_moments.observed_shape.g2
            test_sig_HSM[it] = my_moments.moments_sigma

            test_vignets[it, :, :] = image_epsf.array

        # Build the masks
        test_masks = self.handle_SExtractor_mask(test_vignets, thresh=-1e5)

        # Build the dictionary
        test_dic = {'VIGNET_LIST': test_vignets,
                    'GLOB_POSITION_IMG_LIST': test_positions,
                    'MASK_LIST': test_masks, 'CCD_ID_LIST': test_ccd_id,
                    'TRUE_E1_HSM': test_e1_HSM, 'TRUE_E2_HSM': test_e2_HSM,
                    'TRUE_SIG_HSM': test_sig_HSM}

        # Save the fits file
        mccd.mccd_utils.save_fits(test_dic,
                                  train_bool=False,
                                  cat_id=self.catalog_id,
                                  output_path=self.output_path)

    @staticmethod
    def e1_catalog_fun(x, y):
        r"""Define an e1 ellipticity per position.

        Analytic function for defining the e1 ellipticity as a function
        of the global position (MegaCam).
        """
        # Set the max and min values for the sinc coordinates
        coor_min = -5
        coor_max = 5

        # Model dependent paremeters
        scale_factor = 0.20001648
        # exp_decay_alpha = 702.86105548
        exp_decay_alpha = None

        scaled_x, scaled_y = GenerateSimDataset.scale_coordinates(
            x, y, coor_min, coor_max)
        scaled_d = np.sqrt(scaled_x ** 2 + scaled_y ** 2)

        vals_x = np.sinc(scaled_d)

        if exp_decay_alpha is not None:
            exp_weight = np.exp(
                -(scaled_d - ((coor_max + coor_min) / 2)) / exp_decay_alpha)
            scale_factor *= exp_weight

        return vals_x * scale_factor

    @staticmethod
    def e2_catalog_fun(x, y):
        r"""Define an e2 ellipticity per position.

        Analytic function for defining the e2 ellipticity as a function
        of the global position (MegaCam).
        """
        # Set the max and min values for the bessel coordinates
        coor_min = -15
        coor_max = 15

        # Model dependent paremeters
        max_order = 1
        scale_factor = 0.15001691
        exp_decay_alpha = 201.13767350

        return GenerateSimDataset.bessel_generator(
            x, y, coor_min, coor_max, max_order, scale_factor,
            circular_symetry=True, exp_decay_alpha=exp_decay_alpha)

    @staticmethod
    def scale_coordinates(x, y, coor_min, coor_max, offset=None):
        r"""Scale global coordinates."""
        # Set the max and min values for the coordinate system
        loc2glob = mccd.mccd_utils.Loc2Glob()
        grid_xmin = - 5 * loc2glob.x_npix - 5 * loc2glob.x_gap
        grid_xmax = 6 * loc2glob.x_npix + 5 * loc2glob.x_gap
        grid_ymin = -2 * loc2glob.y_npix - 2 * loc2glob.y_gap
        grid_ymax = 2 * loc2glob.y_npix + 1 * loc2glob.y_gap

        # Scale the input coordinates
        scaled_x = ((x - grid_xmin) / (grid_xmax - grid_xmin)) * (
                coor_max - coor_min) + coor_min
        scaled_y = ((y - grid_ymin) / (grid_ymax - grid_ymin)) * (
                coor_max - coor_min) + coor_min

        if offset is not None:
            scaled_x += offset[0]
            scaled_y += offset[1]

        return scaled_x, scaled_y

    @staticmethod
    def bessel_generator(x, y, coor_min, coor_max, max_order, scale_factor,
                         circular_symetry=False,
                         exp_decay_alpha=None, offset=None):
        r"""Generate a type of Bessel function response."""
        # Scale coordinates
        scaled_x, scaled_y = GenerateSimDataset.scale_coordinates(
            x, y, coor_min, coor_max, offset=offset)

        # Calculate the function
        vals_x = 0
        vals_y = 0

        if not circular_symetry:
            # Sum of several orders
            for it in range(max_order):
                vals_x += sp.special.jv(it, scaled_x)
                vals_y += sp.special.jv(it, scaled_y)

            # Generate the value and scale it
            return vals_x * vals_y * scale_factor
        else:

            scaled_d = np.sqrt(scaled_x ** 2 + scaled_y ** 2)
            for it in range(max_order):
                vals_x += sp.special.jv(it, scaled_d)

            if exp_decay_alpha is not None:
                exp_weight = np.exp(-(scaled_d - (
                        (coor_max + coor_min) / 2)) / exp_decay_alpha)
                scale_factor *= exp_weight

            return vals_x * scale_factor

    @staticmethod
    def handle_SExtractor_mask(stars, thresh):
        r"""Handle SExtractor masks.

        Reads SExtracted star stamps, generates MCCD-compatible masks
        (that is, binary weights), and replaces bad pixels with 0s - they will
        not be used by MCCD, but the ridiculous numerical values can
        otherwise still lead to problems because of convolutions.
        """
        mask = np.ones(stars.shape)
        mask[stars < thresh] = 0
        stars[stars < thresh] = 0
        return mask
