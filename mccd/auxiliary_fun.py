# -*- coding: utf-8 -*-

r"""AUXILIARY FUNCTIONS.

Essential MCCD helper functions. They are needed to:

- Generate simulated datasets.

- Preprocess, fit and validate MCCD PSF models.

- Parse the configuration file.

- Run automatically the MCCD algortithm from the config file parameters.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import os
import numpy as np
import mccd
import mccd.mccd_utils as mccd_utils
import mccd.utils as utils
import scipy as sp
import galsim as gs
import gc
from configparser import ConfigParser
from astropy.io import fits


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
        mccd_utils.save_fits(train_dic, train_bool=True, cat_id=catalog_id,
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
        loc2glob = mccd_utils.Loc2Glob()

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
        mccd_utils.save_fits(test_dic, train_bool=False,
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
        loc2glob = mccd_utils.Loc2Glob()
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


def mccd_fit(starcat, mccd_inst_kw, mccd_fit_kw, output_dir='./',
             catalog_id=1234567, sex_thresh=-1e5, use_SNR_weight=False,
             verbose=False, saving_name='fitted_model'):
    r"""Fits (train) the MCCD model.

    Then saves it.

    Parameters
    ----------
    starcat: fits table
        Opened fits file containing a fits table,
        ie ``astropy.io.fits(path)[1]``.
        Should contain the training dataset with the columns:
        Mandatory: ``CCD_ID_LIST``, ``VIGNET_LIST``,
        ``GLOB_POSITION_IMG_LIST``, ``MASK_LIST``.
        Optional: ``SNR_WIN_LIST``.
    mccd_inst_kw: dict
        Parameters for the MCCD class initialization.
    mccd_fit_kw: dict
        Parameters for the MCCD class fit() function.
    output_dir: str
        Path to the directory to save the fitted model.
    catalog_id: int
        Id of the catalog being trained.
        Default is ``1234567``.
    sex_thresh: float
        Masking threshold, specially for SExtractor catalogs.
        Default is ``-1e5``.
    use_SNR_weight: bool
        Boolean to decide to use the SNR weight strategy. The columns
        ``SNR_WIN_LIST`` should be available on the input fits table.
    verbose: bool
        Verbose mode.
        Default is ``False``.
    saving_name: str
        Name of the fitted model file. Default is ``fitted_model``.

    Notes
    -----
    Saves the fitted model intro a ``.npy`` element on ``output_dir``.

    """
    # Extract the catalog and build the list with each ccd data
    # Stars are already masked and masks are provided
    ccds = np.copy(starcat.data['CCD_ID_LIST']).astype(int)
    ccds_unique = np.unique(np.copy(starcat.data['CCD_ID_LIST'])).astype(int)
    positions = np.copy(starcat.data['GLOB_POSITION_IMG_LIST'])
    stars = np.copy(starcat.data['VIGNET_LIST'])
    masks = np.copy(starcat.data['MASK_LIST'])

    # If masks are not provided they have to be calculated
    if np.sum(masks) == 0:
        masks = utils.handle_SExtractor_mask(stars, sex_thresh)

    # SNR weight calculation
    try:
        if use_SNR_weight:
            SNRs = np.copy(starcat.data['SNR_WIN_LIST'])
            # SNR strategy N6
            max_snr_w = 2.0
            min_snr_w = 0.25
            SNR_weights = SNRs / (np.median(SNRs) + SNRs)
            SNR_weights /= np.max(SNR_weights)
            SNR_weights *= max_snr_w
            SNR_weights[SNR_weights < min_snr_w] = min_snr_w

            SNR_weight_list = [SNR_weights[ccds == ccd] for ccd in
                               ccds_unique]
        else:
            # If no SNR should be use we go to the no-SNR case
            # with the exception
            raise ValueError
    except ValueError:
        SNR_weight_list = None
        if verbose:
            print('No SNR weights are being used.')

    pos_list = [positions[ccds == ccd]
                for ccd in ccds_unique]
    star_list = [utils.rca_format(stars[ccds == ccd])
                 for ccd in ccds_unique]
    mask_list = [utils.rca_format(masks[ccds == ccd])
                 for ccd in ccds_unique]
    ccd_list = [ccds[ccds == ccd].astype(int)
                for ccd in ccds_unique]
    ccd_list = [np.unique(_list)[0].astype(int)
                for _list in ccd_list]

    # Instantiate the method
    mccd_instance = mccd.MCCD(**mccd_inst_kw, verbose=verbose)
    # Launch the training
    _, _, _, _, _ = mccd_instance.fit(
        star_list, pos_list, ccd_list, mask_list,
        SNR_weight_list, **mccd_fit_kw)

    if isinstance(catalog_id, int):
        cat_id = str(catalog_id)
    elif isinstance(catalog_id, str):
        cat_id = catalog_id

    fitted_model_path = output_dir + saving_name + cat_id
    mccd_instance.quicksave(fitted_model_path)

    # Memory management (for clusters)
    del mccd_instance
    gc.collect()


def mccd_validation(mccd_model_path, testcat, apply_degradation=True,
                    mccd_debug=False, global_pol_interp=False,
                    sex_thresh=-1e5):
    r"""Validate a MCCD model.

    Parameters
    ----------
    mccd_model_path: str
        Path to the saved trained MCCD model.
    testcat: fits table
        Opened fits file containing a fits table,
        ie ``astropy.io.fits(path)[1]``.
        Should contain the testing dataset with the columns:
        Mandatory: ``CCD_ID_LIST``, ``VIGNET_LIST``,
        ``GLOB_POSITION_IMG_LIST``, ``MASK_LIST``.
        Optional: ``RA_LIST``, ``DEC_LIST``.
    apply_degradation: bool
        Boolean determining if the returned PSFs should be matched to the
        observed stars by the application of a degradation which consists of
        an intra-pixel shift, fulx matching, etc.
        Default is ``True``.
    mccd_debug: bool
        Boolean to determine if the validation will be run in debug mode.
        Default is ``False``.
    global_pol_interp: bool
        Boolean to determine if the global model interpolation is done with
        a new position matrix (True) or if it is done with the RBF kernel
        interpolation (False).
        Default is ``False``.
    sex_thresh: float
        Masking threshold, specially for SExtractor catalogs.
        Default is ``-1e5``.

    Returns
    -------
    star_dict: dict
        Dictionary containing the data needed for validation purposes.
        Keys: ``PSF_VIGNET_LIST``, ``PSF_MOM_LIST``, ``STAR_MOM_LIST``,
        ``GLOB_POSITION_IMG_LIST``, ``VIGNET_LIST``,
        ``MASK_LIST``, ``CCD_ID_LIST``.
        Optional keys: ``RA_LIST``, ``DEC_LIST``,
        ``PSF_GLOB_VIGNET_LIST``, ``PSF_LOC_VIGNET_LIST``.

    Notes
    -----
    The parameter ``apply_degradation`` needs to be set to True if a pixel
    validation of the model is going to be done.
    """
    # Import
    mccd_model = mccd.mccd_quickload(mccd_model_path)

    # Saving dictionary file
    star_dict = {}

    # Extract data
    ccds = np.copy(testcat.data['CCD_ID_LIST']).astype(int)
    ccds_unique = np.unique(np.copy(testcat.data['CCD_ID_LIST'])).astype(int)
    positions = np.copy(testcat.data['GLOB_POSITION_IMG_LIST'])
    stars = np.copy(testcat.data['VIGNET_LIST'])
    masks = np.copy(testcat.data['MASK_LIST'])
    try:
        RA_pos = np.copy(testcat.data['RA_LIST'])
        DEC_pos = np.copy(testcat.data['DEC_LIST'])
    except KeyError:
        RA_pos = None
        DEC_pos = None

    # If masks are not provided they have to be calculated
    # Mask convention: 1=good pixel / 0=bad pixel
    if ~np.any(masks):
        masks = utils.handle_SExtractor_mask(stars, sex_thresh)

    # Prepare data in ccd-list format
    if RA_pos is not None:
        val_RA_list = [RA_pos[ccds == ccd]
                       for ccd in ccds_unique]
        val_DEC_list = [DEC_pos[ccds == ccd]
                        for ccd in ccds_unique]
    else:
        val_RA_list, val_DEC_list = None, None

    val_pos_list = [positions[ccds == ccd]
                    for ccd in ccds_unique]
    val_star_list = [utils.rca_format(stars[ccds == ccd])
                     for ccd in ccds_unique]
    val_mask_list = [utils.rca_format(masks[ccds == ccd])
                     for ccd in ccds_unique]
    val_ccd_list = [ccds[ccds == ccd].astype(int)
                    for ccd in ccds_unique]
    val_ccd_list_to_save = np.copy(val_ccd_list)
    val_ccd_list = [np.unique(_list)[0].astype(int)
                    for _list in val_ccd_list]

    if apply_degradation:
        if mccd_debug:
            PSF_list = []
            PSF_glob_list = []
            PSF_loc_list = []
            if global_pol_interp:
                print('''The polynomial interpolation for the global model is
                not available in mccd_debug mode.''')

            for it in range(len(val_star_list)):

                deg_PSFs, deg_PSFs_glob, deg_PSFs_loc = \
                    mccd_model.validation_stars(
                        val_star_list[it], val_pos_list[it], val_mask_list[it],
                        val_ccd_list[it], mccd_debug=mccd_debug)

                PSF_list.append(deg_PSFs)
                if deg_PSFs_glob is not None:
                    PSF_glob_list.append(deg_PSFs_glob)
                    PSF_loc_list.append(deg_PSFs_loc)

            star_dict['PSF_GLOB_VIGNET_LIST'] = np.copy(
                np.concatenate(PSF_glob_list, axis=0))
            star_dict['PSF_LOC_VIGNET_LIST'] = np.copy(
                np.concatenate(PSF_loc_list, axis=0))
        else:

            global_pol_interp = False

            if global_pol_interp:
                interp_Pi = mccd_utils.interpolation_Pi(
                    val_pos_list, mccd_model.d_comp_glob)

                PSF_list = [mccd_model.validation_stars(
                    _star, _pos, _mask, _ccd_id,
                    mccd_debug=mccd_debug, global_pol_interp=_iterp_Pi)
                    for _star, _pos, _mask, _ccd_id, _iterp_Pi in
                    zip(val_star_list, val_pos_list, val_mask_list,
                        val_ccd_list, interp_Pi)]
            else:
                PSF_list = [mccd_model.validation_stars(
                    _star, _pos, _mask, _ccd_id, mccd_debug=mccd_debug)
                    for _star, _pos, _mask, _ccd_id in
                    zip(val_star_list, val_pos_list, val_mask_list,
                        val_ccd_list)]

    # Remove the CCDs not used for training from ALL the lists
    # Identify the None elements on a boolean list
    bad_ccds_bool = [psf is None for psf in PSF_list]
    # Get the True indexes
    bad_ccds_indexes = [i for i, x in enumerate(bad_ccds_bool) if x]
    # Delete the None elements from the lists from the last to the first one
    # not to mess up with the order
    for idx in sorted(bad_ccds_indexes, reverse=True):
        del PSF_list[idx]
        del val_pos_list[idx]
        del val_star_list[idx]
        del val_mask_list[idx]
        del val_ccd_list[idx]
        if val_RA_list is not None:
            del val_RA_list[idx]
            del val_DEC_list[idx]

    # Have the PSFs in rca format, to match the stars
    PSF_list = [utils.rca_format(psfs) for psfs in PSF_list]

    # Calculate moments
    star_shapes_list, psf_shapes_list = [], []
    psf_vignet_list = []
    pos_list = []
    star_list = []
    mask_list = []
    ccd_list = []
    RA_list, DEC_list = [], []

    for it in range(len(val_star_list)):
        # For each ccd
        test_stars = val_star_list[it]
        # Galsim's HSM bad pixel convetion thinks 0 means good
        badpix_mask = np.rint(np.abs(val_mask_list[it] - 1))
        matched_psfs = PSF_list[it]

        # Stars
        star_moms = [gs.hsm.FindAdaptiveMom(
            gs.Image(star), badpix=gs.Image(bp), strict=False)
            for star, bp in
            zip(utils.reg_format(test_stars), utils.reg_format(badpix_mask))]
        star_shapes = np.array([[moms.observed_shape.g1,
                                 moms.observed_shape.g2,
                                 moms.moments_sigma,
                                 int(bool(moms.error_message))]
                                for moms in star_moms])
        # PSFs
        psf_moms = [gs.hsm.FindAdaptiveMom(gs.Image(psf), strict=False)
                    for psf in utils.reg_format(matched_psfs)]
        psf_shapes = np.array([[moms.observed_shape.g1,
                                moms.observed_shape.g2,
                                moms.moments_sigma,
                                int(bool(moms.error_message))]
                               for moms in psf_moms])

        star_shapes_list.append(star_shapes)
        psf_shapes_list.append(psf_shapes)
        psf_vignet_list.append(utils.reg_format(matched_psfs))
        pos_list.append(val_pos_list[it])
        star_list.append(utils.reg_format(val_star_list[it]))
        mask_list.append(utils.reg_format(val_mask_list[it]))
        ccd_list.append(val_ccd_list_to_save[it])
        if val_RA_list is not None:
            RA_list.append(val_RA_list[it])
            DEC_list.append(val_DEC_list[it])

    # Prepare the PSF list and the moments in an array form
    # To be able to save them in the fits format
    psf_shapes = np.concatenate(psf_shapes_list, axis=0)
    star_shapes = np.concatenate(star_shapes_list, axis=0)
    psf_vignets = np.concatenate(psf_vignet_list, axis=0)
    pos_ordered = np.concatenate(pos_list, axis=0)
    star_ordered = np.concatenate(star_list, axis=0)
    mask_ordered = np.concatenate(mask_list, axis=0)
    ccd_ordered = np.concatenate(ccd_list, axis=0)
    if val_RA_list is not None:
        RA_ordered = np.concatenate(RA_list, axis=0)
        DEC_ordered = np.concatenate(DEC_list, axis=0)

    # Save the results and the psfs
    star_dict['PSF_VIGNET_LIST'] = np.copy(psf_vignets)
    star_dict['PSF_MOM_LIST'] = np.copy(psf_shapes)
    star_dict['STAR_MOM_LIST'] = np.copy(star_shapes)
    star_dict['GLOB_POSITION_IMG_LIST'] = np.copy(pos_ordered)
    star_dict['VIGNET_LIST'] = np.copy(star_ordered)
    star_dict['MASK_LIST'] = np.copy(mask_ordered)
    star_dict['CCD_ID_LIST'] = np.copy(ccd_ordered)
    if val_RA_list is not None:
        star_dict['RA_LIST'] = np.copy(RA_ordered)
        star_dict['DEC_LIST'] = np.copy(DEC_ordered)

    # Memory management
    del mccd_model
    gc.collect()

    return star_dict


def mccd_preprocessing(input_folder_path, output_path, min_n_stars=20,
                       file_pattern='sexcat-*-*.fits',
                       separator='-',
                       CCD_id_filter_list=None,
                       outlier_std_max=100.,
                       save_masks=True,
                       save_name='train_star_selection',
                       save_extension='.fits', verbose=True):
    r"""Preprocess input catalog.

    Parameters
    ----------
    input_folder_path: str
        Path to the folder containing the files to preprocess.
    output_path: str
        Path to the folder where to save the preprocessed files.
    min_n_stars: int
        Minimum number of stars in order to preprocess the CCD.
        Default is ``20``.
    file_pattern: str
        Input file pattern as a regex expression. Only the files matching
        the ``file_pattern`` in ``input_folder_path`` will be treated.
        Default is ``sexcat-*-*.fits``. Where the first `*` corresponds to
        the catalog_id, the second `*` corresponds to the CCD id and they are
        separated by the ``separator``.
    separator: str
        Separator string that separates the catalog id and the CCD id.
        Default is ``'-'``.
    CCD_id_filter_list: list of int or None
        A list that correspond to the CCDs that should be preprocessed.
        If it is None, all the CCDs are preprocessed.
        (Current version: Hardcoded for the MegaCam scenario).
        Default is ``None``.
    outlier_std_max: float
        Parameter regulating the shape outlier removal. Default is very high
        so as it is not done at all. A decent number would be ``10``.
        Default is ``100.``.
    save_masks: bool
        If masks should be saved in the new file.
        Default is ``True``.
    save_name: str
        Name to save the preprocessed file.
        Default is ``'train_star_selection'``.
    save_extension: str
        Extension of the saved file.
        Default is ``.fits``.
    verbose: bool
        Verbose mode.
        Default is ``True``.

    Returns
    -------
    mccd_inputs: class
        An instance of ``MccdInputs`` class used for the input preprocessing.

    """
    mccd_star_nb = 0

    if CCD_id_filter_list is None:
        CCD_id_filter_list = np.arange(40)
    else:
        CCD_id_filter_list = np.array(CCD_id_filter_list)

    if verbose:
        def print_fun(x):
            print(x)
    else:
        def print_fun(x):
            pass

    # Preprocess
    mccd_inputs = mccd_utils.MccdInputs(separator=separator)
    print_fun('Processing dataset..')
    catalog_ids = mccd_inputs.preprocess_data(folder_path=input_folder_path,
                                              pattern=file_pattern)

    # Loop over the catalogs
    for it in range(catalog_ids.shape[0]):
        # For each observation position
        catalog_id = catalog_ids[it]
        star_list, pos_list, mask_list, ccd_list, SNR_list, RA_list, \
            DEC_list = mccd_inputs.get_inputs(catalog_id)

        star_list, pos_list, mask_list, ccd_list, SNR_list, RA_list, \
            DEC_list, _ = mccd_inputs.outlier_rejection(
                star_list, pos_list, mask_list, ccd_list, SNR_list, RA_list,
                DEC_list, shape_std_max=outlier_std_max, print_fun=print_fun)

        mccd_star_list = []
        mccd_pos_list = []
        mccd_mask_list = []
        mccd_ccd_list = []
        mccd_SNR_list = []
        mccd_RA_list = []
        mccd_DEC_list = []

        for j in range(len(star_list)):
            # For each CCD
            if ccd_list[j] in CCD_id_filter_list:
                try:
                    n_stars = star_list[j].shape[2]

                    if n_stars >= min_n_stars:
                        mccd_star_list.append(star_list[j])
                        mccd_pos_list.append(pos_list[j])
                        mccd_mask_list.append(mask_list[j])
                        mccd_ccd_list.append(ccd_list[j] * np.ones(n_stars))
                        if SNR_list is not None:
                            mccd_SNR_list.append(SNR_list[j])
                        if RA_list is not None:
                            mccd_RA_list.append(RA_list[j])
                            mccd_DEC_list.append(DEC_list[j])
                    else:
                        msg = '''Not enough stars in catalog_id %s
                            ,ccd %d. Total stars = %d''' % (
                            catalog_id, ccd_list[j], n_stars)
                        print_fun(msg)

                except Exception:
                    msg = '''Warning! Problem detected in
                        catalog_id %s ,ccd %d''' % (catalog_id, ccd_list[j])
                    print_fun(msg)

        if mccd_pos_list:
            # If the list is not empty
            # Concatenate, as fits can't handle list of numpy arrays and
            # turn into reg format
            mccd_stars = utils.reg_format(
                np.concatenate(mccd_star_list, axis=2))
            mccd_poss = np.concatenate(mccd_pos_list, axis=0)
            mccd_ccds = np.concatenate(mccd_ccd_list, axis=0)

            if save_masks is True:
                mccd_masks = utils.reg_format(
                    np.concatenate(mccd_mask_list, axis=2))
            else:
                # Send an array of False (None cannot be used in .fits)
                mccd_masks = np.zeros((mccd_poss.shape[0]), dtype=bool)

            if SNR_list is not None:
                mccd_SNRs = np.concatenate(mccd_SNR_list, axis=0)
            else:
                # Send an array of False (None cannot be used in .fits)
                mccd_SNRs = np.zeros((mccd_poss.shape[0]), dtype=bool)

            if RA_list is not None:
                mccd_RAs = np.concatenate(mccd_RA_list)
                mccd_DECs = np.concatenate(mccd_DEC_list)
            else:
                mccd_RAs = np.zeros((mccd_poss.shape[0]), dtype=bool)
                mccd_DECs = np.zeros((mccd_poss.shape[0]), dtype=bool)

            mccd_star_nb += mccd_stars.shape[0]

            # Save the fits file
            train_dic = {'VIGNET_LIST': mccd_stars,
                         'GLOB_POSITION_IMG_LIST': mccd_poss,
                         'MASK_LIST': mccd_masks, 'CCD_ID_LIST': mccd_ccds,
                         'SNR_WIN_LIST': mccd_SNRs,
                         'RA_LIST': mccd_RAs, 'DEC_LIST': mccd_DECs}

            saving_path = output_path + save_name + separator \
                + catalog_id + save_extension
            mccd_utils.save_to_fits(train_dic, saving_path)

    print_fun('Finished the training dataset processing.')
    print_fun('Total stars processed = %d' % mccd_star_nb)

    return mccd_inputs


class MCCDParamsParser(object):
    r"""Parse MCCD config file.

    Set up a parser for the MCCD parameters.

    Parameters
    ----------
    file_path: str
        Path to the config file.

    Raises
    ------
    IOError
        For non existent configuration file.

    """

    def __init__(self, file_path):
        r"""Initialize class."""
        if not os.path.exists(file_path):
            raise IOError('Configuration file {} does not exist.'.format(
                file_path))

        self.file_name = file_path
        self.config = ConfigParser()

        self.processed_inputs = False
        self.mccd_inst_kw = None
        self.mccd_fit_kw = None
        self.mccd_inputs_kw = None
        self.mccd_val_kw = None
        self.mccd_val_prepro_kw = None

        self.mccd_extra_kw = {}

    def _set_inputs_options(self):
        """Set Input Options.

        This method checks the ``INPUTS`` options in the configuration file.

        Raises
        ------
        RuntimeError
            For no input directory specified
        OSError
            For non-existent input directory
        RuntimeError
            For no output directory specified
        OSError
            For non-existent output directory

        """
        if not self.config.has_option('INPUTS', 'OUTPUT_DIR'):
            raise RuntimeError('Not output directory specified.')
        elif not os.path.isdir(self.config['INPUTS']['OUTPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('INPUTS', 'OUTPUT_DIR')))

        if not self.config.has_option('INPUTS', 'PREPROCESSED_OUTPUT_DIR'):
            raise RuntimeError('Not preprocessed output directory specified.')
        elif not os.path.isdir(
                self.config['INPUTS']['PREPROCESSED_OUTPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('INPUTS', 'PREPROCESSED_OUTPUT_DIR')))

        if not self.config.has_option('INPUTS', 'INPUT_DIR'):
            raise RuntimeError('Not output directory specified.')
        elif not os.path.isdir(self.config['INPUTS']['INPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('INPUTS', 'INPUT_DIR')))

        if not self.config.has_option('INPUTS', 'INPUT_REGEX_FILE_PATTERN'):
            self.config.set('INPUTS', 'INPUT_REGEX_FILE_PATTERN',
                            'sexcat-*-*.fits')
        if not self.config.has_option('INPUTS', 'INPUT_SEPARATOR'):
            self.config.set('INPUTS', 'INPUT_SEPARATOR', '-')
        if not self.config.has_option('INPUTS', 'MIN_N_STARS'):
            self.config.set('INPUTS', 'MIN_N_STARS', '20')
        if not self.config.has_option('INPUTS', 'OUTLIER_STD_MAX'):
            self.config.set('INPUTS', 'OUTLIER_STD_MAX', '100.')
        if not self.config.has_option('INPUTS', 'USE_SNR_WEIGHTS'):
            self.config.set('INPUTS', 'USE_SNR_WEIGHTS', 'False')

    def _set_instance_options(self):
        """Set Instance Options.

        This method checks the ``INSTANCE`` options in the configuration file.

        """
        if not self.config.has_option('INSTANCE', 'N_COMP_LOC'):
            self.config.set('INSTANCE', 'N_COMP_LOC', '8')

        if not self.config.has_option('INSTANCE', 'D_COMP_GLOB'):
            self.config.set('INSTANCE', 'D_COMP_GLOB', '3')

        if not self.config.has_option('INSTANCE', 'KSIG_LOC'):
            self.config.set('INSTANCE', 'KSIG_LOC', '1.0')

        if not self.config.has_option('INSTANCE', 'KSIG_GLOB'):
            self.config.set('INSTANCE', 'KSIG_GLOB', '1.0')

        if not self.config.has_option('INSTANCE', 'FILTER_PATH'):
            self.config.set('INSTANCE', 'FILTER_PATH', 'None')

    def _set_fit_options(self):
        """Set Fit Options.

        This method checks the ``FIT`` options in the configuration file.

        """
        if not self.config.has_option('FIT', 'PSF_SIZE'):
            self.config.set('FIT', 'PSF_SIZE', '6.15')

        if not self.config.has_option('FIT', 'PSF_SIZE_TYPE'):
            self.config.set('FIT', 'PSF_SIZE_TYPE', 'R2')

        if not self.config.has_option('FIT', 'N_EIGENVECTS'):
            self.config.set('FIT', 'N_EIGENVECTS', '5')

        if not self.config.has_option('FIT', 'N_ITER_RCA'):
            self.config.set('FIT', 'N_ITER_RCA', '1')

        if not self.config.has_option('FIT', 'N_ITER_GLOB'):
            self.config.set('FIT', 'N_ITER_GLOB', '2')

        if not self.config.has_option('FIT', 'N_ITER_LOC'):
            self.config.set('FIT', 'N_ITER_LOC', '2')

        if not self.config.has_option('FIT', 'NB_SUBITER_S_LOC'):
            self.config.set('FIT', 'NB_SUBITER_S_LOC', '100')

        if not self.config.has_option('FIT', 'NB_SUBITER_A_LOC'):
            self.config.set('FIT', 'NB_SUBITER_A_LOC', '500')

        if not self.config.has_option('FIT', 'NB_SUBITER_S_GLOB'):
            self.config.set('FIT', 'NB_SUBITER_S_GLOB', '30')

        if not self.config.has_option('FIT', 'NB_SUBITER_A_GLOB'):
            self.config.set('FIT', 'NB_SUBITER_A_GLOB', '200')

        if not self.config.has_option('FIT', 'LOC_MODEL'):
            self.config.set('FIT', 'LOC_MODEL', 'hybrid')

    def _set_val_options(self):
        """Set Validation Options.

        Method to check the ``VALIDATION`` options in the configuration file.

        Raises
        ------
        RuntimeError
            For no input directory specified
        OSError
            For non-existent input directory
        RuntimeError
            For no output directory specified
        OSError
            For non-existent output directory

        """
        if not self.config.has_option('VALIDATION', 'VAL_MODEL_INPUT_DIR'):
            raise RuntimeError('Not input model directory specified.')
        elif not os.path.isdir(self.config['VALIDATION'][
                                   'VAL_MODEL_INPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('VALIDATION', 'VAL_MODEL_INPUT_DIR')))

        if not self.config.has_option('VALIDATION', 'VAL_DATA_INPUT_DIR'):
            raise RuntimeError('Not input dataset directory specified.')
        elif not os.path.isdir(
                self.config['VALIDATION']['VAL_DATA_INPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('VALIDATION', 'VAL_DATA_INPUT_DIR')))

        if not self.config.has_option('VALIDATION', 'VAL_OUTPUT_DIR'):
            raise RuntimeError('Not validation output directory specified.')
        elif not os.path.isdir(self.config['VALIDATION']['VAL_OUTPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('VALIDATION', 'VAL_OUTPUT_DIR')))

        if not self.config.has_option('VALIDATION',
                                      'VAL_PREPROCESSED_OUTPUT_DIR'):
            raise RuntimeError('''Not validation preprocessing output
            directory specified.''')
        elif not os.path.isdir(self.config['VALIDATION'][
                                   'VAL_PREPROCESSED_OUTPUT_DIR']):
            raise OSError('Directory {} not found.'.format(
                self.config.has_option('VALIDATION',
                                       'VAL_PREPROCESSED_OUTPUT_DIR')))

        if not self.config.has_option('INPUTS', 'VAL_REGEX_FILE_PATTERN'):
            self.config.set('INPUTS', 'VAL_REGEX_FILE_PATTERN',
                            'test-star_selection-*-*.fits')

        if not self.config.has_option('INPUTS', 'VAL_SEPARATOR'):
            self.config.set('INPUTS', 'VAL_SEPARATOR', '-')

        if not self.config.has_option('INPUTS', 'APPLY_DEGRADATION'):
            self.config.set('INPUTS', 'APPLY_DEGRADATION', 'True')

        if not self.config.has_option('INPUTS', 'MCCD_DEBUG'):
            self.config.set('INPUTS', 'MCCD_DEBUG', 'False')

        if not self.config.has_option('INPUTS', 'GLOBAL_POL_INTERP'):
            self.config.set('INPUTS', 'GLOBAL_POL_INTERP', 'False')

    def parse_document(self):
        r"""Parse config file."""
        if not self.processed_inputs:
            self.config.read(self.file_name)
            self._set_inputs_options()
            self._set_instance_options()
            self._set_fit_options()
            self._set_val_options()
            self.processed_inputs = True

    def _build_instance_kw(self):
        r"""Build ``INSTANCE`` parameter dictionary."""
        if not self.processed_inputs:
            self.parse_document()

        if self.mccd_inst_kw is None:
            n_comp_loc = int(self.config['INSTANCE'].get('N_COMP_LOC'))
            d_comp_glob = int(self.config['INSTANCE'].get('D_COMP_GLOB'))
            ksig_loc = float(self.config['INSTANCE'].get('KSIG_LOC'))
            ksig_glob = float(self.config['INSTANCE'].get('KSIG_GLOB'))
            if self.config['INSTANCE'].get('FILTER_PATH') == 'None':
                filters = None
            else:
                filters = self.config['INSTANCE'].get('FILTER_PATH')

            # Build the parameter dictionaries
            self.mccd_inst_kw = {'n_comp_loc': n_comp_loc,
                                 'd_comp_glob': d_comp_glob,
                                 'filters': filters, 'ksig_loc': ksig_loc,
                                 'ksig_glob': ksig_glob}

    def _build_fit_kw(self):
        r"""Build ``FIT`` parameter dictionary."""
        if not self.processed_inputs:
            self.parse_document()

        if self.mccd_fit_kw is None:
            psf_size = float(self.config['FIT'].get('PSF_SIZE'))
            psf_size_type = self.config['FIT'].get('PSF_SIZE_TYPE')
            n_eigenvects = int(self.config['FIT'].get('N_EIGENVECTS'))
            n_iter_rca = int(self.config['FIT'].get('N_ITER_RCA'))
            nb_iter_glob = int(self.config['FIT'].get('N_ITER_GLOB'))
            nb_iter_loc = int(self.config['FIT'].get('N_ITER_LOC'))
            nb_subit_S_loc = int(self.config['FIT'].get('NB_SUBITER_S_LOC'))
            nb_subit_A_loc = int(self.config['FIT'].get('NB_SUBITER_A_LOC'))
            nb_subit_S_glob = int(self.config['FIT'].get('NB_SUBITER_S_GLOB'))
            nb_subit_A_glob = int(self.config['FIT'].get('NB_SUBITER_A_GLOB'))
            loc_model = self.config['FIT'].get('LOC_MODEL')

            # Build the parameter dictionaries
            self.mccd_fit_kw = {'psf_size': psf_size,
                                'psf_size_type': psf_size_type,
                                'n_eigenvects': n_eigenvects,
                                'nb_iter': n_iter_rca,
                                'nb_iter_glob': nb_iter_glob,
                                'nb_iter_loc': nb_iter_loc,
                                'nb_subiter_S_loc': nb_subit_S_loc,
                                'nb_subiter_A_loc': nb_subit_A_loc,
                                'nb_subiter_S_glob': nb_subit_S_glob,
                                'nb_subiter_A_glob': nb_subit_A_glob,
                                'loc_model': loc_model}

    def _build_inputs_kw(self):
        r"""Build ``INPUTS`` parameter dictionary."""
        if not self.processed_inputs:
            self.parse_document()

        if self.mccd_inputs_kw is None:
            input_folder_path = self.config['INPUTS'].get('INPUT_DIR')
            output_path = self.config['INPUTS'].get('PREPROCESSED_OUTPUT_DIR')
            min_n_stars = int(self.config['INPUTS'].get('MIN_N_STARS'))
            file_pattern = self.config['INPUTS'].get(
                'INPUT_REGEX_FILE_PATTERN')
            separator = self.config['INPUTS'].get('INPUT_SEPARATOR')
            outlier_std_max = float(self.config['INPUTS'].get(
                'OUTLIER_STD_MAX'))
            if self.config['INPUTS'].get('USE_SNR_WEIGHTS') == 'True':
                use_SNR_weight = True
            elif self.config['INPUTS'].get('USE_SNR_WEIGHTS') == 'False':
                use_SNR_weight = False
            else:
                raise RuntimeError('USE_SNR_WEIGHTS should be True or False.')

            self.mccd_inputs_kw = {'input_folder_path': input_folder_path,
                                   'output_path': output_path,
                                   'min_n_stars': min_n_stars,
                                   'file_pattern': file_pattern,
                                   'separator': separator,
                                   'outlier_std_max': outlier_std_max,
                                   'save_name': 'train_star_selection',
                                   'save_extension': '.fits'}

            self.mccd_extra_kw['use_SNR_weight'] = use_SNR_weight
            self.mccd_extra_kw['output_dir'] = self.config['INPUTS'].get(
                'OUTPUT_DIR')

    def _build_val_kw(self):
        r"""Build ``VALIDATION`` parameter dictionary."""
        if not self.processed_inputs:
            self.parse_document()

        if self.mccd_val_kw is None:
            if self.config['VALIDATION'].get('APPLY_DEGRADATION') == 'True':
                apply_degradation = True
            elif self.config['VALIDATION'].get('APPLY_DEGRADATION') == 'False':
                apply_degradation = False
            else:
                raise RuntimeError('APPLY_DEGRADATION must be True or False.')

            if self.config['VALIDATION'].get('MCCD_DEBUG') == 'True':
                mccd_debug = True
            elif self.config['VALIDATION'].get('MCCD_DEBUG') == 'False':
                mccd_debug = False
            else:
                raise RuntimeError('MCCD_DEBUG must be True or False.')

            if self.config['VALIDATION'].get('GLOBAL_POL_INTERP') == 'True':
                global_pol_interp = True
            elif self.config['VALIDATION'].get('GLOBAL_POL_INTERP') == 'False':
                global_pol_interp = False
            else:
                raise RuntimeError('GLOBAL_POL_INTERP must be True or False.')

            # Build the parameter dictionaries
            self.mccd_val_kw = {'apply_degradation': apply_degradation,
                                'mccd_debug': mccd_debug,
                                'global_pol_interp': global_pol_interp}

            val_input_folder_path = self.config['VALIDATION'].get(
                'VAL_DATA_INPUT_DIR')
            val_output_path = self.config['VALIDATION'].get(
                'VAL_PREPROCESSED_OUTPUT_DIR')
            val_file_pattern = self.config['VALIDATION'].get(
                'VAL_REGEX_FILE_PATTERN')
            val_separator = self.config['VALIDATION'].get('VAL_SEPARATOR')
            outlier_std_max = float(self.config['INPUTS'].get(
                'OUTLIER_STD_MAX'))

            # Build the preprocessing validatoin parameter dictionaries
            self.mccd_val_prepro_kw = {
                'input_folder_path': val_input_folder_path,
                'output_path': val_output_path,
                'min_n_stars': 1,
                'file_pattern': val_file_pattern,
                'separator': val_separator,
                'outlier_std_max': outlier_std_max,
                'save_name': 'test_star_selection',
                'save_extension': '.fits'}

            self.mccd_extra_kw['val_model_input_dir'] = self.config[
                'VALIDATION'].get('VAL_MODEL_INPUT_DIR')
            self.mccd_extra_kw['val_model_input_dir'] = self.config[
                'VALIDATION'].get('VAL_OUTPUT_DIR')

    def get_extra_kw(self, param_name):
        r"""Get parameter from extra arguments.

        Returns
        -------
        param_name: str
            Name of the parameter
        """
        if self.mccd_inputs_kw is None:
            self._build_inputs_kw()
        if self.mccd_val_prepro_kw is None:
            self._build_val_kw()

        return self.mccd_extra_kw[param_name]

    def get_fit_kw(self):
        r"""Get fit parameter dictionary.

        Returns
        -------
        mccd_fit_kw: dict
            MCCD fit parameter dictionary.
        """
        if self.mccd_fit_kw is None:
            self._build_fit_kw()

        return self.mccd_fit_kw

    def get_instance_kw(self):
        r"""Get instace parameter dictionary.

        Returns
        -------
        mccd_inst_kw: dict
            MCCD instance parameter dictionary.
        """
        if self.mccd_inst_kw is None:
            self._build_instance_kw()

        return self.mccd_inst_kw

    def get_inputs_kw(self):
        r"""Get paths parameter dictionary.

        Returns
        -------
        mccd_inputs_kw: dict
             MCCD input parameter dictionary.
        """
        if self.mccd_inputs_kw is None:
            self._build_inputs_kw()

        return self.mccd_inputs_kw

    def get_val_prepro_kw(self):
        r"""Get preprocessing validation input dictionary.

        Returns
        -------
        mccd_val_prepro_kw: dict
             MCCD preprocessing validation input dictionary.
        """
        if self.mccd_val_prepro_kw is None:
            self._build_val_kw()

        return self.mccd_val_prepro_kw

    def get_val_kw(self):
        r"""Get validaiton parameter dictionary.

        Returns
        -------
        mccd_val_kw: dict
             MCCD validation dictionary.
        """
        if self.mccd_val_kw is None:
            self._build_val_kw()

        return self.mccd_val_kw


class RunMCCD(object):
    r"""Run the MCCD method.

    This class allows to run the MCCD method using the paramters present on
    the configuration file. Saves the fitted model on the output directory
    found in the MCCD configuration file.

    Parameters
    ----------
    config_file_path: str
        Path to the configuration file.
    fits_table_pos: int
        Position of the Table in the fits file.
        Default is ``1``.
    verbose: bool
        Verbose mode.
        Default is ``True``.

    Notes
    -----
    Missing:
    - Method including the validation.
    - Erase the preprocessed files (?)

    """

    def __init__(self, config_file_path, fits_table_pos=1, verbose=True):
        r"""Initialize class."""
        self.config_file_path = config_file_path

        self.param_parser = None
        self.mccd_inputs_kw = None
        self.mccd_inst_kw = None
        self.mccd_fit_kw = None
        self.mccd_val_prepro_kw = None
        self.mccd_val_kw = None

        self.val_mccd_inputs = None
        self.mccd_inputs = None
        self.val_catalog_ids = None
        self.catalog_ids = None

        self.preprocess_name = 'train_star_selection'
        self.file_extension = '.fits'
        self.fits_table_pos = fits_table_pos
        self.separator = None

        self.fitting_model_saving_name = 'fitted_model'

        self.val_saving_name = 'validation_psf'

        self.use_SNR_weight = None
        self.fit_output_dir = None

        self.parsed_parameters = False

        self.verbose = verbose

    def parse_config_file(self):
        r"""Parse configuration file and recover parameters."""
        if not self.parsed_parameters:
            self.param_parser = MCCDParamsParser(self.config_file_path)
            self.mccd_inputs_kw = self.param_parser.get_inputs_kw()
            self.mccd_inst_kw = self.param_parser.get_instance_kw()
            self.mccd_fit_kw = self.param_parser.get_fit_kw()
            self.mccd_val_prepro_kw = self.param_parser.get_val_prepro_kw()
            self.mccd_val_kw = self.param_parser.get_val_kw()

            self.separator = self.mccd_inputs_kw['separator']
            self.use_SNR_weight = self.param_parser.get_extra_kw(
                'use_SNR_weight')
            self.fit_output_dir = self.param_parser.get_extra_kw('output_dir')

            self.parsed_parameters = True

    def preprocess_inputs(self):
        r"""Preprocess the input data."""
        if not self.parsed_parameters:
            self.parse_config_file()
        self.mccd_inputs = mccd_preprocessing(**self.mccd_inputs_kw)
        self.catalog_ids = self.mccd_inputs.get_catalog_ids()

    def preprocess_val_inputs(self):
        r"""Preprocess validation input data."""
        if not self.parsed_parameters:
            self.parse_config_file()
        self.val_mccd_inputs = mccd_preprocessing(**self.mccd_val_prepro_kw)
        self.val_catalog_ids = self.val_mccd_inputs.get_catalog_ids()

    def fit_models(self):
        r"""Build and save the models to the catalgos found."""
        if not self.parsed_parameters:
            self.parse_config_file()
        if self.mccd_inputs is None:
            self.preprocess_inputs()

        input_dir = self.mccd_inputs_kw['output_path']
        output_dir = self.fit_output_dir

        for _cat_id in self.catalog_ids:
            if not isinstance(_cat_id, str):
                cat_id = '%07d' % _cat_id
            else:
                cat_id = _cat_id

            input_path = input_dir + self.preprocess_name + self.separator \
                + cat_id + self.file_extension

            if os.path.isfile(input_path):
                starcat = fits.open(input_path)[self.fits_table_pos]
            else:
                raise OSError('File {} not found.'.format(input_path))

            mccd_fit(starcat,
                     self.mccd_inst_kw,
                     self.mccd_fit_kw,
                     output_dir=output_dir,
                     catalog_id=int(cat_id),
                     sex_thresh=-1e5,
                     use_SNR_weight=self.use_SNR_weight,
                     verbose=self.verbose,
                     saving_name=self.fitting_model_saving_name +
                     self.separator)

    def validate_models(self):
        r"""Validate MCCD models."""
        if not self.parsed_parameters:
            self.parse_config_file()
        if self.val_mccd_inputs is None:
            self.preprocess_val_inputs()

        # Preprocessed validation dir
        input_dir = self.mccd_val_prepro_kw['output_path']
        # Fit model input dir
        fit_model_input_dir = self.param_parser.get_extra_kw(
            'val_model_input_dir')
        # Validation output dir
        val_output_dir = self.param_parser.get_extra_kw('val_model_input_dir')

        for _cat_id in self.val_catalog_ids:
            if not isinstance(_cat_id, str):
                cat_id = '%07d' % _cat_id
            else:
                cat_id = _cat_id

            if self.verbose:
                print('Validating catalog %s..' % cat_id)

            # Check if there is the fitted model
            fit_model_path = fit_model_input_dir + \
                self.fitting_model_saving_name + self.separator + \
                cat_id + '.npy'

            if os.path.isfile(fit_model_path):
                prepro_name = self.mccd_val_prepro_kw['save_name']
                separator = self.mccd_val_prepro_kw['separator']
                save_extension = self.mccd_val_prepro_kw['save_extension']
                input_val_path = input_dir + prepro_name + separator + \
                    cat_id + save_extension

                testcat = fits.open(input_val_path)[self.fits_table_pos]

                val_dict = mccd_validation(fit_model_path,
                                           testcat,
                                           **self.mccd_val_kw,
                                           sex_thresh=-1e5)

                saving_path = val_output_dir + self.val_saving_name + \
                    separator + cat_id + save_extension
                # Save validation dictionary to fits file
                mccd_utils.save_to_fits(val_dict, saving_path)
                if self.verbose:
                    print('Validation catalog < %s > saved.' % (
                            self.val_saving_name + separator + cat_id +
                            save_extension))

            else:
                print('''Fitted model corresponding to catalog %d was not
                    found.''' % cat_id)

    def fit_MCCD_models(self):
        r"""Fit MCCD models."""
        self.parse_config_file()
        self.preprocess_inputs()
        self.fit_models()

    def validate_MCCD_models(self):
        r"""Validate MCCD models."""
        self.parse_config_file()
        self.preprocess_val_inputs()
        self.validate_models()

    def run_MCCD(self):
        r"""Run the MCCD routines."""
        self.parse_config_file()
        self.preprocess_inputs()
        self.fit_models()
        self.preprocess_val_inputs()
        self.validate_models()

    @staticmethod
    def recover_MCCD_PSFs(mccd_model_path, positions, ccd_id, local_pos=False):
        r"""Recover MCCD PSFs at required positions.

        Parameters
        ----------
        mccd_model_path: str
            Path pointing to the saved fitted MCCD model to be used.
        positions: numpy.ndarray
            Array containing the positions where the PSF should be recovered.
            The shape of the array should be (n,2) [x,y].
        ccd_id: int
            Id of the CCD from where the positions where taken.
        local_pos: bool
            If the positions passed are local to the CCD. If False, the
            positions are considered to be in the same format
            (coordinate system, units, etc.) as the ``obs_pos`` fed
            to :func:`MCCD.fit`.
            Default is ``False``.

        Returns
        -------
        rec_PSFs: numpy.ndarray
            Array containing the recovered PSFs.
            Array dimensions: (n_psf, n_im, n_im).

        Raises
        ------
        OSError
            For non-existent fitted model.
        ValueError
            For ccd_id not being an integer.

        """
        if not os.path.isfile(mccd_model_path):
            raise OSError('Fitted model {} not found.'.format(mccd_model_path))

        if not isinstance(ccd_id, int):
            raise ValueError('Parameter ccd_id should be an integer.')

        if local_pos:
            loc2glob = mccd_utils.Loc2Glob()
            glob_pos = np.array([
                loc2glob.loc2glob_img_coord(ccd_id, _pos[0], _pos[1])
                for _pos in positions])
        else:
            glob_pos = positions

        # Import the model
        mccd_model = mccd.mccd_quickload(mccd_model_path)

        rec_PSFs = mccd_model.estimate_psf(glob_pos, ccd_id)

        return rec_PSFs
