# -*- coding: utf-8 -*-

r"""AUXILIARY FUNCTIONS.

These functions are needed to run the tests.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import mccd
import mccd.mccd_utils as mccd_utils
import mccd.utils as utils
import scipy as sp
import galsim as gs
import gc


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
             catalog_id=1234567, sex_thresh=-1e5, use_SNR_weight=False):
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
    mccd_instance = mccd.MCCD(**mccd_inst_kw, verbose=False)
    # Launch the training
    _, _, _, _, _ = mccd_instance.fit(
        star_list, pos_list, ccd_list, mask_list,
        SNR_weight_list, **mccd_fit_kw)

    fitted_model_path = output_dir + '/fitted_model' + \
        str(catalog_id.astype(int))
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
