# -*- coding: utf-8 -*-

r""" AUXILIARY FUNCTIONS

These functions are needed to run the tests.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>

"""

import numpy as np
import mccd.mccd_utils as mccd_utils
import scipy as sp
import galsim


class GenerateSimDataset(object):
    r""" Class to generate simulated dataset for training and validating PSF models.

    Parameters
    ----------
    input_pos_path: str
        Path to the global positions of the PSF that will be used for the training.
    input_ccd_path: str
        Path to the corresponding CCDs of the global positions.
    output_path: str
        Path to the folder to save the simulated datasets.
    e1_analytic_fun: function
        The analytic e1 ellipticity function that will define an ellipticity e1 for each position in the focal plane.
    e2_analytic_fun: function
        The analytic e2 ellipticity function that will define an ellipticity e2 for each position in the focal plane.

    Notes
    -----
    The simulated PSFs are based on the Moffat profile and we are using Galsim to generate them.
    We base ourselves on two analytic functions that have to output an ellipticity for each position in the
    focal plane.

    """

    def __init__(self, input_pos_path, input_ccd_path, output_path, e1_analytic_fun=None, e2_analytic_fun=None):
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
        try:
            self.positions = np.load(self.input_pos_path, allow_pickle=True)
            self.ccd_id = np.load(self.input_ccd_path, allow_pickle=True)
        except FileNotFoundError:
            print('The positions or ccd path was not found. Check the paths.')
            raise FileNotFoundError

    def generate_train_data(self, sigma=1.6, image_size=51, psf_flux=1., beta_psf=4.8, pix_scale=0.187,
                            desired_SNR=30, catalog_id=2086592):
        r""" Generate the training dataset and saves it in fits format.

        Parameters
        ----------
        sigma: float
            Size of the PSF in sigma's. (Sigma from Galsim's HSM adaptive moments).
            Default is 1.6
        image_size: int
            Dimension of the squared image stamp. (image_size x image_size)
            Default is 51
        psf_flux: float
            Total PSF photometric flux.
            Default is 1.
        beta_psf: float
            Moffat beta parameter.
            Default is 4.8
        pix_scale: float
            Pixel scale.
            Default is 0.187
        desired_SNR: float
            Desired SNR
            Default is 30
        catalog_id: int
            Catalog identifier number.
            Default is 2086592

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
        e1s = np.array([self.e1_catalog_fun(position[0], position[1]) for position in self.positions])
        e2s = np.array([self.e2_catalog_fun(position[0], position[1]) for position in self.positions])

        # Define the constant shape of the stars (before the shearing)
        fwhm_psf = (2 * np.sqrt(2 * np.log(2))) * sigma
        fwhms = np.ones(e1s.shape) * fwhm_psf  # Arround 5. and 6. in sigma

        # Generate the vignets
        new_vignets = np.zeros((e1s.shape[0], image_size, image_size))
        new_e1_HSM = np.zeros(e1s.shape)
        new_e2_HSM = np.zeros(e1s.shape)
        new_R2_HSM = np.zeros(e1s.shape)
        for it in range(e1s.shape[0]):
            # PSF generation. Define size
            psf = galsim.Moffat(fwhm=fwhms[it] * pix_scale, beta=beta_psf)

            # Define the Flux
            psf = psf.withFlux(psf_flux)
            # Define the shear
            psf = psf.shear(g1=e1s[it], g2=e2s[it])
            # Draw the PSF on a vignet
            noisy_image_epsf = galsim.ImageF(image_size, image_size)
            # Define intrapixel shift (uniform distribution in [-0.5,0.5])
            rand_shift = np.random.rand(2) - 0.5
            psf.drawImage(image=noisy_image_epsf, offset=rand_shift, scale=pix_scale)

            sigma_noise = np.sqrt((np.sum(noisy_image_epsf.array ** 2)) / (desired_SNR * image_size ** 2))
            # Generate Gaussian noise for the PSF
            gaussian_noise = galsim.GaussianNoise(sigma=sigma_noise)

            # Before adding the noise, we measure the ellipticity components
            my_moments = galsim.hsm.FindAdaptiveMom(noisy_image_epsf)
            new_e1_HSM[it] = my_moments.observed_shape.g1
            new_e2_HSM[it] = my_moments.observed_shape.g2
            new_R2_HSM[it] = my_moments.moments_sigma

            # Add Gaussian noise to the PSF
            noisy_image_epsf.addNoise(gaussian_noise)

            new_vignets[it, :, :] = noisy_image_epsf.array

        new_masks = self.handle_SExtractor_mask(new_vignets, thresh=-1e5)

        # Build the dictionary
        train_dic = {'VIGNET_LIST': new_vignets, 'GLOB_POSITION_IMG_LIST': self.positions,
                     'MASK_LIST': new_masks, 'CCD_ID_LIST': self.ccd_id,
                     'TRUE_E1_HSM': new_e1_HSM, 'TRUE_E2_HSM': new_e2_HSM, 'TRUE_R2_HSM': new_R2_HSM}

        # Save the fits file
        mccd_utils.save_fits(train_dic, train_bool=True, cat_id=catalog_id, output_path=self.output_path)

    def generate_test_data(self, x_grid=5, y_grid=10, n_ccd=40):
        r"""" Generate the test dataset and save it into a fits file.

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
        ``n_ccd`` should be coherent with the corresponding functions on ``mccd.mccd_utils`` that do the
        change of coordiante system.

        """
        # Parameters
        self.test_grid_xy = [x_grid, y_grid]  # Grid size for the PSF generation
        self.n_ccd = n_ccd

        # Generation of the test positions
        ccd_unique_list = np.arange(self.n_ccd)

        # Saving file dictionary
        loc2glob = mccd_utils.Loc2Glob()

        # Generate local generic grid
        x_lin = np.linspace(start=self.image_size, stop=loc2glob.x_npix - self.image_size, num=self.test_grid_xy[0])
        y_lin = np.linspace(start=self.image_size, stop=loc2glob.y_npix - self.image_size, num=self.test_grid_xy[1])
        xv, yv = np.meshgrid(x_lin, y_lin)
        x_coor = xv.flatten()
        y_coor = yv.flatten()

        position_list = []
        ccd_list = []

        for it in range(len(ccd_unique_list)):
            x_glob, y_glob = loc2glob.loc2glob_img_coord(ccd_n=ccd_unique_list[it],
                                                         x_coor=np.copy(x_coor), y_coor=np.copy(y_coor))
            position_list.append(np.array([x_glob, y_glob]).T)
            ccd_list.append((np.ones(len(x_glob), dtype=int) * ccd_unique_list[it]).astype(int))

        # Obtain final positions and ccd_id list
        test_positions = np.concatenate(position_list, axis=0)
        test_ccd_id = np.concatenate(ccd_list, axis=0)

        # Calculate the ellipticities on the testing positions
        test_e1s = np.array([self.e1_catalog_fun(position[0], position[1]) for position in test_positions])
        test_e2s = np.array([self.e2_catalog_fun(position[0], position[1]) for position in test_positions])

        fwhm_psf = (2 * np.sqrt(2 * np.log(2))) * self.sigma
        test_fwhms = np.ones(test_e1s.shape) * fwhm_psf  # Arround 5. and 6. in sigma

        # Generate the vignets
        test_vignets = np.zeros((test_e1s.shape[0], self.image_size, self.image_size))
        test_e1_HSM = np.zeros(test_e1s.shape)
        test_e2_HSM = np.zeros(test_e1s.shape)
        test_R2_HSM = np.zeros(test_e1s.shape)
        for it in range(test_e1s.shape[0]):
            # PSF generation. Define size
            psf = galsim.Moffat(fwhm=test_fwhms[it] * self.pix_scale, beta=self.beta_psf)
            # Define the Flux
            psf = psf.withFlux(self.psf_flux)
            # Define the shear
            psf = psf.shear(g1=test_e1s[it], g2=test_e2s[it])
            # Draw the PSF on a vignet
            image_epsf = galsim.ImageF(self.image_size, self.image_size)
            psf.drawImage(image=image_epsf, scale=self.pix_scale)

            # Before adding the noise, we measure the ellipticity components
            my_moments = galsim.hsm.FindAdaptiveMom(image_epsf)
            test_e1_HSM[it] = my_moments.observed_shape.g1
            test_e2_HSM[it] = my_moments.observed_shape.g2
            test_R2_HSM[it] = my_moments.moments_sigma

            test_vignets[it, :, :] = image_epsf.array

        # Build the masks
        test_masks = self.handle_SExtractor_mask(test_vignets, thresh=-1e5)

        # Build the dictionary
        test_dic = {'VIGNET_LIST': test_vignets, 'GLOB_POSITION_IMG_LIST': test_positions,
                    'MASK_LIST': test_masks, 'CCD_ID_LIST': test_ccd_id,
                    'TRUE_E1_HSM': test_e1_HSM, 'TRUE_E2_HSM': test_e2_HSM, 'TRUE_R2_HSM': test_R2_HSM}

        # Save the fits file
        mccd_utils.save_fits(test_dic, train_bool=False, cat_id=self.catalog_id, output_path=self.output_path)

    @staticmethod
    def e1_catalog_fun(x, y):
        # Set the max and min values for the sinc coordinates
        coor_min = -5
        coor_max = 5

        # Model dependent paremeters
        scale_factor = 0.20001648
        exp_decay_alpha = 702.86105548
        exp_decay_alpha = None

        scaled_x, scaled_y = GenerateSimDataset.scale_coordinates(x, y, coor_min, coor_max)
        scaled_d = np.sqrt(scaled_x ** 2 + scaled_y ** 2)

        vals_x = np.sinc(scaled_d)

        if exp_decay_alpha is not None:
            exp_weight = np.exp(-(scaled_d - ((coor_max + coor_min) / 2)) / exp_decay_alpha)
            scale_factor *= exp_weight

        return vals_x * scale_factor

    @staticmethod
    def e2_catalog_fun(x, y):
        # Set the max and min values for the bessel coordinates
        coor_min = -15
        coor_max = 15

        # Model dependent paremeters
        max_order = 1
        scale_factor = 0.15001691
        exp_decay_alpha = 201.13767350

        return GenerateSimDataset.bessel_generator(x, y, coor_min, coor_max, max_order, scale_factor,
                                                   circular_symetry=True,
                                                   exp_decay_alpha=exp_decay_alpha)

    @staticmethod
    def scale_coordinates(x, y, coor_min, coor_max, offset=None):
        # Set the max and min values for the coordinate system
        loc2glob = mccd_utils.Loc2Glob()
        grid_xmin = - 5 * loc2glob.x_npix - 5 * loc2glob.x_gap
        grid_xmax = 6 * loc2glob.x_npix + 5 * loc2glob.x_gap
        grid_ymin = -2 * loc2glob.y_npix - 2 * loc2glob.y_gap
        grid_ymax = 2 * loc2glob.y_npix + 1 * loc2glob.y_gap

        # Scale the input coordinates
        scaled_x = ((x - grid_xmin) / (grid_xmax - grid_xmin)) * (coor_max - coor_min) + coor_min
        scaled_y = ((y - grid_ymin) / (grid_ymax - grid_ymin)) * (coor_max - coor_min) + coor_min

        if offset is not None:
            scaled_x += offset[0]
            scaled_y += offset[1]

        return scaled_x, scaled_y

    @staticmethod
    def bessel_generator(x, y, coor_min, coor_max, max_order, scale_factor, circular_symetry=False,
                         exp_decay_alpha=None, offset=None):
        # Scale coordinates
        scaled_x, scaled_y = GenerateSimDataset.scale_coordinates(x, y, coor_min, coor_max, offset=offset)

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
                exp_weight = np.exp(-(scaled_d - ((coor_max + coor_min) / 2)) / exp_decay_alpha)
                scale_factor *= exp_weight

            return vals_x * scale_factor

    @staticmethod
    def handle_SExtractor_mask(stars, thresh):
        """ Reads SExtracted star stamps, generates MCCD-compatible masks (that is, binary weights),
        and replaces bad pixels with 0s - they will not be used by MCCD, but the ridiculous numerical
        values can otherwise still lead to problems because of convolutions."""
        mask = np.ones(stars.shape)
        mask[stars < thresh] = 0
        stars[stars < thresh] = 0
        return mask
