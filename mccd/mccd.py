# -*- coding: utf-8 -*-

r"""MCCD CLASS.

This module contains the main MCCD class.

:Authors:   Tobias Liaudat <tobias.liaudat@cea.fr>

            Jerome Bonnin <https://github.com/jerome-bonnin>

            Morgan Schmitz <https://github.com/MorganSchmitz>

"""

from __future__ import absolute_import, print_function
import numpy as np
from scipy.interpolate import Rbf
from modopt.signal.wavelet import get_mr_filters, filter_convolve
from modopt.opt.cost import costObj
from modopt.opt.reweight import cwbReweight
import modopt.opt.algorithms as optimalg
import galsim as gs
import mccd.proxs as prox
import mccd.grads as grads
import mccd.utils as utils
import mccd.mccd_utils as mccd_utils


def mccd_quickload(path):
    r"""Load pre-fitted MCCD model.

    (saved with :func:`mccd.quicksave`).

    Parameters
    ----------
    path : str
        Path to the npy file containing the saved MCCD model.

    Returns
    -------
    loaded_model : MCCD class
        The MCCD model.
    """
    if path[-4:] != '.npy':
        path += '.npy'
    mccd_params, fitted_model = np.load(path, allow_pickle=True)
    loaded_model = MCCD(**mccd_params)
    loaded_model.n_ccd = fitted_model['n_ccd']
    loaded_model.obs_pos = fitted_model['obs_pos']
    loaded_model.A_loc = fitted_model['A_loc']
    loaded_model.A_glob = fitted_model['A_glob']
    loaded_model.S = fitted_model['S']
    loaded_model.flux_ref = fitted_model['flux_ref']
    loaded_model.psf_size = fitted_model['psf_size']
    loaded_model.VT = fitted_model['VT']
    loaded_model.Pi = fitted_model['Pi']
    loaded_model.alpha = fitted_model['alpha']
    loaded_model.is_fitted = True
    try:
        loaded_model.ccd_list = fitted_model['ccd_list']
    except Exception:
        loaded_model.ccd_list = None

    return loaded_model


class MCCD(object):
    r"""Multi-CCD Resolved Components Analysis.

    Parameters
    ----------
    n_comp_loc: int
        Number of components to learn for each CCD.
        The interpretation may depend on the ``loc_model``
        parameter of the :func:`fit()` function.
    d_comp_glob: int
        Degree of polynomial components for the global model.
    upfact: int
        Upsampling factor. Default is 1 (no superresolution).
    ksig_loc: float
        Value of :math:`K_{\sigma}^{Loc}` for the thresholding in
        wavelet (Starlet by default) domain (taken to be
        :math:`K\sigma`, where :math:`\sigma` is the estimated noise standard
        deviation.). It is used for the thresholding of the
        local eigenPSF :math:`S_{K}` update.
        Default is ``1``.
    ksig_glob: float
        Value of :math:`K_{\sigma}^{Glob}` for the thresholding in Starlet
        domain (taken to be :math:`k\sigma`, where :math:`\sigma` is the
        estimated noise standard deviation.).
        It is used for the thresholding of the
        global eigenPSF :math:`\tilde{S}` update.
        Default is ``1``.
    n_scales: int
        Number of wavelet (Default Starlet) scales to use for the
        sparsity constraint.
        Default is ``3``. Unused if ``filters`` are provided.
    ksig_init: float
        Similar to ``ksig``, for use when estimating shifts and noise levels,
        as it might be desirable to have it set higher than ``ksig``.
        Unused if ``shifts`` are provided when running :func:`RCA.fit`.
        Default is ``5``.
    filters: numpy.ndarray
        Optional filters to the transform domain wherein eigenPSFs are
        assumed to be sparse; convolution by them should amount to
        applying :math:`\Phi`. Optional; if not provided, the
        Starlet transform with `n_scales` scales will be used.
    verbose: bool or int
        If True, will only output RCA-specific lines to stdout.
        If verbose is set to 2, will run ModOpt's optimization
        algorithms in verbose mode.
        Default to ``2``.
    """

    def __init__(self, n_comp_loc, d_comp_glob, d_hyb_loc=2,
                 min_d_comp_glob=None, upfact=1, ksig_loc=1.,
                 ksig_glob=1., n_scales=3, ksig_init=1.,
                 filters=None, verbose=2):
        r"""General parameter initialisations."""
        # [TL] TODO Propagate `d_hyb_loc` & `min_d_comp_glob` into config_file
        self.n_comp_loc = n_comp_loc
        self.d_comp_glob = d_comp_glob
        self.min_d_comp_glob = min_d_comp_glob
        self.n_comp_glob = (self.d_comp_glob + 1) * (self.d_comp_glob + 2) // 2
        if self.min_d_comp_glob is not None:
            if self.min_d_comp_glob > self.d_comp_glob:
                raise ValueError("The total global degree must be" +\
                    " bigger than the minimum degree.")
            print('Reducing the global polynomial degree with d_min = ',
                self.min_d_comp_glob)
            self.n_comp_glob -= (self.min_d_comp_glob + 1) * (
                self.min_d_comp_glob + 2) // 2

        self.d_hyb_loc = d_hyb_loc
        self.upfact = upfact
        self.ksig_loc = ksig_loc
        self.ksig_glob = ksig_glob
        self.ksig_init = ksig_init
        self.iter_outputs = False

        if filters is None:
            # option strings for mr_transform
            # ModOpt sparse2d get_mr_filters() convention
            # self.opt = ['-t2', '-n{}'.format(n_scales)]
            # Pysap sparse2d get_mr_filters() convention
            self.opt = 'BsplineWaveletTransformATrousAlgorithm'
            self.n_scales = n_scales
            self.default_filters = True
        else:
            self.Phi_filters = filters
            self.default_filters = False
            self.n_scales = n_scales
        self.verbose = verbose
        if self.verbose > 1:
            self.modopt_verb = True
        else:
            self.modopt_verb = False
        self.is_fitted = False

        # Define the attributes that will be initialised later.
        self.d_comp_loc = None
        self.obs_data = None
        self.n_ccd = None
        self.loc_model = None
        self.ccd_list = None
        self.SNR_weight_list = None
        self.shap = None
        self.im_hr_shape = None
        self.obs_pos = None
        self.obs_weights = None
        self.loc_model = None
        self.S = None
        self.VT = None
        self.Pi = None
        self.A_loc = None
        self.A_glob = None
        self.alpha = None
        self.shifts = None
        self.psf_size_type = None
        self.psf_size = None
        self.sigmas = None
        self.sigs = None
        self.flux = None
        self.flux_ref = None
        self.nb_iter = None
        self.nb_iter_glob = None
        self.nb_iter_loc = None
        self.nb_subiter_S_loc = None
        self.nb_subiter_A_loc = None
        self.nb_subiter_S_glob = None
        self.nb_subiter_A_glob = None
        self.nb_reweight = None
        self.n_eigenvects = None
        self.pi_degree = None
        self.graph_kwargs = None

        if self.iter_outputs:
            self.iters_glob_A = None
            self.iters_glob_S = None
            self.iters_loc_A = None
            self.iters_loc_S = None

    def quicksave(self, path):
        r"""Save fitted model.

        Save fitted MCCD model for later use. Ideally, you would probably
        want to store the whole MCCD instance, though this might mean
        storing a lot of data you are not likely to use if you do not alter
        the fit that was already performed.
        Stored models can be loaded with :func:`mccd.mccd_quickload`.

        Parameters
        ----------
        path: str
            Path to where the fitted MCCDF model should be saved.

        """
        if not self.is_fitted:
            raise ValueError('MCCD instance has not yet been fitted to\
                observations. Please run the fit method.')
        mccd_params = {'n_comp_loc': self.n_comp_loc,
                       'd_comp_glob': self.d_comp_glob, 'upfact': self.upfact}
        fitted_model = {'n_ccd': self.n_ccd, 'obs_pos': self.obs_pos,
                        'A_loc': self.A_loc, 'A_glob': self.A_glob,
                        'S': self.S, 'flux_ref': self.flux_ref,
                        'psf_size': self.psf_size, 'VT': self.VT,
                        'Pi': self.Pi, 'alpha': self.alpha,
                        'ccd_list': self.ccd_list}

        if self.iter_outputs is True:
            iters_dic = {'iters_glob_A': self.iters_glob_A,
                         'iters_glob_S': self.iters_glob_S,
                         'iters_loc_A': self.iters_loc_A,
                         'iters_loc_S': self.iters_loc_S}
            np.save(path + '__iter_outputs_dic', iters_dic)

        if path[-4:] != '.npy':
            path += '.npy'
        np.save(path, [mccd_params, fitted_model])

    def fit(self, obs_data, obs_pos, ccd_list, obs_weights=None,
            SNR_weight_list=None, S=None, VT=None, Pi=None, alpha=None,
            shifts=None, sigs=None, psf_size=6, psf_size_type='R2',
            flux=None, nb_iter=1, nb_iter_glob=2, nb_iter_loc=2,
            nb_subiter_S_loc=100, nb_reweight=0, nb_subiter_A_loc=500,
            nb_subiter_S_glob=30, nb_subiter_A_glob=200, n_eigenvects=5,
            loc_model='rca', pi_degree=2, graph_kwargs={}):
        r"""Fits MCCD to observed star field.

        Parameters
        ----------
        obs_data: list of numpy.ndarray
            Observed data (each element of the list being one CCD).
        obs_pos: list of numpy.ndarray
            Corresponding positions (global coordinate system).
        ccd_list: list of numpy.ndarray
            List containing the ccd_ids of each set of observations,
            positions and weights.
            It is of utmost importance that the ccd_list contains the ccd_id
            in the same order as in the other lists.
            Ex:  obs_data[0] is the data from the ccd ccd_list[0].
        obs_weights: list of numpy.ndarray
            Corresponding weights. Can be either one per observed star,
            or contain pixel-wise values. Masks can be handled via binary
            weights. Default is None (in which case no weights are applied).
            Note if fluxes and shifts are not provided, weights will be ignored
            for their estimation. Noise level estimation only removes
            bad pixels (with weight strictly equal to 0) and otherwise
            ignores weights.
        SNR_weight_list: list of floats
            List of weights to be used on each star. They can probably be based
            on a specific function of the SNR and should represent the degree
            of significance we give to a specific star.
            The values should be around 1.
            Default is ``None`` meaning that no weights will be used.
        S: list of numpy.ndarray
            First guess (or warm start) eigenPSFs :math:`S`
            (last matrix is global). Default is ``None``.
        VT: list of numpy.ndarray
            Matrices of concatenated eigenvectors of the different
            graph Laplacians. Default is ``None``.
        Pi: list of numpy.ndarray
            Matrices of polynomials in positions. Default is ``None``.
        alpha: list numpy.ndarray
            First guess (or warm start) weights :math:`\alpha`,
            after factorization by ``VT`` (last matrix is global).
            Default is ``None``.
        shifts: list of numpy.ndarray
            Corresponding sub-pixel shifts. Default is ``None``;
            will be estimated from observed data if not provided.
        sigs: numpy.ndarray
            Estimated noise levels. Default is ``None``;
            will be estimated from data if not provided.
        psf_size: float
            Approximate expected PSF size in ``psf_size_type``;
            will be used for the size of the Gaussian window for centroid
            estimation.
            ``psf_size_type`` determines the convention used for
            this size (default is FWHM).
            Ignored if ``shifts`` are provided.
            Default is ``'R2'`` of ``6``.
        psf_size_type: str
            Can be any of ``'R2'``, ``'fwhm'`` or ``'sigma'``, for the size
            defined from quadrupole moments, full width at half maximum
            (e.g. from SExtractor) or 1-sigma width of the best matching
            2D Gaussian. ``'sigma'`` is the value outputed from Galsim's
            HSM adaptive moment estimator.
            Default is ``'R2'``.
        flux: list of numpy.ndarray
            Flux levels. Default is ``None``;
            will be estimated from data if not provided.
        nb_iter: int
            Number of overall iterations (i.e. of alternations).
            Note the weights and global components do not
            get updated the last time around, so they actually get
            ``nb_iter-1`` updates.
            Paramter :math:`l_{max}` on the MCCD article pseudo-algorithm.
            Default is ``1``.
        nb_iter_glob: int
            Number of iterations on the global model estimation.
            The times we go trough the global :math:`S,A` updates.
            Paramter :math:`n_G` on the MCCD article pseudo-algorithm.
            Default value is ``2``.
        nb_iter_loc: int
            Number of iterations on the local model estimation.
            The times we go trough the local :math:`S,A` updates.
            Paramter :math:`n_L` on the MCCD article pseudo-algorithm.
            Default value is ``2``.
        nb_subiter_S_loc: int
            Number of iterations when solving the optimization problem (III)
            concerning the local eigenPSFs :math:`S_{k}`.
            Default is ``100``.
        nb_subiter_A_loc: int
            Number of iterations when solving the optimization problem (IV)
            concerning the local weights :math:`\alpha_{k}`.
            Default is ``500``.
        nb_subiter_S_glob: int
            Number of iterations when solving the optimization problem (I)
            concerning the global eigenPSFs :math:`\tilde{S}`.
            Default is ``30``.
        nb_subiter_A_glob: int
            Number of iterations when solving the optimization problem (II)
            concerning the global weights :math:`\tilde{\alpha}`.
            Default is ``200``.
        nb_reweight: int
            Number of reweightings to apply during :math:`S` updates.
            See equation (33) in RCA paper.
            Default is ``0``.
        n_eigenvects: int
            Maximum number of eigenvectors to consider per :math:`(e,a)`
            couple. Default is ``None``;
            if not provided, *all* eigenvectors will be considered,
            which can lead to a poor selection of graphs, especially when data
            is undersampled. Ignored if ``VT`` and ``alpha`` are provided.
        loc_model: str
            Defines the type of local model to use, it can be:
            ``'rca'``, ``'poly'`` or ``'hybrid'``.
            Thus defining the MCCD-RCA, MCCD-POL and MCCD-HYB.
            When MCCD-POL is used, ``n_comp_loc`` should be used
            as the ``d_comp_glob`` (max degree of the polynomial)
            for the local model.
            When MCCD-HYB is used, ``n_comp_loc`` should be used
            as in MCCD-RCA, the number of graph-based eigenPSFs.
            The max local polynomial degree is set to ``2``.
        pi_degree: int
            Maximum degree of polynomials in Pi. Default is ``2``.
            Ignored if Pi is provided.
        graph_kwargs: dict
            List of optional kwargs to be passed on
            to the :func:`utils.GraphBuilder`.
        """
        # Initialize the needed variables
        self.obs_data = [np.copy(obs_data_k) for obs_data_k in obs_data]
        self.n_ccd = len(self.obs_data)
        self.loc_model = loc_model
        self.ccd_list = ccd_list
        if SNR_weight_list is None:
            self.SNR_weight_list = [np.ones(pos.shape[0]) for pos in obs_pos]
        else:
            self.SNR_weight_list = SNR_weight_list
        self.shap = [self.obs_data[k].shape for k in range(self.n_ccd)]
        self.shap.append(np.concatenate(self.obs_data, axis=2).shape)
        self.im_hr_shape = [(self.upfact * self.shap[k][0],
                             self.upfact * self.shap[k][1],
                             self.shap[k][2])
                            for k in range(self.n_ccd)]
        self.obs_pos = obs_pos

        if obs_weights is None:
            self.obs_weights = [np.ones(self.shap[k]) for k in
                                range(self.n_ccd)]
        elif obs_weights[0].shape == self.shap[0]:
            self.obs_weights = [obs_weights[k] /
                                np.expand_dims(np.sum(obs_weights[k], axis=2),
                                               2) * self.shap[k][2]
                                for k in range(self.n_ccd)]
        elif obs_weights.shape[0] == (self.shap[0][2],):
            self.obs_weights = [obs_weights[k].reshape(1, 1, -1) /
                                np.sum(obs_weights[k]) * self.shap[k][2]
                                for k in range(self.n_ccd)]
        else:
            raise ValueError(
                'Shape mismatch; weights should be of shape:' +
                ' {} (for per-pixel weights) or'.format(self.shap[0]) +
                ' {} (per-observation)'.format(self.shap[0][2:]))

        if self.loc_model == 'poly':
            self.d_comp_loc = self.n_comp_loc
            self.n_comp_loc = (self.n_comp_loc + 1) * (
                    self.n_comp_loc + 2) // 2

        if self.loc_model == 'hybrid':
            # Hardcoded a poly deg 2 for the local polynome [TL] [improve]
            self.n_comp_loc += ((self.d_hyb_loc + 1) * (
                self.d_hyb_loc + 2) // 2)

        if S is None:
            # global eigenPSFs are the last ones
            self.S = [np.zeros(self.im_hr_shape[0][:2] + (self.n_comp_loc,))
                      for k in range(self.n_ccd)]
            self.S.append(
                np.zeros(self.im_hr_shape[0][:2] + (self.n_comp_glob,)))
        else:
            self.S = S
        self.VT = VT
        self.Pi = Pi
        self.alpha = alpha
        self.shifts = shifts
        self.psf_size_type = psf_size_type
        if shifts is None:
            self.psf_size = self._set_psf_size(psf_size, self.psf_size_type)
        self.sigmas = None
        self.sigs = sigs
        self.flux = flux
        self.nb_iter = nb_iter
        self.nb_iter_glob = nb_iter_glob
        self.nb_iter_loc = nb_iter_loc
        self.nb_subiter_S_loc = nb_subiter_S_loc
        if nb_subiter_A_loc is None:
            nb_subiter_A_loc = 2 * nb_subiter_S_loc
        self.nb_subiter_A_loc = nb_subiter_A_loc
        self.nb_subiter_S_glob = nb_subiter_S_glob
        self.nb_subiter_A_glob = nb_subiter_A_glob
        self.nb_reweight = nb_reweight
        self.n_eigenvects = n_eigenvects
        self.pi_degree = pi_degree
        self.graph_kwargs = graph_kwargs

        if self.iter_outputs:
            self.iters_glob_A = []
            self.iters_glob_S = []
            self.iters_loc_A = [[] for _ in range(self.n_ccd)]
            self.iters_loc_S = [[] for _ in range(self.n_ccd)]

        # Start the initialization
        if self.verbose:
            print('Running basic initialization tasks...')
        self._initialize()
        if self.verbose:
            print('... Done.')
        if self.VT is None or self.alpha is None:
            if self.verbose:
                print('Constructing local spatial constraint...')

            if self.loc_model == 'rca':
                self._initialize_graph_constraint()
            elif self.loc_model == 'poly':
                self._initialize_loc_poly_model()
            elif self.loc_model == 'hybrid':
                self._initialize_loc_hybrid_model()
            else:
                raise ValueError(
                    'Local model not undersood. Should be <rca> or <poly>.')
            if self.verbose:
                print('... Done.')
        else:
            self.A_loc = [self.alpha[k].dot(self.VT[k])
                          for k in range(self.n_ccd)]
        if self.Pi is None or len(self.alpha) <= self.n_ccd:
            if self.verbose:
                print('Building position polynomials...')
            self._initialize_poly_model()
            if self.verbose:
                print('... Done.')
        else:
            self.A_glob = [self.alpha[self.n_ccd].dot(self.Pi[k])
                           for k in range(self.n_ccd)]

        # Finally fit the model
        self._fit()
        self.is_fitted = True
        return self.S, self.A_loc, self.A_glob, self.alpha, self.Pi

    @staticmethod
    def _set_psf_size(psf_size, psf_size_type):
        r"""Handle different ``size`` conventions."""
        if psf_size is not None:
            if psf_size_type == 'fwhm':
                return psf_size / (2 * np.sqrt(2 * np.log(2)))
            elif psf_size_type == 'R2':
                return np.sqrt(psf_size / 2)
            elif psf_size_type == 'sigma':
                return psf_size
            else:
                raise ValueError(
                    'psf_size_type should be one of "fwhm", "R2" or "sigma"')
        else:
            print('''WARNING: neither shifts nor an estimated PSF size were
            provided to RCA; the shifts will be estimated from the data using
            the default Gaussian window of 7.5 pixels.''')
            return 7.5

    def _initialize(self):
        r"""Initialize internal tasks.

        Tasks related to noise levels, shifts and flux.
        Note it includes renormalizing observed data,
        so needs to be ran even if all three are provided.
        """
        if self.default_filters:
            init_filters = mccd_utils.get_mr_filters(
                self.shap[0][:2], opt=self.opt, coarse=True)
        else:
            init_filters = self.Phi_filters

        # Initialize sizes with Galsim's HSM
        if self.sigmas is None:
            star_moms = [[gs.hsm.FindAdaptiveMom(
                gs.Image(star), badpix=gs.Image(np.rint(np.abs(badpix - 1))),
                guess_sig=self.psf_size, strict=False)
                for star, badpix in zip(utils.reg_format(self.obs_data[k]),
                                        utils.reg_format(self.obs_weights[k]))]
                for k in range(self.n_ccd)]

            self.sigmas = [np.array(
                [moms.moments_sigma for moms in star_moms[k]])
                for k in range(self.n_ccd)]

        # Initialize noise levels
        if self.sigs is None:
            transf_data = [
                utils.apply_transform(self.obs_data[k], init_filters)
                for k in range(self.n_ccd)]
            transf_mask = [
                utils.transform_mask(self.obs_weights[k], init_filters[0])
                for k in range(self.n_ccd)]
            sigmads = [
                np.array([1.4826 * utils.mad(fs[0], w)
                          for fs, w in zip(transf_data[k],
                                           utils.reg_format(transf_mask[k]))])
                for k in range(self.n_ccd)]
            self.sigs = [sigmads[k] / np.linalg.norm(init_filters[0])
                         for k in range(self.n_ccd)]
        else:
            self.sigs = [np.copy(self.sigs[k]) for k in range(self.n_ccd)]
        self.sig_min = [np.min(self.sigs[k]) for k in range(self.n_ccd)]
        self.sig_min.append(np.min(self.sig_min))

        # Initialize intra-pixel shifts
        if self.shifts is None:
            thresh_data = [np.copy(self.obs_data[k])
                           for k in range(self.n_ccd)]
            cents = [[] for k in range(self.n_ccd)]
            for k in range(self.n_ccd):
                for i in range(self.shap[k][2]):
                    # don't allow thresholding to be
                    # over 80% of maximum observed pixel
                    nsig_shifts = min(self.ksig_init,
                                      0.8 * self.obs_data[k][:, :, i].max() /
                                      self.sigs[k][i])
                    thresh_data[k][:, :, i] = utils.HardThresholding(
                        thresh_data[k][:, :, i],
                        nsig_shifts * self.sigs[k][i])
                    cents[k] += [
                        utils.CentroidEstimator(thresh_data[k][:, :, i],
                                                sig=self.sigmas[k][i])]
            self.shifts = [np.array([ce.return_shifts() for ce in cents[k]])
                           for k in range(self.n_ccd)]
        lanc_rad = np.ceil(3. * np.max(
            np.array([np.max(_sigma) for _sigma in self.sigmas]))).astype(int)
        self.shift_ker_stack, self.shift_ker_stack_adj = zip(
            *[utils.shift_ker_stack(self.shifts[k], self.upfact,
                                    lanc_rad=lanc_rad)
              for k in range(self.n_ccd)])

        # Flux levels
        if self.flux is None:
            centroids = [np.array([[ce.xc, ce.yc] for ce in cents[k]]) for k in
                         range(self.n_ccd)]
            self.flux = [
                utils.flux_estimate_stack(self.obs_data[k], cent=centroids[k],
                                          sigmas=self.sigmas[k]) for k in
                range(self.n_ccd)]
        self.flux_ref = [np.median(self.flux[k]) for k in range(self.n_ccd)]
        self.flux_ref.append(np.median(np.concatenate(self.flux)))

        # Normalize noise levels observed data
        for k in range(self.n_ccd):
            self.sigs[k] /= self.sig_min[k]
            self.obs_data[k] /= self.sigs[k].reshape(1, 1, -1)

    def _initialize_graph_constraint(self):
        r"""Initialize of the graph constraint.

        ``graph_kwards`` parameters used.
        """
        gber = [utils.GraphBuilder(self.obs_data[k], self.obs_pos[k],
                                   self.obs_weights[k], self.n_comp_loc,
                                   n_eigenvects=self.n_eigenvects,
                                   verbose=self.verbose,
                                   **self.graph_kwargs) for k in
                range(self.n_ccd)]
        self.VT, self.alpha, self.distances = (
            [gber[k].VT for k in range(self.n_ccd)],
            [gber[k].alpha for k in range(self.n_ccd)],
            [gber[k].distances for k in range(self.n_ccd)])
        self.sel_e, self.sel_a = ([gber[k].sel_e for k in range(self.n_ccd)],
                                  [gber[k].sel_a for k in range(self.n_ccd)])
        self.A_loc = [self.alpha[k].dot(self.VT[k]) for k in range(self.n_ccd)]

    def _initialize_poly_model(self):
        r"""Initialize the global polynomial model.

        The positions in the global coordinate system
        are used.
        Normalization of the polynomials is being done here.
        """
        # Calculate max and min values of global coordinate system
        # This configuration is specific for CFIS MegaCam configuration
        loc2glob = mccd_utils.Loc2Glob()
        max_x = loc2glob.x_npix * 6 + loc2glob.x_gap * 5
        min_x = loc2glob.x_npix * (-5) + loc2glob.x_gap * (-5)
        max_y = loc2glob.y_npix * 2 + loc2glob.y_gap * 1
        min_y = loc2glob.y_npix * (-2) + loc2glob.y_gap * (-2)

        self.Pi = [
            utils.poly_pos(pos=self.obs_pos[k], max_degree=self.d_comp_glob,
                           center_normalice=True,
                           x_lims=[min_x, max_x], y_lims=[min_y, max_y],
                           normalice_Pi=False, min_degree=self.min_d_comp_glob)
            for k in range(self.n_ccd)]

        self.alpha.append(np.eye(self.n_comp_glob))

        # Global position model normalisation
        # Start with the list Pi
        conc_Pi = np.concatenate((self.Pi), axis=1)
        Pi_norms = np.sqrt(np.sum(conc_Pi**2, axis=1)).reshape(-1, 1)
        self.Pi = [self.Pi[k] / Pi_norms for k in range(self.n_ccd)]

        self.A_glob = [self.alpha[self.n_ccd].dot(self.Pi[k])
                       for k in range(self.n_ccd)]

    def _initialize_loc_poly_model(self):
        r"""Initialize the local polynomial model."""
        self.VT = [
            utils.poly_pos(pos=self.obs_pos[k], max_degree=self.d_comp_loc,
                           center_normalice=True,
                           x_lims=None, y_lims=None)
            for k in range(self.n_ccd)]
        self.alpha = [np.eye(self.n_comp_loc) for _it in range(self.n_ccd)]
        self.A_loc = [self.alpha[k].dot(self.VT[k]) for k in range(self.n_ccd)]

    def _initialize_loc_hybrid_model(self):
        r"""Initialize the hybrid local model.

        Graphs + polynomials.
        """
        max_deg = self.d_hyb_loc
        n_poly_comp = (max_deg + 1) * (max_deg + 2) // 2
        # Take the number of local component top the graph value
        self.n_comp_loc -= n_poly_comp

        # First initialize the graph constraint
        self._initialize_graph_constraint()

        # Calculate the local polynomial and add it to
        # the graph-calculated values
        for k in range(self.n_ccd):
            poly_VT = utils.poly_pos(pos=self.obs_pos[k], max_degree=max_deg,
                                     center_normalice=True,
                                     x_lims=None, y_lims=None)
            poly_alpha = np.eye(n_poly_comp)

            n_comp_hyb = poly_alpha.shape[0]
            n_vec_hyb = poly_alpha.shape[1]

            zero_concat_1 = np.zeros((self.n_comp_loc, n_vec_hyb))
            zero_concat_2 = np.zeros((n_comp_hyb, self.alpha[k].shape[1]))
            tmp_alpha_1 = np.concatenate((self.alpha[k], zero_concat_1),
                                         axis=1)
            tmp_alpha_2 = np.concatenate((zero_concat_2, poly_alpha), axis=1)
            self.alpha[k] = np.concatenate((tmp_alpha_1, tmp_alpha_2), axis=0)
            self.VT[k] = np.concatenate((self.VT[k], poly_VT), axis=0)
            self.A_loc[k] = self.alpha[k].dot(self.VT[k])

        self.n_comp_loc += n_poly_comp

    def _fit(self):
        r"""Perform the main model fitting."""
        # Function variables
        comp = self.S
        alpha = self.alpha
        weights_loc = self.A_loc
        weights_glob = self.A_glob

        # Very useful shortcut
        conc = np.concatenate

        # Estimated models (local and global for each CCD)
        H_loc = [comp[k].dot(weights_loc[k]) for k in range(self.n_ccd)]
        H_glob = [comp[self.n_ccd].dot(weights_glob[k]) for k in
                  range(self.n_ccd)]

        # Dual variables (for Condat algorithm)
        dual_comp = [np.zeros((self.im_hr_shape[k])) for k in
                     range(self.n_ccd)]
        dual_alpha = [np.zeros(self.A_loc[k].shape) for k in range(self.n_ccd)]
        dual_alpha.append(np.zeros(conc(self.A_glob, axis=1).shape))

        # Starlet filters and associated spectral radius
        if self.default_filters:
            self.Phi_filters = mccd_utils.get_mr_filters(
                self.im_hr_shape[0][:2], opt=self.opt,
                coarse=True, trim=False)
        rho_phi = np.sqrt(
            np.sum(np.sum(np.abs(self.Phi_filters), axis=(1, 2)) ** 2))

        # Gradient objects
        source_loc_grad = [grads.SourceLocGrad(
            self.obs_data[k],
            self.obs_weights[k],
            weights_loc[k],
            H_glob[k],
            self.flux[k],
            self.sigs[k],
            self.shift_ker_stack[k],
            self.shift_ker_stack_adj[k],
            self.SNR_weight_list[k],
            self.upfact,
            self.Phi_filters,
            save_iter_cost=self.iter_outputs,
            verbose=self.verbose) for k in range(self.n_ccd)]

        weight_loc_grad = [grads.CoeffLocGrad(
            self.obs_data[k],
            self.obs_weights[k],
            comp[k],
            self.VT[k],
            H_glob[k],
            self.flux[k],
            self.sigs[k],
            self.shift_ker_stack[k],
            self.shift_ker_stack_adj[k],
            self.SNR_weight_list[k],
            self.upfact,
            save_iter_cost=self.iter_outputs,
            verbose=self.verbose) for k in range(self.n_ccd)]

        source_glob_grad = grads.SourceGlobGrad(
            conc(self.obs_data, axis=2),
            conc(self.obs_weights, axis=2),
            conc(weights_glob, axis=1),
            conc(H_loc, axis=2),
            conc(self.flux),
            conc(self.sigs),
            conc(self.shift_ker_stack, axis=2),
            conc(self.shift_ker_stack_adj, axis=2),
            conc(self.SNR_weight_list),
            self.upfact,
            self.Phi_filters,
            save_iter_cost=self.iter_outputs,
            verbose=self.verbose)

        weight_glob_grad = grads.CoeffGlobGrad(
            conc(self.obs_data, axis=2),
            conc(self.obs_weights, axis=2),
            comp[self.n_ccd],
            conc(self.Pi, axis=1),
            conc(H_loc, axis=2),
            conc(self.flux),
            conc(self.sigs),
            conc(self.shift_ker_stack, axis=2),
            conc(self.shift_ker_stack_adj, axis=2),
            self.upfact,
            conc(self.SNR_weight_list),
            save_iter_cost=self.iter_outputs,
            verbose=self.verbose)

        # Proxs for component optimization
        sparsity_prox = prox.StarletThreshold(0)
        pos_prox = [prox.PositityOff(H_k) for H_k in H_glob]
        lin_recombine = [prox.LinRecombine(weights_loc[k], self.Phi_filters)
                         for k in range(self.n_ccd)]

        # Proxs for weight optimization
        # Local model
        # The last (1-steady_state_thresh_loc)*100% elements will have same
        # threshold
        # min_elements_loc: Minimum number of elements to maintain when
        # threshold is the highest
        steady_state_thresh_loc = 0.8
        min_elements_loc = 5

        def iter_func_loc(x, elem_size):
            return np.min(
                [np.floor((elem_size / 2 - 1) * (1 / np.sqrt(
                    self.nb_subiter_A_loc * steady_state_thresh_loc)) \
                          * np.sqrt(x)) + min_elements_loc,
                 np.floor(elem_size / 2)])

        # [TL] Using strong sparsity inducing function
        iter_func_loc = lambda x, elem_size: np.floor(np.sqrt(x)) + 1
        coeff_prox_loc = prox.KThreshold(iter_func_loc)

        # Global model
        # The last (1-steady_state_thresh_glob)*100% elements will have same
        # threshold
        # min_elements_glob: Minimum number of elements to maintain when
        # threshold is the highest
        steady_state_thresh_glob = 0.8
        min_elements_glob = 5

        def iter_func_glob(x, elem_size):
            return np.min(
                [np.floor((elem_size / 2 - 1) * (1 / np.sqrt(
                    self.nb_subiter_A_glob * steady_state_thresh_glob)) \
                          * np.sqrt(x)) + min_elements_glob,
                 np.floor(elem_size / 2)])

        # [TL] Using strong sparsity inducing function
        iter_func_glob_v2 = lambda x, elem_size: np.floor(np.sqrt(x)) + 1
        coeff_prox_glob = prox.KThreshold(iter_func_glob_v2)

        norm_prox = prox.proxNormalization(type='columns')
        lin_recombine_alpha = [prox.LinRecombineAlpha(self.VT[k])
                               for k in range(self.n_ccd)]
        lin_recombine_alpha.append(
            prox.LinRecombineAlpha(conc(self.Pi, axis=1)))

        # Cost functions
        source_loc_cost = [
            costObj([source_loc_grad[k]], verbose=self.modopt_verb)
            for k in range(self.n_ccd)]
        weight_loc_cost = [
            costObj([weight_loc_grad[k]], verbose=self.modopt_verb)
            for k in range(self.n_ccd)]
        source_glob_cost = costObj([source_glob_grad],
                                   verbose=self.modopt_verb)
        weight_glob_cost = costObj([weight_glob_grad],
                                   verbose=self.modopt_verb)

        # Transformed components in wavelet (default: Starlet) domain
        transf_comp = [utils.apply_transform(comp[k], self.Phi_filters)
                       for k in range(self.n_ccd + 1)]

        # Big loop: Main iteration
        for main_it in range(self.nb_iter):

            # Global Optimization
            for l_glob in range(self.nb_iter_glob):
                # Global Components Optimization

                # Components gradient update
                source_glob_grad.update_A(conc(weights_glob, axis=1))
                source_glob_grad.update_H_loc(conc(H_loc, axis=2))

                # Lipschitz constant for ForwardBackward
                beta = source_glob_grad.spec_rad * 1.5 + rho_phi
                tau = 1. / beta

                # Sparsity prox thresholds update
                thresh = utils.reg_format(
                    utils.acc_sig_maps(
                        self.shap[self.n_ccd],
                        conc(self.shift_ker_stack_adj, axis=2),
                        conc(self.sigs),
                        conc(self.flux),
                        self.flux_ref[self.n_ccd],
                        self.upfact,
                        conc(weights_glob, axis=1),
                        sig_data=np.ones(
                                (self.shap[self.n_ccd][2],)) \
                        * self.sig_min[self.n_ccd]))

                thresholds = self.ksig_glob * np.sqrt(
                    np.array(
                        [filter_convolve(Sigma_k ** 2, self.Phi_filters ** 2)
                         for Sigma_k in thresh]))
                sparsity_prox.update_threshold(tau * thresholds)

                # Reweighting. Borrowed from original RCA code
                if self.nb_reweight:
                    reweighter = cwbReweight(self.nb_reweight)
                    for _ in range(self.nb_reweight):
                        # Optimize!
                        source_optim = optimalg.ForwardBackward(
                            transf_comp[self.n_ccd],
                            source_glob_grad, sparsity_prox,
                            cost=source_glob_cost,
                            beta_param=1. / beta, auto_iterate=False,
                            verbose=self.verbose, progress=self.verbose)
                        source_optim.iterate(max_iter=self.nb_subiter_S_glob)
                        transf_comp[self.n_ccd] = source_optim.x_final
                        reweighter.reweight(transf_comp[self.n_ccd])
                        thresholds = reweighter.weights
                else:
                    # Optimize!
                    source_optim = optimalg.ForwardBackward(
                        transf_comp[self.n_ccd],
                        source_glob_grad, sparsity_prox, cost=source_glob_cost,
                        beta_param=1. / beta, auto_iterate=False,
                        verbose=self.verbose, progress=self.verbose)
                    source_optim.iterate(max_iter=self.nb_subiter_S_glob)
                    transf_comp[self.n_ccd] = source_optim.x_final

                    # Save iteration diagnostic data
                    if self.iter_outputs:
                        self.iters_glob_S.append(
                            source_glob_grad.get_iter_cost())
                        source_glob_grad.reset_iter_cost()

                # Update pixel domain global components
                comp[self.n_ccd] = utils.rca_format(
                    np.array([filter_convolve(transf_Sj,
                                              self.Phi_filters,
                                              filter_rot=True) for
                              transf_Sj in transf_comp[self.n_ccd]]))

                # Global Weights Optimization

                # Weights gradient update
                weight_glob_grad.update_S(comp[self.n_ccd])
                weight_glob_grad.update_H_loc(conc(H_loc, axis=2))

                # Coefficient sparsity prox update
                coeff_prox_glob.reset_iter()

                if l_glob < self.nb_iter_glob - 1 or l_glob == 0:
                    # Conda's algorithm parameters
                    # (lipschitz of diff. part and operator norm of lin. part)
                    # See Conda's paper for more details.
                    beta = weight_glob_grad.spec_rad * 1.5
                    tau = 1. / beta
                    sigma = (1. / lin_recombine_alpha[
                        self.n_ccd].norm ** 2) * beta / 2

                    # Optimize !
                    weight_optim = optimalg.Condat(
                        alpha[self.n_ccd],
                        dual_alpha[self.n_ccd],
                        weight_glob_grad,
                        coeff_prox_glob,
                        norm_prox,
                        linear=lin_recombine_alpha[self.n_ccd],
                        cost=weight_glob_cost,
                        max_iter=self.nb_subiter_A_glob,
                        tau=tau,
                        sigma=sigma,
                        verbose=self.verbose,
                        progress=self.verbose)
                    alpha[self.n_ccd] = weight_optim.x_final
                    weights_glob = [alpha[self.n_ccd].dot(self.Pi[k])
                                    for k in range(self.n_ccd)]

                    # Save iteration diagnostic data
                    if self.iter_outputs:
                        self.iters_glob_A.append(
                            weight_glob_grad.get_iter_cost())
                        weight_glob_grad.reset_iter_cost()

                # Global model update
                H_glob = [comp[self.n_ccd].dot(weights_glob[k])
                          for k in range(self.n_ccd)]

            # Local models update
            for l_loc in range(self.nb_iter_loc):
                # Loop on all CCDs
                for k in range(self.n_ccd):
                    # Local Components Optimization

                    # Components gradient update
                    source_loc_grad[k].update_A(weights_loc[k])
                    source_loc_grad[k].update_H_glob(H_glob[k])

                    # Positivity prox update
                    pos_prox[k].update_offset(H_glob[k])
                    lin_recombine[k].update_A(weights_loc[k])

                    # Conda parameters
                    # (lipschitz of diff. part and operator norm of lin. part)
                    beta = source_loc_grad[k].spec_rad * 1.5 + rho_phi
                    tau = 1. / beta
                    sigma = (1. / lin_recombine[k].norm ** 2) * beta / 2

                    # Sparsity prox thresholds update
                    thresh = utils.reg_format(
                        utils.acc_sig_maps(
                            self.shap[k],
                            self.shift_ker_stack_adj[k],
                            self.sigs[k],
                            self.flux[k],
                            self.flux_ref[k],
                            self.upfact,
                            weights_loc[k],
                            sig_data=np.ones((self.shap[k][2],)) \
                            * self.sig_min[k]))

                    thresholds = self.ksig_loc * np.sqrt(
                        np.array([filter_convolve(
                            Sigma_k ** 2, self.Phi_filters ** 2)
                            for Sigma_k in thresh]))
                    sparsity_prox.update_threshold(tau * thresholds)

                    # Reweighting
                    if self.nb_reweight:
                        reweighter = cwbReweight(self.nb_reweight)
                        for _ in range(self.nb_reweight):
                            # Optimize!
                            source_optim = optimalg.Condat(
                                transf_comp[k],
                                dual_comp[k],
                                source_loc_grad[k],
                                sparsity_prox,
                                pos_prox[k],
                                linear=lin_recombine[k],
                                cost=source_loc_cost[k],
                                max_iter=self.nb_subiter_S_loc,
                                tau=tau,
                                sigma=sigma,
                                verbose=self.verbose,
                                progress=self.verbose)
                            transf_comp[k] = source_optim.x_final
                            reweighter.reweight(transf_comp[k])
                            thresholds = reweighter.weights
                    else:
                        # Optimize!
                        source_optim = optimalg.Condat(
                            transf_comp[k],
                            dual_comp[k],
                            source_loc_grad[k],
                            sparsity_prox,
                            pos_prox[k],
                            linear=lin_recombine[k],
                            cost=source_loc_cost[k],
                            max_iter=self.nb_subiter_S_loc,
                            tau=tau, sigma=sigma,
                            verbose=self.verbose,
                            progress=self.verbose)
                        transf_comp[k] = source_optim.x_final

                        # Save iteration diagnostic data
                        if self.iter_outputs:
                            self.iters_loc_S[k].append(
                                source_loc_grad[k].get_iter_cost())
                            source_loc_grad[k].reset_iter_cost()

                    # Update pixel domain local components
                    comp[k] = utils.rca_format(
                        np.array([filter_convolve(transf_Sj,
                                                  self.Phi_filters,
                                                  filter_rot=True) for
                                  transf_Sj in transf_comp[k]]))

                    # Local weights Optimization

                    # Weights gradient update
                    weight_loc_grad[k].update_S(comp[k])
                    weight_loc_grad[k].update_H_glob(H_glob[k])

                    # Coeff sparsity prox update
                    coeff_prox_loc.reset_iter()

                    # Skip alpha updates for the last iterations
                    if l_loc < self.nb_iter_loc - 1 or l_loc == 0:

                        # Conda parameters
                        # (lipschitz of diff. part and
                        # operator norm of lin. part)
                        beta = weight_loc_grad[k].spec_rad * 1.5
                        tau = 1. / beta
                        sigma = (1. / lin_recombine_alpha[k].norm ** 2) * (
                            beta / 2)

                        # Optimize
                        weight_optim = optimalg.Condat(
                            alpha[k],
                            dual_alpha[k],
                            weight_loc_grad[k],
                            coeff_prox_loc,
                            norm_prox,
                            linear=lin_recombine_alpha[k],
                            cost=weight_loc_cost[k],
                            max_iter=self.nb_subiter_A_loc,
                            tau=tau,
                            sigma=sigma,
                            verbose=self.verbose,
                            progress=self.verbose)
                        alpha[k] = weight_optim.x_final
                        weights_loc[k] = alpha[k].dot(self.VT[k])

                        # Save iteration diagnostic data
                        if self.iter_outputs:
                            self.iters_loc_A[k].append(
                                weight_loc_grad[k].get_iter_cost())
                            weight_loc_grad[k].reset_iter_cost()

                    # Local model update
                    H_loc[k] = comp[k].dot(weights_loc[k])

        # Final values
        self.S = comp
        self.alpha = alpha
        self.A_loc = weights_loc
        self.A_glob = weights_glob

    def estimate_psf(self, test_pos, ccd_n, n_loc_neighbors=15,
                     n_glob_neighbors=15, rbf_function='thin_plate',
                     apply_degradation=False, shifts=None, flux=None,
                     sigmas=None, upfact=None, mccd_debug=False,
                     global_pol_interp=None):
        r"""Estimate and return PSF at desired positions.

        Returns the model in "regular" format, (n_stars, n_pixels, n_pixels).

        Parameters
        ----------
        test_pos: numpy.ndarray
            Positions where the PSF should be estimated.
            Should be in the same format (coordinate system, units, etc.) as
            the ``obs_pos`` fed to :func:`MCCD.fit`.
        ccd_n: int
            ``ccd_id`` of the positions to be tested.
        n_loc_neighbors: int
            Number of neighbors for the local model to use for RBF
            interpolation. Default is 15.
        n_glob_neighbors: int
            Number of neighbors for the global model to use for RBF
            interpolation. Default is 15.
        rbf_function: str
            Type of RBF kernel to use. Default is ``'thin_plate'``.
            Check scipy RBF documentation for more models.
        apply_degradation: bool
            Whether PSF model should be degraded (shifted and
            resampled on coarse grid), for instance for comparison with stars.
            If True, expects shifts to be provided.
            Needed if pixel validation against stars will be required.
            Default is False.
        shifts: numpy.ndarray
            Intra-pixel shifts to apply if ``apply_degradation`` is set
            to True.
            Needed to match the observed stars in case of pixel validation.
        flux: numpy.ndarray
            Flux levels by which reconstructed PSF will be multiplied if
            provided. For pixel validation with stars if ``apply_degradation``
            is set to True.
        sigmas: numpy.ndarray
            Sigmas (shapes) are used to have a better estimate of the size
            of the lanczos interpolant that will be used when performing the
            intra-pixel shift.
            Default is None, where a predefined value is used for the
            interpolant size.
        upfact: int
            Upsampling factor; default is None, in which case that of the
            MCCD instance will be used.
        mccd_debug: bool
            Debug option. It returns the model plus the local and the global
            reconstruction components.
            Default is False.
        global_pol_interp: numpy.ndarray
            If is None, the global interpolation is done with th RBF
            interpolation as in the local model.
            If is not None, the global interpolation is done directly using
            position polynomials.
            In this case, ``global_pol_interp`` should be the normalized Pi
            interpolation matrix.
            Default is None.

        Returns
        -------
        PSFs: numpy.ndarray
            Returns the interpolated PSFs in regular format.

        """
        if not self.is_fitted:
            raise ValueError('''MCCD instance has not yet been fitted to
            observations. Please run the fit method.''')
        if upfact is None:
            upfact = self.upfact

        if sigmas is None:
            lanc_rad = 8
        else:
            lanc_rad = np.ceil(3. * np.max(sigmas)).astype(int)

        ntest = test_pos.shape[0]
        test_weights_glob = np.zeros((self.n_comp_glob, ntest))
        test_weights_loc = np.zeros((self.n_comp_loc, ntest))

        # Turn ccd_n into list number.
        # Select the correct indexes of the requested CCD.
        try:
            ccd_idx = np.where(np.array(self.ccd_list) == ccd_n)[0][0]
        except Exception:
            # If the CCD was not used for training the output should be
            # None and be handled by the wrapping function.
            if mccd_debug:
                return None, None, None
            else:
                return None

        # PSF recovery
        for j, pos in enumerate(test_pos):
            # Local model

            # Determine neighbors
            nbs_loc, pos_nbs_loc = mccd_utils.return_loc_neighbors(
                pos,
                self.obs_pos[ccd_idx],
                self.A_loc[ccd_idx].T,
                n_loc_neighbors)
            # Train RBF and interpolate for each component
            for i in range(self.n_comp_loc):
                rbfi = Rbf(pos_nbs_loc[:, 0],
                           pos_nbs_loc[:, 1],
                           nbs_loc[:, i],
                           function=rbf_function)
                test_weights_loc[i, j] = rbfi(pos[0], pos[1])

            # Global model
            if global_pol_interp is None:
                # Use RBF interpolation for the global component
                # Depending on the problem one may want to use neighbors
                # from the corresponding CCD (return_loc_neighbors) or from
                # any CCD (return_glob_neighbors).

                # Using return_loc_neighbors()
                nbs_glob, pos_nbs_glob = mccd_utils.return_loc_neighbors(
                    pos,
                    self.obs_pos[ccd_idx],
                    self.A_glob[ccd_idx].T,
                    n_glob_neighbors)

                # Using return_glob_neighbors()
                # nbs_glob, pos_nbs_glob = mccd_utils.return_glob_neighbors(
                #     pos,
                #     self.obs_pos,
                #     self.A_glob,
                #     n_glob_neighbors)

                for i in range(self.n_comp_glob):
                    rbfi = Rbf(pos_nbs_glob[:, 0],
                               pos_nbs_glob[:, 1],
                               nbs_glob[:, i],
                               function=rbf_function)
                    test_weights_glob[i, j] = rbfi(pos[0], pos[1])
            else:
                # Use classic PSFEx-like position polynomial interpolation
                # for the global component
                test_weights_glob = self.alpha[-1] @ global_pol_interp

        # Reconstruct the global and local PSF contributions
        PSFs_loc = self._loc_transform(test_weights_loc, ccd_idx)
        PSFs_glob = self._glob_transform(test_weights_glob)
        PSFs = PSFs_glob + PSFs_loc

        if apply_degradation:
            shift_kernels, _ = utils.shift_ker_stack(shifts, self.upfact,
                                                     lanc_rad=lanc_rad)
            # PSFs changed into reg_format in the degradation process
            deg_PSFs = np.array(
                [utils.degradation_op(PSFs[:, :, j],
                                      shift_kernels[:, :, j],
                                      upfact)
                 for j in range(ntest)])

            if flux is not None:
                deg_PSFs *= flux.reshape(-1, 1, 1) / self.flux_ref[ccd_idx]

            if mccd_debug:
                deg_PSFs_glob = np.array([utils.degradation_op(
                    PSFs_glob[:, :, j], shift_kernels[:, :, j], upfact)
                    for j in range(ntest)])
                deg_PSFs_loc = np.array([utils.degradation_op(
                    PSFs_loc[:, :, j], shift_kernels[:, :, j], upfact)
                    for j in range(ntest)])
                if flux is not None:
                    deg_PSFs_glob *= flux.reshape(-1, 1, 1) / self.flux_ref[
                        ccd_idx]
                    deg_PSFs_loc *= flux.reshape(-1, 1, 1) / self.flux_ref[
                        ccd_idx]

            if mccd_debug:
                return deg_PSFs, deg_PSFs_glob, deg_PSFs_loc
            else:
                return deg_PSFs

        else:
            # If PSF are not degraded they come in rca_format from before so
            # we change to regular_format.
            # I should normalize the flux of the PSFs before the output when
            # no degradation is done
            PSFs = np.array(
                [PSFs[:, :, j] / np.sum(PSFs[:, :, j]) for j in range(ntest)])

            if mccd_debug:
                return PSFs, PSFs_glob, PSFs_loc
            else:
                return PSFs

    def validation_stars(self, test_stars, test_pos, test_masks=None,
                         ccd_id=None, mccd_debug=False,
                         response_flag=False, global_pol_interp=None):
        r"""Match PSF model to stars.

        The match is done in flux, shift and pixel sampling -
        for validation tests.
        Returns the matched PSFs' stamps.
        Returned matched PSFs will be None if there is no model on that
        specific CCD due to the lack of training stars.

        Parameters
        ----------
        test_stars: numpy.ndarray
            Star stamps to be used for comparison with the PSF model.
            Should be in "rca" format,
            i.e. with axises (n_pixels, n_pixels, n_stars).
        test_pos: numpy.ndarray
            Their corresponding positions in global coordinate system.
        test_masks: numpy.ndarray
            Masks to be used on the star stamps.
            If None, all the pixels will be used.
            Default is None.
        ccd_id: int
            The corresponding ccd_id
            (ie ccd number corresponding to the megacam geometry).
            Do not mistake for ccd_idx (index).
        mccd_debug: bool
            Debug option. It returns the local and the global
            reconstruction components.
        response_flag: bool
            Response option. True if in response mode.
        global_pol_interp: Position pols of numpy.ndarray or None
            If is None, the global interpolation is done with th RBF
            interpolation as in the local model.
            If is not None, the global interpolation is done directly
            using position polynomials.
            In this case, it should be the normalized Pi interpolation matrix.
        """
        if not self.is_fitted:
            raise ValueError('''MCCD instance has not yet been fitted to
            observations. Please run the fit method.''')

        if response_flag:
            test_shifts = np.zeros((test_pos.shape[0], 2))
            test_fluxes = None
            sigmas = np.ones((test_pos.shape[0],)) * self.psf_size

        else:
            if test_masks is None:
                test_masks = np.ones(test_stars.shape)

            star_moms = [gs.hsm.FindAdaptiveMom(gs.Image(star),
                                                badpix=gs.Image(np.rint(
                                                    np.abs(badpix - 1))),
                                                guess_sig=self.psf_size,
                                                strict=False)
                         for star, badpix in zip(utils.reg_format(test_stars),
                                                 utils.reg_format(test_masks))]
            sigmas = np.array([moms.moments_sigma for moms in star_moms])
            cents = [
                utils.CentroidEstimator(test_stars[:, :, it], sig=sigmas[it])
                for it in range(test_stars.shape[2])]
            test_shifts = np.array([ce.return_shifts() for ce in cents])
            test_fluxes = utils.flux_estimate_stack(test_stars, sigmas=sigmas)

        matched_psfs = self.estimate_psf(test_pos, ccd_id,
                                         apply_degradation=True,
                                         shifts=test_shifts,
                                         flux=test_fluxes,
                                         sigmas=sigmas,
                                         mccd_debug=mccd_debug,
                                         global_pol_interp=global_pol_interp)

        # Optimized flux matching
        if matched_psfs is not None:
            # matched_psfs will be None if there is no model on that
            # specific CCD due to the lack of training stars.
            norm_factor = np.array(
                [np.sum(_star * _psf) / np.sum(_psf * _psf)
                 for _star, _psf in zip(utils.reg_format(test_stars),
                                        matched_psfs)]).reshape(-1, 1, 1)
            matched_psfs *= norm_factor

        return matched_psfs

    def _loc_transform(self, A_loc_weights, ccd_idx):
        r"""Transform local weights to local contributions of the PSFs."""
        return self.S[ccd_idx].dot(A_loc_weights)

    def _glob_transform(self, A_glob_weights):
        r"""Transform global weights to global contribution of the PSFs."""
        return self.S[-1].dot(A_glob_weights)
