# -*- coding: utf-8 -*-

r"""UTILS.

These functions include several functions needed by PSF modelling algorithms.

:Authors:   Morgan Schmitz <https://github.com/MorganSchmitz>

            Tobias Liaudat <tobias.liaudat@cea.fr>

"""

from __future__ import absolute_import, print_function
import scipy.signal as scisig
import numpy as np
from modopt.signal.wavelet import filter_convolve
import galsim as gs


def apply_transform(data, filters):
    r"""Transform ``data`` through application of a set of filters.

    Parameters
    ----------
    data: numpy.ndarray
        Data to be transformed. Should be in rca_format, where the image
        index is contained
        on last/2nd axis, ie (n_pix,n_pix,n_images).
    filters: numpy.ndarray
        Set of filters. Usually the wavelet transform filters.

    """
    data = reg_format(np.copy(data))
    return np.array([filter_convolve(im, filters) for im in data])


def acc_sig_maps(shap_im, ker_stack, sig_est, flux_est, flux_ref, upfact, w,
                 sig_data=None):
    r"""Apply acc_sig_map() several times.

    Calls:

    * :func:`utils.acc_sig_map`
    """
    shap = w.shape
    map_out = np.zeros((shap_im[0] * upfact, shap_im[1] * upfact, shap[0]))
    for i in range(0, shap[0]):
        map_out[:, :, i] = acc_sig_map(shap_im, ker_stack, sig_est,
                                       flux_est, flux_ref, upfact, w[i, :],
                                       sig_data=sig_data)
    return map_out


def acc_sig_map(shap_im, ker_stack, sig_est, flux_est, flux_ref, upfact,
                w, sig_data=None):
    r"""Estimate the simga noise maps from the observed data.

    Computes the square root of
    :math:`\mathcal{F}^{2*}(\hat\sigma^2)(A^\top\odot A^\top)`.
    See equation (27) in RCA paper (Ngole et al.).

    Notes
    -----
    :math:`\mathrm{Var}(B)` has been replaced by the noise level as
    estimated from the data, and here we do not have the term
    :math:`\mu` (gradient step size in the paper).
    """
    shap = ker_stack.shape
    nb_im = shap[2]
    if sig_data is None:
        sig_data = np.ones((nb_im,))
    var_stack = np.ones((shap_im[0], shap_im[1], nb_im))
    map2 = np.zeros((shap_im[0] * upfact, shap_im[1] * upfact))
    ker_stack_in = np.copy(ker_stack) ** 2
    for it in range(0, shap[2]):
        var_stack[:, :, it] *= sig_data[it] ** 2
        map2 += ((w[it] * flux_est[it] / (sig_est[it] * flux_ref)) ** 2) * \
            scisig.convolve(transpose_decim(var_stack[:, :, it], upfact),
                            ker_stack_in[:, :, it], mode='same')
    sigmap = np.sqrt(map2)
    return sigmap


def return_neighbors(new_pos, obs_pos, vals, n_neighbors):
    r"""Find the nearest neighbors locally in one ccd."""
    distances = np.linalg.norm(obs_pos - new_pos, axis=1)
    nbs = vals[np.argsort(distances)[:n_neighbors]]
    pos = obs_pos[np.argsort(distances)[:n_neighbors]]
    return nbs, pos


def rca_format(cube):
    r"""Switch from ``regular`` format to ``RCA`` format.

    RCA format: image index is contained on last axis [:,:,it]
    Regular format: image index is contained on first axis [it,:,:]
    """
    return cube.swapaxes(0, 1).swapaxes(1, 2)


def reg_format(rca_cube):
    r"""Switch from ``RCA`` format to ``regular`` format.

    RCA format: image index is contained on last axis [:,:,it]
    Regular format: image index is contained on first axis [it,:,:]
    """
    return rca_cube.swapaxes(2, 1).swapaxes(1, 0)


def decim(im, d, av_en=1, fft=1):
    r"""Decimate image to lower resolution."""
    im_filt = np.copy(im)
    im_d = np.copy(im)
    if d > 1:
        if av_en == 1:
            siz = d + 1 - (d % 2)
            mask = np.ones((siz, siz)) / siz ** 2
            if fft == 1:
                im_filt = scisig.fftconvolve(im, mask, mode='same')
            else:
                im_filt = scisig.convolve(im, mask, mode='same')
        n1 = int(np.floor(im.shape[0] / d))
        n2 = int(np.floor(im.shape[1] / d))
        im_d = np.zeros((n1, n2))
        i, j = 0, 0
        for i in range(0, n1):
            for j in range(0, n2):
                im_d[i, j] = im[i * d, j * d]
    if av_en == 1:
        return im_filt, im_d
    else:
        return im_d


def pairwise_distances(obs_pos):
    r"""Compute pairwise distances."""
    ones = np.ones(obs_pos.shape[0])
    out0 = np.outer(obs_pos[:, 0], ones)
    out1 = np.outer(obs_pos[:, 1], ones)
    return np.sqrt((out0 - out0.T) ** 2 + (out1 - out1.T) ** 2)


def transpose_decim(im, decim_fact, av_en=0):
    r"""Apply the transpose of the decimation matrix."""
    shap = im.shape
    im_out = np.zeros((shap[0] * decim_fact, shap[1] * decim_fact))

    for i in range(0, shap[0]):
        for j in range(0, shap[1]):
            im_out[decim_fact * i, decim_fact * j] = im[i, j]

    if av_en == 1:
        siz = decim_fact + 1 - (decim_fact % 2)
        mask = np.ones((siz, siz)) / siz ** 2
        im_out = scisig.fftconvolve(im, mask, mode='same')

    return im_out


def SoftThresholding(data, thresh):
    r"""Perform element-wise soft thresholding."""
    thresh_data = np.copy(data)
    belowmask = (np.abs(data) <= thresh)
    abovemask = np.array(1. - belowmask).astype(bool)
    thresh_data[belowmask] = 0.
    thresh_data[abovemask] = (data - np.sign(data) * thresh)[abovemask]
    return thresh_data


def HardThresholding(data, thresh):
    r"""Perform element-wise hard thresholding."""
    thresh_data = np.copy(data)
    thresh_data[thresh_data < thresh] = 0.
    return thresh_data


def kthresholding(x, k):
    r"""Apply k-thresholding.

    Keep only ``k`` highest values and set the rest to 0.
    """
    k = int(k)
    if k < 1:
        print("Warning: wrong k value for k-thresholding")
        k = 1
    if k > len(x):
        return x
    else:
        xout = np.copy(x) * 0
        ind = np.argsort(abs(x))
        xout[ind[-k:]] = x[ind[-k:]]
        return xout


def lineskthresholding(mat, k):
    r"""Apply k-thresholding to each line of input matrix.

    Calls:

    * :func:`utils.kthresholding`
    """
    mat_out = np.copy(mat)
    shap = mat.shape
    for j in range(0, shap[0]):
        mat_out[j, :] = kthresholding(mat[j, :], k)
    return mat_out


def mad(x, weight=None):
    r"""Compute MAD (Median Absolute Deviation)."""
    if weight is not None:
        valid_pixels = x[weight > 0]
    else:
        valid_pixels = x
    return np.median(np.abs(valid_pixels - np.median(valid_pixels)))


def transform_mask(weights, filt):
    r"""Propagate bad pixels to 1st wavelet scale and mask all pixels affected.

    Bad pixels are the ones with weight 0.
    """
    stamp_size = weights.shape[0]
    antimask = np.zeros(weights.shape)
    antimask[weights == 0] = 1
    kernel = np.where(filt != 0)[0]
    filt_radius = np.max(kernel) - np.min(kernel)
    bad_pix = np.where(antimask)
    for pixx, pixy, flagged_idx in zip(*bad_pix):
        lx = max(0, pixx - filt_radius)
        ly = max(0, pixy - filt_radius)
        rx = min(pixx + filt_radius, stamp_size)
        ry = min(pixy + filt_radius, stamp_size)
        antimask[lx:rx, ly:ry, flagged_idx] = 1

    mask = np.abs(antimask - 1)
    return mask


def lanczos(U, n=10, n2=None):
    r"""Generate Lanczos kernel for a given shift."""
    if n2 is None:
        n2 = n
    siz = np.size(U)

    if siz == 2:
        U_in = np.copy(U)
        if len(U.shape) == 1:
            U_in = np.zeros((1, 2))
            U_in[0, 0] = U[0]
            U_in[0, 1] = U[1]
        H = np.zeros((2 * n + 1, 2 * n2 + 1))
        if (U_in[0, 0] == 0) and (U_in[0, 1] == 0):
            H[n, n2] = 1
        else:
            i = 0
            j = 0
            for i in range(0, 2 * n + 1):
                for j in range(0, 2 * n2 + 1):
                    H[i, j] = np.sinc(U_in[0, 0] - (i - n)) * np.sinc(
                        (U_in[0, 0] - (i - n)) / n
                    ) * np.sinc(U_in[0, 1] - (j - n)) * np.sinc(
                        (U_in[0, 1] - (j - n)) / n)

    else:
        H = np.zeros((2 * n + 1,))
        for i in range(0, 2 * n):
            H[i] = np.sinc(np.pi * (U - (i - n))) * np.sinc(
                np.pi * (U - (i - n)) / n)
    return H


def flux_estimate(im, cent=None, sigma=4):
    r"""Estimate flux for one image.

    Parameters
    ----------
    im: numpy.ndarray
        Image stamp containing the star.
    cent: numpy.ndarray
        Centroid of the star. If not provided, the centroid is calculated.
        Default is None.
    sigma: float
        Size of the star in sigma that will be used to calculate the flux and
        possibly the centroid too.
        Default is 4.

    Returns
    -------
    flux: float
        Photometric flux value of the star.

    Notes
    -----
    See SPRITE paper (Ngole et al.), section 3.4.1.,
    subsection 'Photometric flux'.
    """
    flux = 0
    rad = 3. * sigma
    if cent is None:
        CE = CentroidEstimator(im, sig=sigma)
        CE.estimate()
        cent = [CE.xc, CE.yc]
    shap = im.shape
    for i in range(0, shap[0]):
        for j in range(0, shap[1]):
            if np.sqrt((i - cent[0]) ** 2 + (j - cent[1]) ** 2) <= rad:
                flux = flux + im[i, j]
    return flux


def flux_estimate_stack(stack, cent=None, sigmas=2.):
    r"""Estimate flux for a bunch of images.

    Calls:

    * :func:`utils.flux_estimate`
    """
    shap = stack.shape
    flux = np.zeros((shap[2],))

    if np.isscalar(sigmas):
        sigmas = sigmas * np.ones((shap[2],))

    for i in range(0, shap[2]):
        if cent is not None:
            flux[i] = flux_estimate(stack[:, :, i], cent=cent[i, :],
                                    sigma=sigmas[i])
        else:
            flux[i] = flux_estimate(stack[:, :, i], sigma=sigmas[i])
    return flux


def shift_ker_stack(shifts, upfact, lanc_rad=8):
    r"""Generate shifting kernels and rotated shifting kernels.

    Calls:

    * :func:`utils.lanczos`
    """
    # lanc_rad = np.ceil(np.max(3*sigmas)).astype(int)
    shap = shifts.shape
    var_shift_ker_stack = np.zeros(
        (2 * lanc_rad + 1, 2 * lanc_rad + 1, shap[0]))
    var_shift_ker_stack_adj = np.zeros(
        (2 * lanc_rad + 1, 2 * lanc_rad + 1, shap[0]))

    for i in range(0, shap[0]):
        uin = shifts[i, :].reshape((1, 2)) * upfact
        var_shift_ker_stack[:, :, i] = lanczos(uin, n=lanc_rad)
        var_shift_ker_stack_adj[:, :, i] = np.rot90(
            var_shift_ker_stack[:, :, i], 2)

    return var_shift_ker_stack, var_shift_ker_stack_adj


def gen_Pea(distances, e, a):
    r"""Compute the graph Laplacian for a given set of parameters.

    Parameters
    ----------
    distances: numpy.ndarray
        Array of pairwise distances
    e: float
        Exponent to which the pairwise distances should be raised.
    a: float
        Constant multiplier along Laplacian's diagonal.

    Returns
    -------
    Pea: numpy.ndarray
        Graph laplacian.

    Notes
    -----
    Computes :math:`P_{e,a}` matrix for given ``e``, ``a`` couple.
    See Equations (16-17) in RCA paper (Ngole et al.).
    Watch out with the ``e`` parameter as it plays a vital role in the graph
    definition as it is a parameter of the distance that defines the
    graph's weights.

    """
    Pea = np.copy(distances ** e)
    np.fill_diagonal(Pea, 1.)
    Pea = -1. / Pea
    for i in range(Pea.shape[0]):
        Pea[i, i] = a * (np.sum(-1. * Pea[i]) - 1.)
    return Pea


def select_vstar(eigenvects, R, weights):
    r"""Pick best eigenvector from a set of :math:`(e,a)`.

    i.e., solve (35) from RCA paper (Ngole et al.).

    Parameters
    ----------
    eigenvects: numpy.ndarray
        Array of eigenvects to be tested over.
    R: numpy.ndarray
        :math:`R_i` matrix.
    weights: numpy.ndarray
        Entry-wise weights for :math:`R_i`.
    """
    loss = np.sum((weights * R) ** 2)
    for i, Pea_eigenvects in enumerate(eigenvects):
        for j, vect in enumerate(Pea_eigenvects):
            colvect = np.copy(vect).reshape(1, -1)
            current_loss = np.sum(
                (weights * R - colvect.T.dot(colvect.dot(weights * R))) ** 2)
            if current_loss < loss:
                loss = current_loss
                eigen_idx = j
                ea_idx = i
                best_VT = np.copy(Pea_eigenvects)

    return ea_idx, eigen_idx, best_VT


class GraphBuilder(object):
    r"""GraphBuilder class.

    This class computes the necessary quantities for RCA's graph constraint.

    Parameters
    ----------
    obs_data: numpy.ndarray
        Observed data.
    obs_pos: numpy.ndarray
        Corresponding positions.
    obs_weights: numpy.ndarray
        Corresponding per-pixel weights.
    n_comp: int
        Number of RCA components.
    n_eigenvects: int
        Maximum number of eigenvectors to consider per :math:`(e,a)` couple.
        Default is ``None``;
        if not provided, *all* eigenvectors will be considered,
        which can lead to a poor selection of graphs, especially when data
        is undersampled.
        Ignored if ``VT`` and ``alpha`` are provided.
    n_iter: int
        How many alternations should there be when optimizing over
        :math:`e` and :math:`a`. Default is 3.
    ea_gridsize: int
        How fine should the logscale grid of :math:`(e,a)` values be.
        Default is 10.
    distances: numpy.ndarray
        Pairwise distances for all positions. Default is ``None``;
        if not provided, will be computed from given positions.
    auto_run: bool
        Whether to immediately build the graph quantities.
        Default is ``True``.
    """

    def __init__(self, obs_data, obs_pos, obs_weights, n_comp,
                 n_eigenvects=None, n_iter=3,
                 ea_gridsize=10, distances=None, auto_run=True, verbose=2):
        r"""Initialize class attributes."""
        self.obs_data = obs_data
        shap = self.obs_data.shape
        self.obs_pos = obs_pos
        self.obs_weights = obs_weights
        # change to same format as that we will use for
        # residual matrix R later on
        self.obs_weights = np.transpose(
            self.obs_weights.reshape((shap[0] * shap[1], shap[2])))
        self.n_comp = n_comp
        if n_eigenvects is None:
            self.n_eigenvects = self.obs_data.shape[2]
        else:
            self.n_eigenvects = n_eigenvects
        self.n_iter = n_iter
        self.ea_gridsize = ea_gridsize
        if verbose > 1:
            self.verbose = True
        else:
            self.verbose = False

        if distances is None:
            self.distances = pairwise_distances(self.obs_pos)
        else:
            self.distances = distances
        if auto_run:
            self._build_graphs()

    def _build_graphs(self):
        r"""Compute graph-constraint related values.

        Notes
        -----
        See RCA paper (Ngole et al.) sections 5.2 and (especially) 5.5.3.
        """
        shap = self.obs_data.shape
        e_max = self.pick_emax()
        if self.verbose:
            print(" > power max = ", e_max)

        # [TL] Modif min from 0.01 to 0.001
        a_range = np.geomspace(0.001, 1.99, self.ea_gridsize)
        e_range = np.geomspace(0.01, e_max, self.ea_gridsize)

        # initialize R matrix with observations
        R = np.copy(
            np.transpose(self.obs_data.reshape((shap[0] * shap[1], shap[2]))))

        self.sel_a = []
        self.sel_e = []
        idx = []
        list_eigenvects = []
        for _ in range(self.n_comp):
            e, a, j, best_VT = self.select_params(R, e_range, a_range)
            self.sel_e += [e]
            self.sel_a += [a]
            idx += [j]
            list_eigenvects += [best_VT]
            vect = best_VT[j].reshape(1, -1)
            R -= vect.T.dot(vect.dot(R))
            if self.verbose:
                print(
                    " > selected e: {}\tselected a:".format(e) +
                    "{}\t chosen index: {}/{}".format(a, j, self.n_eigenvects))
        self.VT = np.vstack((eigenvect for eigenvect in list_eigenvects))
        self.alpha = np.zeros((self.n_comp, self.VT.shape[0]))
        for i in range(self.n_comp):
            self.alpha[i, i * self.n_eigenvects + idx[i]] = 1

    def pick_emax(self, epsilon=1e-15):
        r"""Pick maximum value for ``e`` parameter.

        From now, we fix the maximum :math:`e` to 1 and ignore the old
        procedure that was giving values that were too big.

        Old procedure:
        Select maximum value of :math:`e` for the greedy search over set of
        :math:`(e,a)` couples, so that the graph is still fully connected.
        """
        # nodiag = np.copy(self.distances)
        # nodiag[nodiag==0] = 1e20
        # dist_ratios = np.min(nodiag,axis=1) / np.max(self.distances, axis=1)
        # r_med = np.min(dist_ratios**2)
        # return np.log(epsilon)/np.log(r_med)

        return 1.

    def select_params(self, R, e_range, a_range):
        r"""Select best graph parameters.

        Select :math:`(e,a)` parameters and best eigenvector
        for current :math:`R_i` matrix.

        Parameters
        ----------
        R: numpy.ndarray
            Current :math:`R_i` matrix
            (as defined in RCA paper (Ngole et al.), sect. 5.5.3.)
        e_range: numpy.ndarray
            List of :math:`e` values to be tested.
        a_range: numpy.ndarray
            List of :math:`a` values to be tested.
        """
        current_a = 0.5
        for i in range(self.n_iter):
            # optimize over e
            Peas = np.array([gen_Pea(self.distances, e, current_a)
                             for e in e_range])
            all_eigenvects = np.array(
                [self.gen_eigenvects(Pea) for Pea in Peas])
            ea_idx, eigen_idx, _ = select_vstar(all_eigenvects, R,
                                                self.obs_weights)
            current_e = e_range[ea_idx]

            # optimize over a
            Peas = np.array([gen_Pea(self.distances, current_e, a)
                             for a in a_range])
            all_eigenvects = np.array(
                [self.gen_eigenvects(Pea) for Pea in Peas])
            ea_idx, eigen_idx, best_VT = select_vstar(all_eigenvects, R,
                                                      self.obs_weights)
            current_a = a_range[ea_idx]

        return current_e, current_a, eigen_idx, best_VT

    def gen_eigenvects(self, mat):
        r"""Compute input matrix's eigenvectors.

        Keep only the ``n_eigenvects`` associated
        with the smallest eigenvalues.
        """
        U, s, vT = np.linalg.svd(mat, full_matrices=True)
        vT = vT[-self.n_eigenvects:]
        return vT


def poly_pos(pos, max_degree, center_normalice=True,
             x_lims=None, y_lims=None,
             normalice_Pi=True, min_degree=None):
    r"""Construct polynomial matrix.

    Return a matrix Pi containing polynomials of stars
    positions up to ``max_degree``.

    Defaulting to CFIS CCD limits.

    New method:
    The positions are scaled to the [-0.5, 0.5]x[-0.5, 0.5].
    Then the polynomials are constructed with the normalized positions.

    Old method:
    Positions are centred, the polynomials are constructed.
    Then the polynomials are normalized.

    """
    n_mono = (max_degree + 1) * (max_degree + 2) // 2
    Pi = np.zeros((n_mono, pos.shape[0]))
    _pos = np.copy(pos)

    if x_lims is None:
        x_min = np.min(_pos[:, 0])
        x_max = np.max(_pos[:, 0])
        x_lims = [x_min, x_max]

    if y_lims is None:
        y_min = np.min(_pos[:, 1])
        y_max = np.max(_pos[:, 1])
        y_lims = [y_min, y_max]

    # Center and normalise positions
    if center_normalice:
        _pos[:, 0] = (_pos[:, 0] - x_lims[0]) / (x_lims[1] - x_lims[0]) - 0.5
        _pos[:, 1] = (_pos[:, 1] - y_lims[0]) / (y_lims[1] - y_lims[0]) - 0.5

    # Build position polynomials
    for d in range(max_degree + 1):
        row_idx = d * (d + 1) // 2
        for p in range(d + 1):
            Pi[row_idx + p, :] = _pos[:, 0] ** (d - p) * _pos[:, 1] ** p

    if min_degree is not None:
        # Erase the polynomial degrees up to `min_degree`
        # Monomials to erase
        del_n_mono = (min_degree + 1) * (min_degree + 2) // 2
        Pi = Pi[del_n_mono:, :]

    if normalice_Pi:
        # Normalize polynomial lines
        Pi_norms = np.sqrt(np.sum(Pi**2, axis=1))
        Pi /= Pi_norms.reshape(-1, 1)

    return Pi


class CentroidEstimator(object):
    r"""Estimate intra-pixel shifts.

    It calculates the centroid of the image and compare it with the stamp
    centroid and returns the proper shift.
    The star centroid is calculated following an iterative procedure where a
    matched elliptical gaussian is used to calculate the moments.

    Parameters
    ----------
    im: numpy.ndarray
        Star image stamp.
    sig: float
        Estimated shape of the star in sigma.
        Default is 7.5.
    n_iter: int
        Max iteration number for the iterative estimation procedure.
        Default is 5.
    auto_run: bool
        Auto run the intra-pixel shif calculation in the initialization
        of the class.
        Default is True.
    xc: float
        First guess of the ``x`` component of the star centroid. (optional)
        Default is None.
    yc: float
        First guess of the ``y`` component of the star centroid. (optional)
        Default is None.
    """

    def __init__(self, im, sig=7.5, n_iter=5, auto_run=True,
                 xc=None, yc=None):
        r"""Initialize class attributes."""
        self.im = im
        self.stamp_size = im.shape
        self.ranges = np.array([np.arange(i) for i in self.stamp_size])
        self.sig = sig
        self.n_iter = n_iter
        self.xc0, self.yc0 = float(self.stamp_size[0]) / 2, float(
            self.stamp_size[1]) / 2

        self.window = None
        self.xx = None
        self.yy = None

        if xc is None or yc is None:
            self.xc = self.xc0
            self.yc = self.yc0
        else:
            self.xc = xc
            self.yc = yc
        if auto_run:
            self.estimate()

    def UpdateGrid(self):
        r"""Update the grid where the star stamp is defined."""
        self.xx = np.outer(self.ranges[0] - self.xc,
                           np.ones(self.stamp_size[1]))
        self.yy = np.outer(np.ones(self.stamp_size[0]),
                           self.ranges[1] - self.yc)

    def EllipticalGaussian(self, e1=0, e2=0):
        r"""Compute an elliptical 2D gaussian with arbitrary centroid."""
        # Shear it
        gxx = (1 - e1) * self.xx - e2 * self.yy
        gyy = (1 + e1) * self.yy - e2 * self.xx
        # compute elliptical gaussian
        return np.exp(-(gxx ** 2 + gyy ** 2) / (2 * self.sig ** 2))

    def ComputeMoments(self):
        r"""Compute the star moments.

        Compute the star image normalized first order moments with
        the current window function.
        """
        Q0 = np.sum(self.im * self.window)
        Q1 = np.array(
            [np.sum(np.sum(self.im * self.window, axis=1 - i) * self.ranges[i])
             for i in range(2)])
        # Q2 = np.array([np.sum(
        #     self.im*self.window * self.xx**(2-i) * self.yy**i)
        #     for i in range(3)])
        self.xc = Q1[0] / Q0
        self.yc = Q1[1] / Q0

    def estimate(self):
        r"""Estimate the star image centroid iteratively."""
        for _ in range(self.n_iter):
            self.UpdateGrid()
            self.window = self.EllipticalGaussian()
            # Calculate weighted moments.
            self.ComputeMoments()
        return self.xc, self.yc

    def return_shifts(self):
        r"""Return intra-pixel shifts.

        Intra-pixel shifts are the difference between
        the estimated centroid and the center of the stamp (or pixel grid).
        """
        return [self.xc - self.xc0, self.yc - self.yc0]


def adjoint_degradation_op(x_i, shift_ker, D):
    r"""Apply adjoint of the degradation operator ``degradation_op``."""
    return scisig.fftconvolve(transpose_decim(x_i, D),
                              shift_ker, mode='same')


def degradation_op(X, shift_ker, D):
    r"""Shift and decimate fine-grid image."""
    return decim(scisig.fftconvolve(X, shift_ker, mode='same'),
                 D, av_en=0)


def handle_SExtractor_mask(stars, thresh):
    r"""Handle Sextractor masks.

    Reads SExtracted star stamps, generates MCCD-compatible masks
    (that is, binary weights), and replaces bad pixels with 0s -
    they will not be used by MCCD, but the ridiculous numerical
    values can otherwise still lead to problems because of convolutions.
    """
    mask = np.ones(stars.shape)
    mask[stars < thresh] = 0
    stars[stars < thresh] = 0
    return mask


def match_psfs(test_stars, PSFs):
    r"""Match psfs.DEPRECATED.

    See ``MCCD.validation_stars`` instead.
    Takes as input the test_stars vignets and the PSFs vignets that were
    outputs from the psf modelling method. The function outputs the PSFs
    matching the corresponding test stars.
    This allows to compute the pixel RMSE. Intended to be used with PSFEx
    validation functions.

    Parameters
    ----------
    test_stars: numpy.ndarray
        reg format (n_stars,n_pix,n_pix)
    PSFs: numpy.ndarray
        reg format (n_stars,n_pix,n_pix)

    Returns
    -------
    deg_PSFs: numpy.ndarray
        reg format (n_stars,n_pix,n_pix)
    """
    test_masks = handle_SExtractor_mask(test_stars, thresh=-1e5)
    psf_size_R2 = 6.
    psf_size = np.sqrt(psf_size_R2 / 2)

    test_stars = rca_format(test_stars)
    test_masks = rca_format(test_masks)
    PSFs = rca_format(PSFs)

    # Star calculation
    star_moms = [gs.hsm.FindAdaptiveMom(gs.Image(star), badpix=gs.Image(
        np.rint(np.abs(badpix - 1))),
                                        guess_sig=psf_size, strict=False) for
                 star, badpix in
                 zip(reg_format(test_stars), reg_format(test_masks))]
    sigmas = np.array([moms.moments_sigma for moms in star_moms])
    cents = [CentroidEstimator(test_stars[:, :, it], sig=sigmas[it]) for it in
             range(test_stars.shape[2])]
    test_shifts = np.array([ce.return_shifts() for ce in cents])

    # PSF calculation
    check_psf_moms = [gs.hsm.FindAdaptiveMom(gs.Image(star),
                                             guess_sig=psf_size, strict=False)
                      for star in reg_format(PSFs)]
    check_psf_sigmas = np.array(
        [moms.moments_sigma for moms in check_psf_moms])
    check_psf_cents = [
        CentroidEstimator(PSFs[:, :, it], sig=check_psf_sigmas[it])
        for it in range(PSFs.shape[2])]
    check_psf_test_shifts = np.array(
        [ce.return_shifts() for ce in check_psf_cents])

    # Final calculation
    test_shifts = test_shifts - check_psf_test_shifts
    lanc_rad = np.ceil(3. * np.max(sigmas)).astype(int)
    upfact = 1
    ntest = test_stars.shape[2]

    shift_kernels, _ = shift_ker_stack(test_shifts, upfact, lanc_rad=lanc_rad)

    deg_PSFs = np.array(
        [degradation_op(PSFs[:, :, j], shift_kernels[:, :, j], upfact)
         for j in range(ntest)])

    test_stars = reg_format(test_stars)

    # Optimize flux matching
    # (Changing the way the flux are defined for PSFEx) Instead of:
    # deg_PSFs *= test_fluxes.reshape(-1,1,1)
    # We will use:
    norm_factor = np.array(
        [np.sum(_star * _psf) / np.sum(_psf * _psf) for _star, _psf in
         zip(test_stars, deg_PSFs)]).reshape(-1, 1, 1)
    deg_PSFs *= norm_factor

    return deg_PSFs
