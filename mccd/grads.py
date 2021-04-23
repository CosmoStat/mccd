# -*- coding: utf-8 -*-

r"""GRADIENTS.

Defines the gradient classes that will be used in the optimization
procedures from the ModOpt package.

: Authors: Tobias Liaudat <tobiasliaudat@gmail.com>

           Morgan Schmitz <https://github.com/MorganSchmitz>

           Jerome Bonnin <https://github.com/jerome-bonnin>

"""

from __future__ import absolute_import, print_function
import numpy as np
from modopt.opt.gradient import GradParent
from modopt.math.matrix import PowerMethod
from modopt.signal.wavelet import filter_convolve
import mccd.utils as utils


class CoeffLocGrad(GradParent, PowerMethod):
    r"""Gradient class for the local coefficient update.

    Local Alpha, :math:`\\alpha_{k}`.

    Parameters
    ----------
    data: numpy.ndarray
        Observed data.
    weights: numpy.ndarray
        Corresponding pixel-wise weights.
    S: numpy.ndarray
        Current eigenPSFs :math:`S`.
    VT: numpy.ndarray
        Matrix of spatial constraint enforcement (in the MCCD-RCA case will
        be the matrix of concatenated graph Laplacians.)
    H_glob: numpy.ndarray
        Current estimation of the global model.
    flux: numpy.ndarray
        Per-object flux value.
    sig: numpy.ndarray
        Noise levels.
    ker: numpy.ndarray
        Shifting kernels.
    ker_rot: numpy.ndarray
        Inverted shifting kernels.
    SNR_weights: numpy.ndarray
        Array of per star weights.
    D: float
        Upsampling factor.
    save_iter_cost: bool
        To save iteration diagnostic data.
        Default is ``False``.
    data_type: str
        Data type to be used.
        Default is ``float``.
    """

    def __init__(self, data, weights, S, VT, H_glob, flux, sig, ker,
                 ker_rot, SNR_weights, D, save_iter_cost=False,
                 data_type='float', verbose=True):
        r"""Initialize class attributes."""
        self.verbose = verbose
        self._grad_data_type = data_type
        self.obs_data = data
        self.obs_weights = weights
        self.op = self.MX
        self.trans_op = self.MtX
        self.VT = VT
        self.H_glob = H_glob
        self.flux = flux
        self.sig = sig
        self.normfacs = self.flux / (np.median(self.flux) * self.sig)
        self.ker = ker
        self.ker_rot = ker_rot
        self.D = D
        self.SNR_weights = SNR_weights
        self.iter_cost = []
        self.save_iter_cost = save_iter_cost

        self.S = None
        self.FdS = None
        self.FdH_glob = None

        PowerMethod.__init__(self, self.trans_op_op,
                             (S.shape[-1], VT.shape[0]), auto_run=False)
        self.update_S(np.copy(S), update_spectral_radius=False)

        self._current_rec = None

    def reset_iter_cost(self):
        r"""Reset iteration cost."""
        self.iter_cost = []

    def get_iter_cost(self):
        r"""Get current iteration cost."""
        return self.iter_cost

    def update_S(self, new_S, update_spectral_radius=True):
        r"""Update current eigenPSFs."""
        self.S = new_S
        self.FdS = np.array([[nf * utils.degradation_op(S_j, shift_ker, self.D)
                              for nf, shift_ker in
                              zip(self.normfacs, utils.reg_format(self.ker))]
                             for S_j in utils.reg_format(self.S)])
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)

    def update_H_glob(self, new_H_glob):
        r"""Update current global model."""
        self.H_glob = new_H_glob
        dec_H_glob = np.array(
            [nf * utils.degradation_op(H_i, shift_ker, self.D)
             for nf, shift_ker, H_i in
             zip(self.normfacs,
                 utils.reg_format(self.ker),
                 utils.reg_format(self.H_glob))])
        self.FdH_glob = utils.rca_format(dec_H_glob)

    def MX(self, alpha):
        r"""Apply degradation operator and renormalize.

        Parameters
        ----------
        alpha: numpy.ndarray
            Current coefficients (after factorization by :math:`V^{\\top}`).
        """
        A = alpha.dot(self.VT)
        dec_rec = np.empty(self.obs_data.shape)
        for j in range(dec_rec.shape[-1]):
            dec_rec[:, :, j] = np.sum(A[:, j].reshape(-1, 1, 1) *
                                      self.FdS[:, j], axis=0)
        self._current_rec = dec_rec
        return self._current_rec

    def MtX(self, x):
        r"""Adjoint to degradation operator :func:`MX`.

        Parameters
        ----------
        x : numpy.ndarray
            Set of finer-grid images.
        """
        x = utils.reg_format(x * self.SNR_weights)  # [TL] CHECK
        STx = np.array([np.sum(FdS_i * x, axis=(1, 2)) for FdS_i in self.FdS])
        return STx.dot(self.VT.T)

    def cost(self, x, y=None, verbose=False):
        r"""Compute data fidelity term.

        Notes
        -----
        ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat``
        can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(
            self.obs_weights * (self._current_rec + self.FdH_glob -
                                self.obs_data) * self.SNR_weights) ** 2
        return cost_val

    def get_grad(self, x):
        r"""Compute current iteration's gradient."""
        self.grad = self.MtX(self.obs_weights ** 2 *
                             (self.MX(x) + self.FdH_glob - self.obs_data))
        if self.save_iter_cost:
            self.iter_cost.append(self.cost(x))


class CoeffGlobGrad(GradParent, PowerMethod):
    r"""Gradient class for the local coefficient update.

    Global Alpha, :math: \\tilde{\\alpha}`.

    Parameters
    ----------
    data: numpy.ndarray
        Observed data.
    weights: numpy.ndarray
        Corresponding pixel-wise weights.
    S: numpy.ndarray
        Current eigenPSFs :math:`S`.
    Pi: numpy.ndarray
        Matrix of positions polynomials.
    H_loc: numpy.ndarray
        Current estimation of the local model
    flux: numpy.ndarray
        Per-object flux value.
    sig: numpy.ndarray
        Noise levels.
    ker: numpy.ndarray
        Shifting kernels.
    ker_rot: numpy.ndarray
        Inverted shifting kernels.
    SNR_weights: numpy.ndarray
        Array of per star weights.
    D: float
        Upsampling factor.
    save_iter_cost: bool
        To save iteration diagnostic data.
        Default is ``False``.
    data_type: str
        Data type to be used.
        Default is ``float``.
    """

    def __init__(self, data, weights, S, Pi, H_loc, flux, sig, ker,
                 ker_rot, D, SNR_weights, save_iter_cost=False,
                 data_type='float', verbose=True):
        r"""Initialize class attributes."""
        self.verbose = verbose
        self._grad_data_type = data_type
        self.obs_data = data
        self.obs_weights = weights
        self.op = self.MX
        self.trans_op = self.MtX
        self.Pi = Pi
        self.H_loc = H_loc
        self.flux = flux
        self.sig = sig
        self.normfacs = self.flux / (np.median(self.flux) * self.sig)
        self.ker = ker
        self.ker_rot = ker_rot
        self.D = D
        self.SNR_weights = SNR_weights
        self.iter_cost = []
        self.save_iter_cost = save_iter_cost
        self.S = None
        self.FdS = None
        self.FdH_loc = None

        PowerMethod.__init__(self, self.trans_op_op,
                             (S.shape[-1], Pi.shape[0]), auto_run=False)
        self.update_S(np.copy(S), update_spectral_radius=False)

        self._current_rec = None

    def reset_iter_cost(self):
        r"""Reset iteration cost."""
        self.iter_cost = []

    def get_iter_cost(self):
        r"""Get current iteration cost."""
        return self.iter_cost

    def update_S(self, new_S, update_spectral_radius=True):
        r"""Update current eigenPSFs."""
        self.S = new_S
        self.FdS = np.array([[nf * utils.degradation_op(S_j, shift_ker, self.D)
                              for nf, shift_ker in
                              zip(self.normfacs, utils.reg_format(self.ker))]
                             for S_j in utils.reg_format(self.S)])
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)

    def update_H_loc(self, new_H_loc):
        r"""Update current local models."""
        self.H_loc = new_H_loc
        dec_H_loc = np.array([nf * utils.degradation_op(H_i, shift_ker, self.D)
                              for nf, shift_ker, H_i in
                              zip(self.normfacs,
                                  utils.reg_format(self.ker),
                                  utils.reg_format(self.H_loc))])
        self.FdH_loc = utils.rca_format(dec_H_loc)

    def MX(self, alpha):
        r"""Apply degradation operator and renormalize.

        Parameters
        ----------
        alpha: numpy.ndarray
            Current coefficients (after factorization by :math:`\\Pi`).
        """
        A = alpha.dot(self.Pi)
        dec_rec = np.empty(self.obs_data.shape)
        for j in range(dec_rec.shape[-1]):
            dec_rec[:, :, j] = np.sum(A[:, j].reshape(-1, 1, 1) *
                                      self.FdS[:, j], axis=0)
        self._current_rec = dec_rec
        return self._current_rec

    def MtX(self, x):
        r"""Adjoint to degradation operator :func:`MX`.

        Parameters
        ----------
        x : numpy.ndarray
            Set of finer-grid images.
        """
        x = utils.reg_format(x * self.SNR_weights)  # [TL] CHECK
        STx = np.array([np.sum(FdS_i * x, axis=(1, 2)) for FdS_i in self.FdS])
        return STx.dot(self.Pi.T)

    def cost(self, x, y=None, verbose=False):
        r"""Compute data fidelity term.

        Notes
        -----
        ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat``
        can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(
            self.obs_weights * (self._current_rec + self.FdH_loc -
                                self.obs_data) * self.SNR_weights) ** 2
        return cost_val

    def get_grad(self, x):
        r"""Compute current iteration's gradient."""
        self.grad = self.MtX(self.obs_weights ** 2 *
                             (self.MX(x) + self.FdH_loc - self.obs_data))
        if self.save_iter_cost:
            self.iter_cost.append(self.cost(x))


class SourceLocGrad(GradParent, PowerMethod):
    r"""Gradient class for the local eigenPSF update.

    Local S, :math:`S_{k}`.

    Parameters
    ----------
    data: numpy.ndarray
        Input data array, a array of 2D observed images (i.e. with noise).
    weights: numpy.ndarray
        Corresponding pixel-wise weights.
    A: numpy.ndarray
        Current estimation of corresponding coefficients.
    H_glob: numpy.ndarray
        Current estimation of the global model
    flux: numpy.ndarray
        Per-object flux value.
    sig: numpy.ndarray
        Noise levels.
    ker: numpy.ndarray
        Shifting kernels.
    ker_rot: numpy.ndarray
        Inverted shifting kernels.
    D: float
        Upsampling factor.
    filters: numpy.ndarray
        Set of filters.
    save_iter_cost: bool
        To save iteration diagnostic data.
        Default is ``False``.
    data_type: str
        Data type to be used.
        Default is ``float``.
    """

    def __init__(self, data, weights, A, H_glob, flux, sig, ker, ker_rot,
                 SNR_weights, D, filters, save_iter_cost=False,
                 data_type='float', verbose=True):
        r"""Initialize class attributes."""
        self.verbose = verbose
        self._grad_data_type = data_type
        self.obs_data = data
        self.obs_weights = weights
        self.op = self.MX
        self.trans_op = self.MtX
        self.A = np.copy(A)
        self.H_glob = np.copy(H_glob)
        self.flux = flux
        self.sig = sig
        self.normfacs = self.flux / (np.median(self.flux) * self.sig)
        self.ker = ker
        self.ker_rot = ker_rot
        self.D = D
        self.filters = filters
        self.SNR_weights = SNR_weights
        self.iter_cost = []
        self.save_iter_cost = save_iter_cost
        self.FdH_glob = None

        hr_shape = np.array(data.shape[:2]) * D
        PowerMethod.__init__(self, self.trans_op_op,
                             (A.shape[0], filters.shape[0]) + tuple(hr_shape),
                             auto_run=False)

        self._current_rec = None

    def reset_iter_cost(self):
        r"""Reset iteration cost."""
        self.iter_cost = []

    def get_iter_cost(self):
        r"""Get current iteration cost."""
        return self.iter_cost

    def update_A(self, new_A, update_spectral_radius=True):
        r"""Update current coefficients."""
        self.A = new_A
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)

    def update_H_glob(self, new_H_glob):
        r"""Update current global model."""
        self.H_glob = new_H_glob
        dec_H_glob = np.array(
            [nf * utils.degradation_op(H_i, shift_ker, self.D)
             for nf, shift_ker, H_i in zip(self.normfacs,
                                           utils.reg_format(self.ker),
                                           utils.reg_format(self.H_glob))])
        self.FdH_glob = utils.rca_format(dec_H_glob)

    def MX(self, transf_S):
        r"""Apply degradation operator and renormalize.

        Parameters
        ----------
        transf_S : numpy.ndarray
            Current eigenPSFs in wavelet (by default Starlet) space.

        Returns
        -------
        numpy.ndarray result

        """
        S = utils.rca_format(
            np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                      for transf_Sj in transf_S]))
        dec_rec = np.array(
            [nf * utils.degradation_op(S.dot(A_i), shift_ker, self.D)
             for nf, A_i, shift_ker in zip(self.normfacs,
                                           self.A.T,
                                           utils.reg_format(self.ker))])
        self._current_rec = utils.rca_format(dec_rec)
        return self._current_rec

    def MtX(self, x):
        r"""Adjoint to degradation operator :func:`MX`."""
        x = utils.reg_format(x * self.SNR_weights)
        upsamp_x = np.array(
            [nf * utils.adjoint_degradation_op(x_i, shift_ker, self.D)
             for nf, x_i, shift_ker in zip(self.normfacs,
                                           x,
                                           utils.reg_format(self.ker_rot))])
        x, upsamp_x = utils.rca_format(x), utils.rca_format(upsamp_x)
        return utils.apply_transform(upsamp_x.dot(self.A.T), self.filters)

    def cost(self, x, y=None, verbose=False):
        r"""Compute data fidelity term.

        Notes
        -----
        ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat``
        can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(
            self.obs_weights * (self._current_rec + self.FdH_glob -
                                self.obs_data) * self.SNR_weights) ** 2

        return cost_val

    def get_grad(self, x):
        r"""Compute current iteration's gradient."""
        self.grad = self.MtX(self.obs_weights ** 2 *
                             (self.MX(x) + self.FdH_glob - self.obs_data))
        if self.save_iter_cost:
            self.iter_cost.append(self.cost(x))


class SourceGlobGrad(GradParent, PowerMethod):
    r"""Gradient class for the global eigenPSF update.

    Global S, :math:`\\tilde{S}`.

    Parameters
    ----------
    data: numpy.ndarray
        Input data array, a array of 2D observed images (i.e. with noise).
    weights: numpy.ndarray
        Corresponding pixel-wise weights.
    A: numpy.ndarray
        Current estimation of corresponding coefficients.
    H_loc: numpy.ndarray
        Current estimation of the local models
    flux: numpy.ndarray
        Per-object flux value.
    sig: numpy.ndarray
        Noise levels.
    ker: numpy.ndarray
        Shifting kernels.
    ker_rot: numpy.ndarray
        Inverted shifting kernels.
    D: float
        Upsampling factor.
    filters: numpy.ndarray
        Set of filters.
    save_iter_cost: bool
        To save iteration diagnostic data.
        Default is ``False``.
    data_type: str
        Data type to be used.
        Default is ``float``.
    """

    def __init__(self, data, weights, A, H_loc, flux, sig,
                 ker, ker_rot, SNR_weights, D, filters, save_iter_cost=False,
                 data_type='float', verbose=True):
        r"""Initialize class attributes."""
        self.verbose = verbose
        self._grad_data_type = data_type
        self.obs_data = data
        self.obs_weights = weights
        self.op = self.MX
        self.trans_op = self.MtX
        self.A = np.copy(A)
        self.H_loc = np.copy(H_loc)
        self.flux = flux
        self.sig = sig
        self.normfacs = self.flux / (np.median(self.flux) * self.sig)
        self.ker = ker
        self.ker_rot = ker_rot
        self.D = D
        self.filters = filters
        self.SNR_weights = SNR_weights
        self.iter_cost = []
        self.save_iter_cost = save_iter_cost
        self.FdH_loc = None

        hr_shape = np.array(data.shape[:2]) * D
        PowerMethod.__init__(self, self.trans_op_op,
                             (A.shape[0], filters.shape[0]) + tuple(hr_shape),
                             auto_run=False)

        self._current_rec = None

    def reset_iter_cost(self):
        r"""Reset iteration cost."""
        self.iter_cost = []

    def get_iter_cost(self):
        r"""Get current iteration cost."""
        return self.iter_cost

    def update_A(self, new_A, update_spectral_radius=True):
        r"""Update current coefficients."""
        self.A = new_A
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)

    def update_H_loc(self, new_H_loc):
        r"""Update current local models."""
        self.H_loc = new_H_loc
        dec_H_loc = np.array(
            [nf * utils.degradation_op(H_i, shift_ker, self.D)
             for nf, shift_ker, H_i in
             zip(self.normfacs,
                 utils.reg_format(self.ker),
                 utils.reg_format(self.H_loc))])
        self.FdH_loc = utils.rca_format(dec_H_loc)

    def MX(self, transf_S):
        r"""Apply degradation operator and renormalize.

        Parameters
        ----------
        transf_S : numpy.ndarray
            Current eigenPSFs in Starlet space.

        Returns
        -------
        numpy.ndarray result

        """
        S = utils.rca_format(
            np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                      for transf_Sj in transf_S]))
        dec_rec = np.array(
            [nf * utils.degradation_op(S.dot(A_i), shift_ker, self.D)
             for nf, A_i, shift_ker in zip(self.normfacs,
                                           self.A.T,
                                           utils.reg_format(self.ker))])
        self._current_rec = utils.rca_format(dec_rec)
        return self._current_rec

    def MtX(self, x):
        r"""Adjoint to degradation operator :func:`MX`."""
        x = utils.reg_format(x * self.SNR_weights)
        upsamp_x = np.array(
            [nf * utils.adjoint_degradation_op(x_i, shift_ker, self.D) for
             nf, x_i, shift_ker
             in zip(self.normfacs, x, utils.reg_format(self.ker_rot))])
        x, upsamp_x = utils.rca_format(x), utils.rca_format(upsamp_x)
        return utils.apply_transform(upsamp_x.dot(self.A.T), self.filters)

    def cost(self, x, y=None, verbose=False):
        r"""Compute data fidelity term.

        Notes
        -----
        ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat``
        can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(
            self.obs_weights * (
                    self._current_rec + self.FdH_loc - self.obs_data) *
            self.SNR_weights) ** 2
        return cost_val

    def get_grad(self, x):
        r"""Compute current iteration's gradient."""
        self.grad = self.MtX(self.obs_weights ** 2 * (
                self.MX(x) + self.FdH_loc - self.obs_data))
        if self.save_iter_cost:
            self.iter_cost.append(self.cost(x))
