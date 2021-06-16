# -*- coding: utf-8 -*-

r"""PROXIMAL OPERATORS.

Defines proximal operators to be fed to ModOpt algorithm that are
specific to MCCD(or rather, not currently in ``modopt.opt.proximity``).

: Authors: Tobias Liaudat <tobiasliaudat@gmail.com>,
           Morgan Schmitz <github @MorganSchmitz>

"""

from __future__ import absolute_import, print_function
import numpy as np
from modopt.signal.wavelet import filter_convolve
from modopt.opt.proximity import ProximityParent
import mccd.utils as utils


class LinRecombine(ProximityParent):
    r"""Multiply eigenvectors ``S`` and (factorized) weights ``A``.

    Maintain the knowledge about the linear operator norm which is calculated
    as the spectral norm (highest eigenvalue of the matrix).
    The recombination is done with ``S`` living in the tranformed domain.

    Parameters
    ----------
    A: numpy.ndarray
        Matrix defining the linear operator.
    filters: numpy.ndarray
        Filters used by the wavelet transform.
    compute_norm: bool
        Computation of the matrix spectral radius in the initialization.
    """

    def __init__(self, A, filters, compute_norm=False):
        r"""Initialize class attributes."""
        self.A = A
        self.op = self.recombine
        self.adj_op = self.adj_rec
        self.filters = filters
        if compute_norm:
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T), full_matrices=False)
            self.norm = np.sqrt(s[0])

    def recombine(self, transf_S):
        r"""Recombine new S and return it."""
        S = np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                      for transf_Sj in transf_S])
        return utils.rca_format(S).dot(self.A)

    def adj_rec(self, Y):
        r"""Return the adjoint operator of ``recombine``."""
        return utils.apply_transform(Y.dot(self.A.T), self.filters)

    def update_A(self, new_A, update_norm=True):
        r"""Update the ``A`` matrix.

        Also calculate the operator norm of A.
        """
        self.A = new_A
        if update_norm:
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T), full_matrices=False)
            self.norm = np.sqrt(s[0])


class KThreshold(ProximityParent):
    r"""Define linewise hard-thresholding operator with variable thresholds.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep
        in each line at each iteration.
    """

    def __init__(self, iter_func):
        r"""Initialize class attributes."""
        self.iter_func = iter_func
        self.iter = 0

    def reset_iter(self):
        r"""Set iteration counter to zero."""
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        r"""Return input data after thresholding."""
        self.iter += 1

        return utils.lineskthresholding(data, self.iter_func(self.iter,
                                                             data.shape[1]))

    def cost(self, x):
        r"""Return cost.

        (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0


class StarletThreshold(ProximityParent):
    r"""Apply soft thresholding in wavelet(default Starlet) domain.

    Parameters
    ----------
    threshold: numpy.ndarray
        Threshold levels.
    thresh_type: str
        Whether soft- or hard-thresholding should be used.
        Default is ``'soft'``.
    """

    def __init__(self, threshold, thresh_type='soft'):
        r"""Initialize class attributes."""
        self.threshold = threshold
        self._thresh_type = thresh_type

    def update_threshold(self, new_threshold, new_thresh_type=None):
        r"""Update starlet threshold."""
        self.threshold = new_threshold
        if new_thresh_type in ['soft', 'hard']:
            self._thresh_type = new_thresh_type

    def op(self, transf_data, **kwargs):
        r"""Apply wavelet transform and perform thresholding."""
        # Threshold all scales but the coarse
        transf_data[:, :-1] = utils.SoftThresholding(transf_data[:, :-1],
                                                     self.threshold[:, :-1])
        return transf_data

    def cost(self, x, y):
        r"""Return cost."""
        return 0


class proxNormalization(ProximityParent):
    r"""Normalize rows or columns of :math:`x` relatively to L2 norm.

    Parameters
    ----------
    type: str
        String defining the axis to normalize. If is `lines`` or ``columns``.
        Default is ``columns``.
    """

    def __init__(self, type='columns'):
        r"""Initialize class attributes."""
        self.op = self.normalize
        self.type = type

    def normalize(self, x, extra_factor=1.0):
        r"""Apply normalization.

        Following the prefered type.
        """
        # if self.type == 'lines':
        #     x_norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
        # else:
        #     x_norm = np.linalg.norm(x, axis=0).reshape(1, -1)

        # return x / x_norm

        # Not using a prox normalization as it is constraining the model
        # too strong.
        return x

    def cost(self, x):
        r"""Return cost."""
        return 0


class PositityOff(ProximityParent):
    r"""Project to the positive subset, taking into acount an offset."""

    def __init__(self, offset):
        r"""Initialize class attibutes."""
        self.offset = offset
        self.op = self.off_positive_part

    def update_offset(self, new_offset):
        r"""Update the offset value."""
        self.offset = new_offset

    def off_positive_part(self, x, extra_factor=1.0):
        r"""Perform the projection accounting for the offset."""
        prox_x = np.zeros(x.shape)
        pos_idx = (x > - self.offset)
        neg_idx = np.array(1 - pos_idx).astype(bool)
        prox_x[pos_idx] = x[pos_idx]
        prox_x[neg_idx] = - self.offset[neg_idx]
        return prox_x

    def cost(self, x):
        r"""Return cost."""
        return 0


class LinRecombineAlpha(ProximityParent):
    r"""Compute alpha recombination.

    Multiply alpha and VT/Pi matrices (in this function named M) and
    compute the operator norm.
    """

    def __init__(self, M):
        r"""Initialize class attributes."""
        self.M = M
        self.op = self.recombine
        self.adj_op = self.adj_rec

        U, s, Vt = np.linalg.svd(self.M.dot(self.M.T), full_matrices=False)
        self.norm = np.sqrt(s[0])

    def recombine(self, x):
        r"""Return recombination."""
        return x.dot(self.M)

    def adj_rec(self, y):
        r"""Return adjoint recombination."""
        return y.dot(self.M.T)


class GMCAlikeProxL1(ProximityParent):
    """Classic l1 prox with GMCA-like decreasing weighting values.

    GMCA stand for Generalized Morphological Component Analysis.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep
        in each line at each iteration.

    Notes
    -----
    Not being used by the MCCD algorithm for the moment.
    """

    def __init__(self, iter_func, kmax):
        r"""Initialize class attributes."""
        self.iter_func = iter_func
        self.iter = 0
        self.iter_max = kmax

    def reset_iter(self):
        r"""Set iteration counter to zero."""
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        r"""Return input data after thresholding."""
        self.iter += 1
        return self.op_tobi_prox_l1(data, self.iter, self.iter_max)

    def op_tobi_prox_l1(self, mat, k, kmax):
        r"""Apply GMCA hard-thresholding to each line of input matrix."""
        mat_out = np.copy(mat)
        shap = mat.shape
        for j in range(0, shap[0]):
            # GMCA-like threshold calculation
            line = mat_out[j, :]
            idx = np.floor(
                len(line) * np.max([0.9 - (k / kmax) * 3, 0.2])).astype(int)
            idx_thr = np.argsort(abs(line))[idx]
            thresh = abs(line[idx_thr])

            # Linear norm_inf decrease
            # thresh = np.max(mat_out[j,:])*np.max([0.9-(k/kmax)*3,0.2])
            # mat_out[j,:] = utils.SoftThresholding(mat[j,:],thresh)
            mat_out[j, :] = self.HardThresholding(mat_out[j, :], thresh)
        return mat_out

    @staticmethod
    def HardThresholding(data, thresh):
        r"""Perform element-wise hard thresholding."""
        data[data < thresh] = 0.
        return data

    def cost(self, x):
        r"""Cost function. To do."""
        return 0


class ClassicProxL2(ProximityParent):
    r"""This class defines the classic l2 prox.

    Notes
    -----
    ``prox_weights``: Corresponds to the weights of the weighted norm l_{w,2}.
    They are set by default to ones. Not being used in this implementation.
    ``beta_param``: Corresponds to the beta (or lambda) parameter that goes
    with the fucn tion we will calculate the prox on prox_{lambda f(.)}(y).
    ``iter``: Iteration number, just to follow track of the iterations.
    It could be part of the lambda update strategy for the prox calculation.

    Reference: « Mixed-norm estimates for the M/EEG inverse problem using
    accelerated gradient methods
    Alexandre Gramfort, Matthieu Kowalski, Matti Hämäläinen »
    """

    def __init__(self):
        r"""Initialize class attributes."""
        self.beta_param = 0
        self.iter = 0

    def set_beta_param(self, beta_param):
        r"""Set ``beta_param``."""
        self.beta_param = beta_param

    def reset_iter(self):
        """Set iteration counter to zero."""
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        r"""Return input data after thresholding.

        The extra factor is the beta_param!
        Should be used on the proximal operator function.
        """
        self.iter += 1  # not used in this prox

        return self.op_tobi_prox_l2(data)

    def op_tobi_prox_l2(self, data):
        r"""Apply the opterator on the whole data matrix.

        for a vector: :math:`x = prox_{lambda || . ||^{2}_{w,2}}(y)`
        :math:`=> x_i = y_i /(1 + lambda w_i)`
        The operator can be used for the whole data matrix at once.
        """
        dividing_weight = 1. + self.beta_param

        return data / dividing_weight

    def cost(self, x):
        r"""Cost function. To do."""
        return 0
