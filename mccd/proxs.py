# -*- coding: utf-8 -*-

r"""PROXIMAL OPERATORS

Defines proximal operators to be fed to ModOpt algorithm that are
specific to MCCD(or rather, not currently in ``modopt.opt.proximity``).

: Authors: Tobias Liaudat <tobiasliaudat@gmail.com>

"""

from __future__ import absolute_import, print_function
import numpy as np
from modopt.signal.noise import thresh
from modopt.opt.linear import LinearParent
from modopt.signal.wavelet import filter_convolve
from mccd_rca.utils import lineskthresholding, reg_format, rca_format,
from mccd_rca.utils import SoftThresholding, apply_transform


class LinRecombine(object):
    """ Multiply eigenvectors and (factorized) weights."""
    def __init__(self, A, filters, compute_norm=False):
        self.A = A
        self.op = self.recombine
        self.adj_op = self.adj_rec
        self.filters = filters
        if compute_norm:
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T),full_matrices=False)
            self.norm = np.sqrt(s[0])

    def recombine(self, transf_S):
        S = np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                      for transf_Sj in transf_S])
        return rca_format(S).dot(self.A)

    def adj_rec(self, Y):
        return apply_transform(Y.dot(self.A.T), self.filters)

    def update_A(self, new_A, update_norm=True):
        self.A = new_A
        if update_norm:
            U, s, Vt = np.linalg.svd(self.A.dot(self.A.T),full_matrices=False)
            self.norm = np.sqrt(s[0])


class KThreshold(object):
    """This class defines linewise hard-thresholding operator with variable thresholds.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration.
    """
    def __init__(self, iter_func):

        self.iter_func = iter_func
        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0


    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        """
        self.iter += 1
        # try:
        return lineskthresholding(data,self.iter_func(self.iter,data.shape[1]))
        # except :
        #     return lineskthresholding(data,self.iter_func(self.iter))

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class StarletThreshold(object):
    """Apply soft thresholding in Starlet domain.

    Parameters
    ----------
    threshold: np.ndarray
        Threshold levels.
    thresh_type: str
        Whether soft- or hard-thresholding should be used. Default is ``'soft'``.
    """
    def __init__(self, threshold, thresh_type='soft'):
        self.threshold = threshold
        self._thresh_type = thresh_type

    def update_threshold(self, new_threshold, new_thresh_type=None):
        self.threshold = new_threshold
        if new_thresh_type in ['soft', 'hard']:
            self._thresh_type = new_thresh_type

    def op(self, transf_data, **kwargs):
        """Applies Starlet transform and perform thresholding.
        """
        # Threshold all scales but the coarse
        transf_data[:,:-1] = SoftThresholding(transf_data[:,:-1], self.threshold[:,:-1])
        return transf_data

    def cost(self, x, y):
        return 0 #TODO



class proxNormalization(object):
    """Normalize rows of :math:`x` relatively to L2 norm"""
    def __init__(self,type = 'columns'):
        self.op = self.normalize
        self.type = type

    def normalize(self, x, extra_factor=1.0):
        if self.type == 'lines':
            x_norm = np.linalg.norm(x, axis=1).reshape(-1,1)
        else:
            x_norm = np.linalg.norm(x, axis=0).reshape(1,-1)

        return x / x_norm

    def cost(self, x):
        return 0

class PositityOff(object):
    """Project to the positive subset, taking into acount an offset"""
    def __init__(self, offset):
        self.offset = offset
        self.op = self.off_positive_part

    def update_offset(self, new_offset):
        self.offset = new_offset

    def off_positive_part(self, x, extra_factor=1.0):
        prox_x = np.zeros(x.shape)
        pos_idx = (x > - self.offset)
        neg_idx = np.array(1 - pos_idx).astype(bool)
        prox_x[pos_idx] = x[pos_idx]
        prox_x[neg_idx] = - self.offset[neg_idx]
        return prox_x

    def cost(self, x):
        return 0

class LinRecombineAlpha(object):
    """ Multiply alpha and VT/Pi (= M) matrices and compute the operator norm."""
    def __init__(self, M):
        self.M = M
        self.op = self.recombine
        self.adj_op = self.adj_rec

        U, s, Vt = np.linalg.svd(self.M.dot(self.M.T),full_matrices=False)
        self.norm = np.sqrt(s[0])

    def recombine(self, x):
        return x.dot(self.M)

    def adj_rec(self, y):
        return y.dot(self.M.T)

class identity_prox(object):
    """This class defines the classic l1 prox with GMCA-like decreasing weighting values.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration.
    """
    def __init__(self):

        self.iter = 0

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        """
        self.iter +=1
        return data


class tobi_prox_l1(object):
    """This class defines the classic l1 prox with GMCA-like decreasing weighting values.

    Parameters
    ----------
    iter_func: function
        Input function that calcultates the number of non-zero values to keep in each line at each iteration.
    """
    def __init__(self, iter_func, kmax):

        self.iter_func = iter_func
        self.iter = 0
        self.iter_max = kmax

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        """
        self.iter += 1
        return self.op_tobi_prox_l1(data,self.iter,self.iter_max)


    def op_tobi_prox_l1(self,mat,k,kmax):
        """ Applies GMCA-soft-thresholding to each line of input matrix.

        Calls:

        * :func:`utils.gmca_thresh`

        """
        mat_out = np.copy(mat)
        shap = mat.shape
        for j in range(0,shap[0]):
            # GMCA-like threshold calculation
            line = mat_out[j,:]
            idx = np.floor(len(line)*np.max([0.9-(k/kmax)*3,0.2])).astype(int)
            idx_thr = np.argsort(abs(line))[idx]
            thresh = abs(line[idx_thr])

            #thresh = np.max(mat_out[j,:])*np.max([0.9-(k/kmax)*3,0.2]) # Linear norm_inf decrease
            # mat_out[j,:] = SoftThresholding(mat[j,:],thresh)
            mat_out[j,:] = self.HardThresholding(mat_out[j,:],thresh)
        return mat_out

    def HardThresholding(self,data,thresh):
        """ Performs element-wise hard thresholding."""
        data[data < thresh] = 0.
        return data

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0

class tobi_prox_l2(object):
    """This class defines the classic l2 prox.
    « Mixed-norm estimates for the M/EEG inverse problem using accelerated gradient methods
Alexandre Gramfort, Matthieu Kowalski, Matti Hämäläinen »

    Parameters
    ----------
    prox_weights: Matrix
        Corresponds to the weights of the weighted norm l_{w,2}. They are set by default to ones.
    beta_param: float number
        Corresponds to the beta (or lambda) parameter that goes with the fucn tion we will
        calculate the prox on. prox_{lambda f(.)}(y)
    iter: Integer
        Iteration number, just to follow track of the iterations. It could be part of the lambda
        update strategy for the prox calculation.
    """
    def __init__(self):
        self.beta_param = 0
        self.iter = 0

    def set_beta_param(self,beta_param):
        self.beta_param=beta_param

    def reset_iter(self):
        """Set iteration counter to zero.
        """
        self.iter = 0

    def op(self, data, extra_factor=1.0):
        """Return input data after thresholding.
        The extra factor is the beta_param!!
        Should be used on the proximal operator function
        """
        # self.beta_param = extra_factor
        # print('Using L2 prox.. beta = %.5f it=%d\n'%(self.beta_param,self.iter))

        self.iter += 1 # not used in this prox

        result = self.op_tobi_prox_l2(data)
        # print('Prox results, it = %d'%(self.iter))
        # print(result)
        return result

    def op_tobi_prox_l2(self, data):
        """ Apply the opterator on the whole data matrix
        for a vector: x = prox_{lambda || . ||^{2}_{w,2}}(y) =>
        x_i = y_i /(1 + lambda w_i)
        The operator can be used for the whole data matrix at once.
        """
        dividing_weight =  1. + self.beta_param
        print('self.beta_param = %.5e'%(self.beta_param))
        print('RMS (data) = %.5e'%(np.sqrt(np.mean(data**2))))
        print('normL2(data) = %.5e'%(np.linalg.norm(data)))

        return data/dividing_weight

    def cost(self, x):
        """Returns 0 (Indicator of :math:`\Omega` is either 0 or infinity).
        """
        return 0
