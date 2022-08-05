"""
pylag.mlfit

Implements the method of Zoghbi et al. 2013 to fit correlation functions/covariance matrices
to light curves in the time domain to estimate power and lag spectra

v2.0 05/07/2019 - D.R. Wilkins
"""
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import minimize

import lmfit

from .lightcurve import *
from .binning import *
from .fit import *

class MLPSD(object):
    def __init__(self, lc, Nf=10, fbins=None, psd_model=None):
        self.time = np.array(lc.time)
        self.dt = np.min(np.diff(self.time))

        self.data = np.array(lc.rate)
        # the linear algebra we'll be doing requires the data points as a column vector (in matrix form)
        self.data = self.data[:, np.newaxis]
        # to fit a Gaussian process, we should make sure our data has zero mean
        self.data -= np.mean(self.data)

        # matrix of pairwise separations of time bins
        self.tau = squareform(pdist(np.array([[t] for t in self.time])))

        if fbins is None:
            # set up frequency bins to span
            min_freq = 0.5 / (np.max(self.time) - np.min(self.time))
            max_freq = 0.5 / self.dt
            self.fbins = LogBinning(min_freq, max_freq, Nf)
        else:
            self.fbins = fbins

        # RMS normalisation for PSD
        self.psdnorm = 2 * self.dt / (lc.mean() ** 2 * lc.length)

        # pre-compute the integral of the cosine term in the autocorrelation for each frequency bin
        # so we don't have to do this for every change in parameter values
        # note the factor of 2 to integrate over the negative frequencies too!
        self.cos_integral = np.array([np.array([(1. / (np.pi * t)) * (np.sin(2*np.pi*fmax*t) - np.sin(2*np.pi*fmin*t))
                                                if t > 0 else 2.*(fmax - fmin)
                                                for t in self.tau.flatten()]).reshape(self.tau.shape)
                                      for fmin, fmax in zip(self.fbins.bin_start, self.fbins.bin_end)])

        if psd_model is None:
            self.psd_model = None
        elif isinstance(psd_model, Model):
            self.psd_model = psd_model

        self.params = self.get_params()

    def cov_matrix(params):
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        psd = np.exp(params[:Nf])
        cov = np.sum(np.array([p * c for p, c in zip(psd, cos_integral)]), axis=0)
        return cov

    def cov_matrix_deriv(params, delta):
        psd = np.exp(params[:Nf])

        # in this simple case, the covariance matrix is just a linear sum of each frequency term
        # so the derivative is simple - we multiply by p when we're talking about the log
        return np.stack([c * p for c, p in zip(cos_integral, psd)], axis=-1)

    def log_likelihood(params, eval_gradient=True, delta=1e-3):
        c = self.cov_matrix(params)

        try:
            L = cholesky(c, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # if the matrix is sibgular, perturb the diagonal and try again
            L = cholesky(c + 1e-6 * np.eye(c.shape[0]), lower=True, check_finite=False)
            # return (-np.inf, np.zeros(len(params))) if eval_gradient else -np.inf
        except ValueError:
            return (-np.inf, np.zeros(len(params))) if eval_gradient else -np.inf

        alpha = cho_solve((L, True), self.data, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.data, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= c.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        if eval_gradient:
            c_gradient = self.cov_matrix_deriv(params, delta)
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)
            tmp -= cho_solve((L, True), np.eye(c.shape[0]))[:, :, np.newaxis]
            gradient_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, c_gradient)
            gradient = gradient_dims.sum(-1)

        # note we return -log_likelihood, so we can minimize it!
        return (-1 * log_likelihood, -1 * gradient) if eval_gradient else -1 * log_likelihood

    def get_params(self):
        if self.psd_model is None:
            params = lmfit.Parameters()
            for i in range(len(self.fbins)):
                params.add('%sln_psd%01d' % i, 1., vary=True, min=-50., max=50.)
            return params
        else:
            return self.psd_model.get_params()

    def fit(self, init_params=None):
        if init_params is None:
            init_params = self.params


