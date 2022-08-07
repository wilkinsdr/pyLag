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
import copy

from .lightcurve import *
from .binning import *
from .fit import *
from .plotter import *
from .model import *

class MLFit(object):
    def __init__(self, t, y, psdnorm=1., Nf=10, fbins=None, model=None, comp_name='', model_args={}):
        self.time = np.array(t)
        self.dt = np.min(np.diff(self.time))

        # the linear algebra we'll be doing requires the data points as a column vector (in matrix form)
        self.data = y[:, np.newaxis]

        self.psdnorm = psdnorm

        # matrix of pairwise separations of time bins
        self.tau = squareform(pdist(np.array([[t] for t in self.time])))

        if fbins is None:
            # set up frequency bins to span
            min_freq = 0.5 / (np.max(self.time) - np.min(self.time))
            max_freq = 0.5 / self.dt
            self.fbins = LogBinning(min_freq, max_freq, Nf)
        else:
            self.fbins = fbins

        self.freq = self.fbins.bin_cent
        self.freq_error = self.fbins.x_error()

        # pre-compute the integral of the cosine term in the autocorrelation for each frequency bin
        # so we don't have to do this for every change in parameter values
        # note the factor of 2 to integrate over the negative frequencies too!
        self.cos_integral = np.array([np.array([(1. / (np.pi * t)) * (np.sin(2*np.pi*fmax*t) - np.sin(2*np.pi*fmin*t))
                                                if t > 0 else (fmax - fmin)
                                                for t in self.tau.flatten()]).reshape(self.tau.shape)
                                      for fmin, fmax in zip(self.fbins.bin_start, self.fbins.bin_end)])

        if model is None:
            self.model = None
        elif isinstance(model, Model):
            self.model = model
        else:
            self.model = model(component_name=comp_name, **model_args)

        self.comp_name = comp_name
        self.params = self.get_params()

        self.fit_result = None

    def log_likelihood(self, params, eval_gradient=True, delta=1e-3):
        c = self.cov_matrix(params)

        try:
            L = cholesky(c, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            # try:
            # if the matrix is sibgular, perturb the diagonal and try again
            L = cholesky(c + 1e-3 * np.eye(c.shape[0]), lower=True, check_finite=False)
            # except np.linalg.LinAlgError:
            #     return (-np.inf, np.zeros(len(params))) if eval_gradient else -np.inf
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
        if self.model is None:
            params = lmfit.Parameters()
            for i in range(len(self.fbins)):
                params.add('%sln_psd%01d' % (self.comp_name, i), 1., vary=True, min=-30., max=30.)
            return params
        else:
            return self.model.get_params()

    def _dofit(self, init_params, method='L-BFGS-B', **kwargs):
        initial_par_arr = np.array([init_params[p].value for p in init_params if init_params[p].vary])
        bounds = [(init_params[p].min, init_params[p].max) for p in init_params if init_params[p].vary]

        def objective(par_arr):
            fit_params = copy.copy(init_params)
            for par, value in zip([p for p in init_params if init_params[p].vary], par_arr):
                fit_params[par].value = value
            return self.log_likelihood(fit_params, eval_gradient=True)

        result = minimize(objective, initial_par_arr, method=method, jac=True, bounds=bounds, **kwargs)
        return result

    def fit(self, init_params=None, update_params=True, **kwargs):
        if init_params is None:
            init_params = self.params

        self.fit_result = self._dofit(init_params, **kwargs)
        print(self.fit_result)

        if self.fit_result.success and update_params:
            for par, value in zip([p for p in init_params if init_params[p].vary], self.fit_result.x):
                self.params[par].value = value
            self.param_error = self.fit_result.hess_inv(self.fit_result.x) ** 0.5
            self.process_fit_results(self.fit_result, self.params)


class MLPSD(MLFit):
    def __init__(self, lc, **kwargs):
        t = np.array(lc.time)

        y = np.array(lc.rate)
        # to fit a Gaussian process, we should make sure our data has zero mean
        y -= np.mean(y)

        # RMS normalisation for PSD
        psdnorm = 2 * np.min(np.diff(lc.time)) / (lc.mean() ** 2 * lc.length)

        MLFit.__init__(self, t, y, psdnorm=psdnorm, **kwargs)

        self.psd = self.get_psd()
        self.psd_error = None

    def cov_matrix(self, params):
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        # note the factor of 2 to integrate over the negative frequencies too!
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) / self.psdnorm
        else:
            psd = self.model(params, self.fbins.bin_cent) / self.psdnorm

        cov = np.sum(np.array([p * c for p, c in zip(psd, self.cos_integral)]), axis=0)
        return cov

    def cov_matrix_deriv(self, params, delta):
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) / self.psdnorm

            # in this simple case, the covariance matrix is just a linear sum of each frequency term
            # so the derivative is simple - we multiply by p when we're talking about the log
            return np.stack([c * p for c, p in zip(self.cos_integral, psd)], axis=-1)
        else:
            psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent) / self.psdnorm
            return np.stack([np.sum([c * p for c, p in zip(self.cos_integral, psd_deriv[:, par])], axis=0) for par in range(psd_deriv.shape[-1])], axis=-1)

    def process_fit_results(self, fit_result, params):
        self.psd = self.get_psd()
        if self.model is None:
            self.psd_error = fit_result.hess_inv(fit_result.x) ** 0.5
        else:
            # calculate the error on each PSD point from the error on each parameter
            psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent)
            self.psd_error = np.sum([e * psd_deriv[..., i] for i, e in enumerate(self.param_error)], axis=0) / self.psd
            if np.any(np.isnan(self.psd_error)):
                self.psd_error = None

    def get_psd(self, params=None):
        if params is None:
            params = self.params

        if self.model is None:
            return np.array([self.params[p].value for p in self.params])
        else:
            return np.log(self.model(self.params, self.fbins.bin_cent))

    def _getplotdata(self):
        x = (self.fbins.bin_cent, self.fbins.x_error())
        y = (np.exp(self.psd), np.vstack([np.exp(self.psd) - np.exp(self.psd - self.psd_error), np.exp(self.psd + self.psd_error) - np.exp(self.psd)])) if self.psd_error is not None else np.exp(self.psd)
        return x, y

    def _getplotaxes(self):
        return 'Frequency / Hz', 'log', 'PSD', 'log'


class MLCrossSpectrum(MLFit):
    def __init__(self, lc1, lc2, mlpsd1=None, mlpsd2=None, psd_model=None, cpsd_model=None, lag_model=None, freeze_psd=True, cpsd_model_args={}, lag_model_args={}, **kwargs):
        t = np.array(lc1.time)

        y1 = np.array(lc1.rate)
        y1 -= np.mean(y1)
        y2 = np.array(lc2.rate)
        y2 -= np.mean(y2)
        # to fit a cross spectrum, we stack the data vectors
        y = np.concatenate([y1, y2])

        cpsdnorm = 2 * np.min(np.diff(lc1.time)) / (np.mean(np.concatenate([lc1.rate, lc2.rate])) ** 2 * lc1.length)

        if mlpsd1 is not None:
            self.mlpsd1 = mlpsd1
        else:
            self.mlpsd1 = MLPSD(lc1, model=psd_model, comp_name='psd1', **kwargs)

        if mlpsd2 is not None:
            self.mlpsd2 = mlpsd2
        else:
            self.mlpsd2 = MLPSD(lc2, model=psd_model, comp_name='psd2', **kwargs)

        self.freeze_psd = freeze_psd

        self.autocov1 = None
        self.autocov2 = None

        if cpsd_model is None:
            self.cpsd_model = None
        elif isinstance(cpsd_model, Model):
            self.cpsd_model = cpsd_model
        else:
            self.cpsd_model = cpsd_model(component_name='cpsd_', **cpsd_model_args)

        if lag_model is None:
            self.lag_model = None
        elif isinstance(lag_model, Model):
            self.lag_model = lag_model
        else:
            self.lag_model = lag_model(component_name='lag_', **lag_model_args)

        MLFit.__init__(self, t, y, psdnorm=cpsdnorm, **kwargs)

        # we probably don't need the sine integrals since they go to zero when we integrate over -ve frequencies
        # self.sin_integral = np.array(
        #     [np.array([(1. / (2. * np.pi * t)) * (np.cos(2 * np.pi * fmin * t) - np.cos(2 * np.pi * fmax * t))
        #                if t > 0 else 0.
        #                for t in self.tau.flatten()]).reshape(self.tau.shape)
        #      for fmin, fmax in zip(self.fbins.bin_start, self.fbins.bin_end)])

    def get_params(self):
        params = lmfit.Parameters()
        if not self.freeze_psd:
            params += self.mlpsd1.get_params()
            params += self.mlpsd2.get_params()
        if self.cpsd_model is None:
            for i in range(len(self.fbins)):
                params.add('%sln_cpsd%01d' % (self.comp_name, i), 1., vary=True, min=-30., max=30.)
        else:
            params += self.cpsd_model.get_params()
        if self.lag_model is None:
            for i in range(len(self.fbins)):
                params.add('%slag%01d' % (self.comp_name, i), 1., vary=True, min=-np.pi, max=np.pi)
        else:
            params += self.lag_model.get_params()

        return params

    def cross_cov_matrix(self, params):
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        if self.cpsd_model is None:
            cpsd = np.exp(np.array([params['%sln_cpsd%01d' % (self.comp_name, i)].value for i in range(len(self.fbins))])) / self.psdnorm
        else:
            cpsd = self.cpsd_model(params, self.fbins.bin_cent) / self.psdnorm

        # likewise for the (phase) lags
        if self.lag_model is None:
            lags = np.array([params['%slag%01d' % (self.comp_name, i)].value for i in range(len(self.fbins))])
        else:
            lags = self.lag_model(params, self.fbins.bin_cent)

        # TODO: is this correct or do we need the sine terms? (I think we don't if we're integrating over the -ve freq terms)
        #cov = np.sum(np.array([p * (c * np.cos(phi) + s * np.sin(phi)) for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]), axis=0)
        cov = np.sum(np.array([2. * p * c * np.cos(phi) for p, c, phi in zip(cpsd, self.cos_integral, lags)]), axis=0)
        return cov

    def cov_matrix(self, params):
        if self.freeze_psd:
            ac1 = self.autocov1
            ac2 = self.autocov2
        else:
            ac1 = self.mlpsd1.cov_matrix(params)
            ac2 - self.mlpsd2.cov_matrix(params)

        cc = self.cross_cov_matrix(params)

        return np.vstack([np.hstack([ac1, cc.T]), np.hstack([cc, ac2])])
