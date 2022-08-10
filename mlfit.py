"""
pylag.mlfit

Implements the method of Zoghbi et al. 2013 to fit correlation functions/covariance matrices
to light curves in the time domain to estimate power and lag spectra

v2.0 05/07/2019 - D.R. Wilkins
"""
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, cho_factor, blas
from scipy.optimize import minimize

import lmfit
import copy

from .lightcurve import *
from .binning import *
from .model import *
from .plotter import *

class MLFit(object):
    def __init__(self, t, y, noise=None, psdnorm=1., Nf=10, fbins=None, model=None, component_name=None, model_args={}, eval_sin=False, extend_freq=1.):
        self.time = np.array(t)
        self.dt = np.min(np.diff(self.time))

        # the linear algebra we'll be doing requires the data points as a column vector (in matrix form)
        self.data = y[:, np.newaxis]

        self.noise = noise

        self.psdnorm = psdnorm

        # matrix of pairwise separations of time bins
        self.tau = self.time - np.expand_dims(self.time, 1)

        if fbins is None:
            # set up frequency bins to span
            min_freq = (1. / (np.max(self.time) - np.min(self.time))) / extend_freq
            max_freq = 0.5 / self.dt
            self.fbins = LogBinning(min_freq, max_freq, Nf)
        else:
            self.fbins = fbins

        self.freq = self.fbins.bin_cent
        self.freq_error = self.fbins.x_error()

        # pre-compute the integral of the cosine and sine term in the correlation for each frequency bin
        # so we don't have to do this for every change in parameter values
        # note the factor of 2 to integrate over the negative frequencies too!
        self.cos_integral = np.zeros((len(self.fbins), self.tau.shape[0], self.tau.shape[1]))
        self.sin_integral = np.zeros((len(self.fbins), self.tau.shape[0], self.tau.shape[1])) if eval_sin else None
        diag = np.eye(self.tau.shape[0], dtype=bool)
        for i, (fmin, fmax) in enumerate(zip(self.fbins.bin_start, self.fbins.bin_end)):
            self.cos_integral[i, ~diag] = (1. / (2. * np.pi * self.tau[~diag])) * (
                        np.sin(2. * np.pi * fmax * self.tau[~diag]) - np.sin(2. * np.pi * fmin * self.tau[~diag]))
            self.cos_integral[i, diag] = fmax - fmin
            if eval_sin:
                self.sin_integral[i, ~diag] = (1. / (2. * np.pi * self.tau[~diag])) * (
                            np.cos(2. * np.pi * fmin * self.tau[~diag]) - np.cos(2. * np.pi * fmax * self.tau[~diag]))

        if model is None:
            self.model = None
        elif isinstance(model, Model):
            self.model = model
        else:
            self.model = model(component_name=component_name, **model_args)

        self.comp_name = component_name
        self.prefix = component_name + "_" if component_name is not None else ''
        self.params = self.get_params()

        self.fit_result = None

    def log_likelihood(self, params, eval_gradient=True):
        c = self.cov_matrix(params)

        # add white noise along the leading diagonal
        # this should be the Poisson noise term when calculating a PSD
        if self.noise is not None:
            c += np.diag(self.noise)

        try:
            #print(c.shape)
            L = cho_factor(c, lower=True, check_finite=False)[0]
            #print(L.shape)
        except np.linalg.LinAlgError:
            # try:
            # if the matrix is sibgular, perturb the diagonal and try again
            try:
                L = cho_factor(c + np.diag(self.noise), lower=True, check_finite=False)[0]
            except np.linalg.LinAlgError:
                #par_str = ", ".join(["%s: %g" % (p, params[p].value) for p in params])
                #print("WARNING: Couldn't invert covariance matrix with parameters " + par_str)
                return (1e6, np.zeros(len(params)) - 1e6) if eval_gradient else -np.inf
                # raise np.linalg.LinAlgError("Couldn't invert covariance matrix with parameters " + par_str)
            # except np.linalg.LinAlgError:
            #     return (-np.inf, np.zeros(len(params))) if eval_gradient else -np.inf
        except ValueError:
            return (np.inf, np.zeros(len(params))) if eval_gradient else -np.inf

        alpha = cho_solve((L, True), self.data, check_finite=False)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", self.data, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= c.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        if eval_gradient:
            c_gradient = self.cov_matrix_deriv(params)
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
        bounds = [(init_params[p].min, init_params[p].max) for p in init_params if init_params[p].vary] if method == 'L-BFGS-B' else None

        def objective(par_arr):
            fit_params = copy.copy(init_params)
            for par, value in zip([p for p in init_params if init_params[p].vary], par_arr):
                fit_params[par].value = value
            l, g = self.log_likelihood(fit_params, eval_gradient=True)
            print("\r-log(L) = %6.3g" % l + " for parameters: " + ' '.join(['%6.3g' % p for p in param2array(fit_params)]), end="")
            return l, g

        result = minimize(objective, initial_par_arr, method=method, jac=True, bounds=bounds, **kwargs)
        print("\r-log(L) = %6.3g" % result.fun + " for parameters: " + " ".join(['%6.3g' % p for p in param2array(array2param(result.x, init_params))]))
        return result

    def fit(self, init_params=None, update_params=True, **kwargs):
        if init_params is None:
            init_params = self.params

        self.fit_result = self._dofit(init_params, **kwargs)
        print(self.fit_result)

        if True or self.fit_result.success and update_params:
            for par, value in zip([p for p in init_params if init_params[p].vary], self.fit_result.x):
                self.params[par].value = value
            hess = self.fit_result.hess_inv(self.fit_result.x) if callable(self.fit_result.hess_inv) else np.diag(self.fit_result.hess_inv)
            self.param_error =  hess ** 0.5
            self.process_fit_results(self.fit_result, self.params)


class MLPSD(MLFit):
    def __init__(self, lc=None, t=None, r=None, e=None, noise='errors', **kwargs):
        if lc is not None:
            t = np.array(lc.time)
            r = np.array(lc.rate)
            e = np.array(lc.error)

            # remove the time bins with zero counts
            nonzero = (y > 0)
            t = t[nonzero]
            r = y[nonzero]
            e = e[nonzero]

        # to fit a Gaussian process, we should make sure our data has zero mean
        y = r - np.mean(r)

        dt = np.min(np.diff(t))
        length = (np.max(t) - np.min(t)) // dt

        if noise == 'poisson':
            noise_arr = (1. / lc.mean()**2) * np.ones_like(lc.rate)
        elif noise == 'errors':
            # Note: RMS normalisation
            noise_arr = e ** 2 * length / (2.*dt)
        elif isinstance(noise, (float, int)):
            noise_arr = noise * np.ones_like(lc.rate)
        else:
            noise_arr = None

        # RMS normalisation for PSD
        psdnorm = (np.mean(r) ** 2 * length) / (2 * dt)

        MLFit.__init__(self, t, y, noise=noise_arr, psdnorm=psdnorm, **kwargs)

        self.psd = self.get_psd()
        self.psd_error = None

        self.noise_level = (2. / np.mean(r)**2) * np.ones_like(self.fbins.bin_cent)

    def cov_matrix(self, params):
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        # note the factor of 2 to integrate over the negative frequencies too!
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) * self.psdnorm
        else:
            psd = self.model(params, self.fbins.bin_cent) * self.psdnorm

        cov = np.sum(np.array([p * c for p, c in zip(psd, self.cos_integral)]), axis=0)

        return cov

    def cov_matrix_deriv(self, params, delta=1e-6):
        if self.model is None:
            psd = np.exp(np.array([params[p].value for p in params])) * self.psdnorm

            # in this simple case, the covariance matrix is just a linear sum of each frequency term
            # so the derivative is simple - we multiply by p when we're talking about the log
            return np.stack([c * p for c, p in zip(self.cos_integral, psd)], axis=-1)
        else:
            psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent) * self.psdnorm
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
    def __init__(self, lc1, lc2, mlpsd1=None, mlpsd2=None, psd_model=None, cpsd_model=None, lag_model=None, freeze_psd=True, noise='errors', cpsd_model_args={}, lag_model_args={}, **kwargs):
        t = np.array(lc1.time)

        r1 = np.array(lc1.rate)
        r2 = np.array(lc2.rate)

        e1 = np.array(lc1.error)
        e2 = np.array(lc2.error)

        # remove the time bins where either light curve is zero, since these will cause problems with the likelihood function (due to zero error)
        nonzero = np.logical_and(r1>0, r2>0)
        t = t[nonzero]
        r1 = r1[nonzero]
        r2 = r2[nonzero]
        e1 = e1[nonzero]
        e2 = e2[nonzero]

        y1 = r1 - np.mean(r1)
        y2 = r2 - np.mean(r2)

        # to fit a cross spectrum, we stack the data vectors
        y = np.concatenate([y1, y2])

        if noise == 'poisson':
            noise_arr = np.concatenate([(1. / lc1.mean() ** 2) * np.ones_like(lc1.rate), (1. / lc2.mean() ** 2) * np.ones_like(lc2.rate)])
        elif noise == 'errors':
            # note: RMS normalisation
            noise_arr = np.concatenate([e1 ** 2 * lc1.length / (2.*np.min(np.diff(lc1.time))), e2 ** 2 * lc2.length / (2.*np.min(np.diff(lc2.time)))])
        elif isinstance(noise, (float, int)):
            noise_arr = noise * np.ones_like(np.concatenate([lc1.rate, lc2.rate]))
        else:
            noise_arr = None

        cpsdnorm = (lc1.mean() * lc2.mean() * lc1.length) / (2 * np.min(np.diff(lc1.time)))

        self.ac1 = None
        self.ac2 = None

        if mlpsd1 is not None:
            self.mlpsd1 = mlpsd1
            self.ac1 = self.mlpsd1.cov_matrix(self.mlpsd1.params)
        else:
            self.mlpsd1 = MLPSD(t=t, r=r1, e=e1, noise=noise, model=psd_model, component_name='psd1', **kwargs)

        if mlpsd2 is not None:
            self.mlpsd2 = mlpsd2
            self.ac2 = self.mlpsd2.cov_matrix(self.mlpsd2.params)
        else:
            self.mlpsd2 = MLPSD(t=t, r=r2, e=e2, noise=noise, model=psd_model, component_name='psd2', **kwargs)

        self.freeze_psd = freeze_psd

        if cpsd_model is None:
            self.cpsd_model = None
        elif isinstance(cpsd_model, Model):
            self.cpsd_model = cpsd_model
        else:
            self.cpsd_model = cpsd_model(component_name='cpsd', **cpsd_model_args)

        if lag_model is None:
            self.lag_model = None
        elif isinstance(lag_model, Model):
            self.lag_model = lag_model
        else:
            self.lag_model = lag_model(component_name='lag', **lag_model_args)

        MLFit.__init__(self, t, y, noise=noise_arr, psdnorm=cpsdnorm, eval_sin=True, **kwargs)

    def get_params(self):
        params = lmfit.Parameters()
        if not self.freeze_psd:
            params += self.mlpsd1.get_params()
            params += self.mlpsd2.get_params()
        if self.cpsd_model is None:
            for i in range(len(self.fbins)):
                params.add('%sln_cpsd%01d' % (self.prefix, i), -1., vary=True, min=-30., max=30.)
        else:
            params += self.cpsd_model.get_params()
        if self.lag_model is None:
            for i in range(len(self.fbins)):
                params.add('%slag%01d' % (self.prefix, i), 0., vary=True, min=-np.pi, max=np.pi)
        else:
            params += self.lag_model.get_params()

        return params

    def fit_psd(self):
        print("Fitting PSD of light curve 1...")
        self.mlpsd1.fit()
        self.ac1 = self.mlpsd1.cov_matrix(self.mlpsd1.params)

        print("Fitting PSD of light curve 2...")
        self.mlpsd2.fit()
        self.ac2 = self.mlpsd1.cov_matrix(self.mlpsd2.params)

        # set an initial estimate of the cross power spectrum as the average of the two band powers
        # this helps avoid uninvertable matrices in the first step of the fit!
        # for i in range(len(self.fbins)):
        #     self.params['ln_cpsd%01d' % i].value = 0.5 * (self.mlpsd1.psd[i] + self.mlpsd2.psd[i])

    def cross_cov_matrix(self, params):
        # if no model is specified, the PSD model is just the PSD value in each frequency bin
        if self.cpsd_model is None:
            cpsd = np.exp(np.array([params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])) * self.psdnorm
        else:
            cpsd = self.cpsd_model(params, self.fbins.bin_cent) * self.psdnorm

        # likewise for the (phase) lags
        if self.lag_model is None:
            lags = np.array([params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            lags = self.lag_model(params, self.fbins.bin_cent)

        cov = np.sum(np.array([p * (c * np.cos(phi) - s * np.sin(phi)) for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]), axis=0)
        return cov

    def cov_matrix(self, params):
        if self.freeze_psd:
            if self.ac1 is None or self.ac2 is None:
                raise AssertionError("Autocovariance matrices are not available. Did you fit the PSDs?")
            ac1 = self.ac1
            ac2 = self.ac2
        else:
            ac1 = self.mlpsd1.cov_matrix(params)
            ac2 = self.mlpsd2.cov_matrix(params)

        cc = self.cross_cov_matrix(params)

        return np.vstack([np.hstack([ac1, cc.T]), np.hstack([cc, ac2])])

    def cross_cov_matrix_deriv(self, params, delta=1e-6):
        if self.cpsd_model is None:
            # if no model is specified, the PSD model is just the PSD value in each frequency bin
            if self.cpsd_model is None:
                cpsd = np.exp(np.array(
                    [params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])) * self.psdnorm
            else:
                cpsd = self.cpsd_model(params, self.fbins.bin_cent) * self.psdnorm

            # likewise for the (phase) lags
            if self.lag_model is None:
                lags = np.array([params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
            else:
                lags = self.lag_model(params, self.fbins.bin_cent)

            if self.cpsd_model is None:
                cpsd_derivs = [(c * np.cos(phi) + s * np.sin(phi)) * p for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]
            else:
                # TODO: implement chain rule derivatives for functions
                return NotImplemented

            if self.lag_model is None:
                lag_derivs = [-1 * p * (c * np.sin(phi) + s * np.cos(phi)) for p, c, s, phi in zip(cpsd, self.cos_integral, self.sin_integral, lags)]
            else:
                # TODO: implement chain rule derivatives for functions
                return NotImplemented

            # this is the stack of (1) the derivatives w.r.t. the cross powers (multiplied by p when we're using the log)
            # and (2) the phases
            return np.stack(cpsd_derivs + lag_derivs, axis=-1)

    def cov_matrix_deriv(self, params):
        cc = self.cross_cov_matrix_deriv(params)

        if self.freeze_psd:
            Z = np.zeros_like(self.ac1)
            return np.stack(
                [np.vstack([np.hstack([Z, cc[..., p].T]), np.hstack([cc[..., p], Z])]) for p in
                 range(len(self.params))], axis=-1)

        else:
            ac1 = self.mlpsd1.cov_matrix_deriv(params)
            ac2 = self.mlpsd2.cov_matrix_deriv(params)
            return np.stack(
                [np.vstack([np.hstack([ac1[..., p], cc[..., p].T]), np.hstack([cc[..., p], ac2[..., p]])]) for p in
                 range(len(self.params))], axis=-1)



        return np.stack([np.vstack([np.hstack([ac1[...,p], cc[...,p].T]), np.hstack([cc[...,p], ac2[...,p]])]) for p in range(len(self.params))], axis=-1)

    def get_cpsd(self, params=None):
        if params is None:
            params = self.params

        if self.cpsd_model is None:
            return np.array([self.params['%sln_cpsd%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            return np.log(self.cpsd_model(self.params, self.fbins.bin_cent))

    def get_lag(self, params=None, time_lag=True):
        if params is None:
            params = self.params

        if self.lag_model is None:
            lag = np.array([self.params['%slag%01d' % (self.prefix, i)].value for i in range(len(self.fbins))])
        else:
            lag = np.log(self.cpsd_model(self.params, self.fbins.bin_cent))

        return lag / (2. * np.pi * self.fbins.bin_cent) if time_lag else lag

    def process_fit_results(self, fit_result, params):
        self.cpsd = self.get_cpsd()
        if self.cpsd_model is None:
            self.cpsd_error = fit_result.hess_inv(fit_result.x)[:len(self.fbins)] ** 0.5
        else:
            return NotImplemented
            # # calculate the error on each PSD point from the error on each parameter
            # psd_deriv = self.model.eval_gradient(params, self.fbins.bin_cent)
            # self.psd_error = np.sum([e * psd_deriv[..., i] for i, e in enumerate(self.param_error)], axis=0) / self.psd
            # if np.any(np.isnan(self.psd_error)):
            #     self.psd_error = None

        self.lag = self.get_lag()
        if self.cpsd_model is None:
            self.lag_error = fit_result.hess_inv(fit_result.x)[len(self.fbins):] ** 0.5 / (2. * np.pi * self.fbins.bin_cent)
        else:
            return NotImplemented