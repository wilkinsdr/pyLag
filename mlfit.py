"""
pylag.mlfit

Implements the method of Zoghbi et al. 2013 to fit correlation functions/covariance matrices
to light curves in the time domain to estimate power and lag spectra

Classes
-------
CorrelationModel : base class for model correlation functions
- AutoCorrelationModel_plpsd : autocorrelation function for power law PSD
- CrossCorrelationModel_plpsd_constlag : cross-correlation function for power law PSD and constant lag time at all frequencies

CovarianceMatrixModel : constructs a covariance matrix from a CorrelationModel function for a single light curve
CrossCovarianceMatrixModel : constructs a cross-covariance matrix from CorrelationModel functions for two light curves

MLCovariance : maximum-likelihood fitting of a covariance matrix to a single light curve (for PSD estimation)
MLCrossCovariance : maximum-likelihood fitting of a cross-covariance matrix to two light curves (for lag estimation)

v1.0 16/04/2019 - D.R. Wilkins
"""
import numpy as np
import scipy.fftpack
import scipy.integrate
import lmfit

from .binning import *
from .plotter import *


class CorrelationModel(object):
    def __init__(self, component_name=None):
        self.component_name = component_name

        if self.component_name is None:
            self.prefix = ''
        else:
            self.prefix = '%s_' % component_name

    def get_params(self):
        raise AssertionError("I'm supposed to be overridden to return your parameters!")

    def eval(self, tau):
        raise AssertionError("I'm supposed to be overridden to define your covariance!")

    def eval_points(self, params, lags, **kwargs):
        corr_arr = np.array([self.eval(params, tau, **kwargs) for tau in lags])
        corr_arr -= np.min(corr_arr)
        return corr_arr


class AutoCorrelationModel_plpsd(CorrelationModel):
    def get_params(self, norm=1., slope=1.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)

        return params

    def eval(self, params, tau, freq_arr=None, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value

        if freq_arr is None:
            freq_arr = np.arange(-1, 1, 1e-5)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) > flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) > flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]

        integrand = psd * np.cos(2 * np.pi * tau * freq_arr)
        autocorr = scipy.integrate.trapz(integrand[np.isfinite(integrand)], freq_arr[np.isfinite(integrand)])

        return autocorr

    def get_psd_series(self, params, freq=None, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value

        if isinstance(freq, Binning):
            freq_arr = freq.bin_cent
        elif isinstance(freq, tuple):
            freq_arr = LogBinning(freq[0], freq[1], freq[2]).bin_cent
        elif isinstance(freq, (np.ndarray,list)):
            freq_arr = np.array(freq)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) > flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) > flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]

        return DataSeries(freq_arr, psd, xlabel='Frequency / Hz', ylabel='PSD', xscale='log', yscale='log')


class CrossCorrelationModel_plpsd_constlag(CorrelationModel):
    def get_params(self, norm=1., slope=1., lag=0.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%slag' % self.prefix, value=lag, min=-1e4, max=+1e4)

        return params

    def eval(self, params, tau, freq_arr=None, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        lag = params['%slag' % self.prefix].value

        if freq_arr is None:
            freq_arr = np.arange(-1, 1, 1e-5)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) > flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) > flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]

        integrand = psd * np.cos(2*np.pi*tau*freq_arr + 2*np.pi*freq_arr*lag)
        autocorr = scipy.integrate.trapz(integrand[np.isfinite(integrand)], freq_arr[np.isfinite(integrand)])

        return autocorr


class CovarianceMatrixModel(object):
    def __init__(self, corr_model, time, time2=None, component_name=None, freq_arr=None, eval_args={}):
        self.corr_model = corr_model(component_name=component_name)
        self.dt_matrix = self.dt_matrix(time, time2)

        self.min_tau = np.min(self.dt_matrix[self.dt_matrix > 0])
        self.max_tau = np.max(self.dt_matrix)

        self.tau_arr = np.arange(-1*self.max_tau, self.max_tau + self.min_tau, self.min_tau)

        if freq_arr is None:
            self.fmin = 1. / (20 * self.max_tau)
            self.fmax = 1. / (2 * self.min_tau)
            self.freq_arr = np.arange(-1*self.fmax, self.fmax, self.fmin)
        else:
            self.freq_arr = freq_arr

        self.eval_args = eval_args

    @staticmethod
    def dt_matrix(time1, time2=None):
        if time2 is None:
            time2 = time1
        t1, t2 = np.meshgrid(time1, time2)
        return t2 - t1

    def get_params(self, *args, **kwargs):
        return self.corr_model.get_params(*args, **kwargs)

    def eval(self, params):
        corr_arr = self.corr_model.eval_points(params, self.tau_arr, freq_arr=self.freq_arr, **self.eval_args)
        return np.array([corr_arr[int((tau + self.max_tau) / self.min_tau)] \
                         for tau in self.dt_matrix.reshape(-1)]).reshape(self.dt_matrix.shape)


class CrossCovarianceMatrixModel(object):
    def __init__(self, autocorr_model, crosscorr_model, time1, time2, autocorr1_args={}, autocorr2_args={}, crosscorr_args={}):
        self.autocov_matrix1 = CovarianceMatrixModel(autocorr_model, time1, component_name='autocorr1', eval_args=autocorr1_args)
        self.autocov_matrix2 = CovarianceMatrixModel(autocorr_model, time2, component_name='autocorr2', eval_args=autocorr2_args)
        self.crosscov_matrix = CovarianceMatrixModel(crosscorr_model, time1, time2=time2, component_name='crosscorr', eval_args=crosscorr_args)

    def get_params(self, autocorr1_pars={}, autocorr2_pars={}, crosscorr_pars={}):
        return self.autocov_matrix1.get_params(**autocorr1_pars) \
               + self.autocov_matrix2.get_params(**autocorr1_pars) \
               + self.crosscov_matrix.get_params(**crosscorr_pars)

    def eval(self, params):
        ac1 = self.autocov_matrix1.eval(params)
        ac2 = self.autocov_matrix2.eval(params)
        cc = self.crosscov_matrix.eval(params)
        return np.vstack([np.hstack([ac1, cc.T]), np.hstack([cc, ac2])])


class MLCovariance(object):
    def __init__(self, lc, autocov_model, params=None, **kwargs):
        self.cov_matrix = CovarianceMatrixModel(autocov_model, lc.time, **kwargs)
        self.data = lc.rate

        if isinstance(params, lmfit.Parameters):
            self.params = params
        elif isinstance(params, dict):
            self.params = self.cov_matrix.get_params(**params)
        else:
            self.params = self.cov_matrix.get_params()

        self.minimizer = None
        self.fit_result = None

    def log_likelihood(self, params):
        c = self.cov_matrix.eval(params)
        try:
            ci = np.linalg.inv(c)
            _, ld = np.linalg.slogdet(c)
        except:
            return -1e20

        l = (-len(self.data) / 2) * np.log(2 * np.pi) - 0.5 * ld - 0.5 * np.matmul(self.data.T, np.matmul(ci, self.data))
        return l

    def mlog_likelihood(self, params):
        return -1*self.log_likelihood(params)

    def _dofit(self, params, method='nelder', write_steps=0, **kwargs):
        def fit_progress(params, iter, resid):
            if write_steps > 0:
                if iter % write_steps == 0:
                    parstr = ' %10.4g'*len(params)
                    parvals = [params[p] for p in params]
                    print(("%5d %15.6g" + parstr) % tuple([iter, resid] + parvals))

        if write_steps > 0:
            iter_cb = fit_progress
        else:
            iter_cb = None

        minimizer = lmfit.Minimizer(self.mlog_likelihood, params, nan_policy='omit', iter_cb=iter_cb)
        fit_result = minimizer.minimize(method=method, **kwargs)

        return minimizer, fit_result

    def fit(self, params=None, **kwargs):
        if params is None:
            params = self.params

        self.minimizer, self.fit_result = self._dofit(params, **kwargs)
        self.show_fit()

        return self.fit_result.residual

    def show_fit(self):
        if self.fit_result is None:
            raise AssertionError("Need to run fit first!")
        lmfit.report_fit(self.fit_result)

    def run_mcmc(self, params=None, burn=300, steps=1000, thin=1):
        if params is None:
            if self.fit_result is not None:
                params = self.fit_result.params
            else:
                params = self.params

        mcmc_result = lmfit.minimize(self.log_likelihood, params=params, method='emcee', burn=burn, steps=steps, thin=thin)
        return mcmc_result

    def steppar(self, par, steps, method='nelder'):
        import copy
        if self.fit_result is not None:
            step_params = copy.copy(self.fit_result.params)
        else:
            step_params = copy.copy(self.params)

        step_params[par].vary = False

        print()
        print('%16s  %16s' % (par, "-log(likelihood)"))
        print('=' * 34)

        fit_stats = []
        for val in steps:
            step_params[par].value = val
            _, result = self._dofit(step_params, method)
            fit_stats.append(result.residual)
            print('%16g  %16g' % (val, result.residual))

        print()

        fit_stats = np.array(fit_stats)

        return steps, fit_stats

    def get_psd(self, freq=None):
        if self.fit_result is None:
            raise AssertionError("Need to run fit first!")
        if freq is None:
            freq = self.cov_matrix.freq_arr[self.cov_matrix.freq_arr>1e-10]
        return self.cov_matrix.cov_model.get_psd_series(self.fit_result.params, freq)


class MLCrossCovariance(MLCovariance):
    def __init__(self, lc1, lc2, autocov_model, crosscov_model, params=None, **kwargs):
        self.cov_matrix = CrossCovarianceMatrixModel(autocov_model, crosscov_model, lc1.time, lc2.time, **kwargs)
        self.data = np.hstack([lc1.rate, lc2.rate])

        if isinstance(params, lmfit.Parameters):
            self.params = params
        elif isinstance(params, dict):
            self.params = self.cov_matrix.get_params(**params)
        else:
            self.params = self.cov_matrix.get_params()

        self.minimizer = None
        self.fit_result = None
