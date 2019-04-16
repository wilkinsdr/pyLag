import numpy as np
import scipy.fftpack
import scipy.integrate
import lmfit

from .binning import *
from .plotter import *

class CovarianceModel(object):
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


class AutoCovarianceModel_plpsd(CovarianceModel):
    def get_params(self, norm=1., slope=1.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=0., max=1e10)
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


class CrossCovarianceModel_plpsd_constlag(CovarianceModel):
    def get_params(self, norm=1., slope=1., lag=0.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=0., max=1e10)
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
    def __init__(self, cov_model, time, time2=None, component_name=None, freq_arr=None, eval_args={}):
        self.cov_model = cov_model(component_name=component_name)
        self.dt_matrix = self.dt_matrix(time, time2)

        self.min_tau = np.min(self.dt_matrix[self.dt_matrix > 0])
        self.max_tau = np.max(self.dt_matrix)

        self.tau_arr = np.arange(-1*self.max_tau, self.max_tau + self.min_tau, self.min_tau)

        if freq_arr is None:
            self.fmin = 1. / (2 * self.max_tau)
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
        return self.cov_model.get_params(*args, **kwargs)

    def eval(self, params):
        corr_arr = self.cov_model.eval_points(params, self.tau_arr, freq_arr=self.freq_arr, **self.eval_args)
        return np.array([corr_arr[int((tau + self.max_tau) / self.min_tau)] \
                         for tau in self.dt_matrix.reshape(-1)]).reshape(self.dt_matrix.shape)


class CrossCovarianceMatrixModel(object):
    def __init__(self, autocov_model, crosscov_model, time1, time2, autocov1_args={}, autocov2_args={}, crosscov_args={}):
        self.autocov_matrix1 = CovarianceMatrixModel(autocov_model, time1, component_name='autocov1', eval_args=autocov1_args)
        self.autocov_matrix2 = CovarianceMatrixModel(autocov_model, time2, component_name='autocov2', eval_args=autocov2_args)
        self.crosscov_matrix = CovarianceMatrixModel(crosscov_model, time1, time2=time2, component_name='cross_cov', eval_args=crosscov_args)

    def get_params(self, autocov1_pars={}, autocov2_pars={}, crosscov_pars={}):
        return self.autocov_matrix1.get_params(**autocov1_pars) \
               + self.autocov_matrix2.get_params(**autocov1_pars) \
               + self.crosscov_matrix.get_params(**crosscov_pars)

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
            print("WARNING: Failed to invert covariance matrix")
            print(params)
            return np.nan

        l = (-len(self.data) / 2) * np.log(2 * np.pi) - 0.5 * ld - 0.5 * np.matmul(self.data.T, np.matmul(ci, self.data))
        return l

    def mlog_likelihood(self, params):
        return -1*self.log_likelihood(params)

    def _dofit(self, params, method='nelder'):
        minimizer = lmfit.Minimizer(self.mlog_likelihood, params, nan_policy='omit')
        fit_result = minimizer.minimize(method=method)

        return minimizer, fit_result

    def fit(self, params=None, method='nelder'):
        if params is None:
            params = self.params

        self.minimizer, self.fit_result = self._dofit(params, method)
        self.show_fit()

        return self.fit_result.residual

    def show_fit(self):
        if self.fit_result is None:
            raise AssertionError("Need to run fit first!")
        lmfit.report_fit(self.fit_result)

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


def MLCrossCovariance(MLCovariance):
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
