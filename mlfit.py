"""
pylag.mlfit

Implements the method of Zoghbi et al. 2013 to fit correlation functions/covariance matrices
to light curves in the time domain to estimate power and lag spectra

Classes
-------
CorrelationModel : base class for model correlation functions
- AutoCorrelationModel_plpsd : autocorrelation function for power law PSD
- CrossCorrelationModel_plpsd_constlag : cross-correlation function for power law PSD and constant lag time at all frequencies

FFTCorrelationModel : class for faster calculation of model correlation functions using FFT

CovarianceMatrixModel : constructs a covariance matrix from a CorrelationModel function for a single light curve
CrossCovarianceMatrixModel : constructs a cross-covariance matrix from CorrelationModel functions for two light curves

MLCovariance : maximum-likelihood fitting of a covariance matrix to a single light curve (for PSD estimation)
MLCrossCovariance : maximum-likelihood fitting of a cross-covariance matrix to two light curves (for lag estimation)

v1.0 16/04/2019 - D.R. Wilkins
"""
import numpy as np
import scipy.fftpack
import scipy.integrate
import scipy.interpolate
import scipy.linalg
import scipy.optimize
import lmfit
import copy

from .binning import *
from .plotter import *


class CorrelationModel(object):
    def __init__(self, component_name=None, log_psd=True):
        self.component_name = component_name

        self.log_psd = log_psd # many models will have a normalisation or equivalent. We might want to fit the log

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

    def eval_gradient(self, params, lags, delta=1e-3, **kwargs):
        corr1 = self.eval_points(params, lags, **kwargs)
        gradient = []
        for par in [p for p in params if params[p].vary]:
            new_params = copy.copy(params)
            new_params[par].value += delta * new_params[par].value
            gradient.append((self.eval_points(new_params, lags, **kwargs) - corr1) / (delta * new_params[par].value))
        return np.array(gradient)

    def get_corr_series(self, params, lags, **kwargs):
        return DataSeries(x=lags, y=self.eval_points(params, lags, **kwargs), xlabel='Lag / s', ylabel='Correlation')

    def plot_corr(self, params, lags=None, **kwargs):
        if lags is None:
            lags = np.arange(-1000, 1000, 10)
        return Plot(self.get_corr_series(params, lags, **kwargs), lines=True)


class AutoCorrelationModel_plpsd(CorrelationModel):
    def get_params(self, norm=1., slope=2.):
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
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
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
        elif isinstance(freq, (np.ndarray, list)):
            freq_arr = np.array(freq)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) > flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) > flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[freq_arr > flimit][0]

        return DataSeries(freq_arr, psd, xlabel='Frequency / Hz', ylabel='PSD', xscale='log', yscale='log')


class CrossCorrelationModel_plpsd_constlag(CorrelationModel):
    def get_params(self, norm=1., slope=2., lag=0.):
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
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[freq_arr > flimit][0]

        integrand = psd * np.cos(2 * np.pi * tau * freq_arr + 2 * np.pi * freq_arr * lag)
        autocorr = scipy.integrate.trapz(integrand[np.isfinite(integrand)], freq_arr[np.isfinite(integrand)])

        return autocorr


class CrossCorrelationModel_plpsd_sigmoidlag(CorrelationModel):
    def get_params(self, norm=1., slope=2., lag=0., lag_slope=5., lag_logfcut=-3):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%slag' % self.prefix, value=lag, min=-1e4, max=+1e4)
        params.add('%slag_slope' % self.prefix, value=lag_slope, min=1., max=10.)
        params.add('%slag_logfcut' % self.prefix, value=lag_logfcut, min=-5, max=-1)

        return params

    def eval(self, params, tau, freq_arr=None, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        max_lag = params['%slag' % self.prefix].value
        lag_slope = params['%slag_slope' % self.prefix].value
        lag_fcut = params['%slag_logfcut' % self.prefix].value

        if freq_arr is None:
            freq_arr = np.arange(-1, 1, 1e-5)

        psd = np.zeros(freq_arr.shape)
        lag = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        lag[np.abs(freq_arr) >= flimit] = max_lag * (
                    1. - 1. / (1 + np.exp(-lag_slope * (np.log10(freq_arr[np.abs(freq_arr) >= flimit]) - lag_fcut))))
        lag[np.abs(freq_arr) < flimit] = lag[np.abs(freq_arr) > flimit][0]

        integrand = psd * np.cos(2 * np.pi * tau * freq_arr + 2 * np.pi * freq_arr * lag)
        autocorr = scipy.integrate.trapz(integrand[np.isfinite(integrand)], freq_arr[np.isfinite(integrand)])

        return autocorr


class FFTCorrelationModel(CorrelationModel):
    def __init__(self, oversample_len=10., oversample_freq=1, *args, **kwargs):
        self.oversample_len = int(oversample_len)
        self.oversample_freq = int(oversample_freq)
        CorrelationModel.__init__(self, *args, **kwargs)

    def eval_ft(self, params, freq):
        raise AssertionError("I'm supposed to be overridden to define your function's Fourier transform!")

    def eval_points(self, params, lags, freq_arr=None, **kwargs):
        freq_arr = scipy.fftpack.fftfreq(self.oversample_freq * self.oversample_len * len(lags),
                                         d=np.min(lags[lags > 0]) / self.oversample_freq)
        ft = self.eval_ft(params, freq_arr, **kwargs)
        corr = scipy.fftpack.ifft(ft).real
        # corr -= corr.min()
        corr = scipy.fftpack.fftshift(corr)

        fft_lags = scipy.fftpack.fftfreq(len(freq_arr), d=(freq_arr[1] - freq_arr[0]))
        fft_lags = scipy.fftpack.fftshift(fft_lags)

        if np.array_equal(lags, fft_lags):
            return corr
        else:
            tau0 = fft_lags[0]
            dtau = fft_lags[1] - fft_lags[0]
            return np.array([corr[int((tau - tau0) / dtau)] for tau in lags])

    def get_psd_series(self, params, lags=None, freq_arr=None, **kwargs):
        freq_arr = scipy.fftpack.fftfreq(self.oversample_freq * self.oversample_len * len(lags),
                                         d=np.min(lags[lags > 0]) / self.oversample_freq)
        ft = self.eval_ft(params, freq_arr, **kwargs)
        psd = np.abs(ft)
        return DataSeries(freq_arr[freq_arr > 0], psd[freq_arr > 0], xlabel='Frequency / Hz', xscale='log',
                          ylabel='PSD', yscale='log')

    def plot_psd(self, params, lags=None, freq_arr=None, **kwargs):
        return Plot(self.get_psd_series(params, lags, freq_arr, **kwargs), lines=True)


class FFTAutoCorrelationModel_plpsd(FFTCorrelationModel):
    def get_params(self, norm=None, slope=2.):
        if norm is None:
            norm = -7 if self.log_psd else 1e-3
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=(-50 if self.log_psd else 1e-10), max=(50 if self.log_psd else 1e10))
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = np.exp(params['%snorm' % self.prefix].value) if self.log_psd else params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[freq_arr > flimit][0]

        return psd


class FFTAutoCorrelationModel_brokenplpsd(FFTCorrelationModel):
    def get_params(self, norm=1., slope=2.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope1' % self.prefix, value=slope, min=0., max=3.)
        params.add('%sfbreak' % self.prefix, value=slope, min=1e-6, max=1e-2)
        params.add('%sslope2' % self.prefix, value=slope, min=0., max=3.)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope1 = params['%sslope1' % self.prefix].value
        fbreak = params['%sfbreak' % self.prefix].value
        slope2 = params['%sslope2' % self.prefix].value

        psd = np.zeros(freq_arr.shape)
        psd[np.logical_and(np.abs(freq_arr) >= flimit, np.abs(freq_arr) <= fbreak)] = \
            norm * np.abs(freq_arr[np.logical_and(np.abs(freq_arr) >= flimit, np.abs(freq_arr) <= fbreak)]) ** -slope1

        psd[np.abs(freq_arr) > fbreak] = norm * fbreak ** (slope2 - slope1) * np.abs(
            freq_arr[np.abs(freq_arr) > fbreak]) ** -slope2

        psd[np.abs(freq_arr) < flimit] = psd[freq_arr > flimit][0]

        return psd


class FFTAutoCorrelationModel_plpsd_binned(FFTCorrelationModel):
    def get_params(self, norm=1., slope=2., binsize=1.):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%sbinsize' % self.prefix, value=binsize, min=1., max=1000.)
        params['binsize'].vary = False

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        binsize = params['%sbinsize' % self.prefix].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]

        window = np.zeros(freq_arr.shape)
        window[0] = 1
        window[1:] = (np.sin(np.pi * freq_arr[1:] * binsize) / (np.pi * freq_arr[1:] * binsize)) ** 2

        return psd * window


class FFTAutoCorrelationModel_binpsd(FFTCorrelationModel):
    def __init__(self, fbins, log_psd=True, log_interp=True, *args, **kwargs):
        if isinstance(fbins, Binning):
            self.fbins = fbins.bin_cent
        elif isinstance(fbins, np.ndarray):
            self.fbins = fbins

        self.log_interp = log_interp
        # self.log_psd = log_psd

        FFTCorrelationModel.__init__(self, *args, **kwargs)

    def get_params(self, init_psd=1., init_slope=0.):
        params = lmfit.Parameters()

        if self.log_psd:
            min = -50
            max = 50
        else:
            min = 1e-10
            max = 1e10

        if isinstance(init_psd, (int, float)):
            init_psd = init_psd * self.fbins**-init_slope
            if self.log_psd:
                init_psd = np.log(init_psd)

        for bin_num, psd in enumerate(init_psd):
            params.add('%spsd%02d' % (self.prefix, bin_num), value=psd, min=min, max=max)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        psd_points = np.array([params[key].value for key in params if key.startswith(self.prefix)])

        if self.log_psd:
            psd_interp = scipy.interpolate.interp1d(np.log(self.fbins), psd_points,
                                                    fill_value=(psd_points[0], psd_points[-1]))
        elif self.log_interp:
            psd_interp = scipy.interpolate.interp1d(np.log(self.fbins), np.log(psd_points),
                                                    fill_value=(psd_points[0], psd_points[-1]))
        else:
            psd_interp = scipy.interpolate.interp1d(self.fbins, psd_points, fill_value=(psd_points[0], psd_points[-1]))

        psd = np.zeros(freq_arr.shape)
        if self.log_interp or self.log_psd:
            psd[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())] = \
                np.exp(psd_interp(np.log(np.abs(freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                                                                        np.abs(freq_arr) <= self.fbins.max())]))))
        else:
            psd[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                               np.abs(freq_arr) <= self.fbins.max())] = psd_interp(np.abs(
                freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())]))

        if self.log_psd:
            psd[np.abs(freq_arr) < self.fbins.min()] = np.exp(psd_points[0])
            psd[np.abs(freq_arr) >= self.fbins.max()] = np.exp(psd_points[-1])
        else:
            psd[np.abs(freq_arr) < self.fbins.min()] = psd_points[0]
            psd[np.abs(freq_arr) >= self.fbins.max()] = psd_points[-1]

        return psd


class FFTCrossCorrelationModel_plpsd_constlag(FFTCorrelationModel):
    def get_params(self, norm=None, slope=2., lag=0.):
        if norm is None:
            norm = -7 if self.log_psd else 1e-3
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=(-50 if self.log_psd else 1e-10), max=(50 if self.log_psd else 1e10))
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%slag' % self.prefix, value=lag, min=-1e4, max=+1e4)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = np.exp(params['%snorm' % self.prefix].value) if self.log_psd else params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        lag = params['%slag' % self.prefix].value

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        phase = 2 * np.pi * freq_arr * lag

        ft = psd * np.exp(-1j * phase)
        return ft


class FFTCrossCorrelationModel_plpsd_cutofflag(FFTCorrelationModel):
    def get_params(self, norm=1., slope=2., lag=0., lag_fcut=1):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%slag' % self.prefix, value=lag, min=-1e4, max=+1e4)
        params.add('%slag_fcut' % self.prefix, value=lag_fcut, min=1e-4, max=1)
        params.add('%sfix_fcut' % self.prefix, value=0, min=0, max=1)
        params['%sfix_fcut' % self.prefix].vary = False

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        max_lag = params['%slag' % self.prefix].value
        lag_fcut = params['%slag_fcut' % self.prefix].value

        if params['%sfix_fcut' % self.prefix].value == 1:
            lag_fcut = 1. / (2. * max_lag)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        lag = np.zeros(freq_arr.shape)
        lag[np.abs(freq_arr) <= lag_fcut] = max_lag
        lag[np.abs(freq_arr) > lag_fcut] = 0
        phase = 2 * np.pi * freq_arr * lag

        ft = psd * np.exp(-1j * phase)
        return ft


class FFTCrossCorrelationModel_plpsd_linearcutofflag(FFTCorrelationModel):
    def get_params(self, norm=1., slope=2., lag=0., lag_fcut=1):
        params = lmfit.Parameters()

        params.add('%snorm' % self.prefix, value=norm, min=1e-10, max=1e10)
        params.add('%sslope' % self.prefix, value=slope, min=0., max=3.)
        params.add('%slag' % self.prefix, value=lag, min=-1e4, max=+1e4)
        params.add('%slag_fcut' % self.prefix, value=lag_fcut, min=1e-4, max=1)
        params.add('%slag_fzero' % self.prefix, value=lag_fcut, min=1e-4, max=1)
        params.add('%sfix_fcut' % self.prefix, value=0, min=0, max=1)
        params['%sfix_fcut' % self.prefix].vary = False

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        norm = params['%snorm' % self.prefix].value
        slope = params['%sslope' % self.prefix].value
        max_lag = params['%slag' % self.prefix].value
        lag_fcut = params['%slag_fcut' % self.prefix].value
        lag_fzero = params['%slag_fzero' % self.prefix].value

        if params['%sfix_fcut' % self.prefix].value == 1:
            lag_fcut = 1. / (2. * max_lag)

        psd = np.zeros(freq_arr.shape)
        psd[np.abs(freq_arr) >= flimit] = norm * np.abs(freq_arr[np.abs(freq_arr) >= flimit]) ** -slope
        psd[np.abs(freq_arr) < flimit] = psd[np.abs(freq_arr) > flimit][0]
        lag = np.zeros(freq_arr.shape)
        lag[np.abs(freq_arr) <= lag_fcut] = max_lag
        lag[np.abs(freq_arr) > lag_fzero] = 0
        lag[np.logical_and(np.abs(freq_arr) > lag_fcut, np.abs(freq_arr) <= lag_fzero)] \
            = max_lag * (np.log10(
            np.abs(freq_arr[np.logical_and(np.abs(freq_arr) > lag_fcut, np.abs(freq_arr) <= lag_fzero)])) \
                         - np.log10(lag_fzero)) / (np.log10(lag_fcut) - np.log10(lag_fzero))
        phase = 2 * np.pi * freq_arr * lag

        ft = psd * np.exp(-1j * phase)
        return ft


class FFTCrossCorrelationModel_binned(FFTCorrelationModel):
    def __init__(self, fbins, log_psd=True, log_interp=True, *args, **kwargs):
        if isinstance(fbins, Binning):
            self.fbins = fbins.bin_cent
        elif isinstance(fbins, np.ndarray):
            self.fbins = fbins

        self.log_interp = log_interp
        # self.log_psd = log_psd

        FFTCorrelationModel.__init__(self, *args, **kwargs)

    def get_params(self, init_psd=1., init_slope=2., init_lag=0.):
        params = lmfit.Parameters()

        if self.log_psd:
            min = -50
            max = 50
        else:
            min = 1e-10
            max = 1e10

        if isinstance(init_psd, (int, float)):
            init_psd = init_psd * self.fbins ** -init_slope
            if self.log_psd:
                init_psd = np.log(init_psd)

        if isinstance(init_lag, (int, float)):
            init_lag = init_lag * np.ones(self.fbins.shape)

        for bin_num, (psd, lag) in enumerate(zip(init_psd, init_lag)):
            params.add('%spsd%02d' % (self.prefix, bin_num), value=psd, min=min, max=max)
            params.add('%slag%02d' % (self.prefix, bin_num), value=lag, min=-1000, max=1000)

        return params

    def eval_ft(self, params, freq_arr, flimit=1e-6):
        psd_points = np.array([params[key].value for key in params if key.startswith('%spsd' % self.prefix)])
        lag_points = np.array([params[key].value for key in params if key.startswith('%slag' % self.prefix)])

        if self.log_psd:
            psd_interp = scipy.interpolate.interp1d(np.log(self.fbins), psd_points,
                                                    fill_value=(psd_points[0], psd_points[-1]))
        elif self.log_interp:
            psd_interp = scipy.interpolate.interp1d(np.log(self.fbins), np.log(psd_points),
                                                    fill_value=(psd_points[0], psd_points[-1]))
        else:
            psd_interp = scipy.interpolate.interp1d(self.fbins, psd_points, fill_value=(psd_points[0], psd_points[-1]))

        if self.log_interp or self.log_psd:
            lag_interp = scipy.interpolate.interp1d(np.log(self.fbins), lag_points,
                                                    fill_value=(lag_points[0], lag_points[-1]))
        else:
            lag_interp = scipy.interpolate.interp1d(self.fbins, lag_points, fill_value=(lag_points[0], lag_points[-1]))

        psd = np.zeros(freq_arr.shape)
        if self.log_interp or self.log_psd:
            psd[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())] = \
                np.exp(psd_interp(np.log(np.abs(freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                                                                        np.abs(freq_arr) <= self.fbins.max())]))))
        else:
            psd[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                               np.abs(freq_arr) <= self.fbins.max())] = psd_interp(np.abs(
                freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())]))

        if self.log_psd:
            psd[np.abs(freq_arr) < self.fbins.min()] = np.exp(psd_points[0])
            psd[np.abs(freq_arr) >= self.fbins.max()] = np.exp(psd_points[-1])
        else:
            psd[np.abs(freq_arr) < self.fbins.min()] = psd_points[0]
            psd[np.abs(freq_arr) >= self.fbins.max()] = psd_points[-1]

        lag = np.zeros(freq_arr.shape)
        if self.log_interp or self.log_psd:
            lag[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())] = \
                lag_interp(np.log(np.abs(freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                                                                 np.abs(freq_arr) <= self.fbins.max())])))
        else:
            lag[np.logical_and(np.abs(freq_arr) >= self.fbins.min(), np.abs(freq_arr) <= self.fbins.max())] = \
                lag_interp(np.abs(freq_arr[np.logical_and(np.abs(freq_arr) >= self.fbins.min(),
                                                          np.abs(freq_arr) <= self.fbins.max())]))

        phase = 2 * np.pi * freq_arr * lag

        ft = psd * np.exp(-1j * phase)
        return ft


class CovarianceMatrixModel(object):
    def __init__(self, corr_model, time, time2=None, noise=None, component_name=None, freq_arr=None, eval_args={}, model_args={}):
        self.component_name = component_name

        if self.component_name is None:
            self.prefix = ''
        else:
            self.prefix = '%s_' % component_name

        self.corr_model = corr_model(component_name=component_name, **model_args)
        self.dt_matrix = self.dt_matrix(time, time2)

        self.min_tau = np.min(self.dt_matrix[self.dt_matrix > 0])
        self.max_tau = np.max(self.dt_matrix)

        self.tau_arr = np.arange(-1 * self.max_tau, self.max_tau + self.min_tau, self.min_tau)

        if freq_arr is None:
            self.fmin = 1. / (20 * self.max_tau)
            self.fmax = 1. / (2 * self.min_tau)
            self.freq_arr = np.arange(-1 * self.fmax, self.fmax, self.fmin)
        else:
            self.freq_arr = freq_arr

        self.eval_args = eval_args

        self.noise_par = (noise == 'param')

        if isinstance(noise, (float, int)):
            noise = noise * np.ones_like(time)

        self.noise_matrix = np.diag(noise) if noise is not None else None

    @staticmethod
    def dt_matrix(time1, time2=None):
        if time2 is None:
            time2 = time1
        t1, t2 = np.meshgrid(time1, time2)
        return t2 - t1

    def get_params(self, noise_level=0, *args, **kwargs):
        params = self.corr_model.get_params(*args, **kwargs)
        if self.noise_par:
            params.add('%snoiselevel' % self.prefix, value=noise_level, min=-10, max=20)
        return params

    def eval(self, params):
        corr_arr = self.corr_model.eval_points(params, self.tau_arr, freq_arr=self.freq_arr, **self.eval_args)
        matrix = np.array([corr_arr[int((tau + self.max_tau) / self.min_tau)]
                         for tau in self.dt_matrix.reshape(-1)]).reshape(self.dt_matrix.shape)
        if self.noise_par:
            matrix += np.exp(params['%snoiselevel' % self.prefix].value) * np.eye(matrix.shape[0])
        elif self.noise_matrix is not None:
            matrix += self.noise_matrix
        return matrix

    def eval_gradient(self, params, delta=1e-3):
        gradient_arr = self.corr_model.eval_gradient(params, self.tau_arr, delta=delta, **self.eval_args)
        gradient_matrix = np.zeros((self.dt_matrix.shape[0], self.dt_matrix.shape[1], len(params)))
        for p in range(gradient_arr.shape[0]):
            gradient_matrix[..., p] = np.array([gradient_arr[p, int((tau + self.max_tau) / self.min_tau)]
                                                for tau in self.dt_matrix.reshape(-1)]).reshape(self.dt_matrix.shape)
        return gradient_matrix


class CrossCovarianceMatrixModel(object):
    def __init__(self, autocorr_model, crosscorr_model, time1, time2, noise1=None, noise2=None, autocorr1_args={}, autocorr2_args={},
                 crosscorr_args={}):
        self.autocov_matrix1 = CovarianceMatrixModel(autocorr_model, time1, component_name='autocorr1',
                                                     model_args=autocorr1_args, noise=noise1)
        self.autocov_matrix2 = CovarianceMatrixModel(autocorr_model, time2, component_name='autocorr2',
                                                     model_args=autocorr2_args, noise=noise2)
        self.crosscov_matrix1 = CovarianceMatrixModel(crosscorr_model, time1, time2=time2, component_name='crosscorr',
                                                     model_args=crosscorr_args, noise=None)
        self.crosscov_matrix2 = CovarianceMatrixModel(crosscorr_model, time2, time2=time1, component_name='crosscorr',
                                                     model_args=crosscorr_args, noise=None)

    def get_params(self, autocorr1_pars={}, autocorr2_pars={}, crosscorr_pars={}):
        return self.autocov_matrix1.get_params(**autocorr1_pars) \
               + self.autocov_matrix2.get_params(**autocorr1_pars) \
               + self.crosscov_matrix1.get_params(**crosscorr_pars)

    def eval(self, params):
        ac1 = self.autocov_matrix1.eval(params)
        ac2 = self.autocov_matrix2.eval(params)
        cc1 = self.crosscov_matrix1.eval(params)
        cc2 = self.crosscov_matrix2.eval(params)
        return np.vstack([np.hstack([ac1, cc2]), np.hstack([cc1, ac2])])

    def eval_gradient(self, params, delta=1e-3):
        ac1_grad = self.autocov_matrix1.eval_gradient(params, delta)
        ac2_grad = self.autocov_matrix2.eval_gradient(params, delta)
        cc1_grad = self.crosscov_matrix1.eval_gradient(params, delta)
        cc2_grad = self.crosscov_matrix2.eval_gradient(params, delta)
        return np.vstack([np.hstack([ac1_grad, cc2_grad]), np.hstack([cc1_grad, ac2_grad])])


class MLCovariance(object):
    def __init__(self, lc, autocov_model, noise='mean_error', zero_mean=True, params=None, **kwargs):
        if noise == 'error':
            noise = lc.error**2
        elif noise == 'mean_error':
            noise = np.mean(lc.rate) / lc.dt

        self.cov_matrix = CovarianceMatrixModel(autocov_model, lc.time, noise=noise, **kwargs)

        if zero_mean:
            self.data = lc.rate - np.mean(lc.rate)
        else:
            self.data = lc.rate

        if isinstance(params, lmfit.Parameters):
            self.params = params
        elif isinstance(params, dict):
            self.params = self.cov_matrix.get_params(**params)
        else:
            self.params = self.cov_matrix.get_params()

        self.fit_params = None
        self.fit_stat = None
        self.mcmc_result = None

    def log_likelihood(self, params, eval_gradient=False, delta=1e-3):
        c = self.cov_matrix.eval(params)

        try:
            L = scipy.linalg.cholesky(c, lower=True)
        except np.linalg.LinAlgError:
            return (-np.inf, np.zeros(len(params))) if eval_gradient else -np.inf

        data = self.data
        if data.ndim == 1:
            data = data[:, np.newaxis]

        alpha = scipy.linalg.cho_solve((L, True), data)

        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", data, alpha)
        log_likelihood_dims -= np.log(np.diag(L)).sum()
        log_likelihood_dims -= c.shape[0] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)

        if eval_gradient:
            c_gradient = self.cov_matrix.eval_gradient(params, delta)
            tmp = np.einsum("ik,jk->ijk", alpha, alpha)
            tmp -= scipy.linalg.cho_solve((L, True), np.eye(c.shape[0]))[:, :, np.newaxis]
            gradient_dims = 0.5 * np.einsum("ijl,ijk->kl", tmp, c_gradient)
            gradient = gradient_dims.sum(-1)

        return (log_likelihood, gradient) if eval_gradient else log_likelihood

    def _dofit(self, params, method='l-bfgs-b', write_steps=0, delta=1e-3, restarts=0, par_shift=0.1, **kwargs):
        fit_params = copy.copy(params)

        def obj_with_grad(par_arr):
            for par, value in zip([p for p in params if params[p].vary], par_arr):
                fit_params[par].value = value
            l, g = self.log_likelihood(fit_params, eval_gradient=True, delta=delta)
            return -l, -g

        initial_params = np.array([params[p].value for p in params if params[p].vary])
        bounds = [(params[p].min, params[p].max) for p in params if params[p].vary]

        fit_par_arr, func_min, convergence = scipy.optimize.fmin_l_bfgs_b(obj_with_grad, initial_params, bounds=bounds)

        if restarts > 0:
            fit_par_arr = [fit_par_arr]
            func_min = [func_min]

            for n in range(restarts):
                initial_params = [np.random.uniform(b[0], b[1]) for b in bounds]
                this_initial_params = [p + np.random.randn()*np.abs(p)*par_shift for p in fit_par_arr[np.array(func_min).argmin()]]
                par, fmin, conv = scipy.optimize.fmin_l_bfgs_b(obj_with_grad, this_initial_params, bounds=bounds)
                fit_par_arr.append(par)
                func_min.append(fmin)

            func_min = np.array(func_min)
            fit_par_arr = fit_par_arr[func_min.argmin()]
            func_min = func_min.min()

        for par, value in zip([p for p in params if params[p].vary], fit_par_arr):
            fit_params[par].value = value

        return func_min, fit_params

    def fit(self, params=None, **kwargs):
        if params is None:
            params = self.params

        # self.minimizer, self.fit_result = self._dofit(params, **kwargs)
        # self.show_fit()
        self.fit_stat, self.fit_params = self._dofit(params, **kwargs)

        self.show_fit()

    def show_fit(self):
        if self.fit_params is None:
            raise AssertionError("Need to run fit first!")

        print()
        print('%-16s  %4s  %16s' % ("Parameter", "Vary", "Value"))
        print('=' * 40)

        for p in self.fit_params:
            print('%-16s  %4s  %16g' % (p, ('y' if self.fit_params[p].vary else 'n'), self.fit_params[p].value))

        print()
        print('-log(likelihood) = %g' % self.fit_stat)
        print()

    def run_mcmc(self, params=None, burn=300, steps=1000, thin=1):
        if params is None:
            if self.fit_params is not None:
                params = self.fit_params
            else:
                params = self.params

        mcmc_result = lmfit.minimize(self.log_likelihood, params=params, method='emcee', burn=burn, steps=steps,
                                     thin=thin)
        self.mcmc_result = mcmc_result
        self.process_mcmc()

    def process_mcmc(self, err_percentile=[15.9, 84.2]):
        if self.mcmc_result is None:
            raise AssertionError("Need to run MCMC first!")

        # get the solution corresponding to maximum likelihood from MCMC
        self.mcmc_result.maxprob_params = copy.copy(self.mcmc_result.params)
        maxprob_index = np.unravel_index(self.mcmc_result.lnprob.argmax(), self.mcmc_result.lnprob.shape)
        maxprob_params = self.mcmc_result.chain[maxprob_index]
        for par, value in zip([p for p in self.mcmc_result.maxprob_params if self.mcmc_result.maxprob_params[p].vary], maxprob_params):
            self.mcmc_result.maxprob_params[par].value = value

        # calculate the positive and negative errors from the median based on percentiles
        # [15.9, 84.2] for 1-sigma, [2.28, 97.7] for 2-sigma
        # calculate the error bars based on both the median and maximum likelihood values
        for par in [p for p in self.mcmc_result.params if self.mcmc_result.params[p].vary]:
            quantiles = np.percentile(self.mcmc_result.flatchain[par], err_percentile)

            self.mcmc_result.params[par].err_plus = quantiles[1] - self.mcmc_result.params[par].value
            self.mcmc_result.params[par].err_minus = self.mcmc_result.params[par].value - quantiles[0]

            self.mcmc_result.maxprob_params[par].err_plus = quantiles[1] - self.mcmc_result.maxprob_params[par].value
            self.mcmc_result.maxprob_params[par].err_minus = self.mcmc_result.maxprob_params[par].value - quantiles[0]

    def plot_corner(self, truths='median'):
        try:
            import corner
        except ImportError:
            raise AssertionError("plot_corner requires package corner to be installed")

        if self.mcmc_result is None:
            raise AssertionError("Need to run MCMC first!")

        if truths == 'median':
            truth_values = [self.mcmc_result.params[p].value for p in self.mcmc_result.params if
                            self.mcmc_result.params[p].vary]
        elif truths == 'maxprob':
            try:
                truth_values = [self.mcmc_result.maxprob_params[p].value for p in self.mcmc_result.maxprob_params if
                                self.mcmc_result.maxprob_params[p].vary]
            except AttributeError:
                raise AssertionError("To plot maximum likelihood truth values, need to run process_mcmc first")

        corner.corner(self.mcmc_result.flatchain, labels=self.mcmc_result.var_names, truths=truth_values)

    def steppar(self, par, steps):
        if self.fit_params is not None:
            step_params = copy.copy(self.fit_params)
        else:
            step_params = copy.copy(self.params)

        step_params[par].vary = False

        print()
        print('%16s  %16s' % (par, "-log(likelihood)"))
        print('=' * 34)

        fit_stats = []
        for val in steps:
            step_params[par].value = val
            stat, _ = self._dofit(step_params)
            fit_stats.append(stat)
            print('%16g  %16g' % (val, stat))

        print()

        fit_stats = np.array(fit_stats)

        return steps, fit_stats

    def get_psd(self, freq=None):
        if self.fit_result is None:
            raise AssertionError("Need to run fit first!")
        if freq is None:
            freq = self.cov_matrix.freq_arr[self.cov_matrix.freq_arr > 1e-10]
        return self.cov_matrix.cov_model.get_psd_series(self.fit_result.params, freq)


class MLCrossCovariance(MLCovariance):
    def __init__(self, lc1, lc2, autocov_model, crosscov_model, noise1='mean_error', noise2='mean_error', zero_mean=True, params=None, **kwargs):
        if noise1 == 'error':
            noise1 = lc1.error**2
        elif noise1 == 'mean_error':
            noise1 = np.mean(lc1.rate) / lc1.dt
        if noise2 == 'error':
            noise2 = lc2.error**2
        elif noise2 == 'mean_error':
            noise2 = np.mean(lc2.rate) / lc2.dt

        self.cov_matrix = CrossCovarianceMatrixModel(autocov_model, crosscov_model, lc1.time, lc2.time, noise1=noise1, noise2=noise2, **kwargs)

        if zero_mean:
            self.data = np.hstack([lc1.rate - np.mean(lc1.rate), lc2.rate - np.mean(lc2.rate)])
        else:
            self.data = np.hstack([lc1.rate, lc2.rate])

        if isinstance(params, lmfit.Parameters):
            self.params = params
        elif isinstance(params, dict):
            self.params = self.cov_matrix.get_params(**params)
        else:
            self.params = self.cov_matrix.get_params()

        self.minimizer = None
        self.fit_result = None


class StackedMLCovariance(MLCovariance):
    def __init__(self, lclist, autocov_model, noise='mean_error', params=None, **kwargs):
        if isinstance(params, lmfit.Parameters):
            self.params = params
        else:
            # this is just a prototype covariance matrix for getting the parameter list
            n1 = 'param' if noise == 'param' else 1.
            cov_matrix = CovarianceMatrixModel(autocov_model, lclist[0].time, noise=n1, **kwargs)
            self.params = cov_matrix.get_params(**params) if isinstance(params, dict) else cov_matrix.get_params()

        # we construct a separate MLCovariance object for each pair of light curves
        # we fit each one with its own covariance matrix with the same parameters
        self.ml_covariance = [MLCovariance(lc, autocov_model, params=self.params, noise=noise, **kwargs) for lc in lclist]

        self.minimizer = None
        self.fit_result = None

    def log_likelihood(self, params, eval_gradient=False, delta=1e-3):
        # the likelihood is the product of the likelihood for the individual light curve pairs
        # so the log-likelihood is the sum
        if eval_gradient:
            like_arr = np.array([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance])
            # separate and sum the likelihoods and the gradients
            return np.sum(like_arr[:,0]), np.array([grad for grad in like_arr[:,1]]).sum(axis=0)
        else:
            return np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_covariance])


class StackedMLCrossCovariance(MLCrossCovariance):
    def __init__(self, lc1list, lc2list, autocov_model, crosscov_model, noise1='mean_error', noise2='mean_error', params=None, **kwargs):
        if isinstance(params, lmfit.Parameters):
            self.params = params
        else:
            # this is just a prototype covariance matrix for getting the parameter list
            n1 = 'param' if noise1 == 'param' else 1.
            n2 = 'param' if noise2 == 'param' else 1.
            cov_matrix = CrossCovarianceMatrixModel(autocov_model, crosscov_model, lc1list[0].time, lc2list[0].time,
                                                    noise1=n1, noise2=n2, **kwargs)
            self.params = cov_matrix.get_params(**params) if isinstance(params, dict) else cov_matrix.get_params()

        # we construct a separate MLCrossCovariance object for each pair of light curves
        # we fit each one with its own covariance matrix with the same parameters
        self.ml_cross_covariance = [MLCrossCovariance(lc1, lc2, autocov_model, crosscov_model, noise1=noise1,
                                                      noise2=noise2, params=self.params, **kwargs)
                                    for lc1, lc2 in zip(lc1list, lc2list)]

        self.minimizer = None
        self.fit_result = None

    def log_likelihood(self, params, eval_gradient=False, delta=1e-3):
        # the likelihood is the product of the likelihood for the individual light curve pairs
        # so the log-likelihood is the sum
        if eval_gradient:
            like_arr = np.array([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_cross_covariance])
            return np.sum(like_arr[:, 0]), np.array([grad for grad in like_arr[:, 1]]).sum(axis=0)
        else:
            return np.sum([mlc.log_likelihood(params, eval_gradient, delta) for mlc in self.ml_cross_covariance])
