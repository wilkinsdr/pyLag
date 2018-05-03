"""
pylag.gaussian_process

Provides pyLag functionality for fitting light curves using gaussian processes

Classes
-------


v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C

from .lightcurve import *
from .periodogram import *
from .cross_spectrum import *
from .lag_frequency_spectrum import *
from .lag_energy_spectrum import *


class GPLightCurve(LightCurve):
    def __init__(self, filename=None, t=[], r=[], e=[], lc=None, zero_nan=True, kernel=None, n_restarts_optimizer=9, run_fit=True, use_errors=True, noise_kernel=False, lognorm=False):
        if lc is not None:
            t = lc.time
            r = lc.rate
            e = lc.error
        LightCurve.__init__(self, filename, t, r, e, interp_gaps=False, zero_nan=zero_nan, trim=False)

        # need to remove the gaps from the light curve
        # (only store the non-zero time bins)
        self.remove_gaps(to_self=True)

        self.mean_rate = self.mean()

        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic()
            if noise_kernel:
                noise_level = np.sqrt(self.mean_rate * self.dt) / (self.mean_rate * self.dt)
                self.kernel += WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-10, 1e+1))

        if noise_kernel or not use_errors:
            self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=True)
        else:
            alpha = (self.error / self.rate)**2
            self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer,
                                                         normalize_y=True, alpha=alpha)

        self.lognorm = lognorm
        if self.lognorm:
            self.error = self.error / self.rate
            self.rate = np.log(self.rate)

        if run_fit:
            self.fit()

    def fit(self):
        t = np.atleast_2d(self.time).T
        self.gp_regressor.fit(t, self.rate)

    def predict(self, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        r, e = self.gp_regressor.predict(t_samples, return_std=True)
        if self.lognorm:
            r = np.exp(r)
            e = r * e
        return LightCurve(t=t, r=r, e=e)

    def sample(self, n_samples=1, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        r = self.gp_regressor.sample_y(t_samples, n_samples=n_samples, random_state=None)
        e = np.zeros(t.shape)
        if n_samples == 1:
            if self.lognorm:
                return LightCurve(t=t, r=np.exp(r.ravel()), e=e)
            else:
                return LightCurve(t=t, r=r.ravel(), e=e)
        else:
            if self.lognorm:
                return [LightCurve(t=t, r=np.exp(r[:, n]), e=e) for n in range(n_samples)]
            else:
                return [LightCurve(t=t, r=r[:,n], e=e) for n in range(n_samples)]


class GPPeriodogram(Periodogram):
    def __init__(self, lc=None, gplc=None, n_samples=10, bins=None):
        if gplc is not None:
            self.gplc = gplc
        elif lc is not None:
            self.gplc = GPLightCurve(lc=lc)
            self.gplc.fit()
        else:
            raise ValueError("GPPeriodogram requires a light curve!")

        freq, freq_error, per, err = self.calculate(n_samples, bins)
        Periodogram.__init__(self, f=freq, per=per, err=err, ferr=freq_error)

    def calculate(self, n_samples=10, bins=None):
        sample_lcs = self.gplc.sample(t=None, n_samples=n_samples)
        if not isinstance(sample_lcs, list):
            sample_lcs = [sample_lcs]

        if bins is not None:
            freq = bins.bin_cent
            freq_error = bins.bin_end - bins.bin_cent
            per = np.array([Periodogram(lc).bin(bins).periodogram for lc in sample_lcs])
        else:
            freq = sample_lcs[0].ftfreq()
            freq_error = None
            per = np.array([Periodogram(lc).periodogram for lc in sample_lcs])

        per_avg = np.mean(per, axis=0)
        per_std = np.std(per, axis=0)

        return freq, freq_error, per_avg, per_std


class GPEnergyLCList(EnergyLCList):
    def __init__(self, searchstr=None, lcfiles=None, enlclist=None, enmin=None, enmax=None, lclist=None, **kwargs):
        if enlclist is not None:
            self.enmin = enlclist.enmin
            self.enmax = enlclist.enmax
            lclist = enlclist.lclist
        elif lclist is not None and enmin is not None and enmax is not None:
            self.enmin = np.array(enmin)
            self.enmax = np.array(enmax)
        elif lclist is None:
            self.enmin, self.enmax, lclist = self.find_light_curves(searchstr, lcfiles, **kwargs)

        self.en = 0.5*(self.enmin + self.enmax)
        self.en_error = self.en - self.enmin

        self.lclist = []

        if isinstance(lclist[0], list):
            for en_lclist in lclist:
                self.lclist.append([])
                for lc in en_lclist:
                    self.lclist[-1].append(GPLightCurve(lc=lc, **kwargs))

        elif isinstance(lclist[0], LightCurve):
            for lc in lclist:
                self.lclist.append(GPLightCurve(lc=lc, **kwargs))

    def fit(self):
        if isinstance(lclist[0], list):
            for en_lclist in self.lclist:
                for lc in en_lclist:
                    lc.fit()

        elif isinstance(lclist[0], LightCurve):
            for lc in self.lclist:
                lc.fit()

    def sample(self, n_samples=1, t=None):
        if n_samples == 1:
            sample_lclist = []
            if isinstance(self.lclist[0], list):
                for en_lclist in self.lclist:
                    sample_lclist.append([])
                    for lc in en_lclist:
                        sample_lclist[-1].append(lc.sample(n_samples=1, t=t))

            elif isinstance(self.lclist[0], LightCurve):
                for lc in self.lclist:
                    sample_lclist.append(lc.sample(n_samples=1, t=t))

            return EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=sample_lclist)

        elif n_samples > 1:
            sample_lclist = []
            if isinstance(self.lclist[0], list):
                for en_lclist in self.lclist:
                    sample_lclist.append([])
                    for lc in en_lclist:
                        sample_lclist[-1].append(lc.sample(n_samples=1, t=t))

            elif isinstance(self.lclist[0], LightCurve):
                for lc in self.lclist:
                    sample_lclist.append(lc.sample(n_samples=1, t=t))

            sample_lclist = lclist_separate_segments(sample_lclist)

            return [EnergyLCList(enmin=self.enmin, enmax=self.enmax, lclist=lclist) for lclist in sample_lclist]


class GPLagFrequencySpectrum(LagFrequencySpectrum):
    def __init__(self, bins, lc1=None, lc2=None, gplc1=None, gplc2=None, n_samples=10, low_mem=False):
        if gplc1 is not None:
            self.gplc1 = gplc1
        elif lc1 is not None:
            self.gplc1 = GPLightCurve(lc=lc1, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a pair of light curves!")

        if gplc2 is not None:
            self.gplc2 = gplc2
        elif lc2 is not None:
            self.gplc2 = GPLightCurve(lc=lc2, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a pair of light curves!")

        if low_mem:
            self.freq, self.freq_error, self.lag, self.error = self.calculate_seq(bins, n_samples)
        else:
            self.freq, self.freq_error, self.lag, self.error = self.calculate_batch(bins, n_samples)

    def calculate_batch(self, bins, n_samples=10):
        sample_lcs1 = self.gplc1.sample(t=None, n_samples=n_samples)
        sample_lcs2 = self.gplc2.sample(t=None, n_samples=n_samples)
        if not isinstance(sample_lcs1, list):
            sample_lcs1 = [sample_lcs1]
        if not isinstance(sample_lcs2, list):
            sample_lcs2 = [sample_lcs2]

        freq = bins.bin_cent
        freq_error = bins.bin_end - bins.bin_cent
        lag = np.array([LagFrequencySpectrum(bins, lc1, lc2, calc_error=False).lag for (lc1, lc2) in zip(sample_lcs1, sample_lcs2)])

        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        return freq, freq_error, lag_avg, lag_std

    def calculate_seq(self, bins, n_samples=10):
        freq = bins.bin_cent
        freq_error = bins.bin_end - bins.bin_cent

        lag = []
        for n in range(n_samples):
            sample_lc1 = self.gplc1.sample(t=None, n_samples=1)
            sample_lc2 = self.gplc2.sample(t=None, n_samples=1)
            lag.append(LagFrequencySpectrum(bins, sample_lc1, sample_lc2, calc_error=False).lag)

        lag = np.array(lag)
        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        return freq, freq_error, lag_avg, lag_std


class GPLagEnergySpectrum(LagEnergySpectrum):
    def __init__(self, fmin, fmax, lclist=None, gplclist=None, n_samples=10):
        if gplclist is not None:
            self.gplclist = gplclist
        elif lclist is not None:
            self.gplclist = GPEnergyLCList(enlclist=lclist, run_fit=True)
        else:
            raise ValueError("GPLagFrequencySpectrum requires a light curve list!")

        self.en = np.array(lclist.en)
        self.en_error = np.array(lclist.en_error)

        self.lag, self.error = self.calculate(fmin, fmax, n_samples)

    def calculate(self, fmin, fmax, n_samples=10):
        sample_lclists = self.gplclist.sample(t=None, n_samples=n_samples)
        if not isinstance(sample_lclists, list):
            sample_lclists = [sample_lclists]

        lag = np.array([LagEnergySpectrum(lclist).lag for lclist in sample_lclists])

        lag_avg = np.mean(lag, axis=0)
        lag_std = np.std(lag, axis=0)

        return lag_avg, lag_std