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

class GPLightCurve(LightCurve):
    def __init__(self, filename=None, t=[], r=[], e=[], lc=None, zero_nan=True, kernel=None, n_restarts_optimizer=9, run_fit=True):
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
            #self.kernel = C(1.0, (1e-3,1e3)) * RBF(10, (np.min(np.diff(self.time)),1e10))
            #self.kernel = RBF(1, (1e-5, 1e10))
            self.kernel = k = C(1.0, (1e-3,1e3)) * RationalQuadratic()

        self.gp_regressor = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y=True)

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
        return LightCurve(t=t, r=r, e=e)

    def sample(self, n_samples=1, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        t_samples = np.atleast_2d(t).T
        r = self.gp_regressor.sample_y(t_samples, n_samples=n_samples)
        e = np.zeros(t.shape)
        if n_samples == 1:
            return LightCurve(t=t, r=r.ravel(), e=e)
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
            freq = sample_lcs[0].ftfreq()
            freq_error = None
            per = np.array([Periodogram(lc).bin(bins).periodogram for lc in sample_lcs])
        else:
            freq = bins.bin_cent
            freq_error = bins.bin_end - bins.bin_cent
            per = np.array([Periodogram(lc).periodogram for lc in sample_lcs])

        per_avg = np.mean(per, axis=0)
        per_std = np.std(per, axis=0)

        return freq, freq_error, per_avg, per_std