"""
pylag.gaussian_process

Provides pyLag functionality for fitting light curves using gaussian processes

Classes
-------


v1.0 09/03/2017 - D.R. Wilkins
"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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
        self.remove_gaps()

        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = C(1.0, (1e-3,1e3)) * RBF(10, (1e-2,1e2))

        self.gp_regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)

        if run_fit:
            self.fit()

    def fit(self):
        t = np.atleast_2d(self.time).T
        self.gp_regressor.fit(t, self.rate)

    def predict(self, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        r, e = self.gp_regressor.predict(t, return_std=True)
        return LightCurve(t=t, r=r, e=e)

    def sample(self, t=None):
        if t is None:
            t = np.arange(self.time.min(), self.time.max(), np.min(np.diff(self.time)))
        r = self.gp_regressor.sample_y(t)
        e = np.zeros(r.shape)
        return LightCurve(t=t, r=r, e=e)


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
        per = []
        freq = None
        for n in range(n_samples):
            sample_per = Periodogram(self.gplc.sample())
            if bins is not None:
                sample_per = sample_per.bin(bins, calc_error=False)
            per.append(sample_per.periodogram)
            if n == 0:
                freq = sample_per.freq
                freq_error = sample_per.freq_error

        per_avg = np.mean(np.array(per), axis=0)
        per_std = np.std(np.array(per), axis=0)

        return freq, freq_error, per_avg, per_std