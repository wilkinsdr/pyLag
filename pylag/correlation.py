from .lightcurve import *

import numpy as np
from scipy.signal import correlate


class Correlation(object):
    def __init__(self, lc1=None, lc2=None, lags=None, corr=None):
        if lags is not None and corr is not None:
            self.lags = np.array(lags)
            self.corr = np.array(corr)

        if lc1 is not None:
            if lc2 is None:
                lc2 = lc1
            self.lags, self.corr, self.error = self.calculate(lc1.rate, lc2.rate, lc1.time)

    @staticmethod
    def calculate(lc1, lc2, time=None):
        corr = correlate(lc1, lc2)
        lags = (np.arange(corr.size) - (lc1.size - 1)).astype(float)
        if time is not None:
            lags *= (time[1] - time[0])

        return lags, corr, None

    def _getplotdata(self):
        return self.lags, self.corr

    def _getplotaxes(self):
        return 'Lag / s', 'linear', 'Correlation', 'linear'


class StackedCorrelation(Correlation):
    def __init__(self, lc1list, lc2list, bins=None):
        self.lags = bins.bin_cent
        self.lag_err = bins.x_error()

        corr_obj_list = [Correlation(lc1, lc2) for lc1, lc2 in zip(lc1list, lc2list)]
        lag_list = np.hstack([c.lags for c in corr_obj_list])
        corr_list = np.hstack([c.corr for c in corr_obj_list])
        self.corr = bins.bin(lag_list, corr_list)
        self.error = bins.std(lag_list, corr_list)

    def _getplotdata(self):
        return (self.lags, self.lag_err), (self.corr, self.error)
