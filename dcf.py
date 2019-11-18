"""
pylag.dcf

Provides pyLag class for calculating the discrete correlation function
defined in Edelson & Krolik 1988

Classes
-------
DCF : Calculate the discrete correlation function between two light curves

v1.0 02/08/2019 - D.R. Wilkins
"""
from .lightcurve import *
from .binning import *

import numpy as np

class DCF(object):
    def __init__(self, lc1, lc2, bins=50):

        ulags, udcf = self.calculate_udcf(lc1, lc2)

        if isinstance(bins, int):
            bins = LinearBinning(np.min(np.abs(ulags)), np.max(ulags), bins)

        self.lag = bins.bin_cent
        self.lag_error = bins.x_error()

        self.dcf, self.error = self.bin_dcf(ulags, udcf, bins)

    @staticmethod
    def calculate_udcf(lc1, lc2, err1=None, err2=None):
        t1, t2 = np.meshgrid(lc1.time, lc2.time)
        lags = t2 - t1

        if err1 is None:
            err1 = np.sqrt(np.mean(lc1.rate) / lc1.dt)
        if err2 is None:
            err2 = np.sqrt(np.mean(lc2.rate) / lc2.dt)

        r1, r2 = np.meshgrid(lc1.rate, lc2.rate)
        udcf = (r1 - np.mean(lc1.rate)) * (r2 - np.mean(lc2.rate)) \
            / np.sqrt((np.std(lc1.rate)**2 - err1**2)*(np.std(lc2.rate)**2 - err2**2))

        return lags, udcf

    @staticmethod
    def bin_dcf(ulags, udcf, bins):
        dcf = bins.bin(ulags.flatten(), udcf.flatten())

        bin_num_points = bins.num_points_in_bins(ulags.flatten())
        err = (np.sqrt(bin_num_points) / (bin_num_points - 1.)) * bins.std(ulags.flatten(), udcf.flatten())

        return dcf, err

    def lag_peak(self):
        return self.lag[np.argmax(self.dcf)]

    def _getplotdata(self):
        return (self.lag, self.lag_error), (self.dcf, self.error)

    def _getplotaxes(self):
        return 'Lag / s', 'linear', 'DCF', 'linear'


class StackedDCF(DCF):
    def __init__(self, lc1list, lc2list,bins=50):
        ulags = []
        udcf = []
        for lc1, lc2 in zip(lc1list, lc2list):
            l, u = self.calculate_udcf(lc1, lc2)
            ulags.append(l.flatten())
            udcf.append(u.flatten())

        ulags = np.hstack(ulags)
        udcf = np.hstack(udcf)

        ulags = ulags[np.isfinite(udcf)]
        udcf = udcf[np.isfinite(udcf)]

        if isinstance(bins, int):
            bins = LinearBinning(np.min(np.abs(ulags)), np.max(ulags), bins)

        self.lag = bins.bin_cent
        self.lag_error = bins.x_error()

        self.dcf, self.error = self.bin_dcf(ulags, udcf, bins)
